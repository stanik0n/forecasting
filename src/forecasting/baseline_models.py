"""
Layer 4 (Phase 1): Baseline forecasting models
- Moving Average: predict the next N days using the trailing mean
- Seasonal Naive: predict using the value from the same weekday N weeks ago

Both models operate on the fact_features frame and produce a fact_forecast-shaped output.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def time_split(df: pd.DataFrame, validation_days: int, horizon_days: int):
    """
    Split into train and validation sets by time.
    - train: everything before validation window
    - val:   last `validation_days` days before the forecast horizon
    The final `horizon_days` are left untouched as the test horizon.
    """
    max_date = df["date"].max()
    val_end = max_date - pd.Timedelta(days=horizon_days)
    val_start = val_end - pd.Timedelta(days=validation_days - 1)
    train_end = val_start - pd.Timedelta(days=1)

    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]

    return train, val, val_start, val_end


def moving_average_forecast(
    train: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    window: int = 28,
) -> pd.DataFrame:
    """
    For each item-store, compute the trailing `window`-day mean from the training
    data and repeat it as a flat forecast for each forecast date.
    """
    # Mean of last `window` days per item-store
    cutoff = train["date"].max()
    recent = train[train["date"] > cutoff - pd.Timedelta(days=window)]
    avg_demand = (
        recent.groupby(["item_id", "store_id"])["units_sold"]
        .mean()
        .reset_index()
        .rename(columns={"units_sold": "predicted_units"})
    )
    avg_demand["predicted_units"] = avg_demand["predicted_units"].clip(lower=0)

    # Cross-join item-store pairs with each forecast date
    date_df = pd.DataFrame({"forecast_date": forecast_dates})
    forecasts = avg_demand.merge(date_df, how="cross")
    forecasts["model_name"] = "moving_average"
    forecasts["train_cutoff"] = cutoff

    return forecasts[["item_id", "store_id", "forecast_date", "predicted_units", "model_name", "train_cutoff"]]


def seasonal_naive_forecast(
    train: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    season_weeks: int = 1,
) -> pd.DataFrame:
    """
    For each item-store and each forecast date, predict using the observed demand
    from the same weekday `season_weeks` weeks ago.
    Falls back to the item-store mean if no historical match exists.
    """
    cutoff = train["date"].max()
    train = train.copy()
    train["weekday"] = train["date"].dt.dayofweek

    # Build a lookup: item_id, store_id, weekday → mean of that weekday in recent history
    lookback_days = season_weeks * 7 * 4  # last 4 seasonal cycles
    recent = train[train["date"] > cutoff - pd.Timedelta(days=lookback_days)]
    weekday_avg = (
        recent.groupby(["item_id", "store_id", "weekday"])["units_sold"]
        .mean()
        .reset_index()
        .rename(columns={"units_sold": "predicted_units"})
    )

    date_df = pd.DataFrame({
        "forecast_date": forecast_dates,
        "weekday": [d.dayofweek for d in forecast_dates],
    })

    forecasts = weekday_avg.merge(date_df, on="weekday", how="right")
    # Fill NaN with overall item-store mean
    overall_mean = (
        train.groupby(["item_id", "store_id"])["units_sold"]
        .mean()
        .reset_index()
        .rename(columns={"units_sold": "mean_fallback"})
    )
    forecasts = forecasts.merge(overall_mean, on=["item_id", "store_id"], how="left")
    forecasts["predicted_units"] = forecasts["predicted_units"].fillna(
        forecasts["mean_fallback"]
    ).clip(lower=0)
    forecasts.drop(columns=["mean_fallback", "weekday"], inplace=True)

    forecasts["model_name"] = "seasonal_naive"
    forecasts["train_cutoff"] = cutoff

    return forecasts[["item_id", "store_id", "forecast_date", "predicted_units", "model_name", "train_cutoff"]]


def evaluate_forecast(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Compute per-series RMSE, MAE, and MAPE. Returns a summary DataFrame."""
    merged = forecast_df.merge(
        actuals_df[["item_id", "store_id", "date", "units_sold"]].rename(columns={"date": "forecast_date"}),
        on=["item_id", "store_id", "forecast_date"],
        how="inner",
    )

    if merged.empty:
        print(f"  WARNING: no matching actuals found for {model_name}")
        return pd.DataFrame()

    merged["error"] = merged["predicted_units"] - merged["units_sold"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2
    # MAPE: avoid division by zero
    nonzero = merged[merged["units_sold"] > 0].copy()
    nonzero["pct_error"] = (nonzero["abs_error"] / nonzero["units_sold"]) * 100

    metrics = {
        "model_name": model_name,
        "rmse": np.sqrt(merged["sq_error"].mean()),
        "mae": merged["abs_error"].mean(),
        "mape": nonzero["pct_error"].mean() if len(nonzero) > 0 else np.nan,
        "n_obs": len(merged),
    }
    return pd.DataFrame([metrics])


def run(config_path: str = "config/settings.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_data"]
    tables_dir = cfg["paths"]["outputs_tables"]
    horizon_days = cfg["forecast"]["horizon_days"]
    validation_days = cfg["forecast"]["validation_days"]
    os.makedirs(tables_dir, exist_ok=True)

    print("=== Baseline Models ===")

    fact = pd.read_parquet(os.path.join(processed_dir, "fact_features.parquet"))
    fact["date"] = pd.to_datetime(fact["date"])

    train, val, val_start, val_end = time_split(fact, validation_days, horizon_days)
    forecast_dates = pd.date_range(val_start, val_end, freq="D")

    print(f"  Train cutoff : {train['date'].max().date()}")
    print(f"  Val window   : {val_start.date()} → {val_end.date()} ({len(forecast_dates)} days)")

    # --- Moving Average ---
    print("  Running moving average...")
    ma_forecast = moving_average_forecast(train, forecast_dates, window=28)
    ma_metrics = evaluate_forecast(ma_forecast, val, "moving_average")

    # --- Seasonal Naive ---
    print("  Running seasonal naive...")
    sn_forecast = seasonal_naive_forecast(train, forecast_dates)
    sn_metrics = evaluate_forecast(sn_forecast, val, "seasonal_naive")

    # Combine forecasts and metrics
    all_forecasts = pd.concat([ma_forecast, sn_forecast], ignore_index=True)
    all_metrics = pd.concat([ma_metrics, sn_metrics], ignore_index=True)

    print("\n  Validation metrics:")
    print(all_metrics.to_string(index=False))

    all_forecasts.to_parquet(os.path.join(processed_dir, "fact_forecast_baselines.parquet"), index=False)
    all_metrics.to_csv(os.path.join(tables_dir, "baseline_metrics.csv"), index=False)

    print(f"\nWrote baseline forecasts and metrics.")

    return all_forecasts, all_metrics


if __name__ == "__main__":
    run()
