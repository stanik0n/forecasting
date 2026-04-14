"""
Layer 4 (Phase 2): LightGBM gradient boosting model
Trains on the feature-engineered data using a time-based split,
generates 28-day ahead forecasts using a recursive strategy,
and persists the model artifact.
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features.feature_engineering import get_feature_columns
from src.forecasting.baseline_models import time_split, evaluate_forecast


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_lgbm(
    train: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor on the training set."""
    available = [c for c in feature_cols if c in train.columns]
    X_train = train[available]
    y_train = train["units_sold"]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    return model


def direct_forecast(
    model: lgb.LGBMRegressor,
    fact: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    forecast_dates: pd.DatetimeIndex,
    dim_calendar: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Direct (non-recursive) forecast strategy:
    For each forecast date, use actual lag values from history up to train_cutoff.
    This avoids error compounding from recursive forecasting.
    Lags of 7, 14, 28 days refer to actual history, which we have up to train_cutoff.
    We only forecast dates where all required lags are available from the training set.
    """
    dim_calendar = dim_calendar.copy()
    dim_calendar["date"] = pd.to_datetime(dim_calendar["date"])

    # Calendar features for forecast dates
    cal_cols = [
        "date", "weekday", "month", "year", "quarter",
        "is_weekend", "is_month_end", "is_month_start", "has_event",
        "is_cultural_event", "is_national_event", "is_religious_event", "is_sporting_event",
    ]
    snap_cols = [c for c in dim_calendar.columns if c.startswith("snap_")]
    cal_cols += snap_cols

    # Prepare event dummies if not already present
    if "is_cultural_event" not in dim_calendar.columns:
        dim_calendar["is_cultural_event"] = (dim_calendar.get("event_type_1") == "Cultural").astype(int)
        dim_calendar["is_national_event"] = (dim_calendar.get("event_type_1") == "National").astype(int)
        dim_calendar["is_religious_event"] = (dim_calendar.get("event_type_1") == "Religious").astype(int)
        dim_calendar["is_sporting_event"] = (dim_calendar.get("event_type_1") == "Sporting").astype(int)

    cal_for_forecast = dim_calendar[[c for c in cal_cols if c in dim_calendar.columns]].copy()
    cal_for_forecast = cal_for_forecast[cal_for_forecast["date"].isin(forecast_dates)]

    # All unique item-store pairs
    pairs = fact[["item_id", "store_id", "state_id", "cat_id", "dept_id",
                   "item_id_enc", "store_id_enc", "cat_id_enc", "dept_id_enc", "state_id_enc"]].drop_duplicates()

    rows = []
    for fd in forecast_dates:
        for _, pair in pairs.iterrows():
            item_id = pair["item_id"]
            store_id = pair["store_id"]

            series = fact[(fact["item_id"] == item_id) & (fact["store_id"] == store_id)].sort_values("date")
            history = series[series["date"] <= train_cutoff]

            if history.empty:
                continue

            row = {
                "item_id": item_id,
                "store_id": store_id,
                "forecast_date": fd,
            }

            # Lag features from actual history
            for lag in [7, 14, 28]:
                lag_date = fd - pd.Timedelta(days=lag)
                match = history[history["date"] == lag_date]
                row[f"lag_{lag}"] = match["units_sold"].values[0] if not match.empty else history["units_sold"].mean()

            # Rolling features: use history up to train_cutoff
            recent_vals = history["units_sold"].values
            for w in [7, 28, 56]:
                window_vals = recent_vals[-w:] if len(recent_vals) >= w else recent_vals
                row[f"rolling_mean_{w}d"] = float(np.mean(window_vals)) if len(window_vals) > 0 else 0.0
                row[f"rolling_std_{w}d"] = float(np.std(window_vals)) if len(window_vals) > 1 else 0.0

            # Price: use last known price
            last_price_row = history.dropna(subset=["sell_price"]).tail(1)
            row["sell_price"] = last_price_row["sell_price"].values[0] if not last_price_row.empty else 0.0
            row["price_lag1"] = row["sell_price"]
            row["price_change"] = 0.0
            row["price_rolling_mean_4wk"] = row["sell_price"]
            row["price_discount_proxy"] = 1.0

            # ID encodings
            for enc_col in ["item_id_enc", "store_id_enc", "cat_id_enc", "dept_id_enc", "state_id_enc"]:
                row[enc_col] = pair[enc_col]

            rows.append(row)

    forecast_frame = pd.DataFrame(rows)

    if forecast_frame.empty:
        return pd.DataFrame()

    # Join calendar features
    forecast_frame = forecast_frame.merge(
        cal_for_forecast.rename(columns={"date": "forecast_date"}),
        on="forecast_date",
        how="left",
    )

    available_features = [c for c in feature_cols if c in forecast_frame.columns]
    X_pred = forecast_frame[available_features].fillna(0)
    forecast_frame["predicted_units"] = model.predict(X_pred).clip(min=0)
    forecast_frame["model_name"] = "lgbm"
    forecast_frame["train_cutoff"] = train_cutoff

    return forecast_frame[["item_id", "store_id", "forecast_date", "predicted_units", "model_name", "train_cutoff"]]


def run(config_path: str = "config/settings.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_data"]
    tables_dir = cfg["paths"]["outputs_tables"]
    models_dir = cfg["paths"]["outputs_models"]
    horizon_days = cfg["forecast"]["horizon_days"]
    validation_days = cfg["forecast"]["validation_days"]
    lgbm_params = cfg["model"]["lgbm"]
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=== LightGBM Model ===")

    fact = pd.read_parquet(os.path.join(processed_dir, "fact_features.parquet"))
    dim_calendar = pd.read_parquet(os.path.join(processed_dir, "dim_calendar.parquet"))
    fact["date"] = pd.to_datetime(fact["date"])
    dim_calendar["date"] = pd.to_datetime(dim_calendar["date"])

    feature_cols = get_feature_columns(cfg)
    # Add SNAP columns dynamically
    snap_cols = [c for c in fact.columns if c.startswith("snap_")]
    feature_cols += [c for c in snap_cols if c not in feature_cols]

    train, val, val_start, val_end = time_split(fact, validation_days, horizon_days)
    train_cutoff = train["date"].max()
    forecast_dates = pd.date_range(val_start, val_end, freq="D")

    print(f"  Train rows   : {len(train):,}")
    print(f"  Val rows     : {len(val):,}")
    print(f"  Features     : {len([c for c in feature_cols if c in train.columns])}")
    print(f"  Train cutoff : {train_cutoff.date()}")
    print(f"  Val window   : {val_start.date()} → {val_end.date()}")

    print("\n  Training LightGBM...")
    model = train_lgbm(train, feature_cols, lgbm_params)

    # Evaluate on validation set (in-sample feature rows available)
    print("\n  Generating validation forecasts...")
    available = [c for c in feature_cols if c in val.columns]
    val_pred = val[available].fillna(0)
    val["predicted_units"] = model.predict(val_pred).clip(min=0)

    val_forecast_df = val[["item_id", "store_id", "date", "predicted_units"]].rename(
        columns={"date": "forecast_date"}
    )
    val_forecast_df["model_name"] = "lgbm"
    val_forecast_df["train_cutoff"] = train_cutoff

    metrics = evaluate_forecast(val_forecast_df, val, "lgbm")
    print("\n  Validation metrics:")
    print(metrics.to_string(index=False))

    # Save model artifact
    model_path = os.path.join(models_dir, "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"\n  Model saved to: {model_path}")

    # Feature importances
    available_cols = [c for c in feature_cols if c in train.columns]
    importance_df = pd.DataFrame({
        "feature": available_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(os.path.join(tables_dir, "lgbm_feature_importance.csv"), index=False)

    # Save forecast + metrics
    val_forecast_df.to_parquet(os.path.join(processed_dir, "fact_forecast_lgbm.parquet"), index=False)
    metrics.to_csv(os.path.join(tables_dir, "lgbm_metrics.csv"), index=False)

    print("  Wrote lgbm forecast and metrics.")

    return val_forecast_df, metrics


if __name__ == "__main__":
    run()
