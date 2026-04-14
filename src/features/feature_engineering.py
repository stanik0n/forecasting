"""
Layer 3: Feature engineering
Generate lag, rolling, price, and calendar features with leakage-safe logic.
All features are computed per item-store group sorted by date.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    """
    Lag features: units_sold shifted by N days per item-store group.
    Leakage-safe: shift(N) means we know the value N days ago.
    """
    for lag in lag_days:
        df[f"lag_{lag}"] = df.groupby(["item_id", "store_id"])["units_sold"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Rolling mean, std, and sum per item-store group.
    Uses a minimum shift of 1 so the current day is never included (leakage-safe).
    """
    grp = df.groupby(["item_id", "store_id"])["units_sold"]
    for window in windows:
        shifted = grp.shift(1)
        df[f"rolling_mean_{window}d"] = (
            shifted.transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}d"] = (
            shifted.transform(lambda x: x.rolling(window, min_periods=2).std().fillna(0))
        )
    # 7-day sum: total units sold in the past week — strong weekly demand signal
    df["units_sold_7d_sum"] = (
        grp.shift(1).transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    return df


def add_calendar_features(df: pd.DataFrame, dim_calendar: pd.DataFrame) -> pd.DataFrame:
    """Join calendar-derived features onto the sales frame."""
    cal_cols = [
        "date", "weekday", "month", "year", "quarter",
        "is_weekend", "is_month_end", "is_month_start", "has_event",
        "event_name_1", "event_type_1",
    ]
    # Add SNAP columns dynamically
    snap_cols = [c for c in dim_calendar.columns if c.startswith("snap_")]
    cal_cols += snap_cols

    cal_subset = dim_calendar[cal_cols].drop_duplicates("date")
    df = df.merge(cal_subset, on="date", how="left")

    # Encode event_type_1 as dummy rather than string
    df["is_cultural_event"] = (df["event_type_1"] == "Cultural").astype(int)
    df["is_national_event"] = (df["event_type_1"] == "National").astype(int)
    df["is_religious_event"] = (df["event_type_1"] == "Religious").astype(int)
    df["is_sporting_event"] = (df["event_type_1"] == "Sporting").astype(int)

    df.drop(columns=["event_name_1", "event_type_1"], inplace=True)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price features per item-store:
      - price_lag1: price yesterday (captures recent price change)
      - price_change: sell_price - price_lag1
      - price_rolling_mean_4wk: 4-week rolling average price
      - price_discount_proxy: ratio of current price to 4-week rolling mean
    """
    grp = df.groupby(["item_id", "store_id"])["sell_price"]

    df["price_lag1"] = grp.shift(1)
    df["price_change"] = df["sell_price"] - df["price_lag1"]

    df["price_rolling_mean_4wk"] = grp.transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).mean()
    )
    # discount proxy: < 1 means on sale vs recent average
    df["price_discount_proxy"] = df["sell_price"] / df["price_rolling_mean_4wk"].replace(0, np.nan)

    df["sell_price"] = df["sell_price"].fillna(0)
    df["price_lag1"] = df["price_lag1"].fillna(0)
    df["price_change"] = df["price_change"].fillna(0)
    df["price_rolling_mean_4wk"] = df["price_rolling_mean_4wk"].fillna(0)
    df["price_discount_proxy"] = df["price_discount_proxy"].fillna(1.0)

    return df


def add_id_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical ID columns for tree models."""
    for col in ["item_id", "store_id", "state_id", "cat_id", "dept_id"]:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes
    return df


def get_feature_columns(cfg: dict) -> list[str]:
    """Return the ordered list of model input feature columns."""
    lag_days = cfg["forecast"]["lag_days"]
    rolling_windows = cfg["forecast"]["rolling_windows"]

    features = []
    # Lag features
    features += [f"lag_{d}" for d in lag_days]
    # Rolling features
    for w in rolling_windows:
        features += [f"rolling_mean_{w}d", f"rolling_std_{w}d"]
    # 7-day demand sum
    features += ["units_sold_7d_sum"]
    # Calendar
    features += [
        "weekday", "month", "year", "quarter",
        "is_weekend", "is_month_end", "is_month_start", "has_event",
        "is_cultural_event", "is_national_event", "is_religious_event", "is_sporting_event",
    ]
    # SNAP — we'll add these dynamically if present
    # Price
    features += [
        "sell_price", "price_lag1", "price_change",
        "price_rolling_mean_4wk", "price_discount_proxy",
    ]
    # ID encodings
    features += ["item_id_enc", "store_id_enc", "cat_id_enc", "dept_id_enc", "state_id_enc"]

    return features


def run(config_path: str = "config/settings.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_data"]

    print("=== Feature Engineering ===")

    fact = pd.read_parquet(os.path.join(processed_dir, "fact_sales_daily.parquet"))
    dim_calendar = pd.read_parquet(os.path.join(processed_dir, "dim_calendar.parquet"))
    dim_calendar["date"] = pd.to_datetime(dim_calendar["date"])
    fact["date"] = pd.to_datetime(fact["date"])

    # Sort is critical for correct lag/rolling computation
    fact = fact.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    lag_days = cfg["forecast"]["lag_days"]
    rolling_windows = cfg["forecast"]["rolling_windows"]

    print(f"  Adding lag features: {lag_days}")
    fact = add_lag_features(fact, lag_days)

    print(f"  Adding rolling features: {rolling_windows}")
    fact = add_rolling_features(fact, rolling_windows)

    print("  Adding calendar features...")
    fact = add_calendar_features(fact, dim_calendar)

    # Add SNAP columns to feature list if present
    snap_cols = [c for c in fact.columns if c.startswith("snap_")]
    if snap_cols:
        print(f"  SNAP columns: {snap_cols}")

    print("  Adding price features...")
    fact = add_price_features(fact)

    print("  Adding ID encodings...")
    fact = add_id_encodings(fact)

    # Drop rows where the longest lag feature is NaN (series too short to have full history)
    max_lag = max(lag_days)
    before = len(fact)
    fact = fact.dropna(subset=[f"lag_{max_lag}", "lag_1"]).reset_index(drop=True)
    print(f"  Dropped {before - len(fact):,} rows with insufficient lag history")
    print(f"  Final feature frame: {fact.shape}")

    out_path = os.path.join(processed_dir, "fact_features.parquet")
    fact.to_parquet(out_path, index=False)
    print(f"\nWrote features to: {out_path}")

    return fact


if __name__ == "__main__":
    run()
