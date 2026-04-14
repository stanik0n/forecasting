"""
Layer 1: Raw ingestion
Load source CSV files, profile schemas, and write cleaned parquet outputs to data/interim/.
"""

import os
import yaml
import pandas as pd


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_calendar(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "calendar.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sales(raw_dir: str, pilot_store: str | None = None, pilot_category: str | None = None) -> pd.DataFrame:
    path = os.path.join(raw_dir, "sales_train_validation.csv")
    df = pd.read_csv(path)

    if pilot_store:
        df = df[df["store_id"] == pilot_store].copy()
        print(f"  Filtered to store: {pilot_store} — {len(df):,} item rows")

    if pilot_category:
        df = df[df["cat_id"] == pilot_category].copy()
        print(f"  Filtered to category: {pilot_category} — {len(df):,} item rows")

    return df


def load_sell_prices(raw_dir: str, pilot_store: str | None = None) -> pd.DataFrame:
    path = os.path.join(raw_dir, "sell_prices.csv")
    df = pd.read_csv(path)

    if pilot_store:
        df = df[df["store_id"] == pilot_store].copy()

    return df


def profile_dataframe(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Dtypes      :\n{df.dtypes.to_string()}")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"  Nulls       :\n{null_counts[null_counts > 0].to_string()}")
    else:
        print("  Nulls       : none")


def run(config_path: str = "config/settings.yaml") -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    raw_dir = cfg["paths"]["raw_data"]
    interim_dir = cfg["paths"]["interim_data"]
    pilot_store = cfg["data"].get("pilot_store")
    pilot_category = cfg["data"].get("pilot_category")

    os.makedirs(interim_dir, exist_ok=True)

    print("=== Ingestion ===")
    print(f"Raw data dir : {raw_dir}")
    print(f"Pilot store  : {pilot_store or 'all'}")
    print(f"Pilot category: {pilot_category or 'all'}")

    calendar = load_calendar(raw_dir)
    sales = load_sales(raw_dir, pilot_store=pilot_store, pilot_category=pilot_category)
    sell_prices = load_sell_prices(raw_dir, pilot_store=pilot_store)

    profile_dataframe(calendar, "calendar")
    profile_dataframe(sales, "sales_train_validation")
    profile_dataframe(sell_prices, "sell_prices")

    # Write interim parquet
    calendar.to_parquet(os.path.join(interim_dir, "calendar.parquet"), index=False)
    sales.to_parquet(os.path.join(interim_dir, "sales_raw.parquet"), index=False)
    sell_prices.to_parquet(os.path.join(interim_dir, "sell_prices.parquet"), index=False)

    print(f"\nWrote interim parquet files to: {interim_dir}")

    return {"calendar": calendar, "sales": sales, "sell_prices": sell_prices}


if __name__ == "__main__":
    run()
