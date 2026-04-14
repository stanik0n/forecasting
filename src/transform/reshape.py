"""
Layer 2: Cleaned analytics tables
Melt wide day columns into long format, join calendar and pricing data,
and produce fact_sales_daily and dim_calendar parquet outputs.
"""

import os
import yaml
import pandas as pd


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dim_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Enrich calendar with derived date features."""
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["weekday"] = cal["date"].dt.dayofweek          # 0=Mon
    cal["weekday_name"] = cal["date"].dt.day_name()
    cal["month"] = cal["date"].dt.month
    cal["year"] = cal["date"].dt.year
    cal["quarter"] = cal["date"].dt.quarter
    cal["is_weekend"] = cal["weekday"].isin([5, 6]).astype(int)
    cal["is_month_end"] = cal["date"].dt.is_month_end.astype(int)
    cal["is_month_start"] = cal["date"].dt.is_month_start.astype(int)
    cal["has_event"] = cal["event_name_1"].notna().astype(int)

    # SNAP columns: snap_CA, snap_TX, snap_WI
    snap_cols = [c for c in cal.columns if c.startswith("snap_")]
    for col in snap_cols:
        cal[col] = cal[col].fillna(0).astype(int)

    return cal


def melt_sales_to_long(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide sales (d_1 ... d_N columns) into long format.
    Returns: item_id, store_id, state_id, cat_id, dept_id, d, date, units_sold
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    print(f"  Melting {len(day_cols)} day columns × {len(sales):,} items → long format...")
    long = sales[id_cols + day_cols].melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )

    # Map d column to actual dates via calendar
    d_to_date = calendar[["d", "date"]].set_index("d")["date"]
    long["date"] = long["d"].map(d_to_date)
    long = long.dropna(subset=["date"])
    long["date"] = pd.to_datetime(long["date"])
    long["units_sold"] = long["units_sold"].fillna(0).astype(int)

    return long


def join_prices(long_sales: pd.DataFrame, sell_prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Join weekly sell_prices onto the daily long sales table.
    sell_prices is keyed on store_id, item_id, wm_yr_wk.
    """
    # Map date -> wm_yr_wk via calendar
    wk_map = calendar[["date", "wm_yr_wk"]].copy()
    wk_map["date"] = pd.to_datetime(wk_map["date"])
    long_sales = long_sales.merge(wk_map, on="date", how="left")

    # Join prices
    long_sales = long_sales.merge(
        sell_prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    long_sales["revenue_estimate"] = long_sales["units_sold"] * long_sales["sell_price"].fillna(0)

    return long_sales


def build_fact_sales_daily(long_sales: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for the final fact table."""
    cols = [
        "item_id", "store_id", "state_id", "cat_id", "dept_id",
        "date", "d", "wm_yr_wk", "units_sold", "sell_price", "revenue_estimate",
    ]
    return long_sales[cols].sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def run(config_path: str = "config/settings.yaml") -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    interim_dir = cfg["paths"]["interim_data"]
    processed_dir = cfg["paths"]["processed_data"]
    os.makedirs(processed_dir, exist_ok=True)

    print("=== Transform / Reshape ===")

    calendar = pd.read_parquet(os.path.join(interim_dir, "calendar.parquet"))
    sales = pd.read_parquet(os.path.join(interim_dir, "sales_raw.parquet"))
    sell_prices = pd.read_parquet(os.path.join(interim_dir, "sell_prices.parquet"))

    dim_calendar = build_dim_calendar(calendar)
    print(f"  dim_calendar: {dim_calendar.shape}")

    long_sales = melt_sales_to_long(sales, calendar)
    print(f"  After melt  : {long_sales.shape}")

    long_sales = join_prices(long_sales, sell_prices, calendar)
    print(f"  After price join: {long_sales.shape}")

    fact_sales = build_fact_sales_daily(long_sales)
    print(f"  fact_sales_daily: {fact_sales.shape}")
    print(f"  Date range: {fact_sales['date'].min().date()} → {fact_sales['date'].max().date()}")
    print(f"  Items: {fact_sales['item_id'].nunique():,}  |  Stores: {fact_sales['store_id'].nunique()}")

    dim_calendar.to_parquet(os.path.join(processed_dir, "dim_calendar.parquet"), index=False)
    fact_sales.to_parquet(os.path.join(processed_dir, "fact_sales_daily.parquet"), index=False)

    print(f"\nWrote processed parquet files to: {processed_dir}")

    return {"dim_calendar": dim_calendar, "fact_sales_daily": fact_sales}


if __name__ == "__main__":
    run()
