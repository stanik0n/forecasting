"""
Layer 5: Replenishment engine
Converts 28-day demand forecasts into reorder points, safety stock levels,
and a daily inventory simulation with stockout/overstock flags.

Assumptions (configurable in settings.yaml):
  - Fixed lead time (default 7 days)
  - Order-up-to policy: when inventory drops below reorder point, order up to
    lead_time_demand + safety_stock + target_days_coverage * avg_daily_demand
  - Safety stock = z * demand_std * sqrt(lead_time_days)
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_reorder_parameters(
    forecast_df: pd.DataFrame,
    lead_time_days: int,
    service_level_z: float,
    target_days_coverage: int,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each item-store pair, compute:
      - avg_daily_demand: mean of the 28-day forecast
      - demand_std: std of forecast (or historical demand std if available)
      - lead_time_demand: avg_daily_demand * lead_time_days
      - safety_stock: z * demand_std * sqrt(lead_time_days)
      - reorder_point: lead_time_demand + safety_stock
      - order_up_to: reorder_point + target_days_coverage * avg_daily_demand
    """
    params = (
        forecast_df.groupby(["item_id", "store_id"])["predicted_units"]
        .agg(avg_daily_demand="mean", demand_std="std")
        .reset_index()
    )
    params["demand_std"] = params["demand_std"].fillna(0)

    # If we have historical demand, use its std (more stable than 28-day forecast std)
    if history_df is not None:
        hist_std = (
            history_df.groupby(["item_id", "store_id"])["units_sold"]
            .std()
            .reset_index()
            .rename(columns={"units_sold": "hist_std"})
        )
        params = params.merge(hist_std, on=["item_id", "store_id"], how="left")
        params["demand_std"] = params[["demand_std", "hist_std"]].max(axis=1).fillna(0)
        params.drop(columns=["hist_std"], inplace=True)

    params["lead_time_demand"] = params["avg_daily_demand"] * lead_time_days
    params["safety_stock"] = (
        service_level_z * params["demand_std"] * np.sqrt(lead_time_days)
    )
    params["reorder_point"] = params["lead_time_demand"] + params["safety_stock"]
    params["order_up_to"] = params["reorder_point"] + (
        target_days_coverage * params["avg_daily_demand"]
    )

    # Round to nearest integer unit
    for col in ["lead_time_demand", "safety_stock", "reorder_point", "order_up_to"]:
        params[col] = params[col].round(1)

    return params


def simulate_inventory(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    reorder_params: pd.DataFrame,
    lead_time_days: int,
    initial_inventory: float | None,
) -> pd.DataFrame:
    """
    Day-by-day inventory simulation for each item-store pair.

    Logic:
      - Opening inventory at t = closing inventory at t-1 (plus any delivery arriving today)
      - Demand = actual units_sold (if available) else forecast
      - Closing inventory = max(0, opening - demand)
      - If opening inventory falls below reorder_point at any point, a replenishment
        order is placed and arrives after `lead_time_days` days.
      - Stockout flag: closing inventory == 0 and demand > 0
      - Overstock flag: closing > order_up_to * 2
    """
    forecast_dates = sorted(pd.to_datetime(forecast_df["forecast_date"].unique()))

    # Build O(1) lookup dicts keyed by (item_id, store_id, date)
    forecast_lookup: dict[tuple, float] = {}
    for row in forecast_df.itertuples(index=False):
        forecast_lookup[(row.item_id, row.store_id, pd.Timestamp(row.forecast_date))] = row.predicted_units

    actuals_lookup: dict[tuple, float] = {}
    for row in actuals_df[actuals_df["date"].isin(forecast_dates)].itertuples(index=False):
        actuals_lookup[(row.item_id, row.store_id, pd.Timestamp(row.date))] = row.units_sold

    # Build reorder params dict keyed by (item_id, store_id)
    params_lookup: dict[tuple, dict] = {}
    for row in reorder_params.itertuples(index=False):
        params_lookup[(row.item_id, row.store_id)] = {
            "rop": row.reorder_point,
            "otu": row.order_up_to,
            "avg": row.avg_daily_demand,
        }

    pairs = list(params_lookup.keys())
    sim_rows = []

    for item_id, store_id in pairs:
        p = params_lookup[(item_id, store_id)]
        rop = p["rop"]
        otu = p["otu"]
        avg_demand = p["avg"]

        if initial_inventory is not None:
            inventory = float(initial_inventory)
        else:
            inventory = float(otu) if otu > 0 else max(float(avg_demand) * 14, 1.0)

        pending_orders: list[tuple[pd.Timestamp, float]] = []
        on_order = False

        for date in forecast_dates:
            delivery_date = date + pd.Timedelta(days=lead_time_days)

            # Receive deliveries due today
            deliveries_today = sum(qty for (dd, qty) in pending_orders if dd == date)
            pending_orders = [(dd, qty) for (dd, qty) in pending_orders if dd != date]
            inventory += deliveries_today
            if deliveries_today > 0:
                on_order = False

            opening = inventory

            # O(1) lookups instead of DataFrame scans
            key = (item_id, store_id, date)
            fcast_units = forecast_lookup.get(key, avg_demand)
            actual_units = actuals_lookup.get(key, fcast_units)

            demand = float(actual_units)
            closing = max(0.0, opening - demand)
            inventory = closing

            stockout = int(closing == 0 and demand > 0)
            overstock = int(closing > otu * 2 and otu > 0)

            # Reorder trigger: place order if below rop and no order pending
            reorder_qty = 0.0
            if inventory <= rop and not on_order:
                reorder_qty = max(0.0, otu - inventory)
                if reorder_qty > 0:
                    pending_orders.append((delivery_date, reorder_qty))
                    on_order = True

            sim_rows.append({
                "item_id": item_id,
                "store_id": store_id,
                "date": date,
                "opening_inventory": round(opening, 2),
                "demand": demand,
                "forecast": fcast_units,
                "reorder_point": rop,
                "reorder_qty": round(reorder_qty, 2),
                "closing_inventory": round(closing, 2),
                "stockout_flag": stockout,
                "overstock_flag": overstock,
            })

    return pd.DataFrame(sim_rows)


def run(
    config_path: str = "config/settings.yaml",
    model_name: str = "lgbm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_data"]
    tables_dir = cfg["paths"]["outputs_tables"]
    lead_time = cfg["replenishment"]["lead_time_days"]
    z = cfg["replenishment"]["service_level_z"]
    target_cov = cfg["replenishment"]["target_days_coverage"]
    initial_inv = cfg["replenishment"].get("initial_inventory")
    os.makedirs(tables_dir, exist_ok=True)

    print("=== Replenishment Engine ===")
    print(f"  Lead time     : {lead_time} days")
    print(f"  Service level z: {z} (~{int(z * 34 + 50)}% service level)")
    print(f"  Target coverage: {target_cov} days")

    # Load best forecast (prefer lgbm, fallback to baselines)
    lgbm_path = os.path.join(processed_dir, "fact_forecast_lgbm.parquet")
    baseline_path = os.path.join(processed_dir, "fact_forecast_baselines.parquet")

    if os.path.exists(lgbm_path):
        forecast_df = pd.read_parquet(lgbm_path)
        active_model = "lgbm"
    else:
        forecast_df = pd.read_parquet(baseline_path)
        forecast_df = forecast_df[forecast_df["model_name"] == "moving_average"]
        active_model = "moving_average"

    forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"])
    print(f"  Using forecast model: {active_model}")
    print(f"  Forecast dates: {forecast_df['forecast_date'].min().date()} → {forecast_df['forecast_date'].max().date()}")

    # Historical data for demand_std estimation
    fact = pd.read_parquet(os.path.join(processed_dir, "fact_features.parquet"))
    fact["date"] = pd.to_datetime(fact["date"])

    # Use last 90 days of history for demand std
    history_cutoff = forecast_df["forecast_date"].min() - pd.Timedelta(days=1)
    recent_history = fact[fact["date"] > history_cutoff - pd.Timedelta(days=90)]

    print("  Computing reorder parameters...")
    reorder_params = compute_reorder_parameters(
        forecast_df, lead_time, z, target_cov, history_df=recent_history
    )

    print(f"  Simulating inventory for {len(reorder_params):,} item-store pairs...")
    sim = simulate_inventory(
        forecast_df, fact, reorder_params, lead_time, initial_inv
    )

    print(f"  Simulation rows: {len(sim):,}")
    stockout_rate = sim["stockout_flag"].mean() * 100
    overstock_rate = sim["overstock_flag"].mean() * 100
    print(f"  Stockout rate : {stockout_rate:.1f}%")
    print(f"  Overstock rate: {overstock_rate:.1f}%")

    reorder_params.to_parquet(os.path.join(processed_dir, "reorder_params.parquet"), index=False)
    sim.to_parquet(os.path.join(processed_dir, "fact_inventory_simulation.parquet"), index=False)
    reorder_params.to_csv(os.path.join(tables_dir, "reorder_parameters.csv"), index=False)

    print(f"\nWrote reorder parameters and inventory simulation.")

    return reorder_params, sim


if __name__ == "__main__":
    run()
