"""
Layer 7: Evaluation
Export forecast accuracy metrics, operational replenishment metrics,
and comparison charts across models.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Forecast quality metrics
# ---------------------------------------------------------------------------

def forecast_metrics_by_model(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute RMSE, MAE, MAPE per model across the validation window."""
    actuals = actuals_df[["item_id", "store_id", "date", "units_sold"]].rename(
        columns={"date": "forecast_date"}
    )
    merged = forecast_df.merge(actuals, on=["item_id", "store_id", "forecast_date"], how="inner")

    records = []
    for model_name, grp in merged.groupby("model_name"):
        errors = grp["predicted_units"] - grp["units_sold"]
        abs_errors = errors.abs()
        nonzero = grp[grp["units_sold"] > 0]
        records.append({
            "model": model_name,
            "rmse": round(np.sqrt((errors ** 2).mean()), 4),
            "mae": round(abs_errors.mean(), 4),
            "mape": round(
                ((nonzero["predicted_units"] - nonzero["units_sold"]).abs() / nonzero["units_sold"] * 100).mean(), 2
            ) if len(nonzero) > 0 else np.nan,
            "bias": round(errors.mean(), 4),
            "n_obs": len(grp),
        })

    return pd.DataFrame(records).sort_values("rmse")


def forecast_metrics_by_category(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast accuracy broken down by category and model."""
    actuals = actuals_df[["item_id", "store_id", "cat_id", "date", "units_sold"]].rename(
        columns={"date": "forecast_date"}
    )
    merged = forecast_df.merge(actuals, on=["item_id", "store_id", "forecast_date"], how="inner")

    records = []
    for (model_name, cat), grp in merged.groupby(["model_name", "cat_id"]):
        errors = grp["predicted_units"] - grp["units_sold"]
        abs_errors = errors.abs()
        nonzero = grp[grp["units_sold"] > 0]
        records.append({
            "model": model_name,
            "category": cat,
            "rmse": round(np.sqrt((errors ** 2).mean()), 4),
            "mae": round(abs_errors.mean(), 4),
            "mape": round(
                ((nonzero["predicted_units"] - nonzero["units_sold"]).abs() / nonzero["units_sold"] * 100).mean(), 2
            ) if len(nonzero) > 0 else np.nan,
        })

    return pd.DataFrame(records).sort_values(["category", "rmse"])


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------

def operational_metrics(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute supply-chain KPIs from the inventory simulation:
      - service_level: % of days with no stockout
      - stockout_days: count
      - overstock_days: count
      - avg_days_of_supply: avg closing inventory / avg daily demand
      - avg_closing_inventory
      - total_reorder_events
    """
    sim_df = sim_df.copy()
    sim_df["demand"] = sim_df["demand"].replace(0, np.nan)
    sim_df["days_of_supply"] = sim_df["closing_inventory"] / sim_df["demand"].fillna(1)

    metrics = {
        "service_level_pct": round((1 - sim_df["stockout_flag"].mean()) * 100, 2),
        "stockout_days": int(sim_df["stockout_flag"].sum()),
        "overstock_days": int(sim_df["overstock_flag"].sum()),
        "avg_closing_inventory": round(sim_df["closing_inventory"].mean(), 2),
        "avg_days_of_supply": round(sim_df["days_of_supply"].mean(), 2),
        "total_reorder_events": int((sim_df["reorder_qty"] > 0).sum()),
    }
    return pd.DataFrame([metrics])


def stockout_risk_by_item(sim_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Items with the highest stockout frequency."""
    risk = (
        sim_df.groupby(["item_id", "store_id"])
        .agg(
            stockout_days=("stockout_flag", "sum"),
            total_days=("stockout_flag", "count"),
            avg_closing_inv=("closing_inventory", "mean"),
        )
        .reset_index()
    )
    risk["stockout_rate_pct"] = (risk["stockout_days"] / risk["total_days"] * 100).round(2)
    return risk.sort_values("stockout_days", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_model_comparison(metrics_df: pd.DataFrame, charts_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Forecast Model Comparison", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
        if metric not in metrics_df.columns:
            continue
        data = metrics_df.dropna(subset=[metric])
        bars = ax.bar(data["model"], data[metric], color=["#2196F3", "#FF9800", "#4CAF50"][:len(data)])
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=15)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison.png")


def plot_forecast_vs_actuals(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    item_id: str,
    store_id: str,
    charts_dir: str,
    lookback_days: int = 60,
) -> None:
    actuals = actuals_df[
        (actuals_df["item_id"] == item_id) & (actuals_df["store_id"] == store_id)
    ].sort_values("date")

    cutoff = forecast_df["forecast_date"].min() - pd.Timedelta(days=1)
    history = actuals[actuals["date"] <= cutoff].tail(lookback_days)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(history["date"], history["units_sold"], label="Historical demand", color="#555", linewidth=1.2)

    colors = {"lgbm": "#2196F3", "moving_average": "#FF9800", "seasonal_naive": "#9C27B0"}
    for model_name, grp in forecast_df[
        (forecast_df["item_id"] == item_id) & (forecast_df["store_id"] == store_id)
    ].groupby("model_name"):
        grp = grp.sort_values("forecast_date")
        act_in_window = actuals[actuals["date"].isin(grp["forecast_date"])]
        ax.plot(grp["forecast_date"], grp["predicted_units"],
                label=f"{model_name} forecast", color=colors.get(model_name, "blue"),
                linestyle="--", linewidth=1.5)

    # Actuals in forecast window
    act_window = actuals[actuals["date"] >= forecast_df["forecast_date"].min()]
    if not act_window.empty:
        ax.plot(act_window["date"], act_window["units_sold"],
                label="Actual (val)", color="#333", linewidth=1.2, linestyle=":")

    ax.axvline(x=cutoff, color="red", linestyle="--", alpha=0.5, label="Train cutoff")
    ax.set_title(f"Demand Forecast — {item_id} @ {store_id}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    fname = f"forecast_{item_id}_{store_id}.png".replace("/", "_")
    plt.savefig(os.path.join(charts_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_inventory_simulation(
    sim_df: pd.DataFrame,
    reorder_params: pd.DataFrame,
    item_id: str,
    store_id: str,
    charts_dir: str,
) -> None:
    data = sim_df[(sim_df["item_id"] == item_id) & (sim_df["store_id"] == store_id)].sort_values("date")
    params = reorder_params[(reorder_params["item_id"] == item_id) & (reorder_params["store_id"] == store_id)]

    if data.empty:
        return

    rop = params["reorder_point"].values[0] if not params.empty else None
    otu = params["order_up_to"].values[0] if not params.empty else None

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(data["date"], data["closing_inventory"], alpha=0.3, color="#2196F3", label="Closing inventory")
    ax.plot(data["date"], data["closing_inventory"], color="#2196F3", linewidth=1.2)
    ax.bar(data["date"], data["demand"], alpha=0.4, color="#FF9800", label="Daily demand", width=0.8)

    if rop is not None:
        ax.axhline(y=rop, color="red", linestyle="--", linewidth=1.2, label=f"Reorder point ({rop:.0f})")
    if otu is not None:
        ax.axhline(y=otu, color="green", linestyle=":", linewidth=1.2, label=f"Order-up-to ({otu:.0f})")

    # Mark reorder events
    reorders = data[data["reorder_qty"] > 0]
    if not reorders.empty:
        ax.scatter(reorders["date"], reorders["closing_inventory"],
                   color="purple", zorder=5, s=40, label="Reorder triggered", marker="^")

    # Mark stockouts
    stockouts = data[data["stockout_flag"] == 1]
    if not stockouts.empty:
        ax.scatter(stockouts["date"], [0] * len(stockouts),
                   color="red", zorder=5, s=40, label="Stockout", marker="x")

    ax.set_title(f"Inventory Simulation — {item_id} @ {store_id}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    fname = f"inventory_{item_id}_{store_id}.png".replace("/", "_")
    plt.savefig(os.path.join(charts_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def load_evaluation_actuals(cfg: dict) -> pd.DataFrame | None:
    """
    Load sales_train_evaluation.csv (d_1914–d_1941) as true out-of-sample actuals.
    Returns a long-format DataFrame or None if the file isn't present.
    """
    raw_dir = cfg["paths"]["raw_data"]
    eval_path = os.path.join(raw_dir, "sales_train_evaluation.csv")
    if not os.path.exists(eval_path):
        return None

    interim_dir = cfg["paths"]["interim_data"]
    cal = pd.read_parquet(os.path.join(interim_dir, "calendar.parquet"))
    cal["date"] = pd.to_datetime(cal["date"])

    pilot_store = cfg["data"].get("pilot_store")
    pilot_category = cfg["data"].get("pilot_category")

    sales_eval = pd.read_csv(eval_path)
    if pilot_store:
        sales_eval = sales_eval[sales_eval["store_id"] == pilot_store]
    if pilot_category:
        sales_eval = sales_eval[sales_eval["cat_id"] == pilot_category]

    # Only the extra days beyond d_1913
    all_day_cols = [c for c in sales_eval.columns if c.startswith("d_")]
    extra_days = [c for c in all_day_cols if int(c.split("_")[1]) > 1913]
    if not extra_days:
        return None

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = sales_eval[id_cols + extra_days].melt(
        id_vars=id_cols, value_vars=extra_days, var_name="d", value_name="units_sold"
    )
    d_to_date = cal.set_index("d")["date"]
    long["date"] = long["d"].map(d_to_date)
    long = long.dropna(subset=["date"])
    long["date"] = pd.to_datetime(long["date"])
    long["units_sold"] = long["units_sold"].fillna(0).astype(int)
    print(f"  Loaded evaluation actuals: {len(long):,} rows, dates {long['date'].min().date()} to {long['date'].max().date()}")
    return long


def run(config_path: str = "config/settings.yaml") -> None:
    cfg = load_config(config_path)
    processed_dir = cfg["paths"]["processed_data"]
    tables_dir = cfg["paths"]["outputs_tables"]
    charts_dir = cfg["paths"]["outputs_charts"]
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    print("=== Evaluation ===")

    fact = pd.read_parquet(os.path.join(processed_dir, "fact_features.parquet"))
    fact["date"] = pd.to_datetime(fact["date"])

    # Try to load true out-of-sample actuals from evaluation file
    eval_actuals = load_evaluation_actuals(cfg)
    if eval_actuals is not None:
        # Merge into fact for use as ground truth where available
        eval_actuals_slim = eval_actuals[["item_id", "store_id", "date", "units_sold"]]
        fact_extended = pd.concat([fact, eval_actuals_slim], ignore_index=True).drop_duplicates(
            subset=["item_id", "store_id", "date"], keep="last"
        )
        print("  Using sales_train_evaluation.csv for out-of-sample ground truth")
    else:
        fact_extended = fact
        print("  No evaluation file found — using validation window from training data")

    # Load all forecasts
    forecast_parts = []
    for fname, label in [
        ("fact_forecast_baselines.parquet", None),
        ("fact_forecast_lgbm.parquet", None),
    ]:
        path = os.path.join(processed_dir, fname)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["forecast_date"] = pd.to_datetime(df["forecast_date"])
            forecast_parts.append(df)

    if not forecast_parts:
        print("  No forecast files found. Run forecasting modules first.")
        return

    all_forecasts = pd.concat(forecast_parts, ignore_index=True)

    # --- Forecast metrics ---
    print("  Computing forecast accuracy metrics...")
    model_metrics = forecast_metrics_by_model(all_forecasts, fact_extended)
    cat_metrics = forecast_metrics_by_category(all_forecasts, fact_extended)

    print("\n  === Model Comparison ===")
    print(model_metrics.to_string(index=False))

    model_metrics.to_csv(os.path.join(tables_dir, "model_comparison.csv"), index=False)
    cat_metrics.to_csv(os.path.join(tables_dir, "metrics_by_category.csv"), index=False)

    # --- Operational metrics ---
    sim_path = os.path.join(processed_dir, "fact_inventory_simulation.parquet")
    if os.path.exists(sim_path):
        sim = pd.read_parquet(sim_path)
        sim["date"] = pd.to_datetime(sim["date"])
        reorder_params = pd.read_parquet(os.path.join(processed_dir, "reorder_params.parquet"))

        print("\n  Computing operational metrics...")
        ops_metrics = operational_metrics(sim)
        stockout_risk = stockout_risk_by_item(sim)

        print("\n  === Operational KPIs ===")
        print(ops_metrics.to_string(index=False))

        ops_metrics.to_csv(os.path.join(tables_dir, "operational_metrics.csv"), index=False)
        stockout_risk.to_csv(os.path.join(tables_dir, "stockout_risk_top20.csv"), index=False)

    # --- Charts ---
    print("\n  Generating charts...")
    plot_model_comparison(model_metrics, charts_dir)

    # Pick a sample item for demand + inventory charts
    sample_pairs = fact[["item_id", "store_id"]].drop_duplicates().head(3)
    for _, row in sample_pairs.iterrows():
        item_id, store_id = row["item_id"], row["store_id"]
        plot_forecast_vs_actuals(all_forecasts, fact_extended, item_id, store_id, charts_dir)
        if os.path.exists(sim_path):
            plot_inventory_simulation(sim, reorder_params, item_id, store_id, charts_dir)

    # Write a combined dashboard-ready summary
    summary = all_forecasts.merge(
        fact_extended[["item_id", "store_id", "cat_id", "dept_id", "date", "units_sold"]].rename(
            columns={"date": "forecast_date"}
        ),
        on=["item_id", "store_id", "forecast_date"],
        how="left",
    )
    summary.to_parquet(os.path.join(processed_dir, "dashboard_forecast_summary.parquet"), index=False)

    print(f"\nEvaluation complete. Outputs in: {tables_dir} and {charts_dir}")


if __name__ == "__main__":
    run()
