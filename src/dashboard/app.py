"""
Layer 8: Streamlit Dashboard
Three views:
  - Executive: summary KPIs, top stockout/overstock items
  - Planner: item-store demand + forecast + inventory position
  - Model: forecast method comparison and error metrics
"""

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

PROCESSED_DIR = "data/processed"
TABLES_DIR = "outputs/tables"
CHARTS_DIR = "outputs/charts"


@st.cache_data
def load_data():
    data = {}

    def safe_load(key, path):
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
            if "date" in data[key].columns:
                data[key]["date"] = pd.to_datetime(data[key]["date"])
            if "forecast_date" in data[key].columns:
                data[key]["forecast_date"] = pd.to_datetime(data[key]["forecast_date"])
        else:
            data[key] = pd.DataFrame()

    safe_load("fact_sales", os.path.join(PROCESSED_DIR, "fact_sales_daily.parquet"))
    safe_load("fact_features", os.path.join(PROCESSED_DIR, "fact_features.parquet"))
    safe_load("forecast_summary", os.path.join(PROCESSED_DIR, "dashboard_forecast_summary.parquet"))
    safe_load("sim", os.path.join(PROCESSED_DIR, "fact_inventory_simulation.parquet"))
    safe_load("reorder_params", os.path.join(PROCESSED_DIR, "reorder_params.parquet"))

    for fname, key in [
        ("model_comparison.csv", "model_metrics"),
        ("operational_metrics.csv", "ops_metrics"),
        ("stockout_risk_top20.csv", "stockout_risk"),
        ("reorder_parameters.csv", "reorder_table"),
        ("metrics_by_category.csv", "cat_metrics"),
    ]:
        path = os.path.join(TABLES_DIR, fname)
        data[key] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    return data


def fmt_metric(val, fmt=".1f"):
    if pd.isna(val):
        return "N/A"
    return f"{val:{fmt}}"


def executive_view(data: dict):
    st.header("Executive Summary")

    ops = data.get("ops_metrics", pd.DataFrame())
    if not ops.empty:
        row = ops.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Service Level", f"{fmt_metric(row.get('service_level_pct', 0))}%")
        c2.metric("Stockout Days", fmt_metric(row.get("stockout_days", 0), ".0f"))
        c3.metric("Overstock Days", fmt_metric(row.get("overstock_days", 0), ".0f"))
        c4.metric("Avg Days of Supply", fmt_metric(row.get("avg_days_of_supply", 0)))

    model_metrics = data.get("model_metrics", pd.DataFrame())
    if not model_metrics.empty:
        st.subheader("Forecast Accuracy by Model")
        numeric_cols = [c for c in ["rmse", "mae", "mape"] if c in model_metrics.columns and model_metrics[c].notna().any()]
        styled = model_metrics.style.highlight_min(subset=numeric_cols, color="#d4edda").format(
            {c: "{:.4f}" for c in numeric_cols}
        )
        st.dataframe(styled, use_container_width=True)

    stockout_risk = data.get("stockout_risk", pd.DataFrame())
    if not stockout_risk.empty:
        st.subheader("Top Items by Stockout Risk")
        st.dataframe(
            stockout_risk[["item_id", "store_id", "stockout_days", "stockout_rate_pct", "avg_closing_inv"]].head(10),
            use_container_width=True,
        )

    chart_path = os.path.join(CHARTS_DIR, "model_comparison.png")
    if os.path.exists(chart_path):
        st.subheader("Model Comparison Chart")
        st.image(chart_path)


def planner_view(data: dict):
    st.header("Planner View — Item / Store")

    fact = data.get("fact_features", pd.DataFrame())
    forecast = data.get("forecast_summary", pd.DataFrame())
    sim = data.get("sim", pd.DataFrame())
    reorder_params = data.get("reorder_params", pd.DataFrame())

    if fact.empty:
        st.warning("No feature data found. Run the pipeline first.")
        return

    stores = sorted(fact["store_id"].unique())
    selected_store = st.selectbox("Store", stores)

    items = sorted(fact[fact["store_id"] == selected_store]["item_id"].unique())
    selected_item = st.selectbox("Item", items)

    # --- Demand + forecast chart ---
    st.subheader("Demand History & Forecast")
    item_history = fact[
        (fact["item_id"] == selected_item) & (fact["store_id"] == selected_store)
    ].sort_values("date")

    item_forecast = forecast[
        (forecast["item_id"] == selected_item) & (forecast["store_id"] == selected_store)
    ] if not forecast.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(13, 4))
    lookback = item_history.tail(90)
    ax.plot(lookback["date"], lookback["units_sold"], label="Historical demand", color="#555", linewidth=1.2)

    if not item_forecast.empty:
        colors = {"lgbm": "#2196F3", "moving_average": "#FF9800", "seasonal_naive": "#9C27B0"}
        cutoff = item_forecast["forecast_date"].min() - pd.Timedelta(days=1)
        ax.axvline(x=cutoff, color="red", linestyle="--", alpha=0.5, label="Train cutoff")

        for model_name, grp in item_forecast.groupby("model_name"):
            grp = grp.sort_values("forecast_date")
            ax.plot(
                grp["forecast_date"], grp["predicted_units"],
                label=f"{model_name}", color=colors.get(model_name, "blue"),
                linestyle="--", linewidth=1.5,
            )

        if "units_sold" in item_forecast.columns:
            actuals_in_window = item_forecast.dropna(subset=["units_sold"])
            if not actuals_in_window.empty:
                ax.plot(
                    actuals_in_window["forecast_date"],
                    actuals_in_window["units_sold"],
                    color="#333", linestyle=":", linewidth=1.2, label="Actual (val)",
                )

    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Inventory simulation chart ---
    if not sim.empty:
        st.subheader("Inventory Simulation")
        item_sim = sim[
            (sim["item_id"] == selected_item) & (sim["store_id"] == selected_store)
        ].sort_values("date")

        item_params = reorder_params[
            (reorder_params["item_id"] == selected_item) & (reorder_params["store_id"] == selected_store)
        ] if not reorder_params.empty else pd.DataFrame()

        if not item_sim.empty:
            rop = item_params["reorder_point"].values[0] if not item_params.empty else None
            otu = item_params["order_up_to"].values[0] if not item_params.empty else None

            fig2, ax2 = plt.subplots(figsize=(13, 5))
            ax_demand = ax2.twinx()

            # Demand bars on secondary axis so they don't dwarf the inventory line
            ax_demand.bar(item_sim["date"], item_sim["demand"], alpha=0.25, color="#FF9800", width=0.8, label="Daily demand")
            ax_demand.set_ylabel("Daily demand (units)", color="#FF9800", fontsize=9)
            ax_demand.tick_params(axis="y", labelcolor="#FF9800")
            ax_demand.set_ylim(0, item_sim["demand"].max() * 4 if item_sim["demand"].max() > 0 else 1)

            # Inventory on primary axis
            ax2.fill_between(item_sim["date"], item_sim["closing_inventory"], alpha=0.2, color="#2196F3")
            ax2.plot(item_sim["date"], item_sim["closing_inventory"], color="#2196F3", linewidth=1.8, label="Closing inventory", zorder=3)

            if rop is not None:
                ax2.axhline(y=rop, color="red", linestyle="--", linewidth=1.2, label=f"Reorder point ({rop:.0f})", zorder=2)
            if otu is not None:
                ax2.axhline(y=otu, color="#4CAF50", linestyle=":", linewidth=1.2, label=f"Order-up-to ({otu:.0f})", zorder=2)

            reorders = item_sim[item_sim["reorder_qty"] > 0]
            if not reorders.empty:
                ax2.scatter(reorders["date"], reorders["closing_inventory"],
                            color="purple", s=60, zorder=5, marker="^", label="Reorder triggered")

            stockouts = item_sim[item_sim["stockout_flag"] == 1]
            if not stockouts.empty:
                ax2.scatter(stockouts["date"], [0] * len(stockouts),
                            color="red", s=60, zorder=5, marker="x", label="Stockout")

            ax2.set_xlabel("Date")
            ax2.set_ylabel("Inventory (units)", fontsize=9)

            # Merge legends from both axes
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax_demand.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            fig2.autofmt_xdate(rotation=30)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Reorder parameters table
            if not item_params.empty:
                st.subheader("Reorder Parameters")
                display_cols = ["avg_daily_demand", "demand_std", "lead_time_demand",
                                "safety_stock", "reorder_point", "order_up_to"]
                st.dataframe(item_params[[c for c in display_cols if c in item_params.columns]], use_container_width=True)
        else:
            st.info("No simulation data for this item-store combination.")


def model_view(data: dict):
    st.header("Model Comparison")

    model_metrics = data.get("model_metrics", pd.DataFrame())
    if not model_metrics.empty:
        st.subheader("Overall Metrics")
        st.dataframe(model_metrics, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle("Model Error Metrics", fontsize=13, fontweight="bold")
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
        for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
            data_m = model_metrics.dropna(subset=[metric])
            if data_m.empty:
                continue
            bars = ax.bar(data_m["model"], data_m[metric], color=colors[:len(data_m)])
            ax.set_title(metric.upper())
            ax.tick_params(axis="x", rotation=15)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9,
                )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    cat_metrics = data.get("cat_metrics", pd.DataFrame())
    if not cat_metrics.empty:
        st.subheader("Metrics by Category")
        st.dataframe(cat_metrics, use_container_width=True)


def main():
    st.set_page_config(
        page_title="M5 Inventory & Forecasting",
        page_icon="📦",
        layout="wide",
    )
    st.title("📦 M5 Inventory Replenishment & Demand Forecasting")

    data = load_data()
    if all(v.empty for v in data.values() if isinstance(v, pd.DataFrame)):
        st.error(
            "No pipeline outputs found. Run `python run_pipeline.py` first, "
            "then relaunch the dashboard."
        )
        return

    tab1, tab2, tab3 = st.tabs(["Executive", "Planner", "Model"])
    with tab1:
        executive_view(data)
    with tab2:
        planner_view(data)
    with tab3:
        model_view(data)


if __name__ == "__main__":
    main()
