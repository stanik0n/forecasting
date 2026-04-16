# Demand Forecasting & Inventory Replenishment - Supply Chain Planning

End-to-end demand planning and inventory replenishment solution built on the [M5 Walmart dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy). Forecasts product demand using machine learning, simulates inventory with reorder-point logic, and delivers insights through a live dashboard with Executive, Planner, and Model views.

## Live Demo

[https://forecasting-9axkqjqv2oaebabshqbxjj.streamlit.app/](https://forecasting-9axkqjqv2oaebabshqbxjj.streamlit.app/)

---

## Business Objective

- Predict future demand to improve inventory planning and reduce forecast error
- Reduce stockouts and excess inventory through data-driven replenishment policies
- Support procurement and replenishment decisions with 28-day forward demand forecasts
- Align demand forecasts with supply and operational constraints using configurable safety stock

---

## Supply Chain Impact

- Improved inventory planning accuracy by forecasting demand trends across stores and categories
- Enabled better replenishment timing and order quantity decisions using reorder-point simulation
- Reduced risk of overstocking and stockouts — tracked via service level and stockout KPIs
- Supported data-driven supply-demand alignment through an executive planning dashboard

---

## How This Fits in Supply Chain

```
Demand Forecast → Inventory Planning → Replenishment Trigger → Procurement → Distribution
```

---

## Planning Outputs

| Output | Description |
|--------|-------------|
| 28-day demand forecasts | Forecasted demand volumes per item, store, and category |
| Reorder points | Inventory thresholds that trigger replenishment orders |
| Safety stock levels | Buffer stock calculated from demand variability and lead time |
| Stockout risk ranking | Top 20 items most at risk of running out |
| Service level | % of days without a stockout event |
| Inventory simulation | Day-by-day inventory trajectory with order arrival modelled |

---

## Dashboard

![Executive Summary](docs/screenshot_executive.png)

![Model Comparison](docs/screenshot_model_comparison.png)

![Planner — Demand Forecast](docs/screenshot_planner_forecast.png)

![Inventory Simulation](docs/screenshot_inventory_sim.png)

---

## Demand Analysis & Trend Identification

Historical Walmart M5 sales data is cleaned, reshaped, and enriched with:

- Lag features and rolling demand windows to capture recent trends
- Seasonal and calendar signals (day of week, month, SNAP events, holidays)
- Price elasticity features to capture demand sensitivity to price changes

---

## Forecasting Approach

Two models are compared to assess forecast accuracy:

| Model | Approach |
|-------|----------|
| Seasonal Naive | Rolling average baseline representing current planning practice |
| LightGBM | Gradient-boosted model trained on demand history and contextual features |

Forecasting techniques are applied to improve demand prediction accuracy and support inventory planning decisions. All features use a lag structure to prevent data leakage. Validation uses a time-based holdout window — no shuffling.

---

## Replenishment Logic

```
safety_stock   = z × demand_std × √(lead_time_days)
reorder_point  = lead_time_demand + safety_stock
order_up_to    = reorder_point + target_days_coverage × avg_daily_demand
```

A replenishment order is triggered on any simulation day where `closing_inventory ≤ reorder_point`. The order arrives `lead_time_days` later. All parameters (lead time, service level z-score, coverage target) are configurable in `config/settings.yaml`.

---

## Pipeline Modules

| Step | Module | Output |
|------|--------|--------|
| 1. Ingest | `src/ingest/ingest.py` | Cleaned parquet files |
| 2. Transform | `src/transform/reshape.py` | Daily sales fact + calendar dimension |
| 3. Demand Analysis | `src/features/feature_engineering.py` | Lag, rolling, price, calendar features |
| 4. Baseline Forecast | `src/forecasting/baseline_models.py` | Moving average + seasonal naive 28-day forecasts |
| 5. ML Forecast | `src/forecasting/lgbm_model.py` | LightGBM forecast + feature importances |
| 6. Replenishment | `src/replenishment/replenishment_engine.py` | Reorder points, safety stock, inventory simulation |
| 7. Evaluation | `src/evaluation/evaluate.py` | Forecast accuracy + operational KPI tables |
| 8. Dashboard | `src/dashboard/app.py` | Streamlit planning dashboard |

---

## Operational KPIs

**Forecast Accuracy**
- RMSE, MAE, MAPE per model — broken down by category and store

**Inventory & Supply Planning**
- Service level (% days without stockout)
- Stockout days and overstock days
- Average days of supply on hand
- Total replenishment events triggered
- Top 20 items by stockout risk

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the M5 dataset

Download from Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Place these files in `data/raw/`:
- `calendar.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`

### 3. Configure pilot scope (optional)

Edit `config/settings.yaml` to run on a single store first:

```yaml
data:
  pilot_store: CA_1        # or set to null for all stores
  pilot_category: null     # or filter to FOODS, HOBBIES, HOUSEHOLD
```

### 4. Run the pipeline

```bash
# Full pipeline
python run_pipeline.py

# Fast baseline-only run (no LightGBM)
python run_pipeline.py --skip-lgbm

# Specific steps only
python run_pipeline.py --steps 1,2,3
```

### 5. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Project Structure

```
forecasting/
├── config/
│   └── settings.yaml          # Lead time, safety stock, service level, pilot scope
├── data/
│   ├── raw/                   # Source M5 CSV files (not committed)
│   ├── interim/               # Cleaned parquet after ingestion
│   └── processed/             # Feature tables, forecasts, simulations
├── outputs/
│   ├── tables/                # CSV metrics and KPI tables
│   ├── charts/                # Forecast and inventory charts
│   └── models/                # Saved model artifacts
├── src/
│   ├── ingest/ingest.py
│   ├── transform/reshape.py
│   ├── features/feature_engineering.py
│   ├── forecasting/
│   │   ├── baseline_models.py
│   │   └── lgbm_model.py
│   ├── replenishment/replenishment_engine.py
│   ├── evaluation/evaluate.py
│   └── dashboard/app.py
├── run_pipeline.py
└── requirements.txt
```

---

## Key Design Decisions

- **Pilot scope first**: default config uses CA_1 store only — scale by setting `pilot_store: null`
- **Leakage-safe features**: all lag and rolling features use `shift(1)` — no same-day data leakage
- **Time-based splits only**: validation carved from the tail of history; no shuffling
- **Configurable replenishment policy**: lead time, z-score, and coverage target set in `settings.yaml`
- **Parquet throughout**: columnar storage for fast reads; no database required

---

## Keywords

Demand Planning · Inventory Management · Supply Chain · Replenishment · Procurement · Safety Stock · Supply-Demand Alignment · Demand Forecasting · Inventory Optimization · Stockout Reduction
