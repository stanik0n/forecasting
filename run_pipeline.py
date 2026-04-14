"""
End-to-end pipeline runner.
Executes all 8 layers in sequence:
  1. Ingest raw CSVs -> interim parquet
  2. Transform (melt + join) -> fact_sales_daily, dim_calendar
  3. Feature engineering -> fact_features
  4. Baseline models (moving average + seasonal naive)
  5. LightGBM model
  6. Replenishment engine
  7. Evaluation (metrics + charts)
  8. Dashboard data ready -> launch Streamlit

Usage:
  python run_pipeline.py                  # full pipeline
  python run_pipeline.py --steps 1,2,3   # specific steps only
  python run_pipeline.py --skip-lgbm     # skip LightGBM (fast baseline-only run)
"""

import argparse
import sys
import io

# Force UTF-8 output on Windows so Unicode in docstrings/comments doesn't crash printing
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
import traceback

CONFIG_PATH = "config/settings.yaml"


def step_header(n: int, name: str):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {name}")
    print(f"{'='*60}")


def run_step(n: int, name: str, fn, *args, **kwargs):
    step_header(n, name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n  [OK] Completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        print(f"\n  [FAILED]: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="M5 Inventory Replenishment Pipeline")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers to run (e.g. --steps 1,2,3). Default: all steps.",
    )
    parser.add_argument(
        "--skip-lgbm",
        action="store_true",
        help="Skip the LightGBM step (useful for fast baseline-only runs).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    requested_steps = set(int(s) for s in args.steps.split(",")) if args.steps else set(range(1, 9))

    # Late imports so import errors don't block --help
    from src.ingest import ingest
    from src.transform import reshape
    from src.features import feature_engineering
    from src.forecasting import baseline_models
    from src.forecasting import lgbm_model
    from src.replenishment import replenishment_engine
    from src.evaluation import evaluate

    print("\nM5 Inventory Replenishment & Demand Forecasting Pipeline")
    print(f"Config: {args.config}")

    pipeline_start = time.time()

    if 1 in requested_steps:
        run_step(1, "Data Ingestion", ingest.run, args.config)

    if 2 in requested_steps:
        run_step(2, "Reshape Sales (wide to long + joins)", reshape.run, args.config)

    if 3 in requested_steps:
        run_step(3, "Feature Engineering", feature_engineering.run, args.config)

    if 4 in requested_steps:
        run_step(4, "Baseline Models (Moving Avg + Seasonal Naive)", baseline_models.run, args.config)

    if 5 in requested_steps and not args.skip_lgbm:
        run_step(5, "LightGBM Model", lgbm_model.run, args.config)
    elif 5 in requested_steps:
        print("\n  Step 5: LightGBM - SKIPPED (--skip-lgbm flag)")

    if 6 in requested_steps:
        run_step(6, "Replenishment Engine", replenishment_engine.run, args.config)

    if 7 in requested_steps:
        run_step(7, "Evaluation & Charts", evaluate.run, args.config)

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'='*60}")

    if 8 in requested_steps:
        print("\nTo launch the Streamlit dashboard, run:")
        print("  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
