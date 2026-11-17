"""Step 3: score the no-training baselines on the held-out battery.

Usage:  python scripts/run_baselines.py
Output: outputs/results/baseline_predictions.csv, baseline_scores.csv,
        outputs/figures/baseline_forecasts_B0018.png
"""
import _setup  # noqa: F401

import pandas as pd

from battery_forecast.baselines import run_all_baselines
from battery_forecast.config import FIGURE_DIR, RESULTS_DIR, TEST_BATTERY
from battery_forecast.data import load_cycle_table
from battery_forecast.forecasting import attach_predictions, prepare_holdout, score_predictions
from battery_forecast.plotting import plot_forecasts

table = load_cycle_table()
split = prepare_holdout(table, test_battery=TEST_BATTERY, setting="capacity_only")

predictions = pd.concat([
    attach_predictions(split["test_examples"], preds, name, "capacity_only")
    for name, preds in run_all_baselines(table, split["test_examples"]).items()
], ignore_index=True)

scores = score_predictions(predictions)
pooled = score_predictions(predictions, by_origin=False)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)
scores.to_csv(RESULTS_DIR / "baseline_scores.csv", index=False)
plot_forecasts(table, predictions, TEST_BATTERY, FIGURE_DIR / f"baseline_forecasts_{TEST_BATTERY}.png")

pd.set_option("display.width", 120)
print("Scores by origin (horizons 20-50, mAh):")
print(scores.round(1).to_string(index=False))
print("\nPooled over origins:")
print(pooled.round(1).to_string(index=False))
