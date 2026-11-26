"""Step 4: ridge regression and boosted trees in all three information settings.

Ridge is tuned on B0005-B0007 (scripts/tune_ridge.py; docs/decisions.md).
"ridge" uses log rest hours (final); "ridge_raw_rest" is the first version,
kept so both B0018 results are reported. Boosted trees are still UNTUNED.

Usage:  python scripts/run_models.py                # held-out B0018
        python scripts/run_models.py --all-holdouts # also hold out each other battery
Output: outputs/results/model_predictions.csv, model_scores.csv
"""
import _setup  # noqa: F401

import argparse

import pandas as pd

from battery_forecast.config import BATTERIES, RESULTS_DIR, SETTINGS, TEST_BATTERY
from battery_forecast.data import load_cycle_table
from battery_forecast.forecasting import attach_predictions, prepare_holdout, score_predictions
from battery_forecast.models import MODEL_FACTORIES, fit_predict

parser = argparse.ArgumentParser()
parser.add_argument("--all-holdouts", action="store_true",
                    help="Also hold out each of B0005-B0007 (robustness check; optimistic).")
args = parser.parse_args()

table = load_cycle_table()
test_batteries = BATTERIES if args.all_holdouts else [TEST_BATTERY]

frames = []
for test_battery in test_batteries:
    for setting in SETTINGS:
        split = prepare_holdout(table, test_battery=test_battery, setting=setting)
        for name, factory in MODEL_FACTORIES.items():
            preds = fit_predict(factory(), split)
            frames.append(attach_predictions(split["test_examples"], preds, name, setting))

predictions = pd.concat(frames, ignore_index=True)
pooled = score_predictions(predictions, by_origin=False)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(RESULTS_DIR / "model_predictions.csv", index=False)
score_predictions(predictions).to_csv(RESULTS_DIR / "model_scores.csv", index=False)

# Percent change in RMSE relative to the capacity-only version of each model.
ref = pooled[pooled["setting"] == "capacity_only"].set_index(["battery_id", "model"])["rmse_mah"]
pooled["rmse_change_vs_capacity_only_pct"] = [
    100 * (r / ref[(b, m)] - 1) for b, m, r in pooled[["battery_id", "model", "rmse_mah"]].itertuples(index=False)
]

pd.set_option("display.width", 140)
print("Pooled over origins (horizons 20-50, mAh). Ridge tuned; boosted trees UNTUNED:")
print(pooled.round(1).to_string(index=False))
