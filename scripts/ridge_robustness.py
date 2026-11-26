"""Robustness checks for the ridge result (does future rest still help?).

These do NOT replace the main analysis (B0018; origins 30, 45, 60; horizons
20-50, fixed in the proposal). They check whether the conclusion depends on
choices that could reasonably have been made differently:

  1. which battery is the test battery (each battery held out in turn;
     optimistic for B0005-B0007, which share a test schedule)
  2. which forecast origins are used (every 5th origin from 20 to 80, B0018)
  3. which horizons are scored (1-20, 20-50, 1-50; B0018)

Model: final ridge (log rest hours, alpha from models.py). Nothing is tuned here.

Usage:  python scripts/ridge_robustness.py
Output: outputs/results/ridge_robustness_*.csv
"""
import _setup  # noqa: F401

import pandas as pd

from battery_forecast.config import BATTERIES, RESULTS_DIR, SETTINGS, TEST_BATTERY
from battery_forecast.data import load_cycle_table
from battery_forecast.forecasting import attach_predictions, prepare_holdout, score_predictions
from battery_forecast.models import fit_predict, make_ridge

ORIGINS = list(range(20, 81, 5))
HORIZON_WINDOWS = [(1, 20), (20, 50), (1, 50)]


def ridge_rmse(table, test_battery, setting, horizons=(20, 50), **holdout_kwargs):
    split = prepare_holdout(table, test_battery=test_battery, setting=setting, **holdout_kwargs)
    preds = attach_predictions(split["test_examples"], fit_predict(make_ridge(), split),
                               "ridge", setting)
    score = score_predictions(preds, horizon_min=horizons[0], horizon_max=horizons[1],
                              by_origin=False)
    return score["rmse_mah"].iloc[0]


def summarise(rows, key):
    """One row per key value: RMSE per setting and % reduction vs capacity only."""
    df = pd.DataFrame(rows).pivot_table(index=key, columns="setting", values="rmse_mah")
    for setting in ["past_rest", "future_rest_oracle"]:
        df[f"{setting}_reduction_pct"] = 100 * (1 - df[setting] / df["capacity_only"])
    return df[["capacity_only", "past_rest", "future_rest_oracle",
               "past_rest_reduction_pct", "future_rest_oracle_reduction_pct"]]


table = load_cycle_table()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 140)

# 1. Each battery held out in turn -------------------------------------------
rows = [{"test_battery": b, "setting": s, "rmse_mah": ridge_rmse(table, b, s)}
        for b in BATTERIES for s in SETTINGS]
by_battery = summarise(rows, "test_battery")
by_battery.to_csv(RESULTS_DIR / "ridge_robustness_by_battery.csv")
print("1. Each battery as the test battery (RMSE mAh; % = reduction vs capacity only)")
print("   B0005-B0007 share a test schedule, so their results are optimistic.\n")
print(by_battery.round(1).to_string(), "\n")

# 2. Forecast origin ------------------------------------------------------------
rows = [{"origin": o, "setting": s,
         "rmse_mah": ridge_rmse(table, TEST_BATTERY, s, test_origins=[o])}
        for o in ORIGINS for s in SETTINGS]
by_origin = summarise(rows, "origin")
by_origin.to_csv(RESULTS_DIR / "ridge_robustness_by_origin.csv")
helped = (by_origin["future_rest_oracle_reduction_pct"] > 5).sum()
hurt = (by_origin["future_rest_oracle_reduction_pct"] < -5).sum()
print(f"2. Forecast origin on {TEST_BATTERY} (main analysis uses 30, 45, 60)\n")
print(by_origin.round(1).to_string())
print(f"   Future rest helped (>5% lower RMSE) at {helped} of {len(ORIGINS)} origins; "
      f"hurt (>5% higher) at {hurt}.\n")

# 3. Scored horizons --------------------------------------------------------------
rows = [{"horizons": f"{lo}-{hi}", "setting": s,
         "rmse_mah": ridge_rmse(table, TEST_BATTERY, s, horizons=(lo, hi))}
        for lo, hi in HORIZON_WINDOWS for s in SETTINGS]
by_horizon = summarise(rows, "horizons")
by_horizon.to_csv(RESULTS_DIR / "ridge_robustness_by_horizon.csv")
print(f"3. Scored horizons on {TEST_BATTERY} (main analysis uses 20-50)\n")
print(by_horizon.round(1).to_string())
