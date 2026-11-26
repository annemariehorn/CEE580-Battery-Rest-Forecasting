"""Rest-duration sensitivity: does the ridge result survive shorter rest times?

Rest times are upper bounds because impedance-test durations were not
recorded. Here each impedance test is assumed to have lasted 0, 15 or 32
minutes, and that time is subtracted from every rest containing tests.
32 minutes is about the largest value the timestamps allow: the next
operation typically starts about 32 minutes after an impedance test begins
(this script prints that check). These are scenarios, not estimates of the
true duration.

Everything else is held fixed: same design (train B0005-B0007, test B0018,
origins 30/45/60, horizons 20-50) and the same final ridge (log rest hours,
alpha from models.py; not re-tuned per scenario).

Usage:  python scripts/rest_duration_sensitivity.py
Output: outputs/results/rest_duration_sensitivity.csv
"""
import _setup  # noqa: F401

import numpy as np
import pandas as pd
from scipy.io import loadmat

from battery_forecast.config import BATTERIES, RAW_DIR, RESULTS_DIR, SETTINGS, TEST_BATTERY
from battery_forecast.data import adjust_rest_for_impedance, load_cycle_table, read_timestamp
from battery_forecast.forecasting import attach_predictions, prepare_holdout, score_predictions
from battery_forecast.models import fit_predict, make_ridge
from battery_forecast.uncertainty import (
    block_bootstrap_rmse_ratio,
    target_block_bootstrap_rmse_ratio,
)

MINUTES_PER_TEST = [0, 15, 32]

# 0. How much time do the timestamps allow for an impedance test? ------------
windows = []
for battery in BATTERIES:
    events = loadmat(RAW_DIR / f"{battery}.mat", simplify_cells=True)[battery]["cycle"]
    for i in range(len(events) - 1):
        if events[i]["type"] == "impedance":
            gap = read_timestamp(events[i + 1]["time"]) - read_timestamp(events[i]["time"])
            windows.append(gap.total_seconds() / 60)
windows = np.array(windows)
print("0. Minutes from the start of an impedance test to the next operation "
      f"(all {len(windows)} tests):")
print(f"   10th percentile {np.percentile(windows, 10):.1f}, median {np.median(windows):.1f}, "
      f"minimum {windows.min():.1f}. So tests took at most about 32 minutes.\n")

base = load_cycle_table()
rows, rest_rows, preds_by_d = [], [], {}
for minutes in MINUTES_PER_TEST:
    table = adjust_rest_for_impedance(base, minutes)

    # How much did the rest inputs actually change?
    for scope, t, b in [("all batteries", table, base),
                        (TEST_BATTERY, table[table.battery_id == TEST_BATTERY],
                         base[base.battery_id == TEST_BATTERY])]:
        rest, orig = t["rest_time_upper_bound_h"].dropna(), b["rest_time_upper_bound_h"].dropna()
        rest_rows.append({
            "minutes_per_test": minutes, "scope": scope,
            "median_rest_h": rest.median(), "max_rest_h": rest.max(),
            "share_changed_pct": 100 * (rest != orig).mean(),
            "share_now_zero_pct": 100 * (rest == 0).mean(),
        })

    preds_by_d[minutes] = {}
    for setting in SETTINGS:
        split = prepare_holdout(table, test_battery=TEST_BATTERY, setting=setting)
        preds = attach_predictions(split["test_examples"], fit_predict(make_ridge(), split),
                                   "ridge", setting)
        preds_by_d[minutes][setting] = preds
        rmse = score_predictions(preds, by_origin=False)["rmse_mah"].iloc[0]
        rows.append({"minutes_per_test": minutes, "setting": setting, "rmse_mah": rmse})

# Safety check: capacity-only ridge uses no rest inputs, so it must not change.
for minutes in MINUTES_PER_TEST[1:]:
    a = preds_by_d[0]["capacity_only"]["predicted_capacity_ah"].to_numpy()
    b = preds_by_d[minutes]["capacity_only"]["predicted_capacity_ah"].to_numpy()
    if not np.allclose(a, b):
        raise AssertionError("Capacity-only predictions changed; something other than rest changed.")

rest_summary = pd.DataFrame(rest_rows)
print("1. How much the corrections change the rest inputs")
print(rest_summary.round(2).to_string(index=False), "\n")


def errors(minutes, setting):
    p = preds_by_d[minutes][setting]
    p = p[p["horizon"].between(20, 50)].sort_values(["origin_cycle", "horizon"])
    return p, (p["predicted_capacity_ah"] - p["actual_capacity_ah"]).to_numpy() * 1000


summary = pd.DataFrame(rows).pivot(index="minutes_per_test", columns="setting", values="rmse_mah")
summary = summary[SETTINGS]
for minutes in MINUTES_PER_TEST:
    p, e_cap = errors(minutes, "capacity_only")
    _, e_past = errors(minutes, "past_rest")
    _, e_fut = errors(minutes, "future_rest_oracle")
    for name, ref in [("vs_capacity", e_cap), ("vs_past", e_past)]:
        within = block_bootstrap_rmse_ratio(ref, e_fut, p["origin_cycle"])
        cautious = target_block_bootstrap_rmse_ratio(ref, e_fut, p["target_cycle"])
        summary.loc[minutes, f"future_{name}_pct"] = 100 * within["reduction"]
        summary.loc[minutes, f"future_{name}_cautious"] = (
            f"{100 * cautious['ci_low']:.1f} to {100 * cautious['ci_high']:.1f}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(RESULTS_DIR / "rest_duration_sensitivity.csv")
rest_summary.to_csv(RESULTS_DIR / "rest_duration_sensitivity_inputs.csv", index=False)
pd.set_option("display.width", 160)
print(f"2. Ridge on {TEST_BATTERY}: RMSE (mAh) and improvement from future rest (%),")
print("   with target-block bootstrap sensitivity intervals (block length 10)")
print(summary.round(1).to_string())
