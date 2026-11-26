"""Are the rest-information improvements on B0018 real or noise?

For each model, compares the information settings pairwise on the same B0018
forecasts (horizons 20-50):
  capacity only -> past rest       (does past rest help?)
  capacity only -> future rest     (do past + future rest help?)
  past rest     -> future rest     (does FUTURE rest add anything beyond past rest?)

Two 95% intervals are reported (see uncertainty.py):
  within_origin : blocks of horizons within each origin
  cautious      : blocks of predicted measurements, keeping overlapping
                  origins together (wider; the safer one to quote)
Block lengths 5, 10 and 15 are saved to check sensitivity to that choice.

Usage:  python scripts/run_models.py      (creates the predictions first)
        python scripts/bootstrap_comparisons.py
Output: outputs/results/bootstrap_comparisons.csv
"""
import _setup  # noqa: F401

import pandas as pd

from battery_forecast.config import RESULTS_DIR, SCORING_HORIZON, TEST_BATTERY
from battery_forecast.uncertainty import (
    block_bootstrap_rmse_ratio,
    target_block_bootstrap_rmse_ratio,
)

BLOCK_LENGTHS = [5, 10, 15]
MAIN_BLOCK_LENGTH = 10
COMPARISONS = [
    ("capacity_only", "past_rest"),
    ("capacity_only", "future_rest_oracle"),
    ("past_rest", "future_rest_oracle"),
]
KEYS = ["origin_cycle", "horizon", "target_cycle"]

path = RESULTS_DIR / "model_predictions.csv"
if not path.exists():
    raise SystemExit(f"{path} not found. Run `python scripts/run_models.py` first.")

preds = pd.read_csv(path)
preds = preds[
    (preds["battery_id"] == TEST_BATTERY)
    & preds["horizon"].between(*SCORING_HORIZON)
].copy()
preds["error_mah"] = (preds["predicted_capacity_ah"] - preds["actual_capacity_ah"]) * 1000

rows = []
for model, g in preds.groupby("model"):
    if g.duplicated(KEYS + ["setting"]).any():
        raise ValueError(f"{model}: duplicate forecast rows.")
    # One column per setting, rows matched on origin, horizon and target.
    wide = g.pivot(index=KEYS, columns="setting", values="error_mah").sort_index()
    for reference, comparison in COMPARISONS:
        pair = wide[[reference, comparison]]
        # Every forecast must exist in both settings; otherwise the pairing is broken.
        if pair.isna().any().any():
            raise ValueError(f"{model}: {reference} and {comparison} cover different forecasts.")
        for block_length in BLOCK_LENGTHS:
            within = block_bootstrap_rmse_ratio(
                pair[reference], pair[comparison],
                groups=pair.index.get_level_values("origin_cycle"),
                block_length=block_length,
            )
            cautious = target_block_bootstrap_rmse_ratio(
                pair[reference], pair[comparison],
                targets=pair.index.get_level_values("target_cycle"),
                block_length=block_length,
            )
            rows.append({
                "model": model, "reference": reference, "comparison": comparison,
                "block_length": block_length, "reduction": within["reduction"],
                "within_origin_low": within["ci_low"], "within_origin_high": within["ci_high"],
                "cautious_low": cautious["ci_low"], "cautious_high": cautious["ci_high"],
                "cautious_prob_better": cautious["prob_b_better"],
            })

results = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULTS_DIR / "bootstrap_comparisons.csv", index=False)

show = results[results["block_length"] == MAIN_BLOCK_LENGTH].copy()
pct = ["reduction", "within_origin_low", "within_origin_high", "cautious_low",
       "cautious_high", "cautious_prob_better"]
show[pct] = (100 * show[pct]).round(1)
pd.set_option("display.width", 160)
print(f"RMSE reduction on {TEST_BATTERY}, in % (positive = the comparison setting is better).")
print("95% intervals, block length 10. If an interval includes 0, the difference is not clear.")
print("'cautious' keeps overlapping forecasts together and is the safer interval to quote.\n")
print(show.drop(columns="block_length").to_string(index=False))

spread = results.groupby(["model", "reference", "comparison"])[["cautious_low", "cautious_high"]]
spread = (100 * (spread.max() - spread.min())).round(1)
print("\nSensitivity to block length (5, 10, 15): largest change in the cautious interval ends, % points")
print(spread.to_string())
