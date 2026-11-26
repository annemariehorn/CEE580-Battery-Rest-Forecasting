"""Step 5: are the rest-information improvements on B0018 real or noise?

For each model, compares each rest setting against capacity only on the
same B0018 forecasts (horizons 20-50), with 95% block-bootstrap intervals.
Block lengths 5, 10 and 15 are shown to check the intervals do not depend
much on that choice.

Usage:  python scripts/run_models.py      (creates the predictions first)
        python scripts/bootstrap_comparisons.py
Output: outputs/results/bootstrap_comparisons.csv
"""
import _setup  # noqa: F401

import pandas as pd

from battery_forecast.config import RESULTS_DIR, SCORING_HORIZON, TEST_BATTERY
from battery_forecast.uncertainty import block_bootstrap_rmse_ratio

BLOCK_LENGTHS = [5, 10, 15]
REFERENCE = "capacity_only"

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
    # One column per setting, rows matched on origin and horizon.
    wide = g.pivot_table(index=["origin_cycle", "horizon"], columns="setting", values="error_mah")
    wide = wide.sort_index()
    for setting in [c for c in wide.columns if c != REFERENCE]:
        for block_length in BLOCK_LENGTHS:
            out = block_bootstrap_rmse_ratio(
                wide[REFERENCE], wide[setting],
                groups=wide.index.get_level_values("origin_cycle"),
                block_length=block_length,
            )
            rows.append({"model": model, "setting": setting, **out})

results = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULTS_DIR / "bootstrap_comparisons.csv", index=False)

show = results.copy()
for col in ["reduction", "ci_low", "ci_high", "prob_b_better"]:
    show[col] = (100 * show[col]).round(1)
pd.set_option("display.width", 140)
print("RMSE reduction vs capacity only on B0018, in % (positive = rest information helps).")
print("95% interval from a block bootstrap; if it includes 0, the difference is not clear.\n")
print(show[["model", "setting", "block_length", "reduction", "ci_low", "ci_high",
            "prob_b_better"]].to_string(index=False))
