"""Save the train/test forecast examples for inspection (no models fitted).

Usage:  python scripts/export_forecast_examples.py
Output: outputs/forecast_examples/
"""
import _setup  # noqa: F401

from battery_forecast.config import OUTPUT_DIR, SETTINGS, TEST_BATTERY
from battery_forecast.data import load_cycle_table
from battery_forecast.forecasting import prepare_holdout

table = load_cycle_table()
out = OUTPUT_DIR / "forecast_examples"
out.mkdir(parents=True, exist_ok=True)

for setting in SETTINGS:
    split = prepare_holdout(table, test_battery=TEST_BATTERY, setting=setting)
    split["train_examples"].to_csv(out / f"{setting}_train.csv", index=False)
    split["test_examples"].to_csv(out / f"{setting}_test.csv", index=False)
    print(f"{setting}: {len(split['train_examples'])} training examples; "
          f"{len(split['test_examples'])} test examples")
print(f"Saved to {out}")
