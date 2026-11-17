"""Step 1: build the shared per-discharge table from the raw NASA files.

Usage:  python scripts/build_table.py
Output: data/processed/per_cycle_table.csv
"""
import _setup  # noqa: F401

from battery_forecast.config import EOL_THRESHOLD_AH, NASA_EOL_THRESHOLD_AH
from battery_forecast.data import build_cycle_table, first_cycle_below, save_cycle_table

table = build_cycle_table()
path = save_cycle_table(table)

print("Discharge measurements per battery:")
print(table.groupby("battery_id").size().to_string())
print(f"\nFirst measurement below {EOL_THRESHOLD_AH} Ah:")
print(first_cycle_below(table, EOL_THRESHOLD_AH).to_string())
print(f"\nFirst measurement below {NASA_EOL_THRESHOLD_AH} Ah (NaN = never):")
print(first_cycle_below(table, NASA_EOL_THRESHOLD_AH).to_string())
print(f"\nSaved {len(table)} rows to {path}")
