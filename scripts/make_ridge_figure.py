"""Ridge forecasts on B0018 with and without future rest times (final-report figure).

Usage:  python scripts/run_models.py        (creates the predictions first)
        python scripts/make_ridge_figure.py
Output: outputs/figures/ridge_rest_comparison_B0018.png
"""
import _setup  # noqa: F401

import pandas as pd

from battery_forecast.config import FIGURE_DIR, RESULTS_DIR, TEST_BATTERY
from battery_forecast.data import load_cycle_table
from battery_forecast.plotting import plot_rest_comparison

path = RESULTS_DIR / "model_predictions.csv"
if not path.exists():
    raise SystemExit(f"{path} not found. Run `python scripts/run_models.py` first.")

out = FIGURE_DIR / f"ridge_rest_comparison_{TEST_BATTERY}.png"
plot_rest_comparison(load_cycle_table(), pd.read_csv(path), TEST_BATTERY, out, model="ridge")
print(f"Saved {out}")
