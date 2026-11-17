"""Step 2: exploratory figures (Figures 1 and 2 of the progress report).

Usage:  python scripts/make_figures.py
Output: outputs/figures/
"""
import _setup  # noqa: F401

from battery_forecast.config import FIGURE_DIR
from battery_forecast.data import load_cycle_table
from battery_forecast.plotting import plot_capacity_history, plot_rest_vs_capacity_change

table = load_cycle_table()
plot_capacity_history(table, FIGURE_DIR / "capacity_history.png")
plot_rest_vs_capacity_change(table, FIGURE_DIR / "rest_vs_capacity_change.png")
print(f"Saved figures to {FIGURE_DIR}")
