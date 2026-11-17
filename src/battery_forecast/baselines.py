"""Simple reference forecasts that need no training data.

Each baseline uses only the test battery's own measurements up to the origin.
A learned model is only worth reporting if it beats these.

Owner: Andrew (see CONTRIBUTING.md).
"""

import numpy as np
import pandas as pd

DEFAULT_TREND_WINDOW = 20  # measurements used to fit the trend line


def persistence(test_examples):
    """Predict that capacity stays at its latest observed value."""
    return test_examples["current_capacity_ah"].to_numpy(dtype=float)


def _history(table, battery, origin):
    group = table[table["battery_id"] == battery].sort_values("discharge_cycle")
    return group.iloc[:origin]


def trend(table, test_examples, kind="linear", window=DEFAULT_TREND_WINDOW):
    """Extend a trend fitted to the last `window` measurements up to the origin.

    kind="linear":      capacity = a + b * cycle
    kind="exponential": log(capacity) = a + b * cycle
    """
    if kind not in ("linear", "exponential"):
        raise ValueError("kind must be 'linear' or 'exponential'.")
    if window < 2:
        raise ValueError("window must be at least 2.")

    predictions = np.empty(len(test_examples))
    fits = {}
    for i, (battery, origin, horizon) in enumerate(
        test_examples[["battery_id", "origin_cycle", "horizon"]].itertuples(index=False)
    ):
        key = (battery, origin)
        if key not in fits:
            recent = _history(table, battery, origin).tail(window)
            x = recent["discharge_cycle"].to_numpy(dtype=float)
            y = recent["capacity_ah"].to_numpy(dtype=float)
            if kind == "exponential":
                y = np.log(y)
            fits[key] = np.polyfit(x, y, 1)
        slope, intercept = fits[key]
        value = intercept + slope * (origin + horizon)
        predictions[i] = np.exp(value) if kind == "exponential" else value
    return predictions


def run_all_baselines(table, test_examples, window=DEFAULT_TREND_WINDOW):
    """Return {model_name: predictions} for every baseline."""
    return {
        "persistence": persistence(test_examples),
        "linear_trend": trend(table, test_examples, "linear", window),
        "exponential_trend": trend(table, test_examples, "exponential", window),
    }
