import numpy as np
import pandas as pd

from battery_forecast.baselines import persistence, trend
from battery_forecast.forecasting import make_forecast_examples


def _synthetic(kind):
    cycles = np.arange(1, 101)
    capacity = 2.0 - 0.005 * cycles if kind == "linear" else 2.0 * np.exp(-0.003 * cycles)
    return pd.DataFrame({
        "battery_id": "X", "discharge_cycle": cycles, "capacity_ah": capacity,
        "rest_time_upper_bound_h": 1.0,
    })


def test_linear_trend_exact_on_line():
    table = _synthetic("linear")
    ex = make_forecast_examples(table, origins=[40])
    np.testing.assert_allclose(trend(table, ex, "linear"), ex["actual_capacity_ah"], atol=1e-10)


def test_exponential_trend_exact_on_exponential():
    table = _synthetic("exponential")
    ex = make_forecast_examples(table, origins=[40])
    np.testing.assert_allclose(trend(table, ex, "exponential"), ex["actual_capacity_ah"], atol=1e-10)


def test_persistence_is_last_value():
    table = _synthetic("linear")
    ex = make_forecast_examples(table, origins=[40])
    assert np.allclose(persistence(ex), table.loc[39, "capacity_ah"])


def test_log_rest_only_changes_rest_columns():
    from battery_forecast.models import log_rest_features
    X = pd.DataFrame({"current_capacity_ah": [1.8, 1.7], "last_rest_upper_bound_h": [0.0, 241.0]})
    out = log_rest_features(X)
    assert out["current_capacity_ah"].tolist() == [1.8, 1.7]
    np.testing.assert_allclose(out["last_rest_upper_bound_h"], np.log1p([0.0, 241.0]))
