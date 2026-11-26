"""End-to-end check that the headline ridge results reproduce.

If a change anywhere in the pipeline moves these numbers, this test fails,
so the README and report numbers cannot silently go out of date.
"""
import numpy as np
import pytest

from battery_forecast.config import SETTINGS
from battery_forecast.data import adjust_rest_for_impedance, build_cycle_table
from battery_forecast.forecasting import attach_predictions, prepare_holdout, score_predictions
from battery_forecast.models import fit_predict, make_ridge

EXPECTED_RMSE = {"capacity_only": 34.7, "past_rest": 33.5, "future_rest_oracle": 25.1}


@pytest.fixture(scope="module")
def ridge_predictions():
    table = build_cycle_table()  # rebuilt from the raw files, not the saved CSV
    assert len(table) == 636
    out = {}
    for setting in SETTINGS:
        split = prepare_holdout(table, test_battery="B0018", setting=setting)
        out[setting] = attach_predictions(
            split["test_examples"], fit_predict(make_ridge(), split), "ridge", setting)
    return table, out


def test_headline_ridge_rmse(ridge_predictions):
    _, preds = ridge_predictions
    for setting, expected in EXPECTED_RMSE.items():
        rmse = score_predictions(preds[setting], by_origin=False)["rmse_mah"].iloc[0]
        assert rmse == pytest.approx(expected, abs=0.05), setting


def test_settings_score_identical_forecasts(ridge_predictions):
    _, preds = ridge_predictions
    keys = [set(map(tuple, p[p["horizon"].between(20, 50)][["origin_cycle", "horizon"]].to_numpy()))
            for p in preds.values()]
    assert all(k == keys[0] for k in keys) and len(keys[0]) == 93
    for p in preds.values():
        assert np.isfinite(p["predicted_capacity_ah"]).all()


def test_rest_correction_leaves_capacity_only_unchanged(ridge_predictions):
    table, preds = ridge_predictions
    adjusted = adjust_rest_for_impedance(table, 32)
    split = prepare_holdout(adjusted, test_battery="B0018", setting="capacity_only")
    np.testing.assert_allclose(fit_predict(make_ridge(), split),
                               preds["capacity_only"]["predicted_capacity_ah"])
