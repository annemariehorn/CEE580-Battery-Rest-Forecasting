import numpy as np
import pytest

from battery_forecast.uncertainty import block_bootstrap_rmse_ratio

GROUPS = np.repeat([30, 45, 60], 31)


def _errors(seed=1):
    return np.random.default_rng(seed).normal(0, 40, size=len(GROUPS))


def test_identical_models_give_zero():
    a = _errors()
    out = block_bootstrap_rmse_ratio(a, a, GROUPS, n_boot=200)
    assert out["reduction"] == 0
    assert out["ci_low"] == pytest.approx(0) and out["ci_high"] == pytest.approx(0)


def test_half_the_error_is_exactly_fifty_percent():
    a = _errors()
    out = block_bootstrap_rmse_ratio(a, 0.5 * a, GROUPS, n_boot=200)
    assert out["reduction"] == pytest.approx(0.5)
    assert out["ci_low"] == pytest.approx(0.5) and out["ci_high"] == pytest.approx(0.5)
    assert out["prob_b_better"] == 1


def test_same_seed_same_answer():
    a, b = _errors(1), _errors(2)
    assert block_bootstrap_rmse_ratio(a, b, GROUPS, n_boot=200, seed=3) == \
        block_bootstrap_rmse_ratio(a, b, GROUPS, n_boot=200, seed=3)


def test_interval_contains_estimate_for_unrelated_errors():
    a, b = _errors(1), _errors(2)
    out = block_bootstrap_rmse_ratio(a, b, GROUPS, n_boot=500)
    assert out["ci_low"] < out["reduction"] < out["ci_high"]
