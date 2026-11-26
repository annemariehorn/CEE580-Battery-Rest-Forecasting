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


TARGETS = np.concatenate([np.arange(50, 81), np.arange(65, 96), np.arange(80, 111)])


def test_cautious_identical_and_halved():
    from battery_forecast.uncertainty import target_block_bootstrap_rmse_ratio
    a = _errors()
    same = target_block_bootstrap_rmse_ratio(a, a, TARGETS, n_boot=200)
    assert same["reduction"] == 0 and same["ci_low"] == pytest.approx(0)
    half = target_block_bootstrap_rmse_ratio(a, 0.5 * a, TARGETS, n_boot=200)
    assert half["ci_low"] == pytest.approx(0.5) and half["ci_high"] == pytest.approx(0.5)


def test_both_methods_share_point_estimate():
    from battery_forecast.uncertainty import target_block_bootstrap_rmse_ratio
    a, b = _errors(1), _errors(2)
    within = block_bootstrap_rmse_ratio(a, b, GROUPS, n_boot=1000)
    cautious = target_block_bootstrap_rmse_ratio(a, b, TARGETS, n_boot=1000)
    assert cautious["reduction"] == within["reduction"]  # same data, same point estimate


@pytest.mark.parametrize("bad", ["nan", "zero", "length"])
def test_rejects_bad_inputs(bad):
    a, b, g = _errors(1), _errors(2), GROUPS
    if bad == "nan":
        b = b.copy(); b[3] = np.nan
    elif bad == "zero":
        a = np.zeros_like(a)
    else:
        g = g[:-1]
    with pytest.raises(ValueError):
        block_bootstrap_rmse_ratio(a, b, g, n_boot=10)
