import numpy as np
import pytest

from battery_forecast.forecasting import (
    CAPACITY_FEATURES,
    FUTURE_REST_FEATURES,
    PAST_REST_FEATURES,
    attach_predictions,
    feature_columns,
    make_forecast_examples,
    prepare_holdout,
    score_predictions,
)


def test_origin_and_horizon_mapping(table):
    ex = make_forecast_examples(table[table.battery_id == "B0018"], origins=[30])
    assert ex["horizon"].min() == 1 and ex["horizon"].max() == 50
    assert (ex["target_cycle"] == 30 + ex["horizon"]).all()


def test_settings_add_only_rest_features():
    assert feature_columns("past_rest") == CAPACITY_FEATURES + PAST_REST_FEATURES
    assert feature_columns("future_rest_oracle") == (
        CAPACITY_FEATURES + PAST_REST_FEATURES + FUTURE_REST_FEATURES
    )


@pytest.mark.parametrize("setting", ["capacity_only", "past_rest"])
def test_no_future_leakage(table, setting):
    """Changing data after the origin must not change any non-oracle feature."""
    b18 = table[table.battery_id == "B0018"].copy()
    before = make_forecast_examples(b18, setting=setting, origins=[45])

    altered = b18.copy()
    after_origin = altered["discharge_cycle"] > 45
    altered.loc[after_origin, "capacity_ah"] += 0.5
    altered.loc[after_origin, "rest_time_upper_bound_h"] += 100
    after = make_forecast_examples(altered, setting=setting, origins=[45])

    cols = feature_columns(setting)
    np.testing.assert_allclose(before[cols].to_numpy(), after[cols].to_numpy())


def test_holdout_excludes_test_battery(table):
    split = prepare_holdout(table, test_battery="B0018")
    assert "B0018" not in set(split["train_examples"]["battery_id"])
    assert set(split["test_examples"]["battery_id"]) == {"B0018"}
    assert sorted(split["test_examples"]["origin_cycle"].unique()) == [30, 45, 60]


def test_persistence_reproduces_reported_scores(table):
    """Matches Table 2 of the progress report."""
    split = prepare_holdout(table, test_battery="B0018")
    preds = attach_predictions(
        split["test_examples"], split["test_examples"]["current_capacity_ah"],
        "persistence", "capacity_only",
    )
    scores = score_predictions(preds).set_index("origin_cycle")
    np.testing.assert_allclose(
        scores["rmse_mah"].loc[[30, 45, 60]], [156.535, 134.159, 170.536], atol=1e-2
    )
    assert (scores["n_predictions"] == 31).all()
