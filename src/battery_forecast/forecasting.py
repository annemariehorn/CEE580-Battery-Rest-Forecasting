"""Shared forecasting framework: features, train/test splits and scoring.

Terminology
-----------
origin   : the last discharge measurement a model is allowed to see.
horizon  : how many measurements ahead the forecast targets.
           Origin 30, horizon 1 -> measurement 31; horizon 50 -> measurement 80.
setting  : which information the model gets
           - capacity_only       capacity history only
           - past_rest           + rest times observed up to the origin
           - future_rest_oracle  + the ACTUAL future rest times, supplied
                                   retrospectively ("oracle"); this measures the
                                   most rest information could possibly help.
"""

import numpy as np
import pandas as pd

from .config import (
    FORECAST_ORIGINS,
    MAX_HORIZON,
    MIN_HISTORY,
    SCORING_HORIZON,
    TEST_BATTERY,
)

CAPACITY_FEATURES = [
    "origin_cycle",
    "horizon",
    "current_capacity_ah",
    "mean_capacity_last_5_ah",
    "recent_slope_ah_per_cycle",
]

PAST_REST_FEATURES = [
    "last_rest_upper_bound_h",
    "mean_rest_last_10_h",
]

FUTURE_REST_FEATURES = [
    "target_rest_upper_bound_h",
    "mean_future_rest_h",
    "max_future_rest_h",
]


def feature_columns(setting="capacity_only"):
    """Return the model input columns for one information setting."""
    if setting == "capacity_only":
        return CAPACITY_FEATURES.copy()
    if setting == "past_rest":
        return CAPACITY_FEATURES + PAST_REST_FEATURES
    if setting == "future_rest_oracle":
        return CAPACITY_FEATURES + PAST_REST_FEATURES + FUTURE_REST_FEATURES
    raise ValueError(f"Unknown setting: {setting}")


def make_forecast_examples(
    table,
    setting="capacity_only",
    origins=None,
    max_horizon=MAX_HORIZON,
    min_history=MIN_HISTORY,
):
    """Create one row for each battery/origin/horizon combination.

    origins=None generates every eligible origin (used for training).
    Only measurements 1..origin are used to build features, except the
    oracle future-rest features in the future_rest_oracle setting.
    """
    columns = feature_columns(setting)

    if min_history < 2:
        raise ValueError("min_history must be at least 2.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive.")
    if origins is not None:
        origins = list(origins)
        if len(origins) != len(set(origins)):
            raise ValueError("Origins must not contain duplicates.")

    rows = []
    for battery, group in table.groupby("battery_id"):
        group = group.sort_values("discharge_cycle").reset_index(drop=True)
        eligible_origins = range(min_history, len(group)) if origins is None else origins

        for origin in eligible_origins:
            if not isinstance(origin, (int, np.integer)):
                raise ValueError("Origins must be integers.")
            if not min_history <= origin < len(group):
                raise ValueError(f"Invalid origin {origin} for {battery}.")

            # Only observations through the origin are used.
            history = group.iloc[:origin]
            recent = history.tail(10)

            base = {
                "battery_id": battery,
                "origin_cycle": origin,
                "current_capacity_ah": history["capacity_ah"].iloc[-1],
                "mean_capacity_last_5_ah": history["capacity_ah"].tail(5).mean(),
                "recent_slope_ah_per_cycle": np.polyfit(
                    recent["discharge_cycle"], recent["capacity_ah"], 1
                )[0],
            }

            if setting in ("past_rest", "future_rest_oracle"):
                past_rest = history["rest_time_upper_bound_h"]
                base.update({
                    "last_rest_upper_bound_h": past_rest.iloc[-1],
                    "mean_rest_last_10_h": past_rest.tail(10).mean(),
                })

            available_horizon = min(max_horizon, len(group) - origin)
            for horizon in range(1, available_horizon + 1):
                target = group.iloc[origin + horizon - 1]
                row = {
                    **base,
                    "horizon": horizon,
                    "target_cycle": origin + horizon,
                    "actual_capacity_ah": target["capacity_ah"],
                }
                if setting == "future_rest_oracle":
                    future_rest = group.iloc[origin:origin + horizon]["rest_time_upper_bound_h"]
                    row.update({
                        "target_rest_upper_bound_h": future_rest.iloc[-1],
                        "mean_future_rest_h": future_rest.mean(),
                        "max_future_rest_h": future_rest.max(),
                    })
                rows.append(row)

    if not rows:
        raise ValueError("No forecast examples were generated.")

    examples = pd.DataFrame(rows)
    metadata = ["battery_id", "target_cycle"]  # identifiers kept apart from inputs
    return examples[metadata + columns + ["actual_capacity_ah"]]


def prepare_holdout(
    table,
    test_battery=TEST_BATTERY,
    setting="capacity_only",
    test_origins=FORECAST_ORIGINS,
    max_horizon=MAX_HORIZON,
    train_batteries=None,
):
    """Train on other batteries; evaluate on one unseen battery.

    Training examples use all eligible origins; test examples use only the
    agreed evaluation origins. Cycles are never split randomly.
    train_batteries=None uses every battery except the test battery.
    """
    if test_battery not in set(table["battery_id"]):
        raise ValueError(f"Unknown battery: {test_battery}")

    if train_batteries is None:
        train_mask = table["battery_id"] != test_battery
    else:
        if test_battery in train_batteries:
            raise ValueError("Test battery must not be a training battery.")
        train_mask = table["battery_id"].isin(train_batteries)

    train_table = table[train_mask].copy()
    test_table = table[table["battery_id"] == test_battery].copy()
    if train_table.empty:
        raise ValueError("No training batteries remain.")

    train = make_forecast_examples(train_table, setting=setting, max_horizon=max_horizon)
    test = make_forecast_examples(
        test_table, setting=setting, origins=test_origins, max_horizon=max_horizon
    )
    columns = feature_columns(setting)

    return {
        "train_examples": train,
        "test_examples": test,
        "X_train": train[columns].copy(),
        "y_train": train["actual_capacity_ah"].copy(),
        "X_test": test[columns].copy(),
        "y_test": test["actual_capacity_ah"].copy(),
    }


def leave_one_battery_out(table, setting="capacity_only", batteries=None, **kwargs):
    """Yield (test_battery, split) for each battery held out in turn.

    Robustness check only: B0005-B0007 share a rest schedule, so holding out
    one of them while training on the other two is optimistic.
    """
    batteries = batteries or sorted(table["battery_id"].unique())
    for battery in batteries:
        yield battery, prepare_holdout(table, test_battery=battery, setting=setting, **kwargs)


def attach_predictions(test_examples, predicted_capacity_ah, model_name, setting):
    """Attach a model's predictions to the correct forecast rows."""
    predictions = np.asarray(predicted_capacity_ah, dtype=float)
    if predictions.ndim != 1:
        raise ValueError("Predictions must be a 1D array.")
    if len(predictions) != len(test_examples):
        raise ValueError("Need exactly one prediction per test example.")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite.")

    result = test_examples[[
        "battery_id", "origin_cycle", "target_cycle", "horizon", "actual_capacity_ah",
    ]].copy()
    result["predicted_capacity_ah"] = predictions
    result["model"] = model_name
    result["setting"] = setting
    return result


def score_predictions(
    predictions,
    horizon_min=SCORING_HORIZON[0],
    horizon_max=SCORING_HORIZON[1],
    by_origin=True,
):
    """Score forecasts within the scoring horizon window.

    RMSE: root-mean-square error.  MAE: mean absolute error.
    Bias: mean signed error (positive = overprediction).  All in mAh.
    by_origin=False pools the three origins into one score per model.
    """
    required = {
        "battery_id", "origin_cycle", "horizon", "actual_capacity_ah",
        "predicted_capacity_ah", "model", "setting",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    unique_keys = ["battery_id", "origin_cycle", "model", "setting", "horizon"]
    if predictions.duplicated(unique_keys).any():
        raise ValueError("Duplicate forecast rows.")

    selected = predictions[predictions["horizon"].between(horizon_min, horizon_max)].copy()
    if selected.empty:
        raise ValueError("No predictions in the scoring horizon.")

    values = selected[["actual_capacity_ah", "predicted_capacity_ah"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Actual values and predictions must be finite.")

    selected["error_mah"] = (
        selected["predicted_capacity_ah"] - selected["actual_capacity_ah"]
    ) * 1000

    keys = ["battery_id", "model", "setting"]
    if by_origin:
        keys.insert(1, "origin_cycle")

    return (
        selected.groupby(keys)["error_mah"]
        .agg(
            n_predictions="size",
            rmse_mah=lambda e: np.sqrt(np.mean(e ** 2)),
            mae_mah=lambda e: np.mean(np.abs(e)),
            bias_mah="mean",
        )
        .reset_index()
    )
