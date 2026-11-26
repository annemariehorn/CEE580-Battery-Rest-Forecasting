"""Step 2: choose ridge alpha using only B0005-B0007 (never B0018).

For each rest-feature form (raw hours, log hours), setting and alpha:
hold out each of B0005-B0007 in turn, train on the other two, and record
the RMSE on the held-out battery. One alpha is chosen per rest-feature form,
shared by all three settings, so the settings differ only in rest inputs.
"""
import _setup  # noqa: F401  (lets this script find the project code)

import pandas as pd

from battery_forecast.config import SETTINGS
from battery_forecast.data import load_cycle_table
from battery_forecast.forecasting import (
    attach_predictions,
    prepare_holdout,
    score_predictions,
)
from battery_forecast.models import fit_predict, make_ridge

VALIDATION_BATTERIES = ["B0005", "B0006", "B0007"]
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 3000, 10000, 30000, 100000]

table = load_cycle_table()
# Remove B0018 so it can't be used by accident.
table = table[table["battery_id"] != "B0018"]

rows = []
for log_rest in [False, True]:  # raw rest hours vs log(1 + hours)
    for setting in SETTINGS:
        for alpha in ALPHAS:
            for val_battery in VALIDATION_BATTERIES:
                # Train on the other two validation batteries.
                train_batteries = [b for b in VALIDATION_BATTERIES if b != val_battery]

                split = prepare_holdout(
                    table,
                    test_battery=val_battery,
                    train_batteries=train_batteries,
                    setting=setting,
                )
                model = make_ridge(alpha=alpha, log_rest=log_rest)
                predictions = fit_predict(model, split)
                results = attach_predictions(
                    split["test_examples"], predictions, "ridge", setting
                )
                score = score_predictions(results, by_origin=False)

                # Save this run's error on the held-out battery (mAh).
                rows.append(
                    {
                        "log_rest": log_rest,
                        "setting": setting,
                        "alpha": alpha,
                        "val_battery": val_battery,
                        "rmse_mah": score["rmse_mah"].iloc[0],
                    }
                )

scores = pd.DataFrame(rows)
summary = scores.pivot_table(
    index=["log_rest", "setting", "alpha"], columns="val_battery", values="rmse_mah"
)
summary["mean"] = summary.mean(axis=1)
pd.set_option("display.max_rows", None)
print(summary.round(1))

# One alpha for all three settings: average the mean RMSE across settings.
overall = summary["mean"].groupby(["log_rest", "alpha"]).mean().unstack("log_rest")
print("\nMean RMSE across the three settings (choose the lowest in each column):")
print(overall.round(1))
