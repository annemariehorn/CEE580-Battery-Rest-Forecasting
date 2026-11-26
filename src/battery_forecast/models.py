"""Learned forecasting models: ridge regression and boosted trees.

Ridge is TUNED (scripts/tune_ridge.py; docs/decisions.md): log rest hours,
alpha = 1000. Boosted trees are still an UNTUNED starting point. To tune:
  1. choose settings (tree depth, number of trees, learning rate) by holding
     out each of B0005-B0007 in turn -- never by looking at B0018;
  2. keep the three information settings identical except for rest features;
  3. record final settings and the reason for them in docs/decisions.md.

Owners: ridge -- Annemarie; boosted trees -- Emi (see CONTRIBUTING.md).

Target choice
-------------
By default models predict the CHANGE in capacity from the origin
(future capacity - current capacity) and add it back afterwards. Batteries
start at different capacities, so the change transfers better across
batteries than the absolute level. Set predict_change=False to compare.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .forecasting import FUTURE_REST_FEATURES, PAST_REST_FEATURES

RANDOM_STATE = 0


REST_FEATURES = PAST_REST_FEATURES + FUTURE_REST_FEATURES


def log_rest_features(X):
    """Replace rest times (hours) with log(1 + hours).

    Capacity recovery levels off for long rests, so a 241 h rest should not
    count 241 times as much as a 1 h rest. Capacity inputs are unchanged.
    """
    X = X.copy()
    cols = [c for c in REST_FEATURES if c in X.columns]
    X[cols] = np.log1p(X[cols])
    return X


def make_ridge(alpha=1000, log_rest=True):
    """Ridge regression with features standardized on the training data only.

    Final choice (docs/decisions.md): log rest times, alpha = 1000.
    The original version (raw rest hours, alpha = 3000) is kept as
    "ridge_raw_rest" so both results can be reported.
    """
    steps = [FunctionTransformer(log_rest_features)] if log_rest else []
    return make_pipeline(*steps, StandardScaler(), Ridge(alpha=alpha))


def make_ridge_raw_rest(alpha=3000):
    """Original ridge: raw rest hours (first B0018 result, Nov 26)."""
    return make_ridge(alpha=alpha, log_rest=False)


def make_boosted_trees(
    max_depth=2, n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
):
    """Gradient-boosted trees. Keep trees shallow: only three training batteries."""
    return GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )


MODEL_FACTORIES = {
    "ridge": make_ridge,
    "ridge_raw_rest": make_ridge_raw_rest,
    "boosted_trees": make_boosted_trees,
}


def fit_predict(model, split, predict_change=True):
    """Fit on split['X_train'] and return capacity predictions for split['X_test'].

    The model is fitted on training batteries only, so nothing from the
    test battery leaks into model development.
    """
    X_train, y_train = split["X_train"], split["y_train"]
    X_test = split["X_test"]

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Features contain missing values; handle them before fitting.")

    if predict_change:
        model.fit(X_train, y_train - X_train["current_capacity_ah"])
        return (
            np.asarray(model.predict(X_test)) + X_test["current_capacity_ah"].to_numpy()
        )

    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test))
