"""Uncertainty for forecast-error comparisons.

Owner: Annemarie (see CONTRIBUTING.md). This module is a specification
to implement, not finished code.

Why a BLOCK bootstrap
---------------------
Errors at neighbouring horizons from the same origin are strongly
correlated, and the three origins overlap in the measurements they predict.
Resampling individual errors would pretend they are independent and give
intervals that are far too narrow. Resample contiguous blocks of horizons
instead, separately within each origin.
"""

import numpy as np


def block_bootstrap_rmse_ratio(
    errors_a_mah,
    errors_b_mah,
    block_length=10,
    n_boot=2000,
    seed=0,
):
    """Confidence interval for the relative RMSE reduction of model B over model A.

    Parameters
    ----------
    errors_a_mah, errors_b_mah : arrays of forecast errors (mAh) for the SAME
        forecast rows, ordered by origin then horizon.
    block_length : number of consecutive horizons resampled together.

    Returns
    -------
    dict with the point estimate of 1 - RMSE_B / RMSE_A and a 95% interval.

    Implementation notes
    --------------------
    - Resample the same block positions for A and B (paired comparison).
    - Resample within each origin, then pool.
    - Report sensitivity to block_length (e.g. 5, 10, 15).
    """
    raise NotImplementedError("To be implemented (see docstring).")


def rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))
