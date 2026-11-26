"""Uncertainty for forecast-error comparisons.

Owner: Annemarie (see CONTRIBUTING.md).

The question
------------
On B0018, ridge with future rest has lower RMSE than ridge without it.
Is that difference clearly real, or could it come from which few
measurements happened to be in the test set? The bootstrap answers this by
re-drawing the test errors many times and seeing how much the improvement
moves around.

Why a BLOCK bootstrap
---------------------
Errors at neighbouring horizons from the same origin are strongly
correlated: if a forecast is too high at horizon 20, it is almost surely
too high at horizon 21. Resampling single errors would pretend they are
independent and give intervals that are far too narrow. Instead we resample
contiguous blocks of horizons, separately within each origin.

Paired comparison
-----------------
Both models are scored on the same forecast rows, so every bootstrap draw
uses the SAME rows for model A and model B. Differences in difficulty
between rows then cancel out.

Limitation: the three origins overlap in the measurements they predict, so
they are not fully independent; the intervals are therefore approximate.
"""

import numpy as np


def rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))


def _block_indices(n, block_length, rng):
    """Indices for one moving-block resample of a series of length n."""
    block_length = min(block_length, n)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]).ravel()
    return idx[:n]


def block_bootstrap_rmse_ratio(
    errors_a_mah,
    errors_b_mah,
    groups,
    block_length=10,
    n_boot=2000,
    seed=0,
):
    """Relative RMSE reduction of model B over model A, with a 95% interval.

    Parameters
    ----------
    errors_a_mah, errors_b_mah : forecast errors (mAh) for the SAME forecast
        rows, in the same order, ordered by horizon within each group.
        A is the reference (e.g. capacity only); B is the comparison
        (e.g. with future rest).
    groups : origin of each row (e.g. 30, 45, 60). Blocks never cross origins.
    block_length : number of consecutive horizons resampled together.

    Returns
    -------
    dict with
      reduction : 1 - RMSE_B / RMSE_A on the actual data
                  (0.28 means B has 28% lower RMSE than A)
      ci_low, ci_high : 95% percentile interval for the reduction
      prob_b_better : share of bootstrap draws in which B beats A
    """
    a = np.asarray(errors_a_mah, dtype=float)
    b = np.asarray(errors_b_mah, dtype=float)
    groups = np.asarray(groups)
    if not (a.shape == b.shape == groups.shape) or a.ndim != 1:
        raise ValueError("errors_a_mah, errors_b_mah and groups must be 1D and the same length.")
    if block_length < 1 or n_boot < 1:
        raise ValueError("block_length and n_boot must be positive.")

    positions = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)

    reductions = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([
            pos[_block_indices(len(pos), block_length, rng)] for pos in positions
        ])
        reductions[i] = 1 - rmse(b[idx]) / rmse(a[idx])

    low, high = np.percentile(reductions, [2.5, 97.5])
    return {
        "reduction": 1 - rmse(b) / rmse(a),
        "ci_low": float(low),
        "ci_high": float(high),
        "prob_b_better": float(np.mean(reductions > 0)),
        "block_length": block_length,
        "n_boot": n_boot,
    }
