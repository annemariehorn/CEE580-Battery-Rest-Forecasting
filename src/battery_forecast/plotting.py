"""Figures used in the reports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import BATTERIES, EOL_THRESHOLD_AH  # noqa: E402

# Readable names for figure legends (code names use underscores).
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "linear_trend": "Linear trend",
    "exponential_trend": "Exponential trend",
    "ridge": "Ridge regression",
    "boosted_trees": "Boosted trees",
    "capacity_only": "capacity only",
    "past_rest": "past rest",
    "future_rest_oracle": "future rest, oracle",
}


def _label(name):
    return DISPLAY_NAMES.get(name, name.replace("_", " "))


def plot_capacity_history(table, path, long_rest_quantile=0.90):
    """Figure 1: capacity vs discharge measurement, long rests marked."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for ax, battery in zip(axes.flat, BATTERIES):
        group = table[table["battery_id"] == battery]
        cutoff = group["rest_time_upper_bound_h"].quantile(long_rest_quantile)
        long_rests = group[group["rest_time_upper_bound_h"] > cutoff]

        ax.plot(group["discharge_cycle"], group["capacity_ah"], label="Measured capacity")
        ax.scatter(
            long_rests["discharge_cycle"], long_rests["capacity_ah"],
            color="darkorange", label="Longest 10% of rest estimates", zorder=3,
        )
        ax.axhline(EOL_THRESHOLD_AH, color="crimson", linestyle="--",
                   label=f"Study threshold: {EOL_THRESHOLD_AH} Ah")
        ax.set(title=battery, xlabel="Discharge measurement number", ylabel="Capacity (Ah)")
        ax.legend(fontsize=8)

    fig.suptitle("Battery capacity over time")
    _save(fig, path)


def plot_rest_vs_capacity_change(table, path, fontsize=12):
    """Figure 2: change in capacity vs preceding rest estimate."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for battery in BATTERIES:
        group = table[table["battery_id"] == battery].dropna(
            subset=["rest_time_upper_bound_h", "capacity_change_mah"]
        )
        ax.scatter(group["rest_time_upper_bound_h"], group["capacity_change_mah"],
                   label=battery, alpha=0.65, s=25)

    ax.set_xscale("symlog", linthresh=1)  # linear near zero, logarithmic beyond 1 h
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Are longer rest estimates followed by capacity increases?", fontsize=fontsize + 1)
    ax.set_xlabel("Rest-time upper bound (hours; symmetric log scale)", fontsize=fontsize)
    ax.set_ylabel("Change from previous capacity measurement (mAh)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.legend(title="Battery", fontsize=fontsize - 1)
    _save(fig, path)


def plot_forecasts(table, predictions, battery, path, models=None):
    """Actual capacity with each model's forecasts, one panel per origin."""
    preds = predictions[predictions["battery_id"] == battery]
    if models is not None:
        preds = preds[preds["model"].isin(models)]
    origins = sorted(preds["origin_cycle"].unique())
    actual = table[table["battery_id"] == battery]

    fig, axes = plt.subplots(1, len(origins), figsize=(5 * len(origins), 4),
                             sharey=True, constrained_layout=True, squeeze=False)
    for ax, origin in zip(axes.flat, origins):
        ax.plot(actual["discharge_cycle"], actual["capacity_ah"], color="black",
                linewidth=1, label="Measured")
        ax.axvline(origin, color="gray", linestyle=":", label="Forecast origin")
        for (model, setting), g in preds[preds["origin_cycle"] == origin].groupby(["model", "setting"]):
            label = _label(model) if setting == "capacity_only" else f"{_label(model)} ({_label(setting)})"
            ax.plot(g["target_cycle"], g["predicted_capacity_ah"], label=label)
        ax.set(title=f"{battery}, origin {origin}", xlabel="Discharge measurement number")
    axes.flat[0].set_ylabel("Capacity (Ah)")
    axes.flat[-1].legend(fontsize=8)
    _save(fig, path)


def _save(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
