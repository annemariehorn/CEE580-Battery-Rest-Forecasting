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
    "ridge_raw_rest": "Ridge (raw rest hours)",
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


def plot_rest_comparison(table, predictions, battery, path, model="ridge",
                         settings=("capacity_only", "future_rest_oracle"),
                         scoring_horizon=(20, 50), long_rest_h=10, fontsize=11):
    """One model's forecasts with and without rest information, one panel per origin.

    Shades the scored horizons and marks measurements that follow a long rest,
    so the reader can see where knowing future rests changes the forecast.
    """
    colors = {"capacity_only": "#257de0", "past_rest": "#00b894",
              "future_rest_oracle": "#f47230"}
    styles = {"capacity_only": "--", "past_rest": "-.", "future_rest_oracle": "-"}
    ink, muted = "#191919", "#666666"

    actual = table[table["battery_id"] == battery]
    preds = predictions[(predictions["battery_id"] == battery)
                        & (predictions["model"] == model)]
    origins = sorted(preds["origin_cycle"].unique())
    after_long_rest = actual[actual["rest_time_upper_bound_h"] > long_rest_h]

    fig, axes = plt.subplots(1, len(origins), figsize=(4.6 * len(origins), 4.2),
                             sharey=True, constrained_layout=True, squeeze=False)
    for ax, origin in zip(axes.flat, origins):
        lo, hi = origin + scoring_horizon[0], origin + scoring_horizon[1]
        ax.axvspan(lo, hi, color="#eeeeee", zorder=0,
                   label=f"Scored: {scoring_horizon[0]}–{scoring_horizon[1]} ahead")
        ax.plot(actual["discharge_cycle"], actual["capacity_ah"], color=ink,
                linewidth=1.2, label="Measured capacity", zorder=2)
        ax.scatter(after_long_rest["discharge_cycle"], after_long_rest["capacity_ah"],
                   s=28, color=ink, marker="^", zorder=3,
                   label=f"After a rest > {long_rest_h} h")
        ax.axvline(origin, color=muted, linestyle=":", linewidth=1.2)
        ax.text(origin - 1, 1.83, "forecast\nmade here", ha="right", va="top",
                fontsize=fontsize - 2, color=muted)

        for setting in settings:
            g = preds[(preds["origin_cycle"] == origin) & (preds["setting"] == setting)]
            scored = g[g["horizon"].between(*scoring_horizon)]
            rmse = 1000 * ((scored["predicted_capacity_ah"] - scored["actual_capacity_ah"]) ** 2).mean() ** 0.5
            ax.plot(g["target_cycle"], g["predicted_capacity_ah"], color=colors[setting],
                    linestyle=styles[setting], linewidth=2, zorder=4,
                    label=f"Forecast: {_label(setting)}")
            ax.text(0.98, 0.97 - 0.07 * settings.index(setting),
                    f"{_label(setting)}: {rmse:.0f} mAh",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=fontsize - 2, color=ink)

        ax.set_title(f"Forecast from measurement {origin}", fontsize=fontsize + 1, color=ink)
        ax.set_xlabel("Discharge measurement number", fontsize=fontsize, color=ink)
        ax.set_xlim(0, actual["discharge_cycle"].max() + 2)
        ax.tick_params(labelsize=fontsize - 1, colors=muted)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes.flat[0].set_ylabel("Capacity (Ah)", fontsize=fontsize, color=ink)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels),
               frameon=False, fontsize=fontsize - 1)
    fig.suptitle(f"{battery}: {_label(model)} forecasts with and without future rest times "
                 "(error in mAh, top right)", fontsize=fontsize + 1, color=ink)
    _save(fig, path)


def _save(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
