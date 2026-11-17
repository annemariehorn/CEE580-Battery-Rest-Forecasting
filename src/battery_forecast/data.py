"""Build and load the shared per-discharge table from the NASA .mat files.

One row per discharge measurement (one full discharge = one cycle).
Rest time before each discharge is estimated as:

    time since the previous discharge ended
    - recorded durations of operations in between (charges, impedance tests)

Impedance tests have no recorded duration, so the estimate is an
UPPER BOUND on true rest time.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import BATTERIES, CYCLE_TABLE_PATH, RAW_DIR

REQUIRED_COLUMNS = {
    "battery_id",
    "discharge_cycle",
    "capacity_ah",
    "rest_time_upper_bound_h",
}


def read_timestamp(values):
    """Convert a MATLAB date vector [Y, M, D, h, m, s] to a datetime."""
    values = np.asarray(values).ravel()
    return datetime(*map(int, values[:5])) + timedelta(seconds=float(values[5]))


def _find_battery_file(raw_dir: Path, battery: str) -> Path:
    files = list(Path(raw_dir).glob(f"{battery}*.mat"))
    if len(files) != 1:
        raise ValueError(f"Expected one file for {battery}; found {len(files)}.")
    return files[0]


def read_battery_rows(raw_dir: Path, battery: str) -> list:
    """Return one dict per discharge for a single battery."""
    events = loadmat(_find_battery_file(raw_dir, battery), simplify_cells=True)[battery]["cycle"]

    rows = []
    previous_discharge_end = None
    active_seconds_between = 0.0
    impedance_tests_between = 0
    discharge_number = 0

    for event in events:
        operation = event["type"]
        data = event["data"]
        start = read_timestamp(event["time"])

        # Recorded elapsed seconds within this operation.
        times = np.asarray(data.get("Time", []), dtype=float).ravel()
        duration_seconds = float(times[-1]) if len(times) else np.nan

        if operation == "discharge":
            discharge_number += 1
            if not np.isfinite(duration_seconds):
                raise ValueError(
                    f"Missing discharge duration: {battery}, measurement {discharge_number}"
                )
            end = start + timedelta(seconds=duration_seconds)

            if previous_discharge_end is None:
                interval_hours = np.nan
                rest_hours = np.nan
            else:
                interval_seconds = (start - previous_discharge_end).total_seconds()
                rest_seconds = interval_seconds - active_seconds_between
                if rest_seconds < -1e-6:
                    raise ValueError("Operation times overlap.")
                interval_hours = interval_seconds / 3600
                rest_hours = max(rest_seconds, 0) / 3600

            rows.append({
                "battery_id": battery,
                "discharge_cycle": discharge_number,
                "discharge_start": start,
                "discharge_end": end,
                "capacity_ah": float(data["Capacity"]),
                "interval_since_previous_discharge_end_h": interval_hours,
                "known_active_time_between_h": (
                    active_seconds_between / 3600
                    if previous_discharge_end is not None else np.nan
                ),
                # Upper bound: includes unrecorded impedance-test duration.
                "rest_time_upper_bound_h": rest_hours,
                "impedance_tests_between": impedance_tests_between,
                "rest_includes_unknown_test_time": (
                    impedance_tests_between > 0
                    if previous_discharge_end is not None else None
                ),
            })

            previous_discharge_end = end
            active_seconds_between = 0.0
            impedance_tests_between = 0

        elif previous_discharge_end is not None:
            if np.isfinite(duration_seconds):
                active_seconds_between += duration_seconds
            if operation == "impedance":
                impedance_tests_between += 1

    return rows


def build_cycle_table(raw_dir: Path = RAW_DIR, batteries=BATTERIES) -> pd.DataFrame:
    """Build the per-discharge table for all batteries."""
    rows = []
    for battery in batteries:
        rows.extend(read_battery_rows(raw_dir, battery))

    table = (
        pd.DataFrame(rows)
        .sort_values(["battery_id", "discharge_cycle"])
        .reset_index(drop=True)
    )
    table["capacity_change_mah"] = table.groupby("battery_id")["capacity_ah"].diff() * 1000
    return table


def save_cycle_table(table: pd.DataFrame, path: Path = CYCLE_TABLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_cycle_table(path: Path = CYCLE_TABLE_PATH) -> pd.DataFrame:
    """Load and validate the shared per-discharge table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/build_table.py` first."
        )
    table = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    table = table.sort_values(["battery_id", "discharge_cycle"]).reset_index(drop=True)

    if table.duplicated(["battery_id", "discharge_cycle"]).any():
        raise ValueError("Duplicate battery/cycle rows.")
    if not np.isfinite(table["capacity_ah"]).all():
        raise ValueError("Capacities must be finite.")

    for battery, group in table.groupby("battery_id"):
        expected = np.arange(1, len(group) + 1)
        if not np.array_equal(group["discharge_cycle"].to_numpy(), expected):
            raise ValueError(
                f"{battery}: discharge numbers must be consecutive and start at 1."
            )
    return table


def first_cycle_below(table: pd.DataFrame, threshold_ah: float) -> pd.Series:
    """First discharge measurement at which capacity falls below a threshold.

    Uses the FIRST crossing; capacity can bounce back above the threshold
    after a long rest. Returns NaN for batteries that never cross it.
    """
    def _first(group):
        below = group.loc[group["capacity_ah"] < threshold_ah, "discharge_cycle"]
        return below.iloc[0] if len(below) else np.nan

    return table.groupby("battery_id").apply(_first, include_groups=False)
