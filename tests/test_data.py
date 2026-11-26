import numpy as np

from battery_forecast.config import BATTERIES
from battery_forecast.data import first_cycle_below


def test_row_counts(table):
    counts = table.groupby("battery_id").size().to_dict()
    assert counts == {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}
    assert len(table) == 636


def test_one_row_per_discharge(table):
    assert not table.duplicated(["battery_id", "discharge_cycle"]).any()
    assert set(table["battery_id"]) == set(BATTERIES)


def test_capacities_plausible(table):
    assert np.isfinite(table["capacity_ah"]).all()
    assert table["capacity_ah"].between(1.0, 2.1).all()


def test_rest_estimates(table):
    rest = table["rest_time_upper_bound_h"]
    # Only each battery's first discharge has no previous discharge.
    assert rest.isna().sum() == 4
    assert (rest.dropna() >= 0).all()
    assert (rest.dropna() <= table["interval_since_previous_discharge_end_h"].dropna() + 1e-9).all()


def test_end_of_life_crossings(table):
    first = first_cycle_below(table, 1.5).to_dict()
    assert first == {"B0005": 99, "B0006": 76, "B0007": 126, "B0018": 70}
    assert np.isnan(first_cycle_below(table, 1.4)["B0007"])


def test_raw_event_chronology():
    """Operations start in time order, and none starts before the previous one ends.

    The rest estimate subtracts recorded operation durations from the gap
    between discharges, which is only valid if operations do not overlap.
    """
    from scipy.io import loadmat
    from battery_forecast.config import RAW_DIR
    from battery_forecast.data import read_timestamp

    for battery in BATTERIES:
        events = loadmat(RAW_DIR / f"{battery}.mat", simplify_cells=True)[battery]["cycle"]
        starts = [read_timestamp(e["time"]) for e in events]
        for i in range(len(events) - 1):
            gap = (starts[i + 1] - starts[i]).total_seconds()
            assert gap >= 0, f"{battery}: event {i + 1} starts before event {i}"
            times = np.asarray(events[i]["data"].get("Time", []), dtype=float).ravel()
            if len(times):
                assert gap - times[-1] > -60, f"{battery}: event {i} overlaps the next one"


def test_adjust_rest_for_impedance(table):
    import pandas as pd
    import pytest
    from battery_forecast.data import adjust_rest_for_impedance

    same = adjust_rest_for_impedance(table, 0)
    pd.testing.assert_frame_equal(same, table)

    shorter = adjust_rest_for_impedance(table, 15)
    has_test = table["impedance_tests_between"].fillna(0) > 0
    # Rests without impedance tests are unchanged; others shrink but never below 0.
    pd.testing.assert_series_equal(shorter.loc[~has_test, "rest_time_upper_bound_h"],
                                   table.loc[~has_test, "rest_time_upper_bound_h"])
    changed = shorter.loc[has_test, "rest_time_upper_bound_h"].dropna()
    assert (changed <= table.loc[changed.index, "rest_time_upper_bound_h"]).all()
    assert (adjust_rest_for_impedance(table, 10_000)["rest_time_upper_bound_h"].dropna() >= 0).all()
    with pytest.raises(ValueError):
        adjust_rest_for_impedance(table, -1)


def test_impedance_windows_match_table_counts(table):
    """Every impedance test counted in the table has exactly one timing window."""
    from battery_forecast.data import impedance_test_windows
    w = impedance_test_windows()
    counts = w.groupby(["battery_id", "discharge_cycle"]).size()
    t = table.set_index(["battery_id", "discharge_cycle"])["impedance_tests_between"]
    t = t[t.fillna(0) > 0]
    assert counts.sort_index().to_dict() == t.astype(int).sort_index().to_dict()


def test_adjustment_never_exceeds_a_tests_window(table):
    from battery_forecast.data import adjust_rest_for_impedance, impedance_test_windows
    w = impedance_test_windows()
    big = adjust_rest_for_impedance(table, 10_000, windows=w)
    removed_h = (table["rest_time_upper_bound_h"] - big["rest_time_upper_bound_h"]).fillna(0)
    allowed_h = (w.groupby(["battery_id", "discharge_cycle"])["window_min"].sum() / 60)
    key = list(zip(table["battery_id"], table["discharge_cycle"]))
    allowed = np.array([allowed_h.get(k, 0.0) for k in key])
    assert (removed_h.to_numpy() <= allowed + 1e-9).all()
