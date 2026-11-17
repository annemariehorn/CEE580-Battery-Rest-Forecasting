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
