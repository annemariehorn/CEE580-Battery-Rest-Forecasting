import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_forecast.config import CYCLE_TABLE_PATH  # noqa: E402
from battery_forecast.data import build_cycle_table, load_cycle_table  # noqa: E402


@pytest.fixture(scope="session")
def table():
    """Use the saved table if present; otherwise build it from the raw files."""
    if CYCLE_TABLE_PATH.exists():
        return load_cycle_table()
    return build_cycle_table()
