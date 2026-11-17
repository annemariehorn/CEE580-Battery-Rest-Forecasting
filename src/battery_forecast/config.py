"""Shared settings. Every group member should use these values."""

from pathlib import Path

# Repository layout -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

CYCLE_TABLE_PATH = PROCESSED_DIR / "per_cycle_table.csv"

# Data -----------------------------------------------------------------------
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH = 2.0          # nominal capacity of every cell
EOL_THRESHOLD_AH = 1.5           # study end-of-life threshold (25% fade)
NASA_EOL_THRESHOLD_AH = 1.4      # NASA's 30% fade criterion (B0007 never reaches it)

# Forecasting design ---------------------------------------------------------
TEST_BATTERY = "B0018"           # only battery tested on its own schedule
FORECAST_ORIGINS = [30, 45, 60]  # last measurement the model may see
MAX_HORIZON = 50                 # predict 1-50 measurements ahead
SCORING_HORIZON = (20, 50)       # primary metric uses horizons 20-50 inclusive
MIN_HISTORY = 10                 # earliest origin used for training examples
EOL_MAX_HORIZON = 100            # longer forecasts for the end-of-life analysis

SETTINGS = ["capacity_only", "past_rest", "future_rest_oracle"]
