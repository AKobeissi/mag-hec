"""
config.py — Central configuration for MAG Energy Solutions Data Challenge.
All paths, hyperparameters, and constants live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Update BASE_DIR to your local data folder
BASE_DIR        = Path(r"C:\Users\akobe\OneDrive\Documents\mag-hec")
DATA_DIR        = BASE_DIR / "data"

COSTS_DIR       = DATA_DIR / "costs"
PRICES_DIR      = DATA_DIR / "prices"
SIM_MONTHLY_DIR = DATA_DIR / "sim_monthly"
SIM_DAILY_DIR   = DATA_DIR / "sim_daily"

OUTPUT_DIR      = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OPPORTUNITIES_PATH = BASE_DIR / "opportunities.csv"

# ── Sim file lists (explicit — avoids glob issues on Windows) ─────────────────
YEARS = [2020, 2021, 2022, 2023, 2024]

SIM_MONTHLY_PATHS = [SIM_MONTHLY_DIR / f"sim_monthly_{y}.parquet" for y in YEARS]
SIM_DAILY_PATHS   = [SIM_DAILY_DIR   / f"sim_daily_{y}.parquet"   for y in YEARS]

# Filter to only files that actually exist on disk
SIM_MONTHLY_PATHS = [p for p in SIM_MONTHLY_PATHS if p.exists()]
SIM_DAILY_PATHS   = [p for p in SIM_DAILY_PATHS   if p.exists()]

# ── Date ranges ───────────────────────────────────────────────────────────────
# Full development data
TRAIN_START     = "2020-01"
TRAIN_END       = "2022-12"
VALID_START     = "2023-01"
VALID_END       = "2023-12"
# 2024 will be added when provided by organizers
TEST_START      = "2024-01"
TEST_END        = "2024-12"

# Minimum months of history required before making a prediction
MIN_HISTORY_MONTHS = 3

# ── Selection constraints ─────────────────────────────────────────────────────
MIN_SELECT_PER_MONTH = 10
MAX_SELECT_PER_MONTH = 100
DEFAULT_SELECT       = 50      # default K — tune this on validation set

# ── Model hyperparameters ─────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        6,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ── Feature windows ───────────────────────────────────────────────────────────
LOOKBACK_SHORT  = 3    # months
LOOKBACK_MEDIUM = 6    # months
LOOKBACK_LONG   = 12   # months
LOOKBACK_LONG2  = 24   # months

# ── Columns to load from each dataset ────────────────────────────────────────
COSTS_COLS          = ["EID", "MONTH", "PEAKID", "C"]
PRICES_COLS         = ["EID", "DATETIME", "PEAKID", "PRICEREALIZED"]
SIM_MONTHLY_COLS    = ["EID", "DATETIME", "PEAKID", "SCENARIOID",
                        "PSM", "ACTIVATIONLEVEL",
                        "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
                        "NONRENEWBALIMPACT", "EXTERNALIMPACT",
                        "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT"]
SIM_DAILY_COLS      = ["EID", "DATETIME", "PEAKID", "SCENARIOID",
                        "PSD", "ACTIVATIONLEVEL"]

# ── Peak mapping ──────────────────────────────────────────────────────────────
PEAKID_TO_NAME  = {0: "OFF", 1: "ON"}
NAME_TO_PEAKID  = {"OFF": 0, "ON": 1}

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42