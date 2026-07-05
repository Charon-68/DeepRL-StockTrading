"""
Global configuration file for the DeepRL Stock Trading project.
"""

from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

# Project Structure:
#
# DeepRL-StockTrading/
# ├── configs/
# │   └── config.py
# ├── data/
# ├── scripts/
# ├── src/
# ├── results/
# ├── models/
# └── FinRL/
#

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
BACKTEST_DIR = RESULTS_DIR / "backtests"
FIGURES_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ==========================================================
# Dataset Configuration
# ==========================================================

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"

INITIAL_CAPITAL = 1_000_000  # ₹10 Lakhs


# ==========================================================
# NIFTY-50 Stocks
# ==========================================================

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "HINDUNILVR.NS",
]


# ==========================================================
# File Paths
# ==========================================================

RAW_DATA_FILE = RAW_DATA_DIR / "stock.csv"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "stock_processed.csv"


# ==========================================================
# Training Split
# ==========================================================

TRAIN_START = "2014-01-01"
TRAIN_END = "2021-12-31"

VALIDATION_START = "2022-01-01"
VALIDATION_END = "2023-12-31"

TEST_START = "2024-01-01"
TEST_END = "2025-12-31"