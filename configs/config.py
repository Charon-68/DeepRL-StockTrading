from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "stock.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "stock_processed.csv"

# =============================================================================
# Dataset Configuration
# =============================================================================

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
]

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"

# =============================================================================
# Feature Engineering
# =============================================================================

TECHNICAL_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]