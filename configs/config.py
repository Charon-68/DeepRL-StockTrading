from pathlib import Path

from finrl.config import INDICATORS

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "stock.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "stock_processed.csv"

# =============================================================================
# Dataset
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
    "AXISBANK.NS",
    "HINDUNILVR.NS",
]

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"

# =============================================================================
# Feature Engineering
# =============================================================================

TECHNICAL_INDICATORS = INDICATORS

# =============================================================================
# Trading Environment
# =============================================================================

INITIAL_CASH = 1_000_000

HMAX = 100

BUY_COST_PCT = 0.001      # 0.1%

SELL_COST_PCT = 0.001     # 0.1%

REWARD_SCALING = 1e-4

# =============================================================================
# Risk-Aware Reward
# =============================================================================

VOLATILITY_WINDOW = 20

LAMBDA_VOLATILITY = 0.01

MU_TRANSACTION = 0.01