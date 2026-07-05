"""
Sanity check for FinRL StockTradingEnv
"""

import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from configs.config import (
    PROCESSED_DATA_PATH,
    TECHNICAL_INDICATORS,
)


def main():

    print("=" * 80)
    print("Loading Processed Dataset")
    print("=" * 80)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # ------------------------------------------------------------------
    # Sort dataframe
    # ------------------------------------------------------------------

    df = (
        df.sort_values(["date", "tic"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # VERY IMPORTANT
    # FinRL expects all stocks of the same trading day to have
    # the same dataframe index.
    # ------------------------------------------------------------------

    df.index = pd.factorize(df["date"])[0]

    stock_dim = len(df["tic"].unique())

    state_space = (
        1
        + 2 * stock_dim
        + len(TECHNICAL_INDICATORS) * stock_dim
    )

    print(f"Stocks       : {stock_dim}")
    print(f"Trading Days : {len(df.index.unique())}")
    print(f"Rows         : {len(df)}")
    print(f"State Space  : {state_space}")

    print("\nCreating Environment...\n")

    env = StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=TECHNICAL_INDICATORS,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots=False,
        print_verbosity=1,
    )

    print("=" * 80)
    print("Reset Environment")
    print("=" * 80)

    state, info = env.reset()

    print(f"State Length : {len(state)}")
    print(f"Initial Cash : {state[0]:,.2f}")
    print()

    # ============================================================
    # BUY
    # ============================================================

    print("=" * 80)
    print("BUY ACTION")
    print("=" * 80)

    action = np.ones(stock_dim)

    state, reward, terminated, truncated, info = env.step(action)

    print(f"Reward           : {reward}")
    print(f"Portfolio Value  : {env.asset_memory[-1]:,.2f}")
    print(f"Cash Remaining   : {state[0]:,.2f}")
    print(f"Shares Held      : {state[1+stock_dim:1+2*stock_dim]}")
    print()

    # ============================================================
    # HOLD
    # ============================================================

    print("=" * 80)
    print("HOLD ACTION")
    print("=" * 80)

    action = np.zeros(stock_dim)

    state, reward, terminated, truncated, info = env.step(action)

    print(f"Reward           : {reward}")
    print(f"Portfolio Value  : {env.asset_memory[-1]:,.2f}")
    print(f"Cash Remaining   : {state[0]:,.2f}")
    print()

    # ============================================================
    # SELL
    # ============================================================

    print("=" * 80)
    print("SELL ACTION")
    print("=" * 80)

    action = -np.ones(stock_dim)

    state, reward, terminated, truncated, info = env.step(action)

    print(f"Reward           : {reward}")
    print(f"Portfolio Value  : {env.asset_memory[-1]:,.2f}")
    print(f"Cash Remaining   : {state[0]:,.2f}")
    print(f"Shares Held      : {state[1+stock_dim:1+2*stock_dim]}")
    print()

    print("=" * 80)
    print("Environment Test Successful")
    print("=" * 80)


if __name__ == "__main__":
    main()