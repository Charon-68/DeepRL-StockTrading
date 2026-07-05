import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from configs.config import (
    INITIAL_CASH,
    HMAX,
    BUY_COST_PCT,
    SELL_COST_PCT,
    REWARD_SCALING,
    TECHNICAL_INDICATORS,
)


def create_stock_env(df: pd.DataFrame):

    stock_dim = len(df.tic.unique())

    state_space = (
        1
        + 2 * stock_dim
        + stock_dim * len(TECHNICAL_INDICATORS)
    )

    env = StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=HMAX,
        initial_amount=INITIAL_CASH,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[SELL_COST_PCT] * stock_dim,
        reward_scaling=REWARD_SCALING,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=TECHNICAL_INDICATORS,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots=False,
        print_verbosity=1,
    )

    return env