import numpy as np


def run_buy_and_hold(env):
    """
    Buy all stocks equally on the first day and hold.

    Returns
    -------
    np.ndarray
        Portfolio value over time.
    """

    obs, info = env.reset()

    stock_dim = env.unwrapped.stock_dim

    portfolio = [env.unwrapped.asset_memory[-1]]

    # Buy all stocks on first step
    first_action = np.ones(stock_dim, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(first_action)

    portfolio.append(env.unwrapped.asset_memory[-1])

    done = terminated or truncated

    # Hold afterwards
    hold_action = np.zeros(stock_dim, dtype=np.float32)

    while not done:

        obs, reward, terminated, truncated, info = env.step(hold_action)

        portfolio.append(env.unwrapped.asset_memory[-1])

        done = terminated or truncated

    return np.array(portfolio)