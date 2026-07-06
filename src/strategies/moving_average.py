import numpy as np
import pandas as pd


def run_ma_strategy(env, df, window=20):
    """
    Simple Moving Average strategy.

    Buy:
        Close > MA

    Sell:
        Close < MA
    """

    obs, info = env.reset()

    stock_dim = env.unwrapped.stock_dim

    portfolio = [env.unwrapped.asset_memory[-1]]

    done = False

    dates = sorted(df["date"].unique())

    for i in range(len(dates) - 1):

        if done:
            break

        current = df[df["date"] == dates[i]]

        action = np.zeros(stock_dim, dtype=np.float32)

        for j, (_, row) in enumerate(current.iterrows()):

            history = df[df["tic"] == row["tic"]]

            history = history[history["date"] <= row["date"]]

            if len(history) < window:
                continue

            ma = history["close"].tail(window).mean()

            if row["close"] > ma:
                action[j] = 1.0

            elif row["close"] < ma:
                action[j] = -1.0

        obs, reward, terminated, truncated, info = env.step(action)

        portfolio.append(env.unwrapped.asset_memory[-1])

        done = terminated or truncated

    return np.array(portfolio)