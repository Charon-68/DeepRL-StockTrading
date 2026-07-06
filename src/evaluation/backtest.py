from stable_baselines3 import DQN, DDPG
import numpy as np


def run_backtest(model, env):

    obs, info = env.reset()

    portfolio = [env.unwrapped.asset_memory[-1]]

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        portfolio.append(
            env.unwrapped.asset_memory[-1]
        )

        done = terminated or truncated

    return np.array(portfolio)