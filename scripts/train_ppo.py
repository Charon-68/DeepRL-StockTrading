from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from configs.config import (
    PROCESSED_DATA_PATH,
    PPO_TOTAL_TIMESTEPS,
)

from src.envs.stock_env import create_train_env
from src.agents.ppo_agent import PPOAgent


def main():

    print("=" * 60)
    print("PPO TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------------------
    print("\n[1/3] Loading processed dataset...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at:\n  {PROCESSED_DATA_PATH}\n\n"
            "Run this first:\n  python -m scripts.build_features"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    print(f"  Rows   : {len(df):,}")
    print(f"  Dates  : {df['date'].min()} → {df['date'].max()}")
    print(f"  Stocks : {df['tic'].nunique()}")

    # ------------------------------------------------------------------
    # 2. Create training environment (2014–2021)
    # ------------------------------------------------------------------
    print("\n[2/3] Creating training environment...")

    train_env = create_train_env(df)

    print(f"  Obs space    : {train_env.observation_space.shape}")
    print(f"  Action space : {train_env.action_space.shape}")

    # ------------------------------------------------------------------
    # 3. Train PPO
    # ------------------------------------------------------------------
    print("\n[3/3] Initialising PPO agent...")

    agent = PPOAgent(env=train_env, run_name="ppo_run_1")

    agent.train(total_timesteps=PPO_TOTAL_TIMESTEPS)

    agent.save()

    print("\n" + "=" * 60)
    print("DONE — run tensorboard to view training curves:")
    print("  tensorboard --logdir results/logs")
    print("=" * 60)


if __name__ == "__main__":
    main()