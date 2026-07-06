from pathlib import Path
import sys
from src.strategies.buy_and_hold import run_buy_and_hold
from src.strategies.moving_average import run_ma_strategy
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.visualization.plots import (
    save_equity_curve,
    save_metric_bar,
    save_drawdown_curve,
)
import pandas as pd
from stable_baselines3 import DQN, DDPG

from src.envs.stock_env import (
    create_test_env,
    create_test_env_discrete,
)

from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import compute_metrics


def evaluate_model(model, env, name):
    print(f"\nEvaluating {name}...")

    portfolio = run_backtest(model, env)

    metrics = compute_metrics(portfolio)

    print(f"{name} Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")

    return metrics


def main():

    print("=" * 60)
    print("Loading processed dataset...")
    print("=" * 60)

    df = pd.read_csv("data/processed/stock_processed.csv")

    results = {}
    
    portfolios = {} 

    # ============================================================
    # DQN
    # ============================================================

    # ===========================
# DQN
# ===========================

    dqn_env = create_test_env_discrete(df)

    dqn = DQN.load("models/dqn/dqn_run_1_final.zip")

    dqn_portfolio = run_backtest(dqn, dqn_env)

    portfolios["DQN"] = dqn_portfolio

    results["DQN"] = compute_metrics(dqn_portfolio)

    print(results["DQN"])

    # ============================================================
    # DDPG
    # ============================================================

    # ===========================
# DDPG
# ===========================

    ddpg_env = create_test_env(df)

    ddpg = DDPG.load("models/ddpg/ddpg_run_1_final.zip")

    ddpg_portfolio = run_backtest(ddpg, ddpg_env)

    portfolios["DDPG"] = ddpg_portfolio

    results["DDPG"] = compute_metrics(ddpg_portfolio)

    print(results["DDPG"])

        # ===========================
    # Buy & Hold
    # ===========================

    buy_env = create_test_env(df)

    buy_portfolio = run_buy_and_hold(buy_env)

    results["Buy & Hold"] = compute_metrics(
        buy_portfolio
    )
    portfolios["Buy & Hold"] = buy_portfolio

    # ===========================
    # Moving Average
    # ===========================

    ma_env = create_test_env(df)

    ma_portfolio = run_ma_strategy(
        ma_env,
        df,
    )

    results["MA Strategy"] = compute_metrics(
        ma_portfolio
    )
    portfolios["MA Strategy"] = ma_portfolio
    # ============================================================
    # Comparison table
    # ============================================================

    table = pd.DataFrame(results).T

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(table)

    Path("results").mkdir(exist_ok=True)

    table.to_csv(
        "results/evaluation_metrics.csv"
    )
    
    print("\nSaved metrics to results/evaluation_metrics.csv")
    save_equity_curve(portfolios)

    save_metric_bar(
        results,
        metric="Sharpe",
        filename="results/figures/sharpe_bar.png",
    )

    save_metric_bar(
        results,
        metric="Total Return",
        filename="results/figures/return_bar.png",
    )

    best_name = max(
        results,
        key=lambda x: results[x]["Sharpe"],
    )

    save_drawdown_curve(
        portfolios[best_name],
        filename="results/figures/drawdown.png",
    )

    print("\nFigures saved to results/figures/")

if __name__ == "__main__":
    main()