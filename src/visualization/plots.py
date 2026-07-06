from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_equity_curve(portfolios, filename="results/figures/equity_curve.png"):
    """
    portfolios:
        dict[str, np.ndarray]
    """

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for name, values in portfolios.items():
        plt.plot(values, label=name)

    plt.title("Portfolio Value Over Time")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_metric_bar(results,
                    metric="Sharpe",
                    filename="results/figures/sharpe_bar.png"):

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    values = [results[k][metric] for k in names]

    plt.figure(figsize=(8, 5))

    plt.bar(names, values)

    plt.title(metric)
    plt.ylabel(metric)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_drawdown_curve(portfolio,
                        filename="results/figures/drawdown.png"):

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    running_max = np.maximum.accumulate(portfolio)

    drawdown = (
        portfolio - running_max
    ) / running_max

    plt.figure(figsize=(12, 5))

    plt.plot(drawdown)

    plt.title("Drawdown Curve")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()