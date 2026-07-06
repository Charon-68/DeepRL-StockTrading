import numpy as np


def compute_metrics(portfolio_values):
    portfolio_values = np.asarray(portfolio_values)

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    total_return = portfolio_values[-1] / portfolio_values[0] - 1

    years = len(portfolio_values) / 252

    cagr = (
        portfolio_values[-1] /
        portfolio_values[0]
    ) ** (1 / years) - 1

    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        sharpe = 0.0
    else:
        sharpe = (
            np.mean(daily_returns)
            / np.std(daily_returns)
        ) * np.sqrt(252)

    running_max = np.maximum.accumulate(portfolio_values)

    drawdown = (
        portfolio_values
        - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    win_rate = np.mean(daily_returns > 0)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": abs(max_drawdown),
        "Win Rate": win_rate,
    }