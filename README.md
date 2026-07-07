# DeepRL-StockTrading
> A research-oriented Deep Reinforcement Learning framework for multi-asset stock trading using custom risk-aware rewards, transaction costs, and FinRL-based market environments.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FinRL](https://img.shields.io/badge/Framework-FinRL-green)
![StableBaselines3](https://img.shields.io/badge/RL-SB3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

DeepRL-StockTrading is a modular reinforcement learning framework for algorithmic stock trading built on top of the FinRL ecosystem. The project investigates the effectiveness of Deep Reinforcement Learning algorithms under realistic market conditions by incorporating:

- Multi-asset portfolio trading
- Technical indicator based state representations
- Transaction costs
- Risk-aware reward engineering
- Portfolio volatility penalties
- Multiple RL agents (DQN, DDPG, PPO, SAC)
- Backtesting and performance evaluation

The framework is designed for experimentation, reproducibility, and research-oriented benchmarking.

---

# Project Objectives

This project aims to study:

- How Deep RL agents learn trading strategies
- The impact of transaction costs on policy learning
- Risk-aware reward shaping in financial markets
- Comparison of value-based and policy-gradient methods
- Portfolio performance under realistic market constraints

---

# Current Progress

### Data Pipeline
- [x] Yahoo Finance market data ingestion
- [x] Multi-asset dataset creation
- [x] Feature engineering using FinRL

### Trading Environment
- [x] Custom FinRL stock trading environment
- [x] Transaction cost modeling
- [x] Portfolio accounting
- [x] Environment validation

### Reward Engineering
- [x] Risk-aware reward implementation
- [x] Volatility penalty
- [x] Transaction penalty

### RL Agents
- [x] PPO (Proximal Policy Optimization)
- [x] SAC (Soft Actor-Critic)
- [x] DDPG (Deep Deterministic Policy Gradient)
- [x] DQN (Deep Q-Network, discrete action wrapper)

### Evaluation
- [ ] Backtesting
- [ ] Benchmark comparison
- [ ] Risk metrics
- [ ] Visualization dashboard

---

# Project Structure 

configs/: configuration files
data/: datasets and processed artifacts
docs/: documentation
models/: trained model checkpoints
notebooks/: exploratory notebooks
results/: evaluation outputs and plots
scripts/: utility scripts
src/: source code package
tests/: tests
---

# Project Architecture

```text
Market Data
      │
      ▼
Data Download Pipeline
      │
      ▼
Feature Engineering
      │
      ▼
Stock Trading Environment
      │
      ▼
Risk-Aware Reward Function
      │
      ▼
RL Agent Training
      │
      ▼
Backtesting & Evaluation
      │
      ▼
Performance Analysis
```

---

# Repository Structure

```text
DeepRL-StockTrading/

├── configs/                # Global configuration files
├── data/
│   ├── raw/                # Downloaded market data
│   ├── processed/          # Feature-engineered datasets
│   ├── interim/
│   └── external/
│
├── experiments/            # Experiment configurations
│   ├── dqn/
│   ├── ddpg/
│   ├── ppo/
│   └── sac/
│
├── scripts/                # Executable pipelines
│   ├── download_data.py
│   ├── build_features.py
│   ├── test_environment.py
│   ├── train_dqn.py
│   ├── train_ddpq.py
│   └── evaluate.py
│
├── src/
│   ├── agents/
│   │   ├── ppo_agent.py
│   │   ├── sac_agent.py
│   │   ├── ddpg_agent.py
│   │   └── dqn_agent.py
│   ├── envs/
│   │   ├── stock_env.py
│   │   └── discrete_wrapper.py
│   ├── reward/
│   ├── backtesting/
│   ├── evaluation/
│   ├── training/
│   ├── visualization/
│   └── utils/
│
├── results/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   ├── metrics/
│   └── backtests/
│
└── FinRL/                  # Cloned FinRL framework
```

---

# Dataset

### Source
- Yahoo Finance

### Assets

Current portfolio consists of:

| Asset |
|--------|
| RELIANCE.NS |
| TCS.NS |
| INFY.NS |
| HDFCBANK.NS |
| ICICIBANK.NS |
| SBIN.NS |
| ITC.NS |
| LT.NS |
| AXISBANK.NS |
| HINDUNILVR.NS |

### Time Period

```text
2014-01-01 → 2025-12-31
```

### Dataset Statistics

| Metric | Value |
|---------|--------|
| Assets | 10 |
| Trading Days | 2959 |
| Samples | 29590 |

---

# Feature Engineering

Technical indicators are generated using FinRL's `FeatureEngineer`.

Current state features include:

| Feature |
|---------|
| MACD |
| Bollinger Upper Band |
| Bollinger Lower Band |
| RSI(30) |
| CCI(30) |
| DX(30) |
| SMA(30) |
| SMA(60) |

The resulting state vector contains:

```text
State Space =
Cash
+ Stock Prices
+ Portfolio Holdings
+ Technical Indicators
```

Current state dimension:

```text
101
```

---

# Trading Environment

The environment is based on:

```text
FinRL.meta.env_stock_trading.StockTradingEnv
```

### Environment Settings

| Parameter | Value |
|-----------|--------|
| Initial Cash | ₹1,000,000 |
| Max Shares Per Trade | 100 |
| Buy Transaction Cost | 0.1% |
| Sell Transaction Cost | 0.1% |
| Reward Scaling | 1e-4 |

### Actions

Continuous portfolio actions:

```text
[-1, 1]

-1 : Sell
 0 : Hold
+1 : Buy
```

---

# Risk-Aware Reward Function

The default portfolio-profit reward is replaced with a custom reward:

```math
R_t =
\Delta PortfolioValue
-
\lambda \times Volatility
-
\mu \times TransactionCost
```

where:

- PortfolioValue = change in portfolio value
- Volatility = rolling portfolio volatility
- TransactionCost = current transaction cost

### Hyperparameters

| Parameter | Value |
|-----------|--------|
| λ (volatility penalty) | 0.01 |
| μ (transaction penalty) | 0.01 |
| Rolling Window | 20 |

---

# Reinforcement Learning Algorithms

The framework supports:

| Algorithm | Type | Action Space | Status |
|-----------|------|-------------|--------|
| PPO | On-policy | Continuous | Complete |
| SAC | Off-policy | Continuous | Complete |
| DDPG | Off-policy | Continuous | Complete |
| DQN | Off-policy | Discrete (21 actions) | Complete |

Implementation uses:

- Stable-Baselines3
- PyTorch
- FinRL

---

# Installation

Clone repository:

```bash
git clone <repository-url>
cd DeepRL-StockTrading
```

Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install FinRL:

```bash
cd FinRL
pip install -e .
cd ..
```

---

# Running the Pipeline

### 1. Download Data

```bash
python -m scripts.download_data
```

---

### 2. Generate Features

```bash
python -m scripts.build_features
```

---

### 3. Test Environment

```bash
python -m scripts.test_environment
```

---

### 4. Train Agents

Run each agent sequentially (each takes ~2 hours on GPU):

```bash
# PPO — on-policy, continuous actions
python -m scripts.train_ppo

# SAC — off-policy, entropy regularization
python -m scripts.train_sac

# DDPG — off-policy, deterministic policy
python -m scripts.train_ddpg

# DQN — off-policy, discrete action space
python -m scripts.train_dqn
```

Monitor training in real time:

```bash
tensorboard --logdir results/logs --port 6007
```

Trained models are saved to:

```text
models/
├── ppo/ppo_run_1_final.zip
├── sac/sac_run_1_final.zip
├── ddpg/ddpg_run_1_final.zip
└── dqn/dqn_run_1_final.zip
```

---

### 5. Evaluate

```bash
python -m scripts.evaluate
```

---

# Planned Experiments

- [x] Vanilla DQN
- [x] DDPG
- [x] PPO
- [x] SAC
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Ensemble Models
- [ ] Hyperparameter Optimization
- [ ] Risk-aware Reward Ablation
- [ ] Transaction Cost Sensitivity Analysis

---

# Evaluation Metrics

Performance will be evaluated using:

- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Portfolio Volatility
- Win Rate
- Profit Factor

---

# Technologies Used

- Python 3.11
- PyTorch
- Stable-Baselines3
- FinRL
- Gymnasium
- NumPy
- Pandas
- Matplotlib
- Yahoo Finance API

---

# References

1. Liu et al., "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance", NeurIPS 2020.

2. Sutton & Barto, "Reinforcement Learning: An Introduction".

3. Stable-Baselines3 Documentation.

4. FinRL Official Repository:
https://github.com/AI4Finance-Foundation/FinRL

---

# License

This project is released under the MIT License.
