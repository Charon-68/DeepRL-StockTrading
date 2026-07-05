# DeepRL-StockTrading

This repository contains a structured project layout for a deep reinforcement learning stock trading project.

## Project Structure

- configs/: configuration files
- data/: datasets and processed artifacts
- docs/: documentation
- models/: trained model checkpoints
- notebooks/: exploratory notebooks
- results/: evaluation outputs and plots
- scripts/: utility scripts
- src/: source code package
- tests/: tests

.
├── assets
├── configs
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   └── config.cpython-311.pyc
│   └── config.py
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   │   └── stock_processed.csv
│   └── raw
│       └── stock.csv
├── docs
├── experiments
│   ├── ddpg
│   ├── dqn
│   ├── ppo
│   └── sac
├── FinRL
│   ├── 1.66.32
│   ├── docker
│   │   ├── bin
│   │   │   ├── build_container.sh
│   │   │   ├── start_notebook.sh
│   │   │   └── test.sh
│   │   └── Dockerfile
│   ├── docs
│   │   ├── make.bat
│   │   ├── Makefile
│   │   └── source
│   │       ├── conf.py
│   │       ├── developer_guide
│   │       │   ├── contributing.rst
│   │       │   ├── development_setup.rst
│   │       │   └── file_architecture.rst
│   │       ├── faq.rst
│   │       ├── finrl_meta
│   │       │   ├── background.rst
│   │       │   ├── Benchmark.rst
│   │       │   ├── Data_layer.rst
│   │       │   ├── Environment_layer.rst
│   │       │   └── overview.rst
│   │       ├── image
│   │       │   ├── alg_compare.png
│   │       │   ├── download_FinRL.png
│   │       │   ├── ElegantRL_icon.jpeg
│   │       │   ├── ExplainableFinRL-CumulativeReturn.png
│   │       │   ├── ExplainableFinRL-PerformanceAlgs.png
│   │       │   ├── ExplainableFinRL-ReferenceFeature.png
│   │       │   ├── ExplainableFinRL-ReferenceModel.png
│   │       │   ├── ExplainableFinRL-SingleStepPrediction.png
│   │       │   ├── finrl_framework.png
│   │       │   ├── finrl_meta_dataops.png
│   │       │   ├── finrl_overview_drl.png
│   │       │   ├── FinRL_Tutorials.png
│   │       │   ├── FinRL-Architecture.png
│   │       │   ├── finrl-meta_data_layer.png
│   │       │   ├── finrl-meta_data_source.png
│   │       │   ├── finrl-meta_overview.png
│   │       │   ├── FinRL-Meta-Data-layer.png
│   │       │   ├── join_slack.png
│   │       │   ├── logo_finrl_2.jpg
│   │       │   ├── logo_finrl.jpg
│   │       │   ├── logo_transparent_background.png
│   │       │   ├── multiple_1.jpeg
│   │       │   ├── pycharm_MarkDirectoryAsSourcesRoot.png
│   │       │   ├── pycharm_push_PR.png
│   │       │   ├── pycharm_status_bar.png
│   │       │   ├── result_NeurIPS.png
│   │       │   └── timeline.png
│   │       ├── index.rst
│   │       ├── reference
│   │       │   ├── publication.md
│   │       │   ├── publication.rst
│   │       │   └── reference.md
│   │       ├── start
│   │       │   ├── first_glance.rst
│   │       │   ├── installation.rst
│   │       │   ├── introduction.rst
│   │       │   ├── quick_start.rst
│   │       │   ├── three_layer
│   │       │   │   ├── agents.rst
│   │       │   │   ├── applications.rst
│   │       │   │   └── environments.rst
│   │       │   └── three_layer.rst
│   │       └── tutorial
│   │           ├── 1-Introduction.rst
│   │           ├── 2-Advance.rst
│   │           ├── 3-Practical.rst
│   │           ├── 4-Optimization.rst
│   │           ├── 5-Others.rst
│   │           ├── Guide.rst
│   │           ├── Homegrown_example.rst
│   │           ├── Introduction
│   │           │   ├── MultipleStockTrading.rst
│   │           │   ├── PortfolioAllocation.rst
│   │           │   └── SingleStockTrading.rst
│   │           └── stocktrading
│   │               └── 1-data.rst
│   ├── examples
│   │   ├── FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb
│   │   ├── FinRL_GPM_Demo.ipynb
│   │   ├── FinRL_PaperTrading_Demo_refactored.py
│   │   ├── FinRL_PaperTrading_Demo.ipynb
│   │   ├── FinRL_PortfolioOptimizationEnv_Demo.ipynb
│   │   ├── FinRL_StockTrading_2026_1_data.py
│   │   ├── FinRL_StockTrading_2026_2_train.py
│   │   ├── FinRL_StockTrading_2026_3_Backtest.py
│   │   └── README.md
│   ├── figs
│   │   ├── alg_compare.PNG
│   │   ├── example_data.PNG
│   │   ├── finrl_framework.png
│   │   ├── FinRL_Tutorials.png
│   │   ├── FinRL-Architecture.png
│   │   ├── logo_size.jpg
│   │   ├── logo_transparent_background.png
│   │   ├── okx.jpeg
│   │   ├── performance.PNG
│   │   └── Poster_FinRL.jpg
│   ├── finrl
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── config_tickers.cpython-311.pyc
│   │   │   ├── config.cpython-311.pyc
│   │   │   ├── test.cpython-311.pyc
│   │   │   └── train.cpython-311.pyc
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── elegantrl
│   │   │   │   ├── __init__.py
│   │   │   │   └── models.py
│   │   │   ├── portfolio_optimization
│   │   │   │   ├── __init__.py
│   │   │   │   ├── algorithms.py
│   │   │   │   ├── architectures.py
│   │   │   │   ├── models.py
│   │   │   │   ├── README.md
│   │   │   │   └── utils.py
│   │   │   ├── rllib
│   │   │   │   ├── __init__.py
│   │   │   │   ├── drllibv2.py
│   │   │   │   └── models.py
│   │   │   └── stablebaselines3
│   │   │       ├── __init__.py
│   │   │       ├── hyperparams_opt.py
│   │   │       ├── models.py
│   │   │       └── tune_sb3.py
│   │   ├── applications
│   │   │   ├── __init__.py
│   │   │   ├── cryptocurrency_trading
│   │   │   │   ├── __init__.py
│   │   │   │   ├── actor.pth
│   │   │   │   └── recorder.npy
│   │   │   ├── high_frequency_trading
│   │   │   │   ├── __init__.py
│   │   │   │   ├── actor.pth
│   │   │   │   └── recorder.npy
│   │   │   ├── imitation_learning
│   │   │   │   ├── Imitation_Sandbox.ipynb
│   │   │   │   ├── README.md
│   │   │   │   ├── Stock_Selection.ipynb
│   │   │   │   └── Weight_Initialization.ipynb
│   │   │   ├── portfolio_allocation
│   │   │   │   └── __init__.py
│   │   │   ├── Stock_NeurIPS2018
│   │   │   │   ├── README.md
│   │   │   │   ├── Stock_NeurIPS2018_1_Data.ipynb
│   │   │   │   ├── Stock_NeurIPS2018_2_Train.ipynb
│   │   │   │   └── Stock_NeurIPS2018_3_Backtest.ipynb
│   │   │   └── stock_trading
│   │   │       ├── __init__.py
│   │   │       ├── ensemble_stock_trading.py
│   │   │       ├── fundamental_stock_trading.py
│   │   │       ├── stock_trading_rolling_window.py
│   │   │       └── stock_trading.py
│   │   ├── config_private.py
│   │   ├── config_tickers.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── meta
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   └── data_processor.cpython-311.pyc
│   │   │   ├── data_processor.py
│   │   │   ├── data_processors
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   │   ├── processor_alpaca.cpython-311.pyc
│   │   │   │   │   └── processor_yahoofinance.cpython-311.pyc
│   │   │   │   ├── func.py
│   │   │   │   ├── processor_alpaca.py
│   │   │   │   ├── processor_ccxt.py
│   │   │   │   ├── processor_eodhd.py
│   │   │   │   ├── processor_joinquant.py
│   │   │   │   ├── processor_quantconnect.py
│   │   │   │   ├── processor_sinopac.py
│   │   │   │   ├── processor_wrds.py
│   │   │   │   └── processor_yahoofinance.py
│   │   │   ├── env_cryptocurrency_trading
│   │   │   │   ├── __init__.py
│   │   │   │   ├── env_btc_ccxt.py
│   │   │   │   └── env_multiple_crypto.py
│   │   │   ├── env_portfolio_allocation
│   │   │   │   ├── __init__.py
│   │   │   │   └── env_portfolio.py
│   │   │   ├── env_portfolio_optimization
│   │   │   │   ├── __init__.py
│   │   │   │   ├── env_portfolio_optimization.py
│   │   │   │   └── README.md
│   │   │   ├── env_stock_trading
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   │   └── env_stocktrading.cpython-311.pyc
│   │   │   │   ├── env_nas100_wrds.py
│   │   │   │   ├── env_stock_papertrading.py
│   │   │   │   ├── env_stocktrading_cashpenalty.py
│   │   │   │   ├── env_stocktrading_np.py
│   │   │   │   ├── env_stocktrading_stoploss.py
│   │   │   │   └── env_stocktrading.py
│   │   │   ├── meta_config.py
│   │   │   ├── paper_trading
│   │   │   │   ├── alpaca.py
│   │   │   │   └── common.py
│   │   │   └── preprocessor
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-311.pyc
│   │   │       │   ├── preprocessors.cpython-311.pyc
│   │   │       │   └── yahoodownloader.cpython-311.pyc
│   │   │       ├── example_of_shioaji_api.py
│   │   │       ├── ibkrdownloader.py
│   │   │       ├── preprocessors.py
│   │   │       ├── shioajidownloader.py
│   │   │       ├── tusharedownloader.py
│   │   │       └── yahoodownloader.py
│   │   ├── plot.py
│   │   ├── README.md
│   │   ├── test.py
│   │   ├── trade.py
│   │   └── train.py
│   ├── LICENSE
│   ├── my_test.py
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.py
│   ├── tests
│   │   └── test_lazy_imports.py
│   └── unit_tests
│       ├── downloaders
│       │   ├── __init__.py
│       │   ├── test_alpaca_downloader.py
│       │   └── test_yahoo_downloader.py
│       ├── environments
│       │   ├── __init__.py
│       │   └── test_cash_penalty.py
│       ├── preprocessors
│       │   ├── test_groupby_scaler.py
│       │   └── test_yahoodownloader.py
│       └── test_core.py
├── LICENSE
├── models
├── notebooks
├── pyproject.toml
├── README.md
├── requirements.txt
├── results
│   ├── backtests
│   ├── checkpoints
│   ├── figures
│   ├── logs
│   └── metrics
├── scripts
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── build_features.cpython-311.pyc
│   │   ├── download_data.cpython-311.pyc
│   │   └── test_environment.cpython-311.pyc
│   ├── build_features.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── test_environment.py
│   ├── train_ddpq.py
│   └── train_dqn.py
├── src
│   ├── agents
│   ├── backtesting
│   ├── data
│   ├── envs
│   │   ├── __pycache__
│   │   │   └── stock_env.cpython-311.pyc
│   │   └── stock_env.py
│   ├── evaluation
│   ├── features
│   ├── reward
│   │   └── reward_function.py
│   ├── training
│   ├── utils
│   └── visualization
└── tests

85 directories, 227 files
