# Cryptocurrency Trading Backtesting Framework

## Overview
This project is a Python-based framework for backtesting cryptocurrency trading strategies using Binance data. It supports data collection, strategy execution, performance analysis, and interactive visualization of trading signals. Currently, two strategies are implemented: **Supertrend** and **Moving Average Crossover**, both supporting spot and futures markets, stop-loss, and leverage.

### Features
- **Data Collection**: Fetches historical 1m OHLCV data from Binance and stores it in HDF5 (`data/binance.h5`).
- **Backtesting**: Runs strategies on resampled data (e.g., 1h, 1d), computing metrics like total return, max drawdown, win rate, and Sharpe ratio.
- **Visualization**: Generates interactive Plotly charts with candlesticks, strategy indicators (e.g., Supertrend, EMAs), buy/sell signals, and PNL/drawdown subplots.
- **Strategies**:
  - **Supertrend**: Trend-following with ATR-based bands, stop-loss, and leverage.
  - **Moving Average Crossover**: Uses short/long EMA crossovers for signals, with stop-loss and leverage.
- **Extensible**: Add new strategies by inheriting from `Strategy` and registering in `StrategyFactory`.

## Project Structure
```
crypto-backtest/
├── backtest_config.json          # Backtest configuration
├── backtest_results.json         # Backtest output (metrics, signals/indicators paths)
├── data_config.json              # Data collection configuration
├── visualize_config.json         # Visualization configuration
├── main.py                       # CLI entry point for data collection/backtesting
├── visualize_signals.py          # Generates interactive Plotly charts
├── backtester.py                 # Runs backtests
├── strategy_factory.py           # Creates strategy instances
├── strategy_base.py              # Base class for strategies
├── utils.py                      # Helpers (resampling, param validation)
├── config_handler.py             # Manages JSON configs
├── signals_collector.py          # Collects signals/indicators for CSV output
├── database.py                   # HDF5 client for data storage/retrieval
├── data_collector.py             # Fetches Binance data
├── info.log                      # Logs (data collection, backtest, visualization)
├── data/                         # Stores HDF5 data
│   └── binance.h5                # OHLCV data (e.g., spot/BTCUSDT/1m)
├── exchanges/                    # Exchange API clients
│   └── binance.py                # Binance client (spot/futures)
└── strategies/                   # Trading strategies
    ├── supertrend.py             # Supertrend strategy
    └── moving_average_crossover.py # Moving Average Crossover strategy
```

## Setup
1. **Clone Repository** (or ensure files are in place):
   ```bash
   git clone <your-repo> crypto-backtest
   cd crypto-backtest
   ```
2. **Create Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install pandas numpy requests h5py plotly
   ```
4. **Ensure Folders**:
   - Create `data/`, `exchanges/`, `strategies/` if missing.
   - Place `binance.py` in `exchanges/`, `supertrend.py` and `moving_average_crossover.py` in `strategies/`.
5. **Verify Files**: Ensure all `.py` and `.json` files from the project structure are present.

## Usage

### 1. Data Collection
Fetch historical 1m OHLCV data from Binance and store in `data/binance.h5`.

**Example `data_config.json`**:
```json
{
    "mode": "data",
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "futures": false
}
```

**Command**:
```bash
python main.py --config data_config.json
```

**Output**:
- Updates `data/binance.h5` with `spot/BTCUSDT/1m` (or `futures/BTCUSDT/1m` if `futures: true`).
- Logs to `info.log` (e.g., "Collected X candles from 2023-01-01").

**Note**: Binance API limits 1m klines to ~1-2 years. For older data (e.g., 2023), multiple runs may be needed.

### 2. Backtesting
Run a strategy on historical data, outputting metrics and signals/indicators CSVs.

#### Supertrend Strategy
**Example `backtest_config.json`** for Supertrend:
```json
{
    "mode": "backtest",
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "futures": false,
    "strategy": "supertrend",
    "timeframe": "1d",
    "from_time": "2023-01-01",
    "to_time": "2024-01-01",
    "strategy_params": {
        "atr_period": 10,
        "atr_multiplier": 3.0,
        "leverage": 1.0,
        "commission_rate": 0.04,
        "initial_capital": 10000.0,
        "stop_loss_pct": 0.02,
        "use_stop_loss": true
    },
    "save_signals": true,
    "output_file": "backtest_results.json"
}
```

**Parameters**:
- `atr_period`: ATR calculation period (int, 1-100, default 14).
- `atr_multiplier`: Multiplier for bands (float, 0.1-10.0, default 3.0).
- `leverage`: Futures leverage (float, 1.0-100.0, default 1.0).
- `commission_rate`: % fee per trade (float, 0.0-1.0, default 0.04).
- `initial_capital`: Starting capital (float, >=100, default 10000.0).
- `stop_loss_pct`: Stop-loss % (float, 0.001-0.2, default 0.02).
- `use_stop_loss`: Enable stop-loss (bool, default true).

#### Moving Average Crossover Strategy
**Example `backtest_config.json`** for Moving Average Crossover:
```json
{
    "mode": "backtest",
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "futures": false,
    "strategy": "moving_average_crossover",
    "timeframe": "1d",
    "from_time": "2023-01-01",
    "to_time": "2024-01-01",
    "strategy_params": {
        "short_period": 12,
        "long_period": 26,
        "leverage": 1.0,
        "commission_rate": 0.04,
        "initial_capital": 10000.0,
        "stop_loss_pct": 0.02,
        "use_stop_loss": true
    },
    "save_signals": true,
    "output_file": "backtest_results.json"
}
```

**Parameters**:
- `short_period`: Short EMA period (int, 1-100, default 12).
- `long_period`: Long EMA period (int, 1-200, default 26).
- `leverage`, `commission_rate`, `initial_capital`, `stop_loss_pct`, `use_stop_loss`: Same as Supertrend.

**Command** (for either strategy):
```bash
python main.py --config backtest_config.json
```

**Output**:
- `backtest_results.json`: Metrics (total_return, max_drawdown, total_trades, win_rate, sharpe_ratio, final_capital).
- CSVs: `{strategy}_{symbol}_{timeframe}_signals.csv` (signals), `_indicators.csv` (indicators like Supertrend or EMAs).
- Logs to `info.log` (e.g., "Generated X signals").

### 3. Visualization
Generate interactive Plotly charts with candlesticks, strategy indicators, buy/sell signals (green triangle-up for Buy/Long, red triangle-down for Sell/Short), and PNL/drawdown subplots.

**Example `visualize_config.json`**:
```json
{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "futures": false,
    "hdf5_path": "data/binance.h5",
    "signals_path": "moving_average_crossover_BTCUSDT_1d_signals.csv",
    "output_file": "signals_plot_spot.html"
}
```

**Command**:
```bash
python visualize_signals.py --config visualize_config.json
```

**Output**:
- HTML file (`signals_plot_spot.html`) with:
  - Candlesticks for price.
  - Indicators: For Supertrend (`supertrend`, `upper_band`, `lower_band`); for Moving Average Crossover (`short_ema`, `long_ema`).
  - Signals: Green triangle-up for Buy (spot) or Long (futures), red triangle-down for Sell/Short, with hover text (entry/exit price, PNL).
  - Subplots: Cumulative PNL (green), drawdown (red).
- Logs to `info.log` (e.g., "Added Short EMA to chart").

### 4. Debugging
- **Inspect HDF5**:
  ```bash
  python inspect_hdf5.py
  ```
  - Checks `data/binance.h5` structure (e.g., `spot/BTCUSDT/1m`).
- **Check Signals**:
  ```python
  import pandas as pd
  df = pd.read_csv('{strategy}_BTCUSDT_1d_signals.csv')
  print("Signal count:", len(df[df['signal'] != 0]))
  df = pd.read_csv('{strategy}_BTCUSDT_1d_indicators.csv')
  print(df.columns)
  ```
- **Logs**: Check `info.log` for errors or signal counts.

## Extending the Framework
- **Add New Strategy**:
  1. Create a new file in `strategies/` (inherit from `Strategy`).
  2. Implement `calculate_indicators`, `generate_signals`, `calculate_performance`, `collect_signals`.
  3. Register in `strategy_factory.py`.
  4. Add parameters to `utils.py` (STRAT_PARAMS).
  5. Update `config_handler.py` defaults.
- **Add Indicators to Visualization**: Update `STRATEGY_INDICATORS` in `visualize_signals.py`.

## Notes
- **Dependencies**: Requires `pandas`, `numpy`, `requests`, `h5py`, `plotly`.
- **Data Limits**: Binance API limits 1m klines to ~1-2 years. For older data, adjust dates or modify `data_collector.py`.
- **Performance**: Strategies use vectorized pandas operations for efficiency. Stop-loss adds minimal overhead.
- **Issues**: If no signals, check `info.log` for flip counts or try lower `atr_multiplier` (Supertrend) or different periods (Moving Average Crossover).