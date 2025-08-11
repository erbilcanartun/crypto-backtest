# Cryptocurrency Trading Strategy Backtester

A comprehensive framework for backtesting cryptocurrency trading strategies with visualization tools.

## Features

- Data collection from Binance and FTX exchanges
- Support for spot and futures markets
- Backtesting of various trading strategies
- Interactive visualization of backtest results
- Detailed performance metrics
- Configurable via JSON parameters files or command-line

## Supported Strategies

- SuperTrend (spot and futures)

## Quick Start

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running a Backtest

There are two ways to run a backtest:

#### Using a Parameters File (Recommended)

1. Create a JSON parameters file (see example below)
2. Run the backtest:
   ```
   python main.py --config backtest_params.json
   ```

#### Using Command Line Arguments

```
python main.py --mode backtest --exchange binance --symbol BTCUSDT --strategy supertrend_futures --timeframe 1h --from-time 2024-01-01 --to-time 2024-04-01
```

### Visualizing Results

After running a backtest, you can visualize the results:

#### Using a Parameters File (Recommended)

1. Create a visualization parameters file:
   ```
   {
     "results_file": "supertrend_futures_BTCUSDT_1h_results.json",
     "signals_file": "supertrend_futures_BTCUSDT_1h_signals.csv",
     "indicators_file": "supertrend_futures_BTCUSDT_1h_indicators.csv",
     "output_file": "supertrend_futures_BTCUSDT_1h_visualization.html",
     "show_all_indicators": true
   }
   ```

2. Run the visualization:
   ```
   python visualize.py --params visualization_params.json
   ```

#### Using Command Line Arguments

```
python visualize.py --results supertrend_futures_BTCUSDT_1h_results.json
```

The command above will automatically look for signal and indicator files referenced in the results file.

## Parameter File Examples

### Backtest Parameters

```json
{
    "mode": "backtest",
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "strategy": "supertrend_futures",
    "timeframe": "1d",
    "from_time": "2024-01-01",
    "to_time": "2025-01-01",
    "strategy_params": {
        "atr_period": 14,
        "atr_multiplier": 3.0,
        "leverage": 1.0,
        "commission_rate": 0.0,
        "initial_capital": 10000
    },
    "save_signals": true,
    "output_file": "btc_supertrend_results.json"
}
```

### Visualization Parameters

```json
{
  "results_file": "backtest_results.json",
  "signals_file": "supertrend_futures_BTCUSDT_1h_signals.csv",
  "indicators_file": "supertrend_futures_BTCUSDT_1h_indicators.csv",
  "output_file": "btc_supertrend_visualization.html",
  "show_all_indicators": true
}
```

## File Structure

- `main.py` - Main entry point for data collection and backtesting
- `visualize.py` - Tool for creating interactive visualizations
- `backtester.py` - Core backtesting functionality
- `config_handler.py` - Handles configuration loading and validation
- `data_collector.py` - Collects historical data from exchanges
- `database.py` - Manages HDF5 database for OHLC data
- `signals_collector.py` - Collects and stores trading signals
- `strategies/` - Trading strategies implementation
  - `spot/` - Spot market strategies
  - `futures/` - Futures market strategies
- `utils.py` - Utility functions

## Available Strategies

### SuperTrend (spot)

A trend-following indicator that identifies uptrends and downtrends.

Parameters:
- `atr_period`: Period for Average True Range calculation
- `atr_multiplier`: Multiplier for ATR to determine band width

### SuperTrend Futures

The SuperTrend strategy adapted for futures markets with leverage.

Parameters:
- `atr_period`: Period for Average True Range calculation
- `atr_multiplier`: Multiplier for ATR to determine band width
- `leverage`: Leverage multiplier (e.g. 2.0 for 2x leverage)
- `commission_rate`: Commission rate in percentage
- `initial_capital`: Initial capital for the backtest

## Customization

You can create your own strategies by:
1. Adding a new strategy file in the appropriate directory
2. Adding an entry in the `STRAT_PARAMS` dictionary in `utils.py`
3. Implementing the backtesting logic for the new strategy

## Visualization

The visualization tool creates interactive HTML reports with:
- Price charts with candlesticks
- Strategy indicators
- Trade signals
- Performance metrics
- Drawdown chart

## Troubleshooting

If you encounter any issues:

1. **Files not found** - Ensure all referenced files exist in the working directory or provide full paths.
2. **Invalid parameters** - Check your configuration file for typos or invalid parameters.
3. **Visualization problems** - Make sure your signal and indicator files match what was generated during the backtest.