#!/usr/bin/env python3
"""
Main entry point for cryptocurrency trading strategy backtesting framework.
Provides CLI interface for data collection and backtesting.
"""

import logging
import datetime
import json
import argparse
from typing import Dict, Any, Optional

from backtester import run as backtest_run
from config_handler import ConfigHandler
from data_collector import collect_all
from exchanges.binance import BinanceClient
from utils import TF_EQUIV

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("info.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def run_data_collection(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run data collection with the given parameters.
    """
    exchange = params.get("exchange", "").lower()
    if exchange != "binance":
        logger.error(f"Invalid exchange: {exchange}. Only 'binance' is supported.")
        return None

    client = BinanceClient(params.get("futures", False))
    symbol = params.get("symbol", "").upper()
    if symbol not in client.symbols:
        logger.error(f"Invalid symbol: {symbol}")
        return None

    logger.info(f"Collecting data for {symbol} on {exchange} ({'futures' if params.get('futures', False) else 'spot'})")
    collect_all(client, exchange, symbol, params.get("futures", False))
    return {"mode": "data", "status": "completed", "exchange": exchange, "symbol": symbol}

def run_backtest_operation(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run backtest with the given parameters.
    """
    exchange = params.get("exchange", "").lower()
    if exchange != "binance":
        logger.error(f"Invalid exchange: {exchange}. Only 'binance' is supported.")
        return None

    client = BinanceClient(params.get("futures", False))
    symbol = params.get("symbol", "").upper()
    if symbol not in client.symbols:
        logger.error(f"Invalid symbol: {symbol}")
        return None

    from_time = params.get("from_time")
    to_time = params.get("to_time")
    if isinstance(from_time, str):
        from_time = int(datetime.datetime.strptime(from_time, "%Y-%m-%d").timestamp() * 1000)
    if isinstance(to_time, str):
        to_time = int(datetime.datetime.strptime(to_time, "%Y-%m-%d").timestamp() * 1000) + 86399999  # End of day

    strategy_params = params.get("strategy_params", {})
    # Ensure futures consistency
    futures = params.get("futures", False)
    strategy_params["futures"] = futures

    logger.info(f"Running backtest for {symbol} on {exchange} ({'futures' if futures else 'spot'}) with strategy {params['strategy']}")

    results = backtest_run(
        exchange=exchange,
        symbol=symbol,
        strategy=params["strategy"],
        tf=params["timeframe"],
        from_time=from_time,
        to_time=to_time,
        strategy_params=strategy_params,
        save_signals=params.get("save_signals", True),
        futures=futures  # Pass futures
    )

    if results:
        output_file = params.get("output_file", "backtest_results.json")  # Default if not specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4)
                logger.info(f"Backtest results saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_file}: {e}")
        else:
            logger.info("Backtest completed but no output_file specified; results not saved to file.")
        return results

def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency trading strategy backtesting framework")
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, choices=['data', 'backtest'], help='Operation mode: data or backtest')
    parser.add_argument('--exchange', type=str, help='Exchange name (e.g., binance)')
    parser.add_argument('--symbol', type=str, help='Symbol (e.g., BTCUSDT)')
    parser.add_argument('--futures', action='store_true', help='Use futures data (default: spot)')
    parser.add_argument('--strategy', type=str, help='Strategy name (e.g., supertrend)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 1h, 1d)')
    parser.add_argument('--from_time', type=str, help='Start time (YYYY-MM-DD)')
    parser.add_argument('--to_time', type=str, help='End time (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--create-template', type=str, help='Create config template file')
    parser.add_argument('--template-mode', type=str, choices=['data', 'backtest'], help='Template mode: data or backtest')

    args = parser.parse_args()

    if args.create_template:
        # ... (template creation logic assumed unchanged)
        return 0

    if not args.config and not args.mode:
        print("Error: Either a configuration file (--config) or a mode (--mode) must be specified.")
        print("Examples:")
        print("  python main.py --config backtest_config.json")
        print("  python main.py --mode data --exchange binance --symbol BTCUSDT")
        print("  python main.py --mode backtest --exchange binance --symbol BTCUSDT --strategy supertrend --timeframe 1h")
        print("  python main.py --create-template backtest_template.json --template-mode backtest")
        return 1

    config = {}
    if args.config:
        loaded_config = ConfigHandler.load_config(args.config)
        if not loaded_config:
            print(f"Error loading config file: {args.config}")
            return 1
        config = loaded_config

    if args.mode:
        config['mode'] = args.mode
    if args.exchange:
        config['exchange'] = args.exchange
    if args.symbol:
        config['symbol'] = args.symbol
    if args.futures:
        config['futures'] = args.futures
    if args.strategy:
        config['strategy'] = args.strategy
    if args.timeframe:
        config['timeframe'] = args.timeframe
    if args.from_time:
        config['from_time'] = args.from_time
    if args.to_time:
        config['to_time'] = args.to_time
    if args.output:
        config['output_file'] = args.output

    if 'mode' not in config:
        print("Error: Mode ('data' or 'backtest') not specified.")
        return 1

    if not config.get('exchange') or not config.get('symbol'):
        print("Error: exchange and symbol are required.")
        return 1

    if config['mode'] == 'backtest' and (not config.get('strategy') or not config.get('timeframe')):
        print("Error: strategy and timeframe are required for backtesting.")
        return 1

    if config['mode'] == 'data':
        results = run_data_collection(config)
    else:
        results = run_backtest_operation(config)

    if not results:
        print("Operation failed. Check logs for details.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())