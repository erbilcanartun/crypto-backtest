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

    logger.info(f"Collecting data for {symbol} on {exchange}")
    collect_all(client, exchange, symbol)
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

    strategy = params.get("strategy", "").lower()
    if strategy != "supertrend":
        logger.error(f"Invalid strategy: {strategy}. Only 'supertrend' is supported.")
        return None

    timeframe = params.get("timeframe")
    if timeframe not in TF_EQUIV:
        logger.error(f"Invalid timeframe: {timeframe}. Available: {', '.join(TF_EQUIV.keys())}")
        return None

    try:
        from_time = int(datetime.datetime.strptime(params["from_time"], "%Y-%m-%d").timestamp() * 1000)
        to_time = int(datetime.datetime.strptime(params["to_time"], "%Y-%m-%d").timestamp() * 1000)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return None

    strategy_params = params.get("strategy_params", {})
    futures = params.get("futures", False)  # Use top-level futures

    logger.info(f"Running backtest for {symbol} on {exchange} with {strategy} strategy")
    results = backtest_run(
        exchange=exchange,
        symbol=symbol,
        strategy=strategy,
        tf=timeframe,
        from_time=from_time,
        to_time=to_time,
        strategy_params={**strategy_params, "futures": futures},  # Pass futures explicitly
        save_signals=params.get("save_signals", True)
    )

    if results:
        output_file = params.get("output_file", "backtest_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Backtest results saved to {output_file}")
        return results
    return None

def run_backtest(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run backtest or data collection based on parameters.
    """
    errors = ConfigHandler.validate_backtest_config(params)
    if errors:
        for error in errors:
            logger.error(error)
        return None

    mode = params.get("mode", "").lower()
    if mode == "data":
        return run_data_collection(params)
    elif mode == "backtest":
        return run_backtest_operation(params)
    else:
        logger.error(f"Invalid mode: {mode}. Must be 'data' or 'backtest'")
        return None


def create_config_template(output_file: str, mode: str = "backtest") -> bool:
    """
    Create a configuration template file.
    """
    try:
        config = ConfigHandler.create_default_config(mode)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created {mode} configuration template: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating configuration template: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Strategy Backtester')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--create-template', help='Create a configuration template file')
    parser.add_argument('--template-mode', choices=['data', 'backtest'], default='backtest',
                        help='Mode for template creation')
    parser.add_argument('--mode', choices=['data', 'backtest'],
                        help='Operation mode: collect data or run backtest')
    parser.add_argument('--exchange', help='Exchange to use (only binance supported)')
    parser.add_argument('--symbol', help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--futures', action='store_true', help='Use futures market')
    parser.add_argument('--strategy', help='Strategy to backtest (only supertrend supported)')
    parser.add_argument('--timeframe', help='Timeframe for backtest')
    parser.add_argument('--from-time', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-time', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', help='Output file for results')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.create_template:
        return 0 if create_config_template(args.create_template, args.template_mode) else 1

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

    results = run_backtest(config)
    if not results:
        print("Operation failed. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())