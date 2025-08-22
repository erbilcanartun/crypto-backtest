import json
import os
import logging
import datetime
from typing import Dict, Any, Optional, List
from utils import validate_numeric_param

logger = logging.getLogger(__name__)

class ConfigHandler:
    """
    Handler for managing backtest configuration.
    """
    
    @staticmethod
    def load_config(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from a JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {file_path}")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration file: {e}")
            return None
    
    @staticmethod
    def save_config(config: Dict[str, Any], file_path: str) -> bool:
        """
        Save configuration to a JSON file.
        """
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration file: {e}")
            return False
    
    @staticmethod
    def validate_backtest_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate backtest configuration.
        """
        errors = []
        if 'mode' not in config:
            errors.append("Missing required field: mode")
            return errors

        if config['mode'] not in ['data', 'backtest']:
            errors.append(f"Invalid mode: {config['mode']}. Must be 'data' or 'backtest'")
            return errors

        if config['mode'] == 'data':
            if 'exchange' not in config:
                errors.append("Missing required field for data mode: exchange")
            elif config['exchange'].lower() != 'binance':
                errors.append(f"Invalid exchange: {config['exchange']}. Must be 'binance'")
            if 'symbol' not in config:
                errors.append("Missing required field for data mode: symbol")
            return errors

        required_fields = ['exchange', 'symbol', 'strategy', 'timeframe']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        if 'exchange' in config and config['exchange'].lower() != 'binance':
            errors.append(f"Invalid exchange: {config['exchange']}. Must be 'binance'")

        if 'timeframe' in config and config['timeframe'] not in ['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d']:
            errors.append(f"Invalid timeframe: {config['timeframe']}")

        if 'strategy' in config:
            strategy = config['strategy']
            from utils import STRAT_PARAMS
            if strategy not in STRAT_PARAMS:
                errors.append(f"Unknown strategy: {strategy}")
            else:
                strat_param_def = STRAT_PARAMS[strategy]
                if 'strategy_params' not in config:
                    config['strategy_params'] = {}
                for param_code, param_info in strat_param_def.items():
                    if param_code not in config['strategy_params']:
                        errors.append(f"Missing required parameter for {strategy}: {param_info['name']} ({param_code})")
                    else:
                        valid, error_msg, _ = validate_numeric_param(
                            config['strategy_params'][param_code],
                            param_info['type'],
                            param_info.get('min'),
                            param_info.get('max'),
                            param_info['name']
                        )
                        if not valid:
                            errors.append(f"Invalid {param_info['name']}: {error_msg}")
        return errors

    @staticmethod
    def create_data_config() -> Dict[str, Any]:
        """
        Create a default data collection configuration.
        """
        return {
            "mode": "data",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "futures": False
        }

    @staticmethod
    def create_backtest_config() -> Dict[str, Any]:
        """
        Create a default backtest configuration.
        """
        return {
            "mode": "backtest",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "futures": False,
            "strategy": "moving_average_crossover",  # Updated to use the other strategy as default example
            "timeframe": "1h",
            "from_time": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
            "to_time": datetime.datetime.now().strftime("%Y-%m-%d"),
            "strategy_params": {
                "short_period": 12,
                "long_period": 26,
                "leverage": 1.0,
                "commission_rate": 0.04,
                "initial_capital": 10000.0,
                "stop_loss_pct": 0.02,
                "use_stop_loss": True,
                "futures": False  # Added for consistency
            },
            "save_signals": True,
            "output_file": "backtest_results.json"
        }

    @staticmethod
    def create_default_config(mode: str = "backtest") -> Dict[str, Any]:
        """
        Create a default configuration for the specified mode.
        """
        return ConfigHandler.create_data_config() if mode.lower() == "data" else ConfigHandler.create_backtest_config()