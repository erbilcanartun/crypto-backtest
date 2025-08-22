import datetime
import functools
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

TF_EQUIV = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", 
            "1h": "1h", "4h": "4h", "12h": "12h", "1d": "D"}

STRAT_PARAMS = {
    "supertrend": {
        "atr_period": {
            "name": "ATR Period", 
            "type": int,
            "min": 1,
            "max": 100,
            "default": 14
        },
        "atr_multiplier": {
            "name": "ATR Multiplier", 
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "default": 3.0
        },
        "leverage": {
            "name": "Leverage", 
            "type": float,
            "min": 1.0,
            "max": 100.0,
            "default": 1.0
        },
        "commission_rate": {
            "name": "Commission Rate (%)", 
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.04
        },
        "initial_capital": {
            "name": "Initial Capital", 
            "type": float,
            "min": 100.0,
            "default": 10000.0
        },
        "stop_loss_pct": {
            "name": "Stop Loss (%)", 
            "type": float,
            "min": 0.001,
            "max": 0.2,
            "default": 0.02
        },
        "use_stop_loss": {
            "name": "Use Stop Loss", 
            "type": bool,
            "default": True
        },
        "futures": {
            "name": "Futures Mode", 
            "type": bool,
            "default": False
        }
    },
    "moving_average_crossover": {
        "short_period": {
            "name": "Short Period", 
            "type": int,
            "min": 1,
            "max": 100,
            "default": 12
        },
        "long_period": {
            "name": "Long Period", 
            "type": int,
            "min": 1,
            "max": 200,
            "default": 26
        },
        "leverage": {
            "name": "Leverage", 
            "type": float,
            "min": 1.0,
            "max": 100.0,
            "default": 1.0
        },
        "commission_rate": {
            "name": "Commission Rate (%)", 
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.04
        },
        "initial_capital": {
            "name": "Initial Capital", 
            "type": float,
            "min": 100.0,
            "default": 10000.0
        },
        "stop_loss_pct": {
            "name": "Stop Loss (%)", 
            "type": float,
            "min": 0.001,
            "max": 0.2,
            "default": 0.02
        },
        "use_stop_loss": {
            "name": "Use Stop Loss", 
            "type": bool,
            "default": True
        },
        "futures": {  # Added for consistency
            "name": "Futures Mode", 
            "type": bool,
            "default": False
        }
    }
}

def ms_to_dt(ms: int) -> datetime.datetime:
    """
    Convert timestamp in milliseconds to datetime object.
    """
    return datetime.datetime.utcfromtimestamp(ms / 1000)

def resample_timeframe(data: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    """
    if tf not in TF_EQUIV:
        valid_tfs = ', '.join(TF_EQUIV.keys())
        raise ValueError(f"Invalid timeframe: {tf}. Must be one of: {valid_tfs}")

    return data.resample(TF_EQUIV[tf]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

def validate_numeric_param(value, param_type, min_val=None, max_val=None, param_name=None):
    """
    Validate a numeric parameter.
    """
    param_name = param_name if param_name else "Parameter"
    try:
        if param_type == bool:
            try:
                if isinstance(value, str):
                    value = value.lower() in ('true', '1')
                elif isinstance(value, (int, float)):
                    value = bool(value)
                else:
                    raise ValueError
            except ValueError:
                error_msg = f"{param_name} must be a valid boolean"
                logger.warning(error_msg)
                return False, error_msg, None
            return True, None, value
        else:
            converted = param_type(value)
            if min_val is not None and converted < min_val:
                error_msg = f"{param_name} must be at least {min_val}"
                logger.warning(error_msg)
                return False, error_msg, None
            if max_val is not None and converted > max_val:
                error_msg = f"{param_name} must not exceed {max_val}"
                logger.warning(error_msg)
                return False, error_msg, None
            return True, None, converted
    except (ValueError, TypeError):
        error_msg = f"{param_name} must be a valid {param_type.__name__}"
        logger.warning(error_msg)
        return False, error_msg, None

def handle_errors(func):
    """
    Decorator for handling errors in functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return None
    return wrapper

def get_default_params(strategy: str) -> Dict:
    """
    Get default parameters for a strategy.
    """
    if strategy not in STRAT_PARAMS:
        return {}
    return {param_name: param_info["default"] for param_name, param_info in STRAT_PARAMS[strategy].items() if "default" in param_info}