import datetime
import functools
import logging
import pandas as pd
from typing import Dict

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
    
    Args:
        data: DataFrame with OHLCV data
        tf: Target timeframe (e.g., '1m', '1h')
        
    Returns:
        Resampled DataFrame
        
    Raises:
        ValueError: If timeframe is invalid
    """
    if tf not in TF_EQUIV:
        valid_tfs = ', '.join(TF_EQUIV.keys())
        raise ValueError(f"Invalid timeframe: {tf}. Must be one of: {valid_tfs}")
    return data.resample(TF_EQUIV[tf]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

def validate_numeric_param(value, param_type, min_val=None, max_val=None, name=None):
    """
    Validate a numeric parameter.
    
    Args:
        value: Value to validate
        param_type: Expected type (e.g., int, float)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Parameter name for error messages
        
    Returns:
        Tuple (is_valid, error_message, converted_value)
    """
    param_name = name if name else "Parameter"
    try:
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