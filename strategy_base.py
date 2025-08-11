from typing import Dict, Any, Optional
import logging
import pandas as pd
from signals_collector import SignalsCollector

logger = logging.getLogger(__name__)

class Strategy:
    """
    Base class for all trading strategies.
    """
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.params = params
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators.
        """
        raise NotImplementedError("Subclasses must implement calculate_indicators")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def calculate_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.
        """
        raise NotImplementedError("Subclasses must implement calculate_performance")
    
    def run(self, df: pd.DataFrame, signals_collector: Optional[SignalsCollector] = None) -> Dict[str, float]:
        """
        Run the strategy on historical data.
        """
        df_with_indicators = self.calculate_indicators(df)
        df_with_signals = self.generate_signals(df_with_indicators)
        
        if signals_collector is not None:
            self.collect_signals(df_with_signals, signals_collector)
            signals_collector.set_indicators(df_with_signals)
        
        return self.calculate_performance(df_with_signals)
    
    def collect_signals(self, df: pd.DataFrame, signals_collector: SignalsCollector):
        """
        Collect signals for analysis.
        """
        raise NotImplementedError("Subclasses must implement collect_signals")


def validate_parameters(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and convert strategy parameters.
    """
    from utils import STRAT_PARAMS, validate_numeric_param, get_default_params
    
    if strategy not in STRAT_PARAMS:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    params_def = STRAT_PARAMS[strategy]
    validated_params = get_default_params(strategy)
    
    for param_code, param_info in params_def.items():
        if param_code in params:
            valid, error_msg, converted_value = validate_numeric_param(
                params[param_code],
                param_info["type"],
                param_info.get("min"),
                param_info.get("max"),
                param_info["name"]
            )
            if not valid:
                raise ValueError(f"Invalid parameter '{param_info['name']}': {error_msg}")
            validated_params[param_code] = converted_value
        elif param_code not in validated_params:
            raise ValueError(f"Required parameter '{param_info['name']}' ({param_code}) missing")
    
    return validated_params