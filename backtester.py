from signals_collector import SignalsCollector
from database import Hdf5Client
from utils import resample_timeframe, handle_errors, validate_numeric_param, STRAT_PARAMS
from strategy_factory import StrategyFactory
import logging
from typing import Dict, Any, Optional
import datetime

logger = logging.getLogger(__name__)

@handle_errors
def run(exchange: str, symbol: str, strategy: str, tf: str, from_time: int, to_time: int, 
        strategy_params: Optional[Dict] = None, save_signals: bool = True, futures: bool = False) -> Dict:
    """
    Run a backtest for the specified strategy and parameters.
    """
    # Store original parameters
    original_params = {
        "exchange": exchange,
        "symbol": symbol,
        "strategy": strategy,
        "timeframe": tf,
        "from_time": from_time,
        "to_time": to_time,
        "strategy_params": strategy_params.copy() if strategy_params else {},
        "save_signals": save_signals,
        "futures": futures  # Add for config
    }

    # Prepare strategy parameters
    strategy_params = strategy_params or {}

    # Validate parameters
    from strategy_base import validate_parameters
    validated_params = validate_parameters(strategy, strategy_params)

    # Get data from database
    h5_db = Hdf5Client(exchange)
    data = h5_db.get_data(symbol, from_time, to_time, futures)

    if data is None or data.empty:
        raise ValueError(f"No data found for {symbol} ({'futures' if futures else 'spot'}) from {from_time} to {to_time}")

    data = resample_timeframe(data, tf)

    # Initialize results
    results = {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": tf,
        "exchange": exchange,
        "from_time": from_time,
        "to_time": to_time,
        "params": validated_params,
        "config": original_params,
        "mode": 'futures' if futures else 'spot'  # Add for clarity
    }

    # Initialize signals collector
    signals_collector = SignalsCollector(strategy, symbol, tf, exchange) if save_signals else None

    # Create strategy instance
    strategy_instance = StrategyFactory.create_strategy(strategy, validated_params)

    # Run the strategy
    performance = strategy_instance.run(data, signals_collector)

    # Add performance metrics
    results.update(performance)

    # Save signals if requested
    if save_signals and signals_collector:
        files_info = signals_collector.save()
        results.update({
            "signals_file": files_info.get("signals_file"),
            "indicators_file": files_info.get("indicators_file")
        })

    return results