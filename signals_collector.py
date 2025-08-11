import pandas as pd
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SignalsCollector:
    """
    Class to collect and store trading signals during backtesting.
    """
    def __init__(self, strategy: str, symbol: str, timeframe: str, exchange: str):
        """
        Initialize the signals collector.
        """
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = exchange
        self.signals_df = pd.DataFrame(
            columns=['timestamp', 'price', 'signal', 'position', 'entry_price', 
                     'exit_price', 'trade_pnl', 'cum_pnl', 'drawdown']
        )
        self.indicators_df = pd.DataFrame()
    
    def add_signal(self, timestamp, price: float, signal: int, position: int, 
                   entry_price: Optional[float] = None, exit_price: Optional[float] = None,
                   trade_pnl: Optional[float] = None, cum_pnl: Optional[float] = None,
                   drawdown: Optional[float] = None):
        """
        Add a signal to the collection.
        """
        signal_data = {
            'timestamp': timestamp,
            'price': price,
            'signal': signal,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_pnl': trade_pnl,
            'cum_pnl': cum_pnl,
            'drawdown': drawdown
        }
        self.signals_df = pd.concat([self.signals_df, pd.DataFrame([signal_data])], ignore_index=True)
    
    def set_indicators(self, indicators_df: pd.DataFrame):
        """
        Set indicator values.
        """
        self.indicators_df = indicators_df
    
    def save(self, signals_file: Optional[str] = None, indicators_file: Optional[str] = None) -> dict:
        """
        Save signals and indicators to files.
        """
        signals_file = signals_file or f"{self.strategy}_{self.symbol}_{self.timeframe}_signals.csv"
        indicators_file = indicators_file or f"{self.strategy}_{self.symbol}_{self.timeframe}_indicators.csv"

        result = {"signals_file": None, "indicators_file": None}
        if not self.signals_df.empty:
            self.signals_df.to_csv(signals_file, index=False)
            logger.info(f"Saved {len(self.signals_df)} signals to {signals_file}")
            result["signals_file"] = signals_file
        else:
            logger.warning(f"No signals to save for {self.strategy} on {self.symbol}")

        if not self.indicators_df.empty:
            self.indicators_df.to_csv(indicators_file, index=True)
            logger.info(f"Saved indicators to {indicators_file}")
            result["indicators_file"] = indicators_file
        else:
            logger.warning(f"No indicators to save for {self.strategy} on {self.symbol}")

        return result