import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from signals_collector import SignalsCollector
from strategy_base import Strategy
import logging

logger = logging.getLogger(__name__)

class SupertrendStrategy(Strategy):
    """
    Supertrend strategy for spot and futures markets.
    """
    def __init__(self, atr_period: int, atr_multiplier: float, 
                 leverage: float = 1.0, commission_rate: float = 0.04, 
                 initial_capital: float = 10000.0, futures: bool = False):
        """
        Initialize the Supertrend strategy.
        """
        super().__init__(atr_period=atr_period, atr_multiplier=atr_multiplier, 
                        leverage=leverage, commission_rate=commission_rate, 
                        initial_capital=initial_capital, futures=futures)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.initial_capital = initial_capital
        self.futures = futures
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicators.
        """
        df = df.copy()
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ])
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        hl2 = (df['high'] + df['low']) / 2
        df['upper_band'] = hl2 + (self.atr_multiplier * df['atr'])
        df['lower_band'] = hl2 - (self.atr_multiplier * df['atr'])
        df['in_uptrend'] = True
        df['supertrend'] = df['lower_band']

        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            curr_upper = df['upper_band'].iloc[i]
            curr_lower = df['lower_band'].iloc[i]
            prev_supertrend = df['supertrend'].iloc[i-1]
            prev_in_uptrend = df['in_uptrend'].iloc[i-1]

            if prev_in_uptrend:
                if curr_close < prev_supertrend:
                    df.loc[df.index[i], 'in_uptrend'] = False
                    df.loc[df.index[i], 'supertrend'] = curr_upper
                else:
                    df.loc[df.index[i], 'in_uptrend'] = True
                    df.loc[df.index[i], 'supertrend'] = curr_lower
            else:
                if curr_close > prev_supertrend:
                    df.loc[df.index[i], 'in_uptrend'] = True
                    df.loc[df.index[i], 'supertrend'] = curr_lower
                else:
                    df.loc[df.index[i], 'in_uptrend'] = False
                    df.loc[df.index[i], 'supertrend'] = curr_upper

            logger.debug(f"Index {i}: close={curr_close:.2f}, supertrend={df['supertrend'].iloc[i]:.2f}, in_uptrend={df['in_uptrend'].iloc[i]}, upper={curr_upper:.2f}, lower={curr_lower:.2f}")

        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Supertrend.
        """
        df = df.copy()
        df['signal'] = 0
        df['position'] = 0
        df['entry_price'] = 0.0
        df['exit_price'] = 0.0
        df['trade_pnl'] = 0.0
        df['capital'] = float(self.initial_capital)
        df['cum_pnl'] = 0.0
        df['drawdown'] = 0.0

        position = 0
        entry_price = 0.0
        capital = self.initial_capital
        max_capital = self.initial_capital

        for i in range(1, len(df)):
            curr_trend = df['in_uptrend'].iloc[i]
            prev_trend = df['in_uptrend'].iloc[i-1]
            current_price = df['close'].iloc[i]

            df.loc[df.index[i], 'capital'] = capital
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'entry_price'] = entry_price

            if df['capital'].iloc[i] > max_capital:
                max_capital = df['capital'].iloc[i]
            df.loc[df.index[i], 'drawdown'] = (max_capital - df['capital'].iloc[i]) / max_capital if max_capital > 0 else 0
            df.loc[df.index[i], 'cum_pnl'] = (df['capital'].iloc[i] / self.initial_capital) - 1

            if curr_trend and not prev_trend:
                df.loc[df.index[i], 'signal'] = 1
                logger.debug(f"Buy signal at index {i}: price={current_price:.2f}, curr_trend={curr_trend}, prev_trend={prev_trend}")
            elif not curr_trend and prev_trend:
                df.loc[df.index[i], 'signal'] = -1
                logger.debug(f"Sell signal at index {i}: price={current_price:.2f}, curr_trend={curr_trend}, prev_trend={prev_trend}")

            if df['signal'].iloc[i] != 0:
                if position != 0:
                    if self.futures:
                        pnl = self.leverage * ((current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price) * 100
                        pnl -= self.commission_rate * 2
                    else:
                        if position == 1:  # Only calculate PNL for long positions in spot
                            pnl = (current_price / entry_price - 1) * 100
                            pnl -= self.commission_rate  # Single commission on exit
                        else:
                            pnl = 0  # No PNL for short positions in spot
                    if pnl != 0:  # Only update if a valid trade is closed
                        capital *= (1 + pnl / 100)
                        df.loc[df.index[i], 'trade_pnl'] = pnl
                        df.loc[df.index[i], 'exit_price'] = current_price
                        logger.debug(f"Closed position at index {i}: mode={'futures' if self.futures else 'spot'}, position={position}, pnl={pnl:.2f}%, capital={capital:.2f}")
                    position = 0
                    entry_price = 0.0

                if self.futures:
                    if (df['signal'].iloc[i] == 1 and position == 0) or (df['signal'].iloc[i] == -1 and position == 0):
                        capital *= (1 - self.commission_rate / 100)  # Commission on entry
                        position = df['signal'].iloc[i]
                        entry_price = current_price
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        logger.debug(f"Opened position at index {i}: mode=futures, position={position}, entry_price={entry_price:.2f}")
                else:
                    if df['signal'].iloc[i] == 1 and position == 0:  # Only open long positions in spot
                        position = 1
                        entry_price = current_price
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        logger.debug(f"Opened position at index {i}: mode=spot, position={position}, entry_price={entry_price:.2f}")

        # Close any open position at the last candle
        if position != 0:
            current_price = df['close'].iloc[-1]
            if self.futures:
                pnl = self.leverage * ((current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price) * 100
                pnl -= self.commission_rate * 2
            else:
                if position == 1:  # Only close long positions in spot
                    pnl = (current_price / entry_price - 1) * 100
                    pnl -= self.commission_rate
                else:
                    pnl = 0  # No PNL for short positions in spot
            if pnl != 0:
                capital *= (1 + pnl / 100)
                df.loc[df.index[-1], 'trade_pnl'] = pnl
                df.loc[df.index[-1], 'exit_price'] = current_price
                df.loc[df.index[-1], 'capital'] = capital
                df.loc[df.index[-1], 'cum_pnl'] = (capital / self.initial_capital) - 1
                df.loc[df.index[-1], 'drawdown'] = (max_capital - capital) / max_capital if max_capital > 0 else 0
                logger.debug(f"Closed final position at index {len(df)-1}: mode={'futures' if self.futures else 'spot'}, position={position}, pnl={pnl:.2f}%, capital={capital:.2f}")

        logger.info(f"Generated {len(df[df['signal'] != 0])} signals, {len(df[df['trade_pnl'] != 0])} trades in {'futures' if self.futures else 'spot'} mode")
        return df
    
    def calculate_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.
        """
        final_capital = df['capital'].iloc[-1]
        total_return = ((final_capital / self.initial_capital) - 1) * 100
        max_drawdown = df['drawdown'].max() * 100 if not df['drawdown'].empty else 0

        trades = df[df['trade_pnl'] != 0]
        total_trades = len(trades)
        win_trades = trades[trades['trade_pnl'] > 0]
        win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0

        sharpe_ratio = 0
        if total_trades > 0:
            returns = np.array(trades['trade_pnl'])
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        logger.info(f"Performance in {'futures' if self.futures else 'spot'} mode: total_trades={total_trades}, total_return={total_return:.2f}%, max_drawdown={max_drawdown:.2f}%, win_rate={win_rate:.2f}%")
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "final_capital": final_capital
        }
    
    def collect_signals(self, df: pd.DataFrame, signals_collector: SignalsCollector):
        """
        Collect signals for analysis.
        """
        for i in range(len(df)):
            if df['signal'].iloc[i] != 0:
                signals_collector.add_signal(
                    timestamp=df.index[i],
                    price=df['close'].iloc[i],
                    signal=df['signal'].iloc[i],
                    position=df['position'].iloc[i],
                    entry_price=df['entry_price'].iloc[i] if df['entry_price'].iloc[i] != 0 else None,
                    exit_price=df['exit_price'].iloc[i] if df['exit_price'].iloc[i] != 0 else None,
                    trade_pnl=df['trade_pnl'].iloc[i] if df['trade_pnl'].iloc[i] != 0 else None,
                    cum_pnl=df['cum_pnl'].iloc[i],
                    drawdown=df['drawdown'].iloc[i]
                )
        logger.info(f"Collected {len(df[df['signal'] != 0])} signals for storage in {'futures' if self.futures else 'spot'} mode")