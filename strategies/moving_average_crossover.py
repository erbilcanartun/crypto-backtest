import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from signals_collector import SignalsCollector
from strategy_base import Strategy
import logging

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving Average Crossover strategy for spot and futures markets.
    """
    def __init__(self, short_period: int, long_period: int,
                 leverage: float = 1.0, commission_rate: float = 0.04,
                 initial_capital: float = 10000.0, futures: bool = False,
                 stop_loss_pct: float = 0.02, use_stop_loss: bool = True):
        """
        Initialize the Moving Average Crossover strategy.
        """
        super().__init__(short_period=short_period, long_period=long_period,
                         leverage=leverage, commission_rate=commission_rate,
                         initial_capital=initial_capital, futures=futures,
                         stop_loss_pct=stop_loss_pct, use_stop_loss=use_stop_loss)
        self.short_period = short_period
        self.long_period = long_period
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.initial_capital = initial_capital
        self.futures = futures
        self.stop_loss_pct = stop_loss_pct
        self.use_stop_loss = use_stop_loss

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA indicators.
        """
        df = df.copy()
        df['short_ema'] = df['close'].ewm(span=self.short_period, adjust=False).mean()
        df['long_ema'] = df['close'].ewm(span=self.long_period, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with optional stop-loss.
        """
        df = df.copy()
        df['signal'] = 0
        df['position'] = 0
        df['entry_price'] = 0.0
        df['exit_price'] = 0.0
        df['trade_pnl'] = 0.0
        df['cum_pnl'] = 0.0
        df['drawdown'] = 0.0

        position = 0
        entry_price = 0.0
        capital = self.initial_capital
        cum_pnl = 0.0
        max_capital = capital

        for i in range(1, len(df)):
            curr_close = df['close'].iloc[i]
            curr_low = df['low'].iloc[i]
            curr_high = df['high'].iloc[i]
            prev_short = df['short_ema'].iloc[i-1]
            prev_long = df['long_ema'].iloc[i-1]
            curr_short = df['short_ema'].iloc[i]
            curr_long = df['long_ema'].iloc[i]

            # Check stop-loss only if enabled
            stop_loss_triggered = False
            exit_price = 0.0
            if self.use_stop_loss and position != 0:
                stop_loss_price = 0.0
                if position == 1:  # Long
                    stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                elif position == -1:  # Short
                    stop_loss_price = entry_price * (1 + self.stop_loss_pct)

                if position == 1 and curr_low <= stop_loss_price:
                    stop_loss_triggered = True
                    exit_price = min(curr_close, stop_loss_price)
                elif position == -1 and curr_high >= stop_loss_price:
                    stop_loss_triggered = True
                    exit_price = max(curr_close, stop_loss_price)

            if stop_loss_triggered:
                df['signal'].iloc[i] = -position
                df['position'].iloc[i] = 0
                df['exit_price'].iloc[i] = exit_price
                df['entry_price'].iloc[i] = entry_price
                trade_return = 0.0
                if self.futures:
                    trade_return = (exit_price - entry_price) / entry_price * position * self.leverage
                else:
                    trade_return = (exit_price - entry_price) / entry_price * position
                commission = abs(trade_return) * self.commission_rate / 100
                trade_pnl = (trade_return - commission) * capital
                cum_pnl += trade_pnl
                capital += trade_pnl
                df['trade_pnl'].iloc[i] = trade_pnl
                df['cum_pnl'].iloc[i] = cum_pnl
                df['drawdown'].iloc[i] = (max_capital - capital) / max_capital
                max_capital = max(max_capital, capital)
                position = 0
                entry_price = 0.0
                continue

            # Regular crossover signals
            if prev_short < prev_long and curr_short > curr_long:  # Crossover: buy
                if position == -1:  # Exit short
                    df['signal'].iloc[i] = 1  # Exit short and enter long
                    df['position'].iloc[i] = 1
                    df['exit_price'].iloc[i] = curr_close
                    df['entry_price'].iloc[i] = entry_price
                    trade_return = (curr_close - entry_price) / entry_price * position * (self.leverage if self.futures else 1)
                    commission = abs(trade_return) * self.commission_rate / 100
                    trade_pnl = (trade_return - commission) * capital
                    cum_pnl += trade_pnl
                    capital += trade_pnl
                    df['trade_pnl'].iloc[i] = trade_pnl
                    df['cum_pnl'].iloc[i] = cum_pnl
                    df['drawdown'].iloc[i] = (max_capital - capital) / max_capital
                    max_capital = max(max_capital, capital)

                position = 1
                entry_price = curr_close
                df['signal'].iloc[i] = 1
                df['position'].iloc[i] = 1
                df['entry_price'].iloc[i] = entry_price
            elif prev_short > prev_long and curr_short < curr_long:  # Crossunder: sell
                if position == 1:  # Exit long
                    df['signal'].iloc[i] = -1  # Exit long and enter short
                    df['position'].iloc[i] = -1
                    df['exit_price'].iloc[i] = curr_close
                    df['entry_price'].iloc[i] = entry_price
                    trade_return = (curr_close - entry_price) / entry_price * position * (self.leverage if self.futures else 1)
                    commission = abs(trade_return) * self.commission_rate / 100
                    trade_pnl = (trade_return - commission) * capital
                    cum_pnl += trade_pnl
                    capital += trade_pnl
                    df['trade_pnl'].iloc[i] = trade_pnl
                    df['cum_pnl'].iloc[i] = cum_pnl
                    df['drawdown'].iloc[i] = (max_capital - capital) / max_capital
                    max_capital = max(max_capital, capital)

                position = -1
                entry_price = curr_close
                df['signal'].iloc[i] = -1
                df['position'].iloc[i] = -1
                df['entry_price'].iloc[i] = entry_price
            else:
                df['position'].iloc[i] = position
                df['entry_price'].iloc[i] = entry_price
                df['cum_pnl'].iloc[i] = cum_pnl
                df['drawdown'].iloc[i] = (max_capital - capital) / max_capital

        logger.info(f"Generated {len(df[df['signal'] != 0])} signals.")
        return df

    def calculate_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.
        """
        final_capital = self.initial_capital + df['cum_pnl'].iloc[-1] if not df['cum_pnl'].empty else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        max_drawdown = df['drawdown'].max() if not df['drawdown'].empty else 0

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