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
        if self.futures:  # Added for mode/leverage confirmation
            logger.info(f"Futures mode enabled with leverage={leverage}x")
        else:
            logger.info("Spot mode enabled (leverage ignored)")

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
        peak_capital = self.initial_capital

        for i in range(1, len(df)):
            if df['short_ema'].iloc[i] > df['long_ema'].iloc[i] and df['short_ema'].iloc[i-1] <= df['long_ema'].iloc[i-1]:
                df['signal'].iloc[i] = 1 if not self.futures else 1  # Buy/Long
            elif df['short_ema'].iloc[i] < df['long_ema'].iloc[i] and df['short_ema'].iloc[i-1] >= df['long_ema'].iloc[i-1]:
                df['signal'].iloc[i] = -1  # Sell/Short or close

            price = df['close'].iloc[i]

            # Handle stop-loss (futures only)
            if self.use_stop_loss and self.futures and position != 0:
                if position > 0 and price <= entry_price * (1 - self.stop_loss_pct):
                    df['signal'].iloc[i] = -1  # Close long on SL
                elif position < 0 and price >= entry_price * (1 + self.stop_loss_pct):
                    df['signal'].iloc[i] = 1  # Close short on SL

            # Execute trades
            if df['signal'].iloc[i] == 1:  # Buy/Long signal
                if position <= 0:  # Enter or flip to long
                    if position < 0:  # Close short first
                        exit_price = price
                        trade_pnl = position * (exit_price - entry_price) / entry_price * 100 - self.commission_rate * 2
                        cum_pnl += trade_pnl
                        capital *= (1 + trade_pnl / 100)
                        df['exit_price'].iloc[i] = exit_price
                        df['trade_pnl'].iloc[i] = trade_pnl
                        df['cum_pnl'].iloc[i] = cum_pnl

                    # Enter long
                    entry_price = price
                    if self.futures:
                        position = (capital * self.leverage) / entry_price
                    else:
                        position = capital / entry_price
                    df['entry_price'].iloc[i] = entry_price
                    capital -= capital * self.commission_rate / 100  # Commission on entry

            elif df['signal'].iloc[i] == -1:  # Sell/Short signal
                if position >= 0:  # Exit or flip to short
                    if position > 0:  # Close long first
                        exit_price = price
                        trade_pnl = position * (exit_price - entry_price) / entry_price * 100 - self.commission_rate * 2
                        cum_pnl += trade_pnl
                        capital *= (1 + trade_pnl / 100)
                        df['exit_price'].iloc[i] = exit_price
                        df['trade_pnl'].iloc[i] = trade_pnl
                        df['cum_pnl'].iloc[i] = cum_pnl

                    if self.futures:
                        # Enter short
                        entry_price = price
                        position = - (capital * self.leverage) / entry_price
                        df['entry_price'].iloc[i] = entry_price
                        capital -= capital * self.commission_rate / 100  # Commission on entry

            df['position'].iloc[i] = position
            current_value = capital + position * (price - entry_price) if position != 0 else capital
            drawdown = (peak_capital - current_value) / peak_capital * 100 if peak_capital > 0 else 0
            df['drawdown'].iloc[i] = drawdown
            if current_value > peak_capital:
                peak_capital = current_value

        return df

    def calculate_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.
        """
        final_capital = df['cum_pnl'].iloc[-1] + self.initial_capital if not df.empty else self.initial_capital
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