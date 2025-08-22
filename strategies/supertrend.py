import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from signals_collector import SignalsCollector
from strategy_base import Strategy
import logging

logger = logging.getLogger(__name__)

class SupertrendStrategy(Strategy):
    def __init__(self, atr_period: int, atr_multiplier: float, 
                 leverage: float = 1.0, commission_rate: float = 0.04, 
                 initial_capital: float = 10000.0, futures: bool = False,
                 stop_loss_pct: float = 0.02, use_stop_loss: bool = True):
        super().__init__(atr_period=atr_period, atr_multiplier=atr_multiplier, 
                        leverage=leverage, commission_rate=commission_rate, 
                        initial_capital=initial_capital, futures=futures,
                        stop_loss_pct=stop_loss_pct, use_stop_loss=use_stop_loss)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.initial_capital = initial_capital
        self.futures = futures
        self.stop_loss_pct = stop_loss_pct
        self.use_stop_loss = use_stop_loss
        if self.futures:
            logger.info(f"Futures mode enabled with leverage={leverage}x")
        else:
            logger.info("Spot mode enabled (leverage ignored)")
    
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
        # Use EMA for ATR (standard in many implementations)
        df['atr'] = df['tr'].ewm(alpha=1/self.atr_period, min_periods=self.atr_period).mean()

        hl2 = (df['high'] + df['low']) / 2
        df['upper_band'] = hl2 + (self.atr_multiplier * df['atr'])
        df['lower_band'] = hl2 - (self.atr_multiplier * df['atr'])
        df['in_uptrend'] = np.nan  # Initialize as NaN to avoid bias
        df['supertrend'] = np.nan

        # Find first valid ATR index
        first_valid_idx = df['atr'].first_valid_index()
        if first_valid_idx is not None:
            first_idx = df.index.get_loc(first_valid_idx)
            # Set initial supertrend to lower_band, assume uptrend
            df['supertrend'].iloc[first_idx] = df['lower_band'].iloc[first_idx]
            df['in_uptrend'].iloc[first_idx] = True
            # Immediate flip check for initial
            if df['close'].iloc[first_idx] < df['supertrend'].iloc[first_idx]:
                df['in_uptrend'].iloc[first_idx] = False
                df['supertrend'].iloc[first_idx] = df['upper_band'].iloc[first_idx]

        # Update for subsequent rows
        for i in range(first_idx + 1, len(df)):
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            curr_upper = df['upper_band'].iloc[i]
            curr_lower = df['lower_band'].iloc[i]
            prev_lower = df['lower_band'].iloc[i-1]
            prev_upper = df['upper_band'].iloc[i-1]
            prev_supertrend = df['supertrend'].iloc[i-1]

            if pd.isna(prev_supertrend):
                continue

            # Standard flip: compare to prev_supertrend, not basic bands
            if prev_close > prev_supertrend:
                df['in_uptrend'].iloc[i] = True
            elif prev_close < prev_supertrend:
                df['in_uptrend'].iloc[i] = False
            else:
                df['in_uptrend'].iloc[i] = df['in_uptrend'].iloc[i-1]
            
            # Set supertrend based on trend
            if df['in_uptrend'].iloc[i]:
                df['supertrend'].iloc[i] = curr_lower
            else:
                df['supertrend'].iloc[i] = curr_upper

            # Adjust ratchet
            if df['in_uptrend'].iloc[i]:
                if df['in_uptrend'].iloc[i-1] and df['supertrend'].iloc[i] < prev_supertrend:
                    df['supertrend'].iloc[i] = prev_supertrend  # max(curr, prev)
            else:
                if not df['in_uptrend'].iloc[i-1] and df['supertrend'].iloc[i] > prev_supertrend:
                    df['supertrend'].iloc[i] = prev_supertrend  # min(curr, prev)

            # Log proximity for debugging (optional, remove if not needed)
            distance_to_super = (prev_close - prev_supertrend) / prev_supertrend * 100
            if abs(distance_to_super) < 5:
                logger.debug(f"Row {i}: Close {prev_close:.2f}, Supertrend {prev_supertrend:.2f} ({distance_to_super:.2f}%)")

        # Log trend flip count for debugging
        flip_count = (df['in_uptrend'] != df['in_uptrend'].shift(1)).sum(skipna=True)
        logger.info(f"Supertrend trend flips detected: {flip_count}. If 0, no signals possible.")

        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with optional stop-loss.
        """
        # Drop rows with NaN indicators to avoid invalid signals
        df = df.dropna(subset=['atr', 'upper_band', 'lower_band', 'supertrend', 'in_uptrend']).copy()

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
            prev_in_uptrend = df['in_uptrend'].iloc[i-1]
            curr_in_uptrend = df['in_uptrend'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_low = df['low'].iloc[i]
            curr_high = df['high'].iloc[i]

            # Check stop-loss only if enabled
            stop_loss_triggered = False
            exit_price = 0.0
            if self.use_stop_loss and position != 0:
                # Calculate stop-loss levels
                stop_loss_price = 0.0
                if position == 1:  # Long position
                    stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                elif position == -1:  # Short position
                    stop_loss_price = entry_price * (1 + self.stop_loss_pct)

                if position == 1 and curr_low <= stop_loss_price:
                    stop_loss_triggered = True
                    exit_price = min(curr_close, stop_loss_price)  # Exit at stop-loss or worse
                elif position == -1 and curr_high >= stop_loss_price:
                    stop_loss_triggered = True
                    exit_price = max(curr_close, stop_loss_price)  # Exit at stop-loss or worse

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

            # Regular Supertrend signals
            if prev_in_uptrend != curr_in_uptrend:
                if position != 0:  # Exit current position
                    df['signal'].iloc[i] = -position
                    df['position'].iloc[i] = 0
                    df['exit_price'].iloc[i] = curr_close
                    df['entry_price'].iloc[i] = entry_price
                    trade_return = 0.0
                    if self.futures:
                        trade_return = (curr_close - entry_price) / entry_price * position * self.leverage
                    else:
                        trade_return = (curr_close - entry_price) / entry_price * position
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

                # Enter new position
                if curr_in_uptrend and position == 0:  # Enter long
                    position = 1
                    entry_price = curr_close
                    df['signal'].iloc[i] = 1
                    df['position'].iloc[i] = 1
                    df['entry_price'].iloc[i] = entry_price
                elif not curr_in_uptrend and position == 0:  # Enter short
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

        # Log number of signals for debugging
        signal_count = (df['signal'] != 0).sum()
        logger.info(f"Generated {signal_count} signals. If 0, check for trend flips in indicators.")

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