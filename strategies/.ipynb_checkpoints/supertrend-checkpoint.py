import pandas as pd
import numpy as np


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)


def backtest(df: pd.DataFrame, atr_period: int, atr_multiplier: float):
    """
    Backtest the Supertrend strategy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    atr_period : int
        Period for ATR calculation
    atr_multiplier : float
        Multiplier for ATR to determine the band width
    
    Returns:
    --------
    tuple
        (PnL, Maximum Drawdown)
    """
    
    # Calculate ATR (Average True Range)
    df["tr0"] = abs(df["high"] - df["low"])
    df["tr1"] = abs(df["high"] - df["close"].shift(1))
    df["tr2"] = abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["tr0", "tr1", "tr2"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=atr_period).mean()
    
    # Calculate Supertrend bands
    df["basic_upperband"] = (df["high"] + df["low"]) / 2 + (atr_multiplier * df["atr"])
    df["basic_lowerband"] = (df["high"] + df["low"]) / 2 - (atr_multiplier * df["atr"])
    
    # Initialize the final upperband and lowerband columns
    df["upperband"] = np.nan
    df["lowerband"] = np.nan
    
    # Initialize the supertrend column
    df["supertrend"] = np.nan
    
    # Calculate final upperband and lowerband values
    for i in range(atr_period, len(df)):
        # Current values
        curr_upper = df["basic_upperband"].iloc[i]
        curr_lower = df["basic_lowerband"].iloc[i]
        
        # Previous values
        prev_upper = df["upperband"].iloc[i-1] if not np.isnan(df["upperband"].iloc[i-1]) else df["basic_upperband"].iloc[i-1]
        prev_lower = df["lowerband"].iloc[i-1] if not np.isnan(df["lowerband"].iloc[i-1]) else df["basic_lowerband"].iloc[i-1]
        
        # Previous close price
        prev_close = df["close"].iloc[i-1]
        
        # Final upperband
        if curr_upper < prev_upper or prev_close > prev_upper:
            df["upperband"].iloc[i] = curr_upper
        else:
            df["upperband"].iloc[i] = prev_upper
        
        # Final lowerband
        if curr_lower > prev_lower or prev_close < prev_lower:
            df["lowerband"].iloc[i] = curr_lower
        else:
            df["lowerband"].iloc[i] = prev_lower
    
    # Calculate Supertrend values
    for i in range(atr_period, len(df)):
        curr_upper = df["upperband"].iloc[i]
        curr_lower = df["lowerband"].iloc[i]
        curr_close = df["close"].iloc[i]

        # Previous supertrend value (if available)
        prev_supertrend = df["supertrend"].iloc[i-1] if i > atr_period and not np.isnan(df["supertrend"].iloc[i-1]) else None

        # Logic for supertrend value
        if prev_supertrend is None:
            # First valid supertrend value
            if curr_close <= curr_upper:
                df["supertrend"].iloc[i] = curr_upper
            else:
                df["supertrend"].iloc[i] = curr_lower
        else:
            # Subsequent supertrend values
            if prev_supertrend == df["upperband"].iloc[i-1] and curr_close <= curr_upper:
                df["supertrend"].iloc[i] = curr_upper
            elif prev_supertrend == df["upperband"].iloc[i-1] and curr_close > curr_upper:
                df["supertrend"].iloc[i] = curr_lower
            elif prev_supertrend == df["lowerband"].iloc[i-1] and curr_close >= curr_lower:
                df["supertrend"].iloc[i] = curr_lower
            elif prev_supertrend == df["lowerband"].iloc[i-1] and curr_close < curr_lower:
                df["supertrend"].iloc[i] = curr_upper

    # Generate trading signals
    df["trend"] = np.where(df["close"] > df["supertrend"], 1, -1)
    df["signal"] = df["trend"].diff().fillna(0)

    # Calculate PnL
    df["signal_shift"] = df["trend"].shift(1)
    df["pnl"] = df["close"].pct_change() * df["signal_shift"]

    # Remove NaNs for accurate calculations
    df.dropna(inplace=True)

    # Calculate cumulative PnL and drawdown
    df["cum_pnl"] = df["pnl"].cumsum()
    df["max_cum_pnl"] = df["cum_pnl"].cummax()
    df["drawdown"] = df["max_cum_pnl"] - df["cum_pnl"]

    return df["pnl"].sum(), df["drawdown"].max()