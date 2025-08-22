import pandas as pd
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import logging
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s :: %(message)s',
    handlers=[
        logging.FileHandler('info.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def load_price_data(hdf5_path: str, symbol: str, futures: bool) -> pd.DataFrame:
    """
    Load OHLCV data from HDF5 file.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            market_type = 'futures' if futures else 'spot'
            dataset_path = f"{market_type}/{symbol}"
            if dataset_path not in f:
                raise KeyError(f"Dataset {dataset_path} not found in {hdf5_path}")
            data = f[dataset_path][:]
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"Loaded {len(df)} candles from {dataset_path}")
            return df
    except Exception as e:
        logger.error(f"Failed to load price data: {e}")
        raise

def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to the specified timeframe.
    """
    tf_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1H',
        '4h': '4H',
        '12h': '12H',
        '1d': '1D'
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(tf_map.keys())}")
    if timeframe != '1m':
        df = df.resample(tf_map[timeframe]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        logger.info(f"Resampled data to {timeframe} ({len(df)} candles)")
    return df

def load_signals(signals_path: str) -> pd.DataFrame:
    """
    Load signals from CSV file.
    """
    try:
        df = pd.read_csv(signals_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"Loaded {len(df)} signals from {signals_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load signals: {e}")
        raise

def load_indicators(indicators_path: str) -> pd.DataFrame:
    """
    Load indicators from CSV file.
    """
    try:
        df = pd.read_csv(indicators_path, index_col='timestamp')
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded indicators from {indicators_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load indicators: {e}")
        raise

def create_candlestick_chart(price_df: pd.DataFrame, signals_df: pd.DataFrame, indicators_df: pd.DataFrame, 
                            symbol: str, futures: bool, strategy: str) -> go.Figure:
    """
    Create a candlestick chart with signals and indicators.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"{symbol} {'Futures' if futures else 'Spot'} Price", "Cumulative PnL"),
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Calculate offset for signal markers (0.5% of price range)
    price_range = price_df['high'].max() - price_df['low'].min()
    offset = price_range * 0.005

    # Buy signals (below candlestick low)
    buy_signals = signals_df[signals_df['signal'] == 1]
    buy_y = price_df.loc[buy_signals.index, 'low'] - offset
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_y,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Buy Signal'
        ),
        row=1, col=1
    )

    # Sell signals (above candlestick high)
    sell_signals = signals_df[signals_df['signal'] == -1]
    sell_y = price_df.loc[sell_signals.index, 'high'] + offset
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_y,
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Sell Signal'
        ),
        row=1, col=1
    )

    # Plot indicators based on strategy
    if strategy == 'supertrend' and not indicators_df.empty:
        if 'upper_band' in indicators_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators_df.index,
                    y=indicators_df['upper_band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='orange', dash='dash')
                ),
                row=1, col=1
            )
        if 'lower_band' in indicators_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators_df.index,
                    y=indicators_df['lower_band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='blue', dash='dash')
                ),
                row=1, col=1
            )
    elif strategy == 'moving_average_crossover' and not indicators_df.empty:
        if 'short_ema' in indicators_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators_df.index,
                    y=indicators_df['short_ema'],
                    mode='lines',
                    name='Short EMA',
                    line=dict(color='cyan')
                ),
                row=1, col=1
            )
        if 'long_ema' in indicators_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators_df.index,
                    y=indicators_df['long_ema'],
                    mode='lines',
                    name='Long EMA',
                    line=dict(color='purple')
                ),
                row=1, col=1
            )

    # Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=signals_df.index,
            y=signals_df['cum_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} {'Futures' if futures else 'Spot'} - {strategy.replace('_', ' ').title()} Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template='plotly_dark'
    )

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL", row=2, col=1)

    return fig

def main():
    """
    Main function to generate and save the signals visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize trading signals with price data")
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    parser.add_argument('--symbol', type=str, help='Symbol (e.g., BTCUSDT), overrides config')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 1h, 1d), overrides config')
    parser.add_argument('--futures', action='store_true', help='Use futures data (default: spot), overrides config')
    parser.add_argument('--hdf5-path', type=str, help='Path to HDF5 data file, overrides config')
    parser.add_argument('--signals-path', type=str, help='Path to signals CSV file, overrides config')
    parser.add_argument('--indicators-path', type=str, help='Path to indicators CSV file, overrides config')
    parser.add_argument('--strategy', type=str, help='Strategy name, overrides inferred strategy')
    parser.add_argument('--output', type=str, help='Output HTML file, overrides config')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    symbol = args.symbol or config.get('symbol', 'BTCUSDT')
    timeframe = args.timeframe or config.get('timeframe', '1h')
    futures = args.futures or config.get('futures', False)
    hdf5_path = args.hdf5_path or config.get('hdf5_path', 'data/binance.h5')
    signals_path = args.signals_path or config.get('signals_path', f"supertrend_{symbol}_{timeframe}_signals.csv")
    
    # Infer strategy from signals_path if not provided
    strategy = args.strategy or config.get('strategy')
    if not strategy:
        if 'supertrend' in signals_path.lower():
            strategy = 'supertrend'
        elif 'moving_average_crossover' in signals_path.lower():
            strategy = 'moving_average_crossover'
        else:
            strategy = 'supertrend'  # Default fallback
    logger.info(f"Using strategy: {strategy}")

    indicators_path = args.indicators_path or config.get('indicators_path', f"{strategy}_{symbol}_{timeframe}_indicators.csv")
    output_file = args.output or config.get('output_file', 'signals_plot.html')

    logger.info(f"Loading price data for {symbol} ({'futures' if futures else 'spot'}) from {hdf5_path}")
    price_df = load_price_data(hdf5_path, symbol, futures)
    
    logger.info(f"Resampling data to {timeframe}")
    price_df = resample_timeframe(price_df, timeframe)

    logger.info(f"Loading signals from {signals_path}")
    signals_df = load_signals(signals_path)

    logger.info(f"Loading indicators from {indicators_path}")
    try:
        indicators_df = load_indicators(indicators_path)
    except Exception as e:
        logger.warning(f"Could not load indicators: {e}. Proceeding without indicators.")
        indicators_df = pd.DataFrame()

    # Filter price data to match signals period
    start_date = signals_df.index.min()
    end_date = signals_df.index.max()
    price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
    if not indicators_df.empty:
        indicators_df = indicators_df[(indicators_df.index >= start_date) & (indicators_df.index <= end_date)]

    logger.info("Creating candlestick chart")
    fig = create_candlestick_chart(price_df, signals_df, indicators_df, symbol, futures, strategy)

    logger.info(f"Saving chart to {output_file}")
    fig.write_html(output_file)
    logger.info(f"Chart saved. Open {output_file} in a browser to view.")

if __name__ == "__main__":
    main()