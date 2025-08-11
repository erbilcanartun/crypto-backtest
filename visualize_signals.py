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
            dataset_path = f"{market_type}/{symbol}/1m"
            if dataset_path not in f:
                raise KeyError(f"Dataset {dataset_path} not found in {hdf5_path}")
            data = f[dataset_path][:]
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
    except Exception as e:
        logger.error(f"Failed to load price data: {e}")
        raise

def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to the specified timeframe.
    """
    tf_map = {
        '1h': '1H',
        '1d': '1D'
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(tf_map.keys())}")
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_resampled = df.resample(tf_map[timeframe]).agg(agg_dict).dropna()
    return df_resampled

def load_signals(signals_path: str) -> pd.DataFrame:
    """
    Load signals from CSV file.
    """
    try:
        df = pd.read_csv(signals_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load signals: {e}")
        raise

def create_candlestick_chart(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    symbol: str,
    futures: bool
) -> go.Figure:
    """
    Create an interactive candlestick chart with signal markers.
    """
    # Create figure
    fig = make_subplots(rows=1, cols=1)

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='Candlestick'
        )
    )

    # Filter buy/long and sell/short signals
    buy_signals = signals_df[signals_df['signal'] == 1]
    sell_signals = signals_df[signals_df['signal'] == -1]

    # Define signal labels based on mode
    buy_label = 'Long' if futures else 'Buy'
    sell_label = 'Short' if futures else 'Sell'

    # Add buy/long signals (green upward triangles)
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name=buy_label,
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green'
                ),
                text=[
                    f"{buy_label}<br>Price: {price:.2f}<br>Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}<br>PNL: {pnl if pd.notna(pnl) else 'N/A'}%"
                    for ts, price, pnl in zip(buy_signals.index, buy_signals['price'], buy_signals['trade_pnl'])
                ],
                hoverinfo='text'
            )
        )

    # Add sell/short signals (red downward triangles)
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name=sell_label,
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red'
                ),
                text=[
                    f"{sell_label}<br>Price: {price:.2f}<br>Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}<br>PNL: {pnl if pd.notna(pnl) else 'N/A'}%"
                    for ts, price, pnl in zip(sell_signals.index, sell_signals['price'], sell_signals['trade_pnl'])
                ],
                hoverinfo='text'
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Candlestick Chart with Supertrend Signals ({'Futures' if futures else 'Spot'})",
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template='plotly_dark',
        hovermode='x unified'
    )

    # Update x-axis to handle datetime
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[17, 9], pattern="hour")  # Hide non-trading hours (adjust as needed)
        ]
    )

    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Supertrend signals.')
    parser.add_argument('--config', type=str, default='visualize_config.json', help='Path to visualization config file')
    parser.add_argument('--symbol', type=str, help='Trading pair (e.g., BTCUSDT), overrides config')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 1h, 1d), overrides config')
    parser.add_argument('--futures', action='store_true', help='Use futures data (default: spot), overrides config')
    parser.add_argument('--hdf5-path', type=str, help='Path to HDF5 data file, overrides config')
    parser.add_argument('--signals-path', type=str, help='Path to signals CSV file, overrides config')
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
    output_file = args.output or config.get('output_file', 'signals_plot.html')

    logger.info(f"Loading price data for {symbol} ({'futures' if futures else 'spot'}) from {hdf5_path}")
    price_df = load_price_data(hdf5_path, symbol, futures)
    
    logger.info(f"Resampling data to {timeframe}")
    price_df = resample_timeframe(price_df, timeframe)

    logger.info(f"Loading signals from {signals_path}")
    signals_df = load_signals(signals_path)

    # Filter price data to match signals period
    start_date = signals_df.index.min()
    end_date = signals_df.index.max()
    price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]

    logger.info("Creating candlestick chart")
    fig = create_candlestick_chart(price_df, signals_df, symbol, futures)

    logger.info(f"Saving chart to {output_file}")
    fig.write_html(output_file)
    logger.info(f"Chart saved. Open {output_file} in a browser to view.")

if __name__ == "__main__":
    main()