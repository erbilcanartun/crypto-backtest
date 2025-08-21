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

# Map strategies to their expected indicator columns
STRATEGY_INDICATORS = {
    'supertrend': [
        {'column': 'supertrend', 'name': 'Supertrend', 'color': 'purple', 'dash': None},
        {'column': 'upper_band', 'name': 'Upper Band', 'color': 'red', 'dash': 'dash'},
        {'column': 'lower_band', 'name': 'Lower Band', 'color': 'green', 'dash': 'dash'}
    ],
    'moving_average_crossover': [
        {'column': 'short_ema', 'name': 'Short EMA', 'color': 'blue', 'dash': None},
        {'column': 'long_ema', 'name': 'Long EMA', 'color': 'orange', 'dash': None}
    ]
}

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def load_price_data(hdf5_path: str, symbol: str, futures: bool) -> pd.DataFrame:
    """Load OHLCV data from HDF5 file."""
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
    """Resample OHLCV data to the specified timeframe."""
    tf_map = {'1h': '1h', '1d': '1d'}
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(tf_map.keys())}")
    return df.resample(tf_map[timeframe]).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

def load_signals(signals_path: str) -> pd.DataFrame:
    """Load signals from CSV file."""
    try:
        df = pd.read_csv(signals_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"Loaded {len(df)} signals from {signals_path}")
        if len(df) == 0:
            logger.warning("No signals found in CSV. Chart may lack signal markers.")
        return df
    except Exception as e:
        logger.error(f"Failed to load signals: {e}")
        raise

def create_candlestick_chart(price_df: pd.DataFrame, signals_df: pd.DataFrame, symbol: str, futures: bool, signals_path: str) -> go.Figure:
    """Create interactive candlestick chart with signals, indicators, PNL, and drawdown."""
    # Detect strategy from signals_path
    strategy = None
    for strat in STRATEGY_INDICATORS:
        if strat in signals_path:
            strategy = strat
            break
    if not strategy:
        logger.warning(f"Could not detect strategy from {signals_path}. Skipping indicator plots.")
        indicators_to_plot = []
    else:
        indicators_to_plot = STRATEGY_INDICATORS[strategy]
        logger.info(f"Detected strategy: {strategy}")

    # Initialize subplots: price (main), PNL, drawdown
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.15, 0.15],
        subplot_titles=('Price Chart', 'Cumulative PNL', 'Drawdown')
    )

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='Candlestick'
        ),
        row=1, col=1
    )

    # Load and plot indicators
    indicators_path = signals_path.replace('_signals.csv', '_indicators.csv')
    if os.path.exists(indicators_path):
        indicators_df = pd.read_csv(indicators_path, index_col='timestamp', parse_dates=True)
        for indicator in indicators_to_plot:
            if indicator['column'] in indicators_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicators_df.index,
                        y=indicators_df[indicator['column']],
                        mode='lines',
                        name=indicator['name'],
                        line=dict(color=indicator['color'], dash=indicator['dash'])
                    ),
                    row=1, col=1
                )
                logger.info(f"Added {indicator['name']} to chart.")
            else:
                logger.warning(f"{indicator['name']} ({indicator['column']}) not found in indicators CSV.")
    else:
        logger.warning(f"Indicators file not found: {indicators_path}. Skipping indicator plots.")

    # Signal markers with futures/spot labels and hover text
    signal_text = {'buy': 'Long' if futures else 'Buy', 'sell': 'Short' if futures else 'Sell'}
    buy_signals = signals_df[signals_df['signal'] > 0]
    sell_signals = signals_df[signals_df['signal'] < 0]

    # Buy/Long signals
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=price_df.loc[buy_signals.index]['low'] * 0.99,  # Slightly below low
                mode='markers+text',
                marker=dict(symbol='triangle-up', color='green', size=10),
                text=[signal_text['buy']] * len(buy_signals),
                textposition='bottom center',
                hovertext=[
                    f"{signal_text['buy']} at {row['entry_price']:.2f}, PNL: {row['trade_pnl']:.2f}"
                    for _, row in buy_signals.iterrows()
                ],
                hoverinfo='text',
                name=f"{signal_text['buy']} Signals"
            ),
            row=1, col=1
        )

    # Sell/Short signals
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=price_df.loc[sell_signals.index]['high'] * 1.01,  # Slightly above high
                mode='markers+text',
                marker=dict(symbol='triangle-down', color='red', size=10),
                text=[signal_text['sell']] * len(sell_signals),
                textposition='top center',
                hovertext=[
                    f"{signal_text['sell']} at {row['exit_price']:.2f}, PNL: {row['trade_pnl']:.2f}"
                    for _, row in sell_signals.iterrows()
                ],
                hoverinfo='text',
                name=f"{signal_text['sell']} Signals"
            ),
            row=1, col=1
        )

    # PNL subplot
    if 'cum_pnl' in signals_df.columns and not signals_df['cum_pnl'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['cum_pnl'],
                mode='lines',
                name='Cumulative PNL',
                line=dict(color='green')
            ),
            row=2, col=1
        )
    else:
        logger.warning("No valid cum_pnl data for PNL subplot.")

    # Drawdown subplot
    if 'drawdown' in signals_df.columns and not signals_df['drawdown'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            ),
            row=3, col=1
        )
    else:
        logger.warning("No valid drawdown data for drawdown subplot.")

    # Layout updates
    fig.update_layout(
        title=f"{symbol} Signals and Indicators ({'Futures' if futures else 'Spot'})",
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text='PNL', row=2, col=1)
    fig.update_yaxes(title_text='Drawdown', row=3, col=1)

    return fig

def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(description="Visualize trading signals and indicators.")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--symbol', type=str, help='Symbol (e.g., BTCUSDT), overrides config')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 1h, 1d), overrides config')
    parser.add_argument('--futures', action='store_true', help='Use futures data (default: spot), overrides config')
    parser.add_argument('--hdf5-path', type=str, help='Path to HDF5 data file, overrides config')
    parser.add_argument('--signals-path', type=str, help='Path to signals CSV file, overrides config')
    parser.add_argument('--output', type=str, help='Output HTML file, overrides config')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI arguments
    symbol = args.symbol or config.get('symbol', 'BTCUSDT')
    timeframe = args.timeframe or config.get('timeframe', '1h')
    futures = args.futures or config.get('futures', False)
    hdf5_path = args.hdf5_path or config.get('hdf5_path', 'data/binance.h5')
    signals_path = args.signals_path or config.get('signals_path', f"moving_average_crossover_{symbol}_{timeframe}_signals.csv")
    output_file = args.output or config.get('output_file', 'signals_plot.html')

    logger.info(f"Loading price data for {symbol} ({'futures' if futures else 'spot'}) from {hdf5_path}")
    price_df = load_price_data(hdf5_path, symbol, futures)
    
    logger.info(f"Resampling data to {timeframe}")
    price_df = resample_timeframe(price_df, timeframe)

    logger.info(f"Loading signals from {signals_path}")
    signals_df = load_signals(signals_path)

    # Filter price data to match signals period
    if not signals_df.empty:
        start_date = signals_df.index.min()
        end_date = signals_df.index.max()
        price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
    else:
        logger.warning("Empty signals DataFrame. Using full price data range.")

    logger.info("Creating candlestick chart")
    fig = create_candlestick_chart(price_df, signals_df, symbol, futures, signals_path)

    logger.info(f"Saving chart to {output_file}")
    fig.write_html(output_file)
    logger.info(f"Chart saved. Open {output_file} in a browser to view.")

if __name__ == "__main__":
    main()