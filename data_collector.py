from typing import *
import logging
import time

from database import Hdf5Client
from utils import ms_to_dt
from exchanges.binance import BinanceClient

logger = logging.getLogger(__name__)

def collect_all(client: BinanceClient, exchange: str, symbol: str, futures: bool = False):
    """
    Collect all available historical data for a symbol.
    """
    h5_db = Hdf5Client(exchange)
    h5_db.create_dataset(symbol, futures)

    oldest_ts, most_recent_ts = h5_db.get_first_last_timestamp(symbol, futures)

    # Initial Request
    if oldest_ts is None:
        data = client.get_historical_data(symbol, end_time=int(time.time() * 1000) - 60000)
        if not data:
            logger.warning(f"{exchange} {symbol} ({'futures' if futures else 'spot'}): No initial data found")
            return

        logger.info(f"{exchange} {symbol} ({'futures' if futures else 'spot'}): Collected {len(data)} initial candles from {ms_to_dt(data[0][0])} to {ms_to_dt(data[-1][0])}")
        oldest_ts = data[0][0]
        most_recent_ts = data[-1][0]
        h5_db.write_data(symbol, data, futures)

    data_to_insert = []

    # Most recent data
    while True:
        data = client.get_historical_data(symbol, start_time=int(most_recent_ts + 60000))
        if not data:
            break

        if len(data) < 2:
            break

        data = data[:-1]
        data_to_insert.extend(data)

        if len(data_to_insert) > 10000:
            h5_db.write_data(symbol, data_to_insert, futures)
            data_to_insert.clear()

        if data[-1][0] > most_recent_ts:
            most_recent_ts = data[-1][0]

        logger.info(f"{exchange} {symbol} ({'futures' if futures else 'spot'}): Collected {len(data)} recent candles from {ms_to_dt(data[0][0])} to {ms_to_dt(data[-1][0])}")
        time.sleep(1.1)

    if data_to_insert:
        h5_db.write_data(symbol, data_to_insert, futures)
        data_to_insert.clear()

    # Older data
    while True:
        data = client.get_historical_data(symbol, end_time=int(oldest_ts - 60000))
        if not data:
            logger.info(f"{exchange} {symbol} ({'futures' if futures else 'spot'}): No older data found before {ms_to_dt(oldest_ts)}")
            break

        data_to_insert.extend(data)

        if len(data_to_insert) > 10000:
            h5_db.write_data(symbol, data_to_insert, futures)
            data_to_insert.clear()

        if data[0][0] < oldest_ts:
            oldest_ts = data[0][0]

        logger.info(f"{exchange} {symbol} ({'futures' if futures else 'spot'}): Collected {len(data)} older candles from {ms_to_dt(data[0][0])} to {ms_to_dt(data[-1][0])}")
        time.sleep(1.1)

    if data_to_insert:
        h5_db.write_data(symbol, data_to_insert, futures)