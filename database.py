from typing import *
import logging
import time
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Hdf5Client:
    def __init__(self, exchange: str):
        self.hf = h5py.File(f"data/{exchange}.h5", 'a')
        self.hf.flush()

    def create_dataset(self, symbol: str):
        if symbol not in self.hf.keys():
            self.hf.create_dataset(symbol, (0, 6), maxshape=(None, 6), dtype="float64")
            self.hf.flush()

    def write_data(self, symbol: str, data: List[Tuple]):
        min_ts, max_ts = self.get_first_last_timestamp(symbol)
        min_ts = min_ts if min_ts is not None else float("inf")
        max_ts = max_ts if max_ts is not None else 0

        filtered_data = [d for d in data if d[0] < min_ts or d[0] > max_ts]
        if not filtered_data:
            logger.warning(f"{symbol}: No data to insert")
            return

        data_array = np.array(filtered_data)
        self.hf[symbol].resize(self.hf[symbol].shape[0] + data_array.shape[0], axis=0)
        self.hf[symbol][-data_array.shape[0]:] = data_array
        self.hf.flush()

    def get_data(self, symbol: str, from_time: int, to_time: int) -> Optional[pd.DataFrame]:
        start_query = time.time()
        existing_data = self.hf[symbol][:]

        if len(existing_data) == 0:
            return None

        data = sorted(existing_data, key=lambda x: x[0])
        data = np.array(data)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df[(df["timestamp"] >= from_time) & (df["timestamp"] <= to_time)]

        df["timestamp"] = pd.to_datetime(df["timestamp"].values.astype(np.int64), unit="ms")
        df.set_index("timestamp", drop=True, inplace=True)

        logger.info(f"Retrieved {len(df)} {symbol} candles in {round(time.time() - start_query, 2)} seconds")
        return df

    def get_first_last_timestamp(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        existing_data = self.hf[symbol][:]
        if len(existing_data) == 0:
            return None, None
        return min(existing_data, key=lambda x: x[0])[0], max(existing_data, key=lambda x: x[0])[0]