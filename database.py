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
        if 'spot' not in self.hf:
            self.hf.create_group('spot')
        if 'futures' not in self.hf:
            self.hf.create_group('futures')
        self.hf.flush()

    def _get_dataset_path(self, symbol: str, futures: bool) -> str:
        return f"{'futures' if futures else 'spot'}/{symbol}"

    def create_dataset(self, symbol: str, futures: bool):
        path = self._get_dataset_path(symbol, futures)
        if path not in self.hf:
            self.hf.create_dataset(path, (0, 6), maxshape=(None, 6), dtype="float64")
            self.hf.flush()

    def write_data(self, symbol: str, data: List[Tuple], futures: bool):
        path = self._get_dataset_path(symbol, futures)
        min_ts, max_ts = self.get_first_last_timestamp(symbol, futures)
        min_ts = min_ts if min_ts is not None else float("inf")
        max_ts = max_ts if max_ts is not None else 0

        filtered_data = [d for d in data if d[0] < min_ts or d[0] > max_ts]
        if not filtered_data:
            logger.warning(f"{symbol} ({'futures' if futures else 'spot'}): No data to insert")
            return

        data_array = np.array(filtered_data)
        self.hf[path].resize(self.hf[path].shape[0] + data_array.shape[0], axis=0)
        self.hf[path][-data_array.shape[0]:] = data_array
        self.hf.flush()

    def get_data(self, symbol: str, from_time: int, to_time: int, futures: bool) -> Optional[pd.DataFrame]:
        path = self._get_dataset_path(symbol, futures)
        if path not in self.hf:
            logger.warning(f"No dataset found for {symbol} ({'futures' if futures else 'spot'})")
            return None

        start_query = time.time()
        existing_data = self.hf[path][:]

        if len(existing_data) == 0:
            return None

        data = sorted(existing_data, key=lambda x: x[0])
        data = np.array(data)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df[(df["timestamp"] >= from_time) & (df["timestamp"] <= to_time)]

        df["timestamp"] = pd.to_datetime(df["timestamp"].values.astype(np.int64), unit="ms")
        df.set_index("timestamp", drop=True, inplace=True)

        logger.info(f"Retrieved {len(df)} {symbol} ({'futures' if futures else 'spot'}) candles in {round(time.time() - start_query, 2)} seconds")
        return df

    def get_first_last_timestamp(self, symbol: str, futures: bool) -> Tuple[Optional[float], Optional[float]]:
        path = self._get_dataset_path(symbol, futures)
        if path not in self.hf or len(self.hf[path]) == 0:
            return None, None
        existing_data = self.hf[path][:]
        return min(existing_data, key=lambda x: x[0])[0], max(existing_data, key=lambda x: x[0])[0]