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

    def create_dataset(self, symbol: str, futures: bool = False):
        """Create dataset in market_type/symbol/1m structure."""
        market_type = 'futures' if futures else 'spot'
        dataset_path = f"{market_type}/{symbol}/1m"
        if dataset_path not in self.hf:
            self.hf.create_dataset(dataset_path, (0, 6), maxshape=(None, 6), dtype="float64")
            logger.info(f"Created dataset: {dataset_path}")
        self.hf.flush()

    def write_data(self, symbol: str, data: List[Tuple], futures: bool = False):
        """Write data to market_type/symbol/1m dataset."""
        market_type = 'futures' if futures else 'spot'
        dataset_path = f"{market_type}/{symbol}/1m"
        if dataset_path not in self.hf:
            self.create_dataset(symbol, futures)

        min_ts, max_ts = self.get_first_last_timestamp(symbol, futures)
        min_ts = min_ts if min_ts is not None else float("inf")
        max_ts = max_ts if max_ts is not None else 0

        filtered_data = [d for d in data if d[0] < min_ts or d[0] > max_ts]
        if not filtered_data:
            logger.warning(f"{symbol}: No data to insert")
            return

        data_array = np.array(filtered_data)
        dataset = self.hf[dataset_path]
        dataset.resize(dataset.shape[0] + data_array.shape[0], axis=0)
        dataset[-data_array.shape[0]:] = data_array
        self.hf.flush()
        logger.info(f"Wrote {len(data_array)} rows to {dataset_path}")

    def get_data(self, symbol: str, from_time: int, to_time: int, futures: bool = False) -> Optional[pd.DataFrame]:
        """Retrieve data from market_type/symbol/1m dataset."""
        start_query = time.time()
        market_type = 'futures' if futures else 'spot'
        dataset_path = f"{market_type}/{symbol}/1m"
        if dataset_path not in self.hf:
            logger.error(f"Dataset {dataset_path} not found")
            return None

        existing_data = self.hf[dataset_path][:]
        if len(existing_data) == 0:
            logger.warning(f"No data in {dataset_path}")
            return None

        data = sorted(existing_data, key=lambda x: x[0])
        data = np.array(data)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df[(df["timestamp"] >= from_time) & (df["timestamp"] <= to_time)]

        df["timestamp"] = pd.to_datetime(df["timestamp"].values.astype(np.int64), unit="ms")
        df.set_index("timestamp", drop=True, inplace=True)

        logger.info(f"Retrieved {len(df)} {symbol} candles in {round(time.time() - start_query, 2)} seconds")
        return df

    def get_first_last_timestamp(self, symbol: str, futures: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """Get first and last timestamps from market_type/symbol/1m dataset."""
        market_type = 'futures' if futures else 'spot'
        dataset_path = f"{market_type}/{symbol}/1m"
        if dataset_path not in self.hf:
            return None, None
        existing_data = self.hf[dataset_path][:]
        if len(existing_data) == 0:
            return None, None
        return min(existing_data, key=lambda x: x[0])[0], max(existing_data, key=lambda x: x[0])[0]