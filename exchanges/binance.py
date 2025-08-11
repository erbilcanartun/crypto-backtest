from typing import *
import logging
import requests

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, futures: bool = False):
        self.futures = futures
        self._base_url = "https://fapi.binance.com" if futures else "https://api.binance.com"
        self.symbols = self._get_symbols()

    def _make_request(self, endpoint: str, query_parameters: Dict) -> Dict:
        try:
            response = requests.get(self._base_url + endpoint, params=query_parameters)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            raise

    def _get_symbols(self) -> List[str]:
        endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        data = self._make_request(endpoint, {})
        return [x["symbol"] for x in data["symbols"]]

    def get_historical_data(self, symbol: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Tuple]:
        params = {"symbol": symbol, "interval": "1m", "limit": 1500}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"
        raw_candles = self._make_request(endpoint, params)
        return [(float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])) for c in raw_candles]