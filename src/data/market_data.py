import time

import pandas as pd

from src.core.exchange import ExchangeManager
from src.core.logger import setup_logger

logger = setup_logger(__name__)

# Retry configuration for transient failures
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1  # seconds, doubles each retry


class MarketDataManager:
    """Provides market data (OHLCV, tickers, orderbooks) via the exchange wrapper.

    Args:
        exchange: An initialized ExchangeManager instance.
    """

    def __init__(self, exchange: ExchangeManager):
        self.exchange = exchange

    def _retry_fetch(self, fetch_func, *args, **kwargs):
        """Execute a fetch function with retry logic for transient failures.

        Args:
            fetch_func: The function to call.
            *args, **kwargs: Arguments to pass to fetch_func.

        Returns:
            Result from fetch_func.

        Raises:
            Exception: If all retries are exhausted.
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fetch_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"Market data fetch failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {delay}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Market data fetch failed after {MAX_RETRIES} attempts: {e}")
        raise last_error

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candle data with retry logic.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT").
            timeframe: Candle timeframe (e.g. "1m", "5m", "1h").
            limit: Number of candles.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        df = self._retry_fetch(self.exchange.fetch_ohlcv, symbol, timeframe, limit)
        logger.debug(f"OHLCV fetched: {symbol} {timeframe} ({len(df)} candles)")
        return df

    def get_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data with retry logic.

        Args:
            symbol: Trading pair.

        Returns:
            Dict with bid, ask, last, baseVolume, quoteVolume.
        """
        ticker = self._retry_fetch(self.exchange.fetch_ticker, symbol)
        return {
            "symbol": symbol,
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "last": ticker.get("last"),
            "base_volume": ticker.get("baseVolume"),
            "quote_volume": ticker.get("quoteVolume"),
        }

    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Fetch orderbook with spread calculation and retry logic.

        Args:
            symbol: Trading pair.
            limit: Depth of orderbook per side.

        Returns:
            Dict with bids, asks, spread, mid_price.
        """
        ob = self._retry_fetch(self.exchange.fetch_orderbook, symbol, limit)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": (spread / mid_price * 100) if mid_price else 0,
            "mid_price": mid_price,
        }

    def get_funding_rate(self, symbol: str) -> float | None:
        """Fetch current funding rate for a perpetual futures pair.

        Stub for Phase 2 — returns None until futures trading is implemented.
        """
        logger.debug(f"Funding rate fetch not yet implemented for {symbol}")
        return None
