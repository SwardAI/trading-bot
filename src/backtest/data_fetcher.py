import os
import time
from pathlib import Path

import ccxt
import pandas as pd

from src.core.logger import setup_logger

logger = setup_logger("backtest.data_fetcher")

DATA_DIR = Path("data/historical")


def download_ohlcv(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since: str = "2025-01-01",
    limit_per_request: int = 1000,
) -> pd.DataFrame:
    """Download historical OHLCV data from an exchange.

    Downloads in chunks respecting rate limits, caches to CSV.

    Args:
        exchange_id: Exchange name.
        symbol: Trading pair.
        timeframe: Candle timeframe (1m, 5m, 1h, etc).
        since: Start date string (YYYY-MM-DD).
        limit_per_request: Max candles per API call.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_file = _get_cache_path(exchange_id, symbol, timeframe)
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        logger.info(f"Loaded {len(df)} candles from cache")
        return df

    # Download from exchange
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()

    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    all_data = []

    logger.info(f"Downloading {symbol} {timeframe} from {since}...")

    consecutive_errors = 0
    max_retries = 10

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit_per_request)
            consecutive_errors = 0  # Reset on success
        except ccxt.BaseError as e:
            consecutive_errors += 1
            backoff = min(5 * (2 ** (consecutive_errors - 1)), 300)  # 5s, 10s, 20s... up to 5min
            logger.error(f"API error ({consecutive_errors}/{max_retries}): {e}, retrying in {backoff}s...")
            if consecutive_errors >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached, saving partial data ({len(all_data)} candles)")
                break
            time.sleep(backoff)
            continue

        if not candles:
            break

        all_data.extend(candles)
        since_ts = candles[-1][0] + 1  # Start from next candle

        logger.info(f"  Downloaded {len(all_data)} candles (last: {candles[-1][0]})")

        if len(candles) < limit_per_request:
            break  # No more data

        time.sleep(exchange.rateLimit / 1000)

    if not all_data:
        logger.warning(f"No data downloaded for {symbol}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Save cache
    df.to_csv(cache_file, index=False)
    logger.info(f"Saved {len(df)} candles to {cache_file}")

    return df


def _get_cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    """Get the cache file path for a symbol/timeframe."""
    safe_symbol = symbol.replace("/", "_")
    return DATA_DIR / f"{exchange_id}_{safe_symbol}_{timeframe}.csv"


def load_cached_data(exchange_id: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load cached OHLCV data if available.

    Returns:
        DataFrame or None if no cache exists.
    """
    cache_file = _get_cache_path(exchange_id, symbol, timeframe)
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        return df
    return None


def list_cached_data() -> list[dict]:
    """List all cached data files.

    Returns:
        List of dicts with exchange, symbol, timeframe, rows, file_size.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in DATA_DIR.glob("*.csv"):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            df = pd.read_csv(f, nrows=1)
            row_count = sum(1 for _ in open(f)) - 1  # subtract header
            files.append({
                "exchange": parts[0],
                "symbol": f"{parts[1]}/{parts[2]}",
                "timeframe": parts[3] if len(parts) > 3 else "unknown",
                "rows": row_count,
                "file_size_mb": f.stat().st_size / (1024 * 1024),
            })
    return files
