import csv
import os
import time
from pathlib import Path

import ccxt
import pandas as pd

from src.core.logger import setup_logger

logger = setup_logger("backtest.data_fetcher")

DATA_DIR = Path("data/historical")

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def download_ohlcv(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since: str = "2025-01-01",
    limit_per_request: int = 1000,
) -> int:
    """Download historical OHLCV data from an exchange.

    Streams chunks directly to CSV to keep memory low (works on 1GB VPS).
    Supports resuming partial downloads by reading the last timestamp from
    an existing cache file.

    Args:
        exchange_id: Exchange name.
        symbol: Trading pair.
        timeframe: Candle timeframe (1m, 5m, 1h, etc).
        since: Start date string (YYYY-MM-DD).
        limit_per_request: Max candles per API call.

    Returns:
        Total number of candles saved to cache file.
        Use load_cached_data() or load_cached_data_chunked() to read.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _get_cache_path(exchange_id, symbol, timeframe)

    # Download from exchange
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()

    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    total_rows = 0

    # Resume support: if cache file exists, read last timestamp and continue
    if cache_file.exists():
        last_ts = _get_last_timestamp(cache_file)
        if last_ts is not None:
            since_ts = last_ts + 1
            total_rows = sum(1 for _ in open(cache_file)) - 1  # minus header
            logger.info(f"Resuming download from {total_rows} existing candles")

    # Open file in append mode (or write mode if new)
    is_new_file = not cache_file.exists() or total_rows == 0
    f = open(cache_file, "a" if not is_new_file else "w", newline="")
    writer = csv.writer(f)

    if is_new_file:
        writer.writerow(COLUMNS)

    logger.info(f"Downloading {symbol} {timeframe} from {since}...")

    consecutive_errors = 0
    max_retries = 10

    try:
        while True:
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit_per_request)
                consecutive_errors = 0
            except ccxt.BaseError as e:
                consecutive_errors += 1
                backoff = min(5 * (2 ** (consecutive_errors - 1)), 300)
                logger.error(f"API error ({consecutive_errors}/{max_retries}): {e}, retrying in {backoff}s...")
                if consecutive_errors >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached, saved {total_rows} candles so far")
                    break
                time.sleep(backoff)
                continue

            if not candles:
                break

            # Convert timestamps and write directly to CSV
            for c in candles:
                ts = pd.to_datetime(c[0], unit="ms", utc=True).isoformat()
                writer.writerow([ts, c[1], c[2], c[3], c[4], c[5]])

            total_rows += len(candles)
            since_ts = candles[-1][0] + 1

            # Flush every 10K candles so data isn't lost on crash
            if total_rows % 10_000 < limit_per_request:
                f.flush()

            logger.info(f"  Downloaded {total_rows} candles (last: {candles[-1][0]})")

            if len(candles) < limit_per_request:
                break

            time.sleep(exchange.rateLimit / 1000)
    finally:
        f.close()

    if total_rows == 0:
        logger.warning(f"No data downloaded for {symbol}")
        return 0

    logger.info(f"Saved {total_rows} candles to {cache_file}")
    return total_rows


def _get_last_timestamp(cache_file: Path) -> int | None:
    """Read the last timestamp (ms) from a CSV cache file for resume support."""
    try:
        # Read just the last line efficiently
        with open(cache_file, "rb") as f:
            f.seek(0, 2)  # End of file
            size = f.tell()
            if size < 100:
                return None
            # Read last 200 bytes to find the last complete line
            f.seek(max(0, size - 200))
            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
            if len(lines) < 2:
                return None
            last_line = lines[-1]
            ts_str = last_line.split(",")[0]
            ts = pd.to_datetime(ts_str, utc=True)
            return int(ts.timestamp() * 1000)
    except Exception:
        return None


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


def load_cached_data_chunked(
    exchange_id: str, symbol: str, timeframe: str,
    year: int | None = None,
) -> pd.DataFrame | None:
    """Load cached data, optionally filtered to a single year.

    Reads in chunks to keep memory low for large 1m datasets.
    """
    cache_file = _get_cache_path(exchange_id, symbol, timeframe)
    if not cache_file.exists():
        return None

    if year is None:
        return pd.read_csv(cache_file, parse_dates=["timestamp"])

    # Read in chunks and filter to the requested year
    chunks = []
    for chunk in pd.read_csv(cache_file, parse_dates=["timestamp"], chunksize=100_000):
        filtered = chunk[chunk["timestamp"].dt.year == year]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)


def count_cached_rows(exchange_id: str, symbol: str, timeframe: str) -> int:
    """Count rows in a cached data file without loading it into memory."""
    cache_file = _get_cache_path(exchange_id, symbol, timeframe)
    if not cache_file.exists():
        return 0
    return sum(1 for _ in open(cache_file)) - 1  # minus header


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
