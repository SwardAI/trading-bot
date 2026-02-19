"""Download historical OHLCV data for backtesting.

Usage:
    python scripts/download_historical.py --symbol BTC/USDT --timeframe 1h --since 2025-01-01
    python scripts/download_historical.py --list
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import download_ohlcv, list_cached_data


def main():
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--since", default="2025-01-01", help="Start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    parser.add_argument("--list", action="store_true", help="List cached data files")
    args = parser.parse_args()

    if args.list:
        cached = list_cached_data()
        if not cached:
            print("No cached data found.")
            return
        print(f"\nCached data files ({len(cached)}):")
        for f in cached:
            print(f"  {f['exchange']} {f['symbol']} {f['timeframe']}: {f['rows']} rows ({f['file_size_mb']:.1f} MB)")
        return

    print(f"Downloading {args.symbol} {args.timeframe} from {args.since} ({args.exchange})...")
    df = download_ohlcv(args.exchange, args.symbol, args.timeframe, args.since)
    print(f"\nDownloaded {len(df)} candles")
    if len(df) > 0:
        print(f"  First: {df.iloc[0]['timestamp']}")
        print(f"  Last:  {df.iloc[-1]['timestamp']}")


if __name__ == "__main__":
    main()
