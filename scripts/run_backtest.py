"""Run backtests on historical data.

Usage:
    python scripts/run_backtest.py --strategy grid --symbol BTC/USDT --timeframe 1h
    python scripts/run_backtest.py --strategy momentum --symbol BTC/USDT --timeframe 1h
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data, download_ohlcv
from src.backtest.engine import BacktestEngine
from src.backtest.analyzer import print_backtest_report, compare_with_buy_and_hold, get_trade_stats


def main():
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument("--strategy", required=True, choices=["grid", "momentum"], help="Strategy to backtest")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--exchange", default="binance", help="Exchange")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital (default: 10000)")
    parser.add_argument("--since", default="2025-01-01", help="Start date if downloading data")

    # Grid params
    parser.add_argument("--grid-spacing", type=float, default=0.5, help="Grid spacing %% (default: 0.5)")
    parser.add_argument("--order-size", type=float, default=50, help="Grid order size USD (default: 50)")
    parser.add_argument("--num-grids", type=int, default=20, help="Grid levels per side (default: 20)")

    args = parser.parse_args()

    # Load or download data
    df = load_cached_data(args.exchange, args.symbol, args.timeframe)
    if df is None:
        print(f"No cached data found. Downloading {args.symbol} {args.timeframe}...")
        df = download_ohlcv(args.exchange, args.symbol, args.timeframe, args.since)

    if len(df) < 50:
        print(f"Insufficient data: {len(df)} candles (need at least 50)")
        sys.exit(1)

    print(f"\nData: {len(df)} candles from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

    engine = BacktestEngine(initial_capital=args.capital)

    if args.strategy == "grid":
        result = engine.run_grid_backtest(
            df,
            grid_spacing_pct=args.grid_spacing,
            order_size_usd=args.order_size,
            num_grids=args.num_grids,
        )
    elif args.strategy == "momentum":
        result = engine.run_momentum_backtest(df)

    print_backtest_report(result)
    compare_with_buy_and_hold(result, df)

    stats = get_trade_stats(result)
    if stats.get("count", 0) > 0:
        print(f"\n  Detailed Trade Stats:")
        print(f"    Best trade:  ${stats['best_trade']:,.2f}")
        print(f"    Worst trade: ${stats['worst_trade']:,.2f}")
        print(f"    Max consecutive wins:   {stats['consecutive_wins_max']}")
        print(f"    Max consecutive losses: {stats['consecutive_losses_max']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
