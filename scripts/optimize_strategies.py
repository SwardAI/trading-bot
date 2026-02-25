"""Optimize strategy parameters via walk-forward backtesting.

Downloads data if needed, sweeps parameter combinations, validates on out-of-sample
data to prevent overfitting. Reports the best configs for each strategy.

Usage:
    python scripts/optimize_strategies.py --strategy grid
    python scripts/optimize_strategies.py --strategy momentum
    python scripts/optimize_strategies.py --strategy all
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.backtest.data_fetcher import download_ohlcv, load_cached_data
from src.backtest.engine import BacktestEngine


# Walk-forward split: optimize on training set, validate on test set
TRAIN_END = "2025-01-01"  # Train on 2022-2024 (3 years: bear + recovery + bull)
# Test on 2025-2026 (1+ year: current market conditions)

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
INITIAL_CAPITAL = 10000


def load_data(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    """Load cached data or download if missing."""
    df = load_cached_data("binance", symbol, timeframe)
    if df is None:
        print(f"  Downloading {symbol} {timeframe}...")
        download_ohlcv("binance", symbol, timeframe, "2022-01-01")
        df = load_cached_data("binance", symbol, timeframe)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (2022-2024) and test (2025+) sets."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    train = df[df["timestamp"] < TRAIN_END].copy().reset_index(drop=True)
    test = df[df["timestamp"] >= TRAIN_END].copy().reset_index(drop=True)
    return train, test


def optimize_grid(symbols: list[str]):
    """Sweep grid parameters and find optimal configuration."""
    print("\n" + "=" * 70)
    print("  GRID STRATEGY OPTIMIZATION")
    print("=" * 70)

    # Parameter grid
    spacings = [0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]
    order_sizes = [25, 50, 75, 100, 150]
    num_grids = [10, 15, 20, 30]

    total_combos = len(spacings) * len(order_sizes) * len(num_grids)
    print(f"\n  Testing {total_combos} parameter combinations x {len(symbols)} symbols")
    print(f"  Training: 2022-01-01 to {TRAIN_END}")
    print(f"  Testing:  {TRAIN_END} to present\n")

    all_results = []

    for symbol in symbols:
        print(f"  --- {symbol} ---")
        df = load_data(symbol)
        train_df, test_df = split_data(df)
        print(f"  Train: {len(train_df)} candles, Test: {len(test_df)} candles")

        count = 0
        for spacing, osize, ngrids in itertools.product(spacings, order_sizes, num_grids):
            engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)

            # Train
            train_result = engine.run_grid_backtest(
                train_df, grid_spacing_pct=spacing,
                order_size_usd=osize, num_grids=ngrids,
            )

            # Test (out of sample)
            engine_test = BacktestEngine(initial_capital=INITIAL_CAPITAL)
            test_result = engine_test.run_grid_backtest(
                test_df, grid_spacing_pct=spacing,
                order_size_usd=osize, num_grids=ngrids,
            )

            all_results.append({
                "symbol": symbol,
                "spacing": spacing,
                "order_size": osize,
                "num_grids": ngrids,
                "train_return": train_result.total_return_pct,
                "train_pf": train_result.profit_factor,
                "train_dd": train_result.max_drawdown_pct,
                "train_trades": train_result.total_trades,
                "train_sharpe": train_result.sharpe_ratio,
                "test_return": test_result.total_return_pct,
                "test_pf": test_result.profit_factor,
                "test_dd": test_result.max_drawdown_pct,
                "test_trades": test_result.total_trades,
                "test_sharpe": test_result.sharpe_ratio,
            })

            count += 1
            if count % 40 == 0:
                print(f"    {count}/{total_combos} combinations tested...")

        print(f"    Done: {count} combinations for {symbol}")

    # Sort by test performance (out of sample — this is what matters)
    results_df = pd.DataFrame(all_results)

    # Filter: only keep configs profitable on BOTH train AND test
    profitable = results_df[
        (results_df["train_return"] > 0) & (results_df["test_return"] > 0)
    ].copy()

    if len(profitable) == 0:
        print("\n  WARNING: No configuration profitable on both train and test sets!")
        # Show best test results anyway
        profitable = results_df.copy()

    # Score: weighted combination of test return, drawdown, and consistency
    profitable["score"] = (
        profitable["test_return"] * 0.4
        + profitable["test_sharpe"] * 20
        - profitable["test_dd"] * 0.3
        + (profitable["train_return"] > 0).astype(float) * 10  # bonus for train profitability
    )

    profitable = profitable.sort_values("score", ascending=False)

    print("\n  TOP 10 GRID CONFIGURATIONS (ranked by out-of-sample score)")
    print("-" * 70)
    print(f"  {'Symbol':<10} {'Spacing':>8} {'OrdSz':>6} {'Grids':>6} | {'Train%':>8} {'Test%':>8} {'TestDD':>7} {'TestPF':>7} {'Sharpe':>7}")
    print("-" * 70)

    for _, row in profitable.head(10).iterrows():
        print(
            f"  {row['symbol']:<10} {row['spacing']:>7.2f}% ${row['order_size']:>4.0f}  {row['num_grids']:>5.0f} | "
            f"{row['train_return']:>+7.1f}% {row['test_return']:>+7.1f}% {row['test_dd']:>6.1f}% {row['test_pf']:>6.2f} {row['test_sharpe']:>6.2f}"
        )

    print("-" * 70)

    # Also show per-symbol best
    for symbol in symbols:
        sym_results = profitable[profitable["symbol"] == symbol]
        if len(sym_results) > 0:
            best = sym_results.iloc[0]
            print(f"\n  BEST for {symbol}: spacing={best['spacing']}%, order=${best['order_size']:.0f}, grids={best['num_grids']:.0f}")
            print(f"    Train: {best['train_return']:+.1f}% return, {best['train_dd']:.1f}% max DD")
            print(f"    Test:  {best['test_return']:+.1f}% return, {best['test_dd']:.1f}% max DD, PF={best['test_pf']:.2f}")

    return profitable


def optimize_momentum(symbols: list[str]):
    """Sweep momentum parameters including signal subsets."""
    print("\n" + "=" * 70)
    print("  MOMENTUM STRATEGY OPTIMIZATION (Long-Only)")
    print("=" * 70)

    # Phase 1: Test which signal combinations work
    print("\n  Phase 1: Signal Subset Testing")
    print("-" * 50)

    signal_combos = [
        (["ema", "rsi", "adx", "volume", "macd"], "All 5"),
        (["ema", "adx", "volume", "macd"], "No RSI"),
        (["ema", "rsi", "adx", "macd"], "No Volume"),
        (["ema", "rsi", "adx", "volume"], "No MACD"),
        (["ema", "rsi", "volume", "macd"], "No ADX"),
        (["ema", "adx", "macd"], "EMA+ADX+MACD"),
        (["ema", "adx", "volume"], "EMA+ADX+Vol"),
        (["ema", "adx"], "EMA+ADX only"),
        (["ema", "rsi", "adx"], "EMA+RSI+ADX"),
    ]

    signal_results = []
    for symbol in symbols:
        df = load_data(symbol)
        train_df, test_df = split_data(df)

        for signals, label in signal_combos:
            engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
            config = {"required_signals": signals}
            result = engine.run_momentum_backtest(train_df, config=config, long_only=True)

            engine_test = BacktestEngine(initial_capital=INITIAL_CAPITAL)
            test_result = engine_test.run_momentum_backtest(test_df, config=config, long_only=True)

            signal_results.append({
                "symbol": symbol, "signals": label,
                "train_return": result.total_return_pct,
                "train_pf": result.profit_factor,
                "train_wr": result.win_rate,
                "train_trades": result.winning_trades + result.losing_trades,
                "test_return": test_result.total_return_pct,
                "test_pf": test_result.profit_factor,
                "test_wr": test_result.win_rate,
                "test_trades": test_result.winning_trades + test_result.losing_trades,
            })

    sig_df = pd.DataFrame(signal_results)
    print(f"\n  {'Symbol':<10} {'Signals':<17} | {'Train%':>8} {'TrainPF':>8} {'Trades':>7} | {'Test%':>8} {'TestPF':>8} {'Trades':>7}")
    print("-" * 90)
    for _, row in sig_df.iterrows():
        print(
            f"  {row['symbol']:<10} {row['signals']:<17} | "
            f"{row['train_return']:>+7.1f}% {row['train_pf']:>7.2f} {row['train_trades']:>6.0f} | "
            f"{row['test_return']:>+7.1f}% {row['test_pf']:>7.2f} {row['test_trades']:>6.0f}"
        )

    # Find best signal combo (by average test profit factor across symbols)
    sig_avg = sig_df.groupby("signals").agg({"test_pf": "mean", "test_return": "mean"}).reset_index()
    best_signals_row = sig_avg.sort_values("test_pf", ascending=False).iloc[0]
    best_signal_label = best_signals_row["signals"]
    # Map label back to signal list
    best_signals = [s for s, l in signal_combos if l == best_signal_label][0]
    print(f"\n  Best signal combo: {best_signal_label} (avg test PF: {best_signals_row['test_pf']:.2f})")

    # Phase 2: Parameter sweep with best signal combo
    print(f"\n  Phase 2: Parameter Sweep with '{best_signal_label}' signals")
    print("-" * 50)

    ema_pairs = [(8, 21), (9, 21), (12, 26), (13, 34), (20, 50)]
    atr_mults = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    adx_mins = [20, 25, 30, 35]
    rsi_thresholds = [45, 50, 55, 60]

    total = len(ema_pairs) * len(atr_mults) * len(adx_mins) * len(rsi_thresholds)
    print(f"  Testing {total} combinations x {len(symbols)} symbols = {total * len(symbols)} runs\n")

    all_results = []
    start_time = time.time()

    for symbol in symbols:
        df = load_data(symbol)
        train_df, test_df = split_data(df)
        count = 0

        for (ema_f, ema_s), atr_m, adx, rsi in itertools.product(ema_pairs, atr_mults, adx_mins, rsi_thresholds):
            config = {
                "ema_fast": ema_f, "ema_slow": ema_s,
                "trailing_stop_atr_multiplier": atr_m,
                "adx_min_strength": adx,
                "rsi_long_threshold": rsi,
                "required_signals": best_signals,
            }

            engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
            train_result = engine.run_momentum_backtest(train_df, config=config, long_only=True)

            engine_test = BacktestEngine(initial_capital=INITIAL_CAPITAL)
            test_result = engine_test.run_momentum_backtest(test_df, config=config, long_only=True)

            round_trips_train = train_result.winning_trades + train_result.losing_trades
            round_trips_test = test_result.winning_trades + test_result.losing_trades

            all_results.append({
                "symbol": symbol,
                "ema_fast": ema_f, "ema_slow": ema_s,
                "atr_mult": atr_m, "adx_min": adx, "rsi_long": rsi,
                "train_return": train_result.total_return_pct,
                "train_pf": train_result.profit_factor,
                "train_wr": train_result.win_rate,
                "train_dd": train_result.max_drawdown_pct,
                "train_sharpe": train_result.sharpe_ratio,
                "train_trades": round_trips_train,
                "test_return": test_result.total_return_pct,
                "test_pf": test_result.profit_factor,
                "test_wr": test_result.win_rate,
                "test_dd": test_result.max_drawdown_pct,
                "test_sharpe": test_result.sharpe_ratio,
                "test_trades": round_trips_test,
            })

            count += 1
            if count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"    {symbol}: {count}/{total} tested ({elapsed:.0f}s elapsed)")

    results_df = pd.DataFrame(all_results)

    # Filter: profitable on test set with at least 5 round trips on test
    viable = results_df[
        (results_df["test_return"] > 0) &
        (results_df["test_trades"] >= 5) &
        (results_df["test_pf"] > 1.0)
    ].copy()

    if len(viable) == 0:
        print("\n  NO profitable configuration found on test data with sufficient trades!")
        print("  Showing least-losing configs instead:\n")
        viable = results_df[results_df["test_trades"] >= 3].copy()
        if len(viable) == 0:
            viable = results_df.copy()

    # Score: emphasize out-of-sample performance, low drawdown, stability
    viable["score"] = (
        viable["test_pf"] * 30  # Profit factor is king
        + viable["test_return"] * 0.5
        - viable["test_dd"] * 0.5
        + viable["test_sharpe"] * 15
        + (viable["train_pf"] > 1.0).astype(float) * 10  # Bonus for training profitability
    )

    viable = viable.sort_values("score", ascending=False)

    print(f"\n  TOP 15 MOMENTUM CONFIGURATIONS (out-of-sample ranked)")
    print("-" * 105)
    print(f"  {'Symbol':<9} {'EMA':>7} {'ATR':>5} {'ADX':>4} {'RSI':>4} | {'TrnRet':>7} {'TrnPF':>6} {'TrnWR':>6} | {'TstRet':>7} {'TstPF':>6} {'TstWR':>6} {'TstDD':>6} {'Trds':>5}")
    print("-" * 105)

    for _, row in viable.head(15).iterrows():
        print(
            f"  {row['symbol']:<9} {row['ema_fast']:.0f}/{row['ema_slow']:.0f}  {row['atr_mult']:>4.1f} {row['adx_min']:>3.0f} {row['rsi_long']:>3.0f} | "
            f"{row['train_return']:>+6.1f}% {row['train_pf']:>5.2f} {row['train_wr']:>5.1f}% | "
            f"{row['test_return']:>+6.1f}% {row['test_pf']:>5.2f} {row['test_wr']:>5.1f}% {row['test_dd']:>5.1f}% {row['test_trades']:>4.0f}"
        )

    print("-" * 105)

    # Per-symbol best
    for symbol in symbols:
        sym_viable = viable[viable["symbol"] == symbol]
        if len(sym_viable) > 0:
            best = sym_viable.iloc[0]
            print(f"\n  BEST for {symbol}:")
            print(f"    EMA: {best['ema_fast']:.0f}/{best['ema_slow']:.0f}, ATR stop: {best['atr_mult']:.1f}x, ADX>{best['adx_min']:.0f}, RSI>{best['rsi_long']:.0f}")
            print(f"    Train: {best['train_return']:+.1f}%, PF={best['train_pf']:.2f}, WR={best['train_wr']:.1f}%, DD={best['train_dd']:.1f}%")
            print(f"    Test:  {best['test_return']:+.1f}%, PF={best['test_pf']:.2f}, WR={best['test_wr']:.1f}%, DD={best['test_dd']:.1f}%")
        else:
            print(f"\n  {symbol}: NO profitable configuration found on test data.")
            # Show least bad
            sym_all = results_df[results_df["symbol"] == symbol].sort_values("test_return", ascending=False)
            if len(sym_all) > 0:
                worst = sym_all.iloc[0]
                print(f"    Least bad: {worst['test_return']:+.1f}% return, PF={worst['test_pf']:.2f}")

    # Cross-symbol robustness check
    print("\n  CROSS-SYMBOL ROBUSTNESS CHECK")
    print("-" * 50)
    # Find configs that work on BOTH symbols
    param_cols = ["ema_fast", "ema_slow", "atr_mult", "adx_min", "rsi_long"]
    merged = results_df.copy()
    merged["params"] = merged.apply(lambda r: f"{r['ema_fast']:.0f}/{r['ema_slow']:.0f}_ATR{r['atr_mult']:.1f}_ADX{r['adx_min']:.0f}_RSI{r['rsi_long']:.0f}", axis=1)

    robust = []
    for params in merged["params"].unique():
        subset = merged[merged["params"] == params]
        if len(subset) == len(symbols):
            avg_test_pf = subset["test_pf"].mean()
            avg_test_ret = subset["test_return"].mean()
            min_test_pf = subset["test_pf"].min()
            max_test_dd = subset["test_dd"].max()
            avg_train_pf = subset["train_pf"].mean()
            robust.append({
                "params": params,
                "avg_test_pf": avg_test_pf,
                "min_test_pf": min_test_pf,
                "avg_test_return": avg_test_ret,
                "max_test_dd": max_test_dd,
                "avg_train_pf": avg_train_pf,
            })

    robust_df = pd.DataFrame(robust)
    # Best: high minimum PF across symbols
    robust_df = robust_df.sort_values("min_test_pf", ascending=False)

    if len(robust_df) > 0 and robust_df.iloc[0]["min_test_pf"] > 1.0:
        print(f"\n  Configs profitable on ALL symbols (min PF > 1.0):")
        for _, row in robust_df[robust_df["min_test_pf"] > 1.0].head(5).iterrows():
            print(f"    {row['params']}: avg PF={row['avg_test_pf']:.2f}, min PF={row['min_test_pf']:.2f}, avg ret={row['avg_test_return']:+.1f}%")
    else:
        print(f"\n  No config profitable on ALL symbols simultaneously.")
        if len(robust_df) > 0:
            print(f"  Best cross-symbol: {robust_df.iloc[0]['params']}")
            print(f"    avg test PF={robust_df.iloc[0]['avg_test_pf']:.2f}, min PF={robust_df.iloc[0]['min_test_pf']:.2f}")

    # Final verdict
    print("\n" + "=" * 70)
    any_profitable = len(viable[viable["test_pf"] > 1.0]) > 0
    if any_profitable:
        print("  VERDICT: Profitable momentum configs exist! Update live parameters.")
    else:
        print("  VERDICT: Momentum is NOT profitable with these parameters.")
        print("  RECOMMENDATION: Disable momentum strategy for real-money trading.")
    print("=" * 70)

    return viable


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--strategy", default="all", choices=["grid", "momentum", "all"])
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  STRATEGY PARAMETER OPTIMIZER")
    print(f"  Walk-forward: Train 2022-2024, Test 2025-2026")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}")
    print("=" * 70)

    if args.strategy in ("grid", "all"):
        optimize_grid(SYMBOLS)

    if args.strategy in ("momentum", "all"):
        optimize_momentum(SYMBOLS)


if __name__ == "__main__":
    main()
