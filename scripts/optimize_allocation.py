"""Optimize portfolio allocation across strategies using 4yr backtest data.

Tests many allocation splits (grid vs momentum) with regime filter enabled,
finds the split that maximizes risk-adjusted return (return / max_drawdown).

Also tests: momentum-only, grid-only, and various pair combinations.

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/optimize_allocation.py" bot
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data
from scripts.backtest_regime import (
    compute_weekly_regimes,
    run_grid_backtest_with_regime,
    run_momentum_backtest_with_regime,
    GRID_CONFIGS,
    MOMENTUM_CONFIG,
    MOMENTUM_SYMBOLS,
)

REPORT_FILE = "data/allocation_optimization_report.txt"
TOTAL_CAPITAL = 10000


def log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(REPORT_FILE, "a") as f:
        f.write(line + "\n")


def write_section(title: str):
    border = "=" * 70
    log("")
    log(border)
    log(f"  {title}")
    log(border)


def load_all_data():
    """Load and prepare all dataframes needed."""
    data = {}
    # BTC 1h for regime computation
    btc_1h = load_cached_data("binance", "BTC/USDT", "1h")
    if btc_1h is None:
        log("ERROR: Need BTC/USDT 1h data")
        return None, None
    btc_1h["timestamp"] = pd.to_datetime(btc_1h["timestamp"], utc=True)
    regimes_df = compute_weekly_regimes(btc_1h)

    # Grid data
    for cfg in GRID_CONFIGS:
        df = load_cached_data("binance", cfg["symbol"], "1h")
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.reset_index(drop=True)
            data[("grid", cfg["symbol"])] = df

    # Momentum data
    for symbol in MOMENTUM_SYMBOLS:
        df = load_cached_data("binance", symbol, "1h")
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.reset_index(drop=True)
            data[("momentum", symbol)] = df

    return data, regimes_df


def run_allocation(data, regimes_df, grid_pct, momentum_pct, label=""):
    """Run a specific allocation and return combined results."""
    grid_capital_total = TOTAL_CAPITAL * grid_pct / 100
    momentum_capital_total = TOTAL_CAPITAL * momentum_pct / 100

    total_final = 0
    total_initial = 0
    max_dd_all = 0
    details = []

    # Grid strategies
    num_grid_pairs = sum(1 for cfg in GRID_CONFIGS if ("grid", cfg["symbol"]) in data)
    if num_grid_pairs > 0 and grid_capital_total > 0:
        grid_per_pair = grid_capital_total / num_grid_pairs
        for cfg in GRID_CONFIGS:
            key = ("grid", cfg["symbol"])
            if key not in data:
                continue
            result = run_grid_backtest_with_regime(
                data[key], regimes_df,
                grid_spacing_pct=cfg["spacing"],
                order_size_usd=cfg["order_size"],
                num_grids=cfg["num_grids"],
                stop_loss_pct=cfg["stop_loss"],
                initial_capital=grid_per_pair,
                apply_regime=True,
            )
            total_final += result["final_capital"]
            total_initial += grid_per_pair
            max_dd_all = max(max_dd_all, result["max_drawdown_pct"])
            details.append(f"Grid {cfg['symbol']}: ${grid_per_pair:.0f}->${result['final_capital']:,.0f} ({result['total_return_pct']:+.1f}%)")

    # Momentum strategies
    mom_pairs_available = [s for s in MOMENTUM_SYMBOLS if ("momentum", s) in data]
    if mom_pairs_available and momentum_capital_total > 0:
        mom_per_pair = momentum_capital_total / len(mom_pairs_available)
        for symbol in mom_pairs_available:
            key = ("momentum", symbol)
            result = run_momentum_backtest_with_regime(
                data[key], regimes_df, MOMENTUM_CONFIG, mom_per_pair, apply_regime=True,
            )
            total_final += result["final_capital"]
            total_initial += mom_per_pair
            max_dd_all = max(max_dd_all, result["max_drawdown_pct"])
            details.append(f"Mom  {symbol}: ${mom_per_pair:.0f}->${result['final_capital']:,.0f} ({result['total_return_pct']:+.1f}%)")

    total_return = (total_final - total_initial) / total_initial * 100 if total_initial > 0 else 0
    risk_adjusted = total_return / max_dd_all if max_dd_all > 0 else 0

    return {
        "label": label,
        "grid_pct": grid_pct,
        "momentum_pct": momentum_pct,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd_all,
        "risk_adjusted": risk_adjusted,
        "final_capital": total_final,
        "details": details,
    }


def main():
    start_time = time.time()

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(f"Allocation Optimization — {datetime.utcnow().isoformat()}\n")
        f.write(f"{'=' * 70}\n\n")

    log(f"Allocation optimization started")
    log(f"Testing different grid/momentum splits with regime filter ON")

    # Load all data once
    write_section("LOADING DATA")
    data, regimes_df = load_all_data()
    if data is None:
        return
    log(f"  Loaded {len(data)} datasets, {len(regimes_df)} weekly regimes")

    # ---------------------------------------------------------------
    # Test allocation sweeps
    # ---------------------------------------------------------------
    write_section("ALLOCATION SWEEP (grid% / momentum%)")

    allocations = [
        (0, 100, "100% Momentum"),
        (10, 90, "10% Grid / 90% Momentum"),
        (20, 80, "20% Grid / 80% Momentum"),
        (30, 70, "30% Grid / 70% Momentum"),
        (40, 60, "40% Grid / 60% Momentum"),
        (50, 50, "50/50 Split"),
        (60, 40, "60% Grid / 40% Momentum (CURRENT)"),
        (70, 30, "70% Grid / 30% Momentum"),
        (80, 20, "80% Grid / 20% Momentum"),
        (100, 0, "100% Grid"),
    ]

    results = []
    for grid_pct, mom_pct, label in allocations:
        log(f"\n  Testing: {label}...")
        result = run_allocation(data, regimes_df, grid_pct, mom_pct, label)
        results.append(result)
        log(f"    Return: {result['total_return_pct']:+.1f}%, Max DD: {result['max_drawdown_pct']:.1f}%, "
            f"Risk-adj: {result['risk_adjusted']:.2f}, Final: ${result['final_capital']:,.0f}")
        for d in result["details"]:
            log(f"      {d}")

    # Summary table
    write_section("ALLOCATION COMPARISON TABLE")
    log(f"  {'Allocation':<35} {'Return':>8} {'MaxDD':>7} {'R/DD':>6} {'Final':>10}")
    log(f"  {'-'*70}")

    # Sort by risk-adjusted return
    results.sort(key=lambda r: r["risk_adjusted"], reverse=True)
    for r in results:
        marker = " <-- BEST" if r == results[0] else ""
        current = " (CURRENT)" if "CURRENT" in r["label"] else ""
        log(f"  {r['label']:<35} {r['total_return_pct']:>+7.1f}% {r['max_drawdown_pct']:>6.1f}% "
            f"{r['risk_adjusted']:>5.2f} ${r['final_capital']:>9,.0f}{marker}{current}")

    best = results[0]
    write_section("RECOMMENDATION")
    log(f"  Best risk-adjusted allocation: {best['label']}")
    log(f"    Return: {best['total_return_pct']:+.1f}%")
    log(f"    Max Drawdown: {best['max_drawdown_pct']:.1f}%")
    log(f"    Risk-adjusted ratio: {best['risk_adjusted']:.2f}")
    log(f"    Final capital: ${best['final_capital']:,.0f} from ${TOTAL_CAPITAL:,.0f}")
    log(f"\n  Details:")
    for d in best["details"]:
        log(f"    {d}")

    # ---------------------------------------------------------------
    # Test momentum-only with more pairs
    # ---------------------------------------------------------------
    write_section("BONUS: MOMENTUM-ONLY WITH INDIVIDUAL PAIR PERFORMANCE")

    mom_per_pair = TOTAL_CAPITAL / len(MOMENTUM_SYMBOLS)
    for symbol in MOMENTUM_SYMBOLS:
        key = ("momentum", symbol)
        if key not in data:
            continue
        result = run_momentum_backtest_with_regime(
            data[key], regimes_df, MOMENTUM_CONFIG, mom_per_pair, apply_regime=True,
        )
        log(f"  {symbol}: ${mom_per_pair:.0f} -> ${result['final_capital']:,.0f} "
            f"({result['total_return_pct']:+.1f}%, DD={result['max_drawdown_pct']:.1f}%, "
            f"R/DD={result['total_return_pct']/result['max_drawdown_pct']:.2f})")

    # Also test without regime filter for momentum-only
    log(f"\n  Momentum-only WITHOUT regime filter:")
    total_no_regime = 0
    for symbol in MOMENTUM_SYMBOLS:
        key = ("momentum", symbol)
        if key not in data:
            continue
        result = run_momentum_backtest_with_regime(
            data[key], regimes_df, MOMENTUM_CONFIG, mom_per_pair, apply_regime=False,
        )
        total_no_regime += result["final_capital"]
        log(f"  {symbol}: ${mom_per_pair:.0f} -> ${result['final_capital']:,.0f} "
            f"({result['total_return_pct']:+.1f}%, DD={result['max_drawdown_pct']:.1f}%)")
    log(f"  TOTAL without regime: ${TOTAL_CAPITAL:,.0f} -> ${total_no_regime:,.0f} ({(total_no_regime - TOTAL_CAPITAL) / TOTAL_CAPITAL * 100:+.1f}%)")

    elapsed = time.time() - start_time
    log(f"\nOptimization complete! Time: {elapsed:.0f}s")
    log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
