"""Strategy research: test multiple noise-resistant strategies across 10 pairs.

Tests Donchian breakout with various configurations, compares to baseline
EMA momentum, and identifies the best strategy for robustness testing.

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research.py" bot
"""

import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import download_ohlcv, load_cached_data
from src.backtest.engine import BacktestEngine

REPORT_FILE = "data/strategy_research_report.txt"
INITIAL_CAPITAL = 10000

# All symbols with cached data on droplet
ALL_SYMBOLS = [
    "ETH/USDT", "AVAX/USDT", "LINK/USDT",
    "DOT/USDT", "DOGE/USDT", "ADA/USDT", "SOL/USDT",
    "UNI/USDT", "ATOM/USDT", "NEAR/USDT",
]


@dataclass
class StrategyResult:
    name: str
    config: dict
    symbol: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int
    win_rate: float
    return_over_dd: float  # Risk-adjusted return


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


def load_symbol_data(symbol: str) -> pd.DataFrame | None:
    """Load cached 1h data, download if missing."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        log(f"  Downloading {symbol} 1h data...")
        try:
            download_ohlcv("binance", symbol, "1h", "2022-01-01")
            df = load_cached_data("binance", symbol, "1h")
        except Exception as e:
            log(f"  ERROR downloading {symbol}: {e}")
            return None
    if df is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.reset_index(drop=True)
    return df


def run_breakout(df: pd.DataFrame, config: dict):
    """Run breakout backtest."""
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine.run_breakout_backtest(df, config=config, long_only=True)


def run_momentum(df: pd.DataFrame, config: dict):
    """Run EMA momentum backtest (baseline comparison)."""
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine.run_momentum_backtest(df, config=config, long_only=True)


def result_to_summary(name: str, config: dict, symbol: str, result) -> StrategyResult:
    """Convert BacktestResult to StrategyResult summary."""
    r_dd = (result.total_return_pct / result.max_drawdown_pct
            if result.max_drawdown_pct > 0 else 0)
    return StrategyResult(
        name=name, config=config, symbol=symbol,
        total_return_pct=result.total_return_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        profit_factor=result.profit_factor,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        return_over_dd=r_dd,
    )


# ---------------------------------------------------------------
# STRATEGY CONFIGURATIONS TO TEST
# ---------------------------------------------------------------

# Donchian breakout variants
BREAKOUT_CONFIGS = {
    # Classic Turtle: 20-period entry, 10-period exit
    "Breakout_20_10_3.0": {
        "breakout_period": 20, "exit_period": 10,
        "atr_stop_mult": 3.0, "adx_min": 20,
        "volume_mult": 1.0, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 5,
    },
    # Wider breakout with volume filter
    "Breakout_30_15_3.0_vol": {
        "breakout_period": 30, "exit_period": 15,
        "atr_stop_mult": 3.0, "adx_min": 20,
        "volume_mult": 1.3, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 5,
    },
    # Fast breakout (more trades, catches shorter trends)
    "Breakout_14_7_2.5": {
        "breakout_period": 14, "exit_period": 7,
        "atr_stop_mult": 2.5, "adx_min": 20,
        "volume_mult": 1.0, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 3,
    },
    # Slow breakout (fewer but bigger trends)
    "Breakout_55_20_3.5": {
        "breakout_period": 55, "exit_period": 20,
        "atr_stop_mult": 3.5, "adx_min": 20,
        "volume_mult": 1.0, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 10,
    },
    # No ADX filter (let breakout speak for itself)
    "Breakout_20_10_3.0_noADX": {
        "breakout_period": 20, "exit_period": 10,
        "atr_stop_mult": 3.0, "adx_min": 0,
        "volume_mult": 1.0, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 5,
    },
    # Wider stops (more room to breathe)
    "Breakout_20_10_4.0": {
        "breakout_period": 20, "exit_period": 10,
        "atr_stop_mult": 4.0, "adx_min": 20,
        "volume_mult": 1.0, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 5,
    },
    # Tight stops + fast entry (scalper-like)
    "Breakout_10_5_2.0": {
        "breakout_period": 10, "exit_period": 5,
        "atr_stop_mult": 2.0, "adx_min": 25,
        "volume_mult": 1.3, "risk_per_trade_pct": 1.0,
        "cooldown_bars": 3,
    },
    # Higher risk per trade (for compounding power)
    "Breakout_20_10_3.0_2pct": {
        "breakout_period": 20, "exit_period": 10,
        "atr_stop_mult": 3.0, "adx_min": 20,
        "volume_mult": 1.0, "risk_per_trade_pct": 2.0,
        "cooldown_bars": 5,
    },
}

# Baseline EMA momentum for comparison
MOMENTUM_BASELINE = {
    "ema_fast": 20, "ema_slow": 50,
    "trailing_stop_atr_multiplier": 3.5,
    "adx_min_strength": 20, "rsi_long_threshold": 50,
    "required_signals": ["ema", "adx"],
    "risk_per_trade_pct": 1.0,
}


def phase1_individual_pairs():
    """Phase 1: Test all strategies on all pairs individually."""
    write_section("PHASE 1: Individual Pair Performance")
    log(f"Testing {len(BREAKOUT_CONFIGS)} breakout configs + 1 momentum baseline")
    log(f"Across {len(ALL_SYMBOLS)} pairs = {(len(BREAKOUT_CONFIGS) + 1) * len(ALL_SYMBOLS)} backtests")

    all_results = []

    for symbol in ALL_SYMBOLS:
        log(f"\n--- {symbol} ---")
        df = load_symbol_data(symbol)
        if df is None or len(df) < 200:
            log(f"  Skipping {symbol}: insufficient data")
            continue

        log(f"  Data: {len(df)} candles ({df.iloc[0]['timestamp'].date()} to {df.iloc[-1]['timestamp'].date()})")

        # Test breakout configs
        for name, config in BREAKOUT_CONFIGS.items():
            try:
                result = run_breakout(df, config)
                sr = result_to_summary(name, config, symbol, result)
                all_results.append(sr)
                log(f"  {name:35s} | Return: {sr.total_return_pct:+7.1f}% | DD: {sr.max_drawdown_pct:5.1f}% | "
                    f"Sharpe: {sr.sharpe_ratio:5.2f} | PF: {sr.profit_factor:5.2f} | Trades: {sr.total_trades:3d}")
            except Exception as e:
                log(f"  {name}: ERROR - {e}")

        # Test momentum baseline
        try:
            result = run_momentum(df, MOMENTUM_BASELINE)
            sr = result_to_summary("EMA_Momentum_baseline", MOMENTUM_BASELINE, symbol, result)
            all_results.append(sr)
            log(f"  {'EMA_Momentum_baseline':35s} | Return: {sr.total_return_pct:+7.1f}% | DD: {sr.max_drawdown_pct:5.1f}% | "
                f"Sharpe: {sr.sharpe_ratio:5.2f} | PF: {sr.profit_factor:5.2f} | Trades: {sr.total_trades:3d}")
        except Exception as e:
            log(f"  EMA_Momentum_baseline: ERROR - {e}")

        gc.collect()

    return all_results


def phase2_aggregate_analysis(all_results: list[StrategyResult]):
    """Phase 2: Aggregate results and find best strategies."""
    write_section("PHASE 2: Aggregate Analysis")

    if not all_results:
        log("No results to analyze!")
        return None

    # Group by strategy name
    from collections import defaultdict
    strategy_stats = defaultdict(list)
    for r in all_results:
        strategy_stats[r.name].append(r)

    # Compute aggregate metrics per strategy
    log(f"\n{'Strategy':35s} | {'Avg Ret':>8s} | {'Med Ret':>8s} | {'Avg DD':>7s} | "
        f"{'Avg Sharpe':>10s} | {'Win%':>5s} | {'Avg PF':>7s} | {'R/DD':>6s} | {'#Pairs':>6s}")
    log("-" * 120)

    strategy_rankings = []

    for name, results in sorted(strategy_stats.items()):
        returns = [r.total_return_pct for r in results]
        drawdowns = [r.max_drawdown_pct for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        pfs = [r.profit_factor for r in results]
        profitable = sum(1 for r in returns if r > 0)

        avg_ret = np.mean(returns)
        med_ret = np.median(returns)
        avg_dd = np.mean(drawdowns)
        avg_sharpe = np.mean(sharpes)
        win_pct = profitable / len(results) * 100
        avg_pf = np.mean(pfs)
        avg_r_dd = avg_ret / avg_dd if avg_dd > 0 else 0

        log(f"{name:35s} | {avg_ret:+7.1f}% | {med_ret:+7.1f}% | {avg_dd:6.1f}% | "
            f"{avg_sharpe:9.2f} | {win_pct:4.0f}% | {avg_pf:6.2f} | {avg_r_dd:5.2f} | "
            f"{profitable}/{len(results)}")

        strategy_rankings.append({
            "name": name,
            "avg_return": avg_ret,
            "median_return": med_ret,
            "avg_drawdown": avg_dd,
            "avg_sharpe": avg_sharpe,
            "win_pct": win_pct,
            "avg_pf": avg_pf,
            "risk_adjusted": avg_r_dd,
            "num_profitable": profitable,
            "num_pairs": len(results),
            "results": results,
        })

    # Rank by composite score: 40% risk-adjusted return, 30% win%, 30% avg sharpe
    log("\n--- COMPOSITE RANKING (40% R/DD + 30% win% + 30% Sharpe) ---")
    for s in strategy_rankings:
        # Normalize each metric to 0-1 range
        max_r_dd = max(x["risk_adjusted"] for x in strategy_rankings) or 1
        max_win = max(x["win_pct"] for x in strategy_rankings) or 1
        max_sharpe = max(x["avg_sharpe"] for x in strategy_rankings) or 1

        s["composite"] = (
            0.4 * (s["risk_adjusted"] / max_r_dd if max_r_dd > 0 else 0) +
            0.3 * (s["win_pct"] / max_win if max_win > 0 else 0) +
            0.3 * (s["avg_sharpe"] / max_sharpe if max_sharpe > 0 else 0)
        )

    strategy_rankings.sort(key=lambda x: x["composite"], reverse=True)

    for i, s in enumerate(strategy_rankings):
        marker = " <-- BEST" if i == 0 else ""
        log(f"  #{i+1}: {s['name']:35s} | Composite: {s['composite']:.3f} | "
            f"R/DD: {s['risk_adjusted']:.2f} | Win: {s['win_pct']:.0f}% | "
            f"Sharpe: {s['avg_sharpe']:.2f}{marker}")

    return strategy_rankings


def phase3_best_per_pair(all_results: list[StrategyResult], top_strategy_name: str):
    """Phase 3: Show which pairs work best with the top strategy."""
    write_section(f"PHASE 3: Per-Pair Breakdown for '{top_strategy_name}'")

    pair_results = [r for r in all_results if r.name == top_strategy_name]
    pair_results.sort(key=lambda r: r.total_return_pct, reverse=True)

    log(f"\n{'Pair':12s} | {'Return':>8s} | {'Max DD':>7s} | {'Sharpe':>7s} | "
        f"{'PF':>6s} | {'Trades':>6s} | {'Win%':>5s} | {'R/DD':>6s}")
    log("-" * 80)

    for r in pair_results:
        log(f"{r.symbol:12s} | {r.total_return_pct:+7.1f}% | {r.max_drawdown_pct:6.1f}% | "
            f"{r.sharpe_ratio:6.2f} | {r.profit_factor:5.2f} | {r.total_trades:5d} | "
            f"{r.win_rate:4.1f}% | {r.return_over_dd:5.2f}")

    profitable = [r for r in pair_results if r.total_return_pct > 0]
    log(f"\nProfitable pairs: {len(profitable)}/{len(pair_results)}")

    if profitable:
        avg_ret = np.mean([r.total_return_pct for r in profitable])
        log(f"Average return (profitable only): {avg_ret:+.1f}%")

    # Portfolio simulation: equal allocation across all profitable pairs
    if len(pair_results) >= 3:
        portfolio_return = np.mean([r.total_return_pct for r in pair_results])
        portfolio_max_dd = max(r.max_drawdown_pct for r in pair_results)
        log(f"\nEqual-weight portfolio (all {len(pair_results)} pairs):")
        log(f"  Average return: {portfolio_return:+.1f}%")
        log(f"  Worst single-pair DD: {portfolio_max_dd:.1f}%")
        # Diversified DD estimate (correlation ~ 0.5 for crypto)
        avg_dd = np.mean([r.max_drawdown_pct for r in pair_results])
        div_dd = avg_dd * (1 / len(pair_results) ** 0.3)  # rough diversification
        log(f"  Est. diversified DD: {div_dd:.1f}%")


def phase4_noise_quick_test(top_strategy_name: str, top_config: dict):
    """Phase 4: Quick noise injection test on the best strategy (3 pairs, 20 runs)."""
    write_section("PHASE 4: Quick Noise Resilience Check")
    log("Adding 0.5% Gaussian noise to prices, 20 runs per pair")

    test_symbols = ["ETH/USDT", "AVAX/USDT", "LINK/USDT"]
    noise_std = 0.005  # 0.5%
    n_runs = 20

    for symbol in test_symbols:
        df = load_symbol_data(symbol)
        if df is None or len(df) < 200:
            continue

        # Baseline (no noise)
        baseline = run_breakout(df, top_config)
        baseline_ret = baseline.total_return_pct

        # Noisy runs
        noisy_returns = []
        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            df_noisy = df.copy()
            noise = rng.normal(1.0, noise_std, len(df_noisy))
            df_noisy["close"] = df_noisy["close"] * noise
            df_noisy["open"] = df_noisy["open"] * rng.normal(1.0, noise_std, len(df_noisy))
            df_noisy["high"] = df_noisy[["open", "close", "high"]].max(axis=1) * rng.normal(1.0, noise_std * 0.5, len(df_noisy))
            df_noisy["low"] = df_noisy[["open", "close", "low"]].min(axis=1) * rng.normal(1.0, noise_std * 0.5, len(df_noisy))
            # Fix OHLC consistency
            df_noisy["high"] = df_noisy[["open", "close", "high"]].max(axis=1)
            df_noisy["low"] = df_noisy[["open", "close", "low"]].min(axis=1)

            try:
                result = run_breakout(df_noisy, top_config)
                noisy_returns.append(result.total_return_pct)
            except Exception:
                noisy_returns.append(0.0)

        profitable = sum(1 for r in noisy_returns if r > 0)
        avg_noisy = np.mean(noisy_returns)
        std_noisy = np.std(noisy_returns)

        log(f"  {symbol}: Baseline {baseline_ret:+.1f}% | Noisy avg {avg_noisy:+.1f}% "
            f"(std {std_noisy:.1f}%) | Profitable {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")

        gc.collect()


def phase5_portfolio_simulation(top_config: dict):
    """Phase 5: Simulate a multi-pair portfolio with the best strategy."""
    write_section("PHASE 5: Multi-Pair Portfolio Simulation")
    log("Running breakout strategy on all pairs, simulating portfolio-level equity")

    # Load all data
    pair_equities = {}
    pair_trades = {}
    min_len = float("inf")

    for symbol in ALL_SYMBOLS:
        df = load_symbol_data(symbol)
        if df is None or len(df) < 200:
            continue

        result = run_breakout(df, top_config)
        if result.equity_curve:
            pair_equities[symbol] = result.equity_curve
            pair_trades[symbol] = result.total_trades
            min_len = min(min_len, len(result.equity_curve))

    if not pair_equities:
        log("No valid results for portfolio simulation")
        return

    n_pairs = len(pair_equities)
    log(f"Portfolio: {n_pairs} pairs, equal allocation (${INITIAL_CAPITAL/n_pairs:.0f} each)")

    # Normalize equity curves to same length and compute portfolio
    portfolio_equity = []
    per_pair_alloc = INITIAL_CAPITAL / n_pairs

    for i in range(int(min_len)):
        total = 0
        for symbol, eq in pair_equities.items():
            # Scale from individual backtest capital to allocation
            if i < len(eq):
                total += eq[i] / INITIAL_CAPITAL * per_pair_alloc
        portfolio_equity.append(total)

    if not portfolio_equity:
        log("Empty portfolio equity curve")
        return

    # Compute portfolio metrics
    initial = portfolio_equity[0]
    final = portfolio_equity[-1]
    total_return = (final - initial) / initial * 100

    # Max drawdown
    peak = portfolio_equity[0]
    max_dd = 0
    for eq in portfolio_equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    # Annualized return (4 years of hourly data)
    years = min_len / (365.25 * 24)
    annual_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe
    returns = [(portfolio_equity[i] - portfolio_equity[i-1]) / portfolio_equity[i-1]
               for i in range(1, len(portfolio_equity)) if portfolio_equity[i-1] > 0]
    if returns:
        sharpe = np.mean(returns) / np.std(returns) * (365 ** 0.5) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    total_trades = sum(pair_trades.values())

    log(f"\n  Portfolio Performance ({years:.1f} years):")
    log(f"    Total return:     {total_return:+.1f}%")
    log(f"    Annualized return:{annual_return:+.1f}%")
    log(f"    Max drawdown:     {max_dd:.1f}%")
    log(f"    Sharpe ratio:     {sharpe:.2f}")
    log(f"    Return / DD:      {total_return / max_dd:.2f}" if max_dd > 0 else "    Return / DD: inf")
    log(f"    Total trades:     {total_trades}")
    log(f"    Trades/pair/year: {total_trades / n_pairs / years:.1f}")

    # Is 30%/year achievable?
    log(f"\n  TARGET CHECK: 30%/year")
    if annual_return >= 30:
        log(f"    ACHIEVED! {annual_return:.1f}%/year >= 30%")
    elif annual_return >= 20:
        log(f"    CLOSE: {annual_return:.1f}%/year — may reach 30% with position sizing optimization")
    elif annual_return >= 10:
        log(f"    MODERATE: {annual_return:.1f}%/year — needs strategy improvements")
    else:
        log(f"    BELOW TARGET: {annual_return:.1f}%/year — strategy needs rethinking")


def phase6_parameter_scan(best_base_config: dict):
    """Phase 6: Fine-tune the best strategy with parameter variations."""
    write_section("PHASE 6: Parameter Sensitivity Scan")
    log("Testing parameter variations on 3 core symbols")

    test_symbols = ["ETH/USDT", "AVAX/USDT", "LINK/USDT"]
    data = {}
    for sym in test_symbols:
        df = load_symbol_data(sym)
        if df is not None and len(df) >= 200:
            data[sym] = df

    if not data:
        log("No data available for parameter scan")
        return best_base_config

    # Parameters to vary
    param_ranges = {
        "breakout_period": [10, 14, 20, 30, 40, 55],
        "exit_period": [5, 7, 10, 15, 20],
        "atr_stop_mult": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "adx_min": [0, 15, 20, 25, 30],
        "risk_per_trade_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    best_composite = -999
    best_config = best_base_config.copy()

    for param_name, values in param_ranges.items():
        log(f"\n  Scanning {param_name}: {values}")
        param_results = []

        for val in values:
            test_config = best_base_config.copy()
            test_config[param_name] = val

            returns = []
            drawdowns = []
            for sym, df in data.items():
                try:
                    result = run_breakout(df, test_config)
                    returns.append(result.total_return_pct)
                    drawdowns.append(result.max_drawdown_pct)
                except Exception:
                    returns.append(0)
                    drawdowns.append(100)

            avg_ret = np.mean(returns)
            avg_dd = np.mean(drawdowns)
            r_dd = avg_ret / avg_dd if avg_dd > 0 else 0

            param_results.append((val, avg_ret, avg_dd, r_dd))
            log(f"    {param_name}={val:6.1f} -> Avg ret: {avg_ret:+7.1f}%, "
                f"Avg DD: {avg_dd:5.1f}%, R/DD: {r_dd:5.2f}")

        # Find best value for this parameter
        best_val_result = max(param_results, key=lambda x: x[3])
        log(f"    Best: {param_name}={best_val_result[0]} (R/DD={best_val_result[3]:.2f})")

        # Check if parameter landscape is smooth (low CV = good)
        r_dds = [x[3] for x in param_results]
        cv = np.std(r_dds) / abs(np.mean(r_dds)) * 100 if np.mean(r_dds) != 0 else 999
        log(f"    Sensitivity CV: {cv:.1f}% {'(SMOOTH)' if cv < 30 else '(SENSITIVE!)'}")

        gc.collect()

    return best_config


def main():
    start_time = time.time()

    # Clear report file
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH: Noise-Resistant Trend Following")
    log(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"Testing {len(BREAKOUT_CONFIGS)} breakout configurations + EMA baseline")
    log(f"Across {len(ALL_SYMBOLS)} crypto pairs")
    log(f"4-year backtest: 2022-01-01 to present")

    # Phase 1: Individual pair performance
    all_results = phase1_individual_pairs()

    # Phase 2: Aggregate analysis and ranking
    rankings = phase2_aggregate_analysis(all_results)

    if not rankings:
        log("\nNo valid strategy rankings. Aborting.")
        return

    best_strategy = rankings[0]
    best_name = best_strategy["name"]
    log(f"\n  BEST STRATEGY: {best_name}")

    # Get the config for the best strategy
    if best_name == "EMA_Momentum_baseline":
        best_config = MOMENTUM_BASELINE
    else:
        best_config = BREAKOUT_CONFIGS.get(best_name, BREAKOUT_CONFIGS["Breakout_20_10_3.0"])

    # Phase 3: Per-pair breakdown
    phase3_best_per_pair(all_results, best_name)

    # Phase 4: Quick noise test
    phase4_noise_quick_test(best_name, best_config)

    # Phase 5: Portfolio simulation
    phase5_portfolio_simulation(best_config)

    # Phase 6: Parameter sensitivity scan
    phase6_parameter_scan(best_config)

    # Final summary
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start_time
    log(f"Total runtime: {elapsed/60:.1f} minutes")
    log(f"\nTop 3 strategies:")
    for i, s in enumerate(rankings[:3]):
        log(f"  #{i+1}: {s['name']} (R/DD={s['risk_adjusted']:.2f}, "
            f"Sharpe={s['avg_sharpe']:.2f}, Win={s['win_pct']:.0f}%)")

    log(f"\nBest strategy: {best_name}")
    log(f"Config: {best_config}")
    log(f"\nNext step: Run robustness_test.py with this strategy to validate")
    log(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
