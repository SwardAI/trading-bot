"""Strategy research v2: daily-timeframe trend following.

Based on academic research (Zarattini 2025, Le & Ruthbah 2023, Mesicek 2025):
- Move from hourly to daily signals (reduce trade count by 80%)
- Ensemble Donchian channels (2/3 vote = noise resistant)
- Volatility squeeze breakout (very selective, 2-5 trades/pair/4yr)
- Dual MA regime + pullback (golden cross framework)

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v2.py" bot
"""

import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import download_ohlcv, load_cached_data
from src.backtest.engine import BacktestEngine

REPORT_FILE = "data/strategy_research_v2_report.txt"
INITIAL_CAPITAL = 10000

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
    return_over_dd: float


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
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


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly OHLCV to daily candles."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    daily = daily.reset_index()
    return daily


def load_daily_data(symbol: str) -> pd.DataFrame | None:
    """Load hourly data and resample to daily."""
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
        daily = resample_to_daily(df)
        return daily
    return None


def run_daily(df: pd.DataFrame, config: dict):
    """Run daily trend backtest."""
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine.run_daily_trend_backtest(df, config=config, long_only=True)


def run_momentum_hourly(symbol: str) -> StrategyResult | None:
    """Run original EMA momentum on hourly data for comparison."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.reset_index(drop=True)
    config = {
        "ema_fast": 20, "ema_slow": 50,
        "trailing_stop_atr_multiplier": 3.5,
        "adx_min_strength": 20, "rsi_long_threshold": 50,
        "required_signals": ["ema", "adx"],
        "risk_per_trade_pct": 1.0,
    }
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    result = engine.run_momentum_backtest(df, config=config, long_only=True)
    r_dd = result.total_return_pct / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0
    return StrategyResult(
        name="EMA_Momentum_1h", config=config, symbol=symbol,
        total_return_pct=result.total_return_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        profit_factor=result.profit_factor,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        return_over_dd=r_dd,
    )


def to_summary(name: str, config: dict, symbol: str, result) -> StrategyResult:
    r_dd = result.total_return_pct / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0
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
# DAILY STRATEGY CONFIGURATIONS
# ---------------------------------------------------------------

DAILY_CONFIGS = {
    # Strategy A: Ensemble Donchian — 3 channels vote (paper: Zarattini 2025)
    "Ensemble_20_50_100": {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": [20, 50, 100],
        "donchian_min_votes": 2,
        "donchian_exit_divisor": 2,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Ensemble with volume filter
    "Ensemble_20_50_100_vol": {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": [20, 50, 100],
        "donchian_min_votes": 2,
        "donchian_exit_divisor": 2,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.3,
        "cooldown_bars": 3,
    },
    # Ensemble with tighter stops
    "Ensemble_20_50_100_tight": {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": [20, 50, 100],
        "donchian_min_votes": 2,
        "donchian_exit_divisor": 2,
        "atr_stop_mult": 2.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Ensemble with wider stops (let trades breathe)
    "Ensemble_20_50_100_wide": {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": [20, 50, 100],
        "donchian_min_votes": 2,
        "donchian_exit_divisor": 2,
        "atr_stop_mult": 4.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Faster ensemble (shorter periods, more trades)
    "Ensemble_14_30_60": {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": [14, 30, 60],
        "donchian_min_votes": 2,
        "donchian_exit_divisor": 2,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.0,
        "cooldown_bars": 2,
    },
    # Unanimous vote (3/3 — ultra selective)
    "Ensemble_20_50_100_3of3": {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": [20, 50, 100],
        "donchian_min_votes": 3,
        "donchian_exit_divisor": 2,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,  # Higher risk per trade since fewer trades
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Strategy B: Volatility Squeeze
    "Squeeze_20_2.0_1.5": {
        "signal_mode": "squeeze",
        "squeeze_bb_period": 20,
        "squeeze_bb_std": 2.0,
        "squeeze_kc_mult": 1.5,
        "squeeze_min_bars": 5,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.5,
        "cooldown_bars": 5,
    },
    # Squeeze with longer compression
    "Squeeze_20_2.0_1.5_long": {
        "signal_mode": "squeeze",
        "squeeze_bb_period": 20,
        "squeeze_bb_std": 2.0,
        "squeeze_kc_mult": 1.5,
        "squeeze_min_bars": 10,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.5,
        "cooldown_bars": 5,
    },
    # Strategy C: Dual MA (Golden Cross + Pullback)
    "DualMA_50_200": {
        "signal_mode": "dual_ma",
        "ma_fast": 50,
        "ma_slow": 200,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.0,
        "cooldown_bars": 5,
    },
    # Faster dual MA
    "DualMA_20_100": {
        "signal_mode": "dual_ma",
        "ma_fast": 20,
        "ma_slow": 100,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 2.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
}


def phase1_individual_pairs():
    """Phase 1: Test all daily strategies on all pairs."""
    write_section("PHASE 1: Individual Pair Performance (Daily Timeframe)")
    log(f"Testing {len(DAILY_CONFIGS)} daily configs + 1h EMA baseline")
    log(f"Across {len(ALL_SYMBOLS)} pairs = {(len(DAILY_CONFIGS) + 1) * len(ALL_SYMBOLS)} backtests")

    all_results = []

    for symbol in ALL_SYMBOLS:
        log(f"\n--- {symbol} ---")
        daily_df = load_daily_data(symbol)
        if daily_df is None or len(daily_df) < 250:
            log(f"  Skipping {symbol}: insufficient daily data ({len(daily_df) if daily_df is not None else 0})")
            continue

        log(f"  Daily data: {len(daily_df)} candles ({daily_df.iloc[0]['timestamp'].date()} to {daily_df.iloc[-1]['timestamp'].date()})")

        # Test daily configs
        for name, config in DAILY_CONFIGS.items():
            try:
                result = run_daily(daily_df, config)
                sr = to_summary(name, config, symbol, result)
                all_results.append(sr)
                log(f"  {name:35s} | Return: {sr.total_return_pct:+7.1f}% | DD: {sr.max_drawdown_pct:5.1f}% | "
                    f"Sharpe: {sr.sharpe_ratio:5.2f} | PF: {sr.profit_factor:5.2f} | Trades: {sr.total_trades:3d} | "
                    f"Win: {sr.win_rate:4.1f}%")
            except Exception as e:
                log(f"  {name}: ERROR - {e}")

        # Hourly baseline for comparison
        try:
            sr = run_momentum_hourly(symbol)
            if sr:
                all_results.append(sr)
                log(f"  {'EMA_Momentum_1h':35s} | Return: {sr.total_return_pct:+7.1f}% | DD: {sr.max_drawdown_pct:5.1f}% | "
                    f"Sharpe: {sr.sharpe_ratio:5.2f} | PF: {sr.profit_factor:5.2f} | Trades: {sr.total_trades:3d} | "
                    f"Win: {sr.win_rate:4.1f}%")
        except Exception as e:
            log(f"  EMA_Momentum_1h: ERROR - {e}")

        gc.collect()

    return all_results


def phase2_aggregate(all_results: list[StrategyResult]):
    """Phase 2: Aggregate analysis and ranking."""
    write_section("PHASE 2: Aggregate Analysis")

    if not all_results:
        log("No results!")
        return None

    from collections import defaultdict
    strategy_stats = defaultdict(list)
    for r in all_results:
        strategy_stats[r.name].append(r)

    log(f"\n{'Strategy':35s} | {'Avg Ret':>8s} | {'Med Ret':>8s} | {'Avg DD':>7s} | "
        f"{'Sharpe':>7s} | {'Win%':>5s} | {'PF':>6s} | {'R/DD':>6s} | {'Trades':>7s} | Profitable")
    log("-" * 130)

    rankings = []

    for name, results in sorted(strategy_stats.items()):
        returns = [r.total_return_pct for r in results]
        drawdowns = [r.max_drawdown_pct for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        pfs = [r.profit_factor for r in results]
        trade_counts = [r.total_trades for r in results]
        profitable = sum(1 for r in returns if r > 0)

        avg_ret = np.mean(returns)
        med_ret = np.median(returns)
        avg_dd = np.mean(drawdowns)
        avg_sharpe = np.mean(sharpes)
        win_pct = profitable / len(results) * 100
        avg_pf = np.mean(pfs)
        avg_r_dd = avg_ret / avg_dd if avg_dd > 0 else 0
        avg_trades = np.mean(trade_counts)

        log(f"{name:35s} | {avg_ret:+7.1f}% | {med_ret:+7.1f}% | {avg_dd:6.1f}% | "
            f"{avg_sharpe:6.2f} | {win_pct:4.0f}% | {avg_pf:5.2f} | {avg_r_dd:5.2f} | "
            f"{avg_trades:6.0f} | {profitable}/{len(results)}")

        rankings.append({
            "name": name,
            "avg_return": avg_ret,
            "median_return": med_ret,
            "avg_drawdown": avg_dd,
            "avg_sharpe": avg_sharpe,
            "win_pct": win_pct,
            "avg_pf": avg_pf,
            "risk_adjusted": avg_r_dd,
            "avg_trades": avg_trades,
            "num_profitable": profitable,
            "num_pairs": len(results),
            "results": results,
        })

    # Composite ranking
    log("\n--- COMPOSITE RANKING (40% R/DD + 30% win% + 30% Sharpe) ---")
    for s in rankings:
        max_r_dd = max(x["risk_adjusted"] for x in rankings) or 1
        max_win = max(x["win_pct"] for x in rankings) or 1
        max_sharpe = max(x["avg_sharpe"] for x in rankings) or 1
        s["composite"] = (
            0.4 * (s["risk_adjusted"] / max_r_dd if max_r_dd > 0 else 0) +
            0.3 * (s["win_pct"] / max_win if max_win > 0 else 0) +
            0.3 * (s["avg_sharpe"] / max_sharpe if max_sharpe > 0 else 0)
        )

    rankings.sort(key=lambda x: x["composite"], reverse=True)

    for i, s in enumerate(rankings):
        marker = " <-- BEST" if i == 0 else ""
        log(f"  #{i+1}: {s['name']:35s} | Composite: {s['composite']:.3f} | "
            f"R/DD: {s['risk_adjusted']:.2f} | Win: {s['win_pct']:.0f}% | "
            f"Sharpe: {s['avg_sharpe']:.2f} | Trades: {s['avg_trades']:.0f}{marker}")

    return rankings


def phase3_noise_test(top_name: str, top_config: dict, is_daily: bool = True):
    """Phase 3: Noise resilience test on top strategy."""
    write_section(f"PHASE 3: Noise Resilience — '{top_name}'")
    log("Adding 0.5% Gaussian noise to prices, 30 runs per pair")

    test_symbols = ALL_SYMBOLS[:5]  # Test on 5 pairs
    noise_std = 0.005
    n_runs = 30

    overall_profitable = 0
    overall_total = 0

    for symbol in test_symbols:
        if is_daily:
            df = load_daily_data(symbol)
        else:
            df = load_cached_data("binance", symbol, "1h")
            if df is not None:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.reset_index(drop=True)

        if df is None or len(df) < 250:
            continue

        # Baseline
        if is_daily:
            baseline = run_daily(df, top_config)
        else:
            engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
            baseline = engine.run_momentum_backtest(df, config=top_config, long_only=True)
        baseline_ret = baseline.total_return_pct

        # Noisy runs
        noisy_returns = []
        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            df_noisy = df.copy()
            noise = rng.normal(1.0, noise_std, len(df_noisy))
            df_noisy["close"] = df_noisy["close"] * noise
            df_noisy["open"] = df_noisy["open"] * rng.normal(1.0, noise_std, len(df_noisy))
            h_noise = rng.normal(1.0, noise_std * 0.5, len(df_noisy))
            l_noise = rng.normal(1.0, noise_std * 0.5, len(df_noisy))
            df_noisy["high"] = df_noisy[["open", "close", "high"]].max(axis=1) * h_noise
            df_noisy["low"] = df_noisy[["open", "close", "low"]].min(axis=1) * l_noise
            df_noisy["high"] = df_noisy[["open", "close", "high"]].max(axis=1)
            df_noisy["low"] = df_noisy[["open", "close", "low"]].min(axis=1)

            try:
                if is_daily:
                    result = run_daily(df_noisy, top_config)
                else:
                    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
                    result = engine.run_momentum_backtest(df_noisy, config=top_config, long_only=True)
                noisy_returns.append(result.total_return_pct)
            except Exception:
                noisy_returns.append(0.0)

        profitable = sum(1 for r in noisy_returns if r > 0)
        avg_noisy = np.mean(noisy_returns)
        std_noisy = np.std(noisy_returns)
        overall_profitable += profitable
        overall_total += n_runs

        log(f"  {symbol:12s}: Baseline {baseline_ret:+6.1f}% | Noisy avg {avg_noisy:+6.1f}% "
            f"(std {std_noisy:5.1f}%) | Profitable {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")
        gc.collect()

    if overall_total > 0:
        pct = overall_profitable / overall_total * 100
        log(f"\n  OVERALL NOISE RESILIENCE: {overall_profitable}/{overall_total} ({pct:.0f}%)")
        if pct >= 70:
            log(f"  PASS — Strategy survives 0.5% noise")
        elif pct >= 50:
            log(f"  MARGINAL — Somewhat noise resistant")
        else:
            log(f"  FAIL — Strategy is noise-fragile")


def phase4_portfolio(top_config: dict):
    """Phase 4: Multi-pair portfolio simulation."""
    write_section("PHASE 4: Multi-Pair Portfolio Simulation (Daily)")

    pair_equities = {}
    pair_trades = {}
    min_len = float("inf")

    for symbol in ALL_SYMBOLS:
        df = load_daily_data(symbol)
        if df is None or len(df) < 250:
            continue
        result = run_daily(df, top_config)
        if result.equity_curve:
            pair_equities[symbol] = result.equity_curve
            pair_trades[symbol] = result.total_trades
            min_len = min(min_len, len(result.equity_curve))

    if not pair_equities:
        log("No valid results")
        return

    n_pairs = len(pair_equities)
    log(f"Portfolio: {n_pairs} pairs, equal allocation (${INITIAL_CAPITAL/n_pairs:.0f} each)")

    per_pair = INITIAL_CAPITAL / n_pairs
    portfolio_equity = []
    for i in range(int(min_len)):
        total = sum(eq[i] / INITIAL_CAPITAL * per_pair for eq in pair_equities.values() if i < len(eq))
        portfolio_equity.append(total)

    if not portfolio_equity:
        return

    initial = portfolio_equity[0]
    final = portfolio_equity[-1]
    total_return = (final - initial) / initial * 100

    peak = portfolio_equity[0]
    max_dd = 0
    for eq in portfolio_equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    years = min_len / 365.25
    annual_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    returns = [(portfolio_equity[i] - portfolio_equity[i-1]) / portfolio_equity[i-1]
               for i in range(1, len(portfolio_equity)) if portfolio_equity[i-1] > 0]
    sharpe = np.mean(returns) / np.std(returns) * (365.25 ** 0.5) if returns and np.std(returns) > 0 else 0

    total_trades_all = sum(pair_trades.values())

    log(f"\n  Portfolio Performance ({years:.1f} years):")
    log(f"    Total return:      {total_return:+.1f}%")
    log(f"    Annualized return: {annual_return:+.1f}%")
    log(f"    Max drawdown:      {max_dd:.1f}%")
    log(f"    Sharpe ratio:      {sharpe:.2f}")
    log(f"    Return / DD:       {total_return / max_dd:.2f}" if max_dd > 0 else "    Return / DD: inf")
    log(f"    Total trades:      {total_trades_all}")
    log(f"    Trades/pair/year:  {total_trades_all / n_pairs / years:.1f}")

    log(f"\n  TARGET CHECK: 30%/year")
    if annual_return >= 30:
        log(f"    ACHIEVED! {annual_return:.1f}%/year >= 30%/year")
    elif annual_return >= 20:
        log(f"    CLOSE: {annual_return:.1f}%/year — optimize position sizing")
    elif annual_return >= 10:
        log(f"    MODERATE: {annual_return:.1f}%/year — needs improvement but promising")
    else:
        log(f"    BELOW TARGET: {annual_return:.1f}%/year")

    # Per-pair breakdown
    log(f"\n  Per-pair results:")
    pair_results = []
    for symbol in ALL_SYMBOLS:
        df = load_daily_data(symbol)
        if df is None or len(df) < 250:
            continue
        result = run_daily(df, top_config)
        pair_results.append((symbol, result))

    pair_results.sort(key=lambda x: x[1].total_return_pct, reverse=True)
    for symbol, r in pair_results:
        log(f"    {symbol:12s}: {r.total_return_pct:+7.1f}% | DD: {r.max_drawdown_pct:5.1f}% | "
            f"Trades: {r.total_trades:3d} | PF: {r.profit_factor:.2f}")


def phase5_param_sensitivity(top_config: dict):
    """Phase 5: Quick parameter sensitivity on daily strategy."""
    write_section("PHASE 5: Parameter Sensitivity (Daily)")

    test_symbols = ["ETH/USDT", "AVAX/USDT", "LINK/USDT"]
    data = {}
    for sym in test_symbols:
        df = load_daily_data(sym)
        if df is not None and len(df) >= 250:
            data[sym] = df

    if not data:
        return

    mode = top_config.get("signal_mode", "ensemble_donchian")

    if mode == "ensemble_donchian":
        param_ranges = {
            "atr_stop_mult": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "risk_per_trade_pct": [1.0, 1.5, 2.0, 3.0, 4.0],
            "donchian_min_votes": [2, 3],
            "cooldown_bars": [1, 2, 3, 5, 7],
        }
    elif mode == "squeeze":
        param_ranges = {
            "atr_stop_mult": [2.0, 2.5, 3.0, 3.5, 4.0],
            "risk_per_trade_pct": [1.0, 2.0, 3.0, 4.0],
            "squeeze_min_bars": [3, 5, 7, 10, 15],
            "squeeze_kc_mult": [1.0, 1.25, 1.5, 1.75, 2.0],
        }
    else:
        param_ranges = {
            "atr_stop_mult": [2.0, 2.5, 3.0, 3.5, 4.0],
            "risk_per_trade_pct": [1.0, 2.0, 3.0],
            "ma_fast": [20, 30, 50, 75],
            "ma_slow": [100, 150, 200, 250],
        }

    for param_name, values in param_ranges.items():
        log(f"\n  Scanning {param_name}: {values}")
        results_list = []

        for val in values:
            test_config = top_config.copy()
            # Handle nested list params
            if param_name == "donchian_periods":
                test_config[param_name] = val
            else:
                test_config[param_name] = val

            rets = []
            dds = []
            for sym, df in data.items():
                try:
                    result = run_daily(df, test_config)
                    rets.append(result.total_return_pct)
                    dds.append(result.max_drawdown_pct)
                except Exception:
                    rets.append(0)
                    dds.append(100)

            avg_ret = np.mean(rets)
            avg_dd = np.mean(dds)
            r_dd = avg_ret / avg_dd if avg_dd > 0 else 0
            results_list.append((val, avg_ret, avg_dd, r_dd))
            log(f"    {param_name}={str(val):>6s} -> Ret: {avg_ret:+7.1f}%, DD: {avg_dd:5.1f}%, R/DD: {r_dd:5.2f}")

        r_dds = [x[3] for x in results_list]
        mean_r_dd = np.mean(r_dds)
        cv = np.std(r_dds) / abs(mean_r_dd) * 100 if mean_r_dd != 0 else 999
        best = max(results_list, key=lambda x: x[3])
        log(f"    Best: {param_name}={best[0]} (R/DD={best[3]:.2f}) | CV: {cv:.0f}% "
            f"{'(SMOOTH)' if cv < 30 else '(MODERATE)' if cv < 60 else '(SENSITIVE!)'}")

        gc.collect()


def main():
    start_time = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V2: Daily Trend Following")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("Based on: Zarattini 2025, Le & Ruthbah 2023, Mesicek 2025")
    log(f"Testing {len(DAILY_CONFIGS)} daily configs across {len(ALL_SYMBOLS)} pairs")
    log("Key insight: daily signals reduce trade count by 80%, improving after-fee returns")

    # Phase 1
    all_results = phase1_individual_pairs()

    # Phase 2
    rankings = phase2_aggregate(all_results)
    if not rankings:
        log("No valid rankings!")
        return

    best = rankings[0]
    best_name = best["name"]
    best_config = DAILY_CONFIGS.get(best_name, list(DAILY_CONFIGS.values())[0])
    log(f"\n  BEST STRATEGY: {best_name}")

    # Phase 3: Noise test
    is_daily = best_name != "EMA_Momentum_1h"
    if is_daily:
        phase3_noise_test(best_name, best_config, is_daily=True)
    else:
        mom_config = {
            "ema_fast": 20, "ema_slow": 50,
            "trailing_stop_atr_multiplier": 3.5,
            "adx_min_strength": 20, "rsi_long_threshold": 50,
            "required_signals": ["ema", "adx"],
            "risk_per_trade_pct": 1.0,
        }
        phase3_noise_test(best_name, mom_config, is_daily=False)

    # Phase 4: Portfolio
    if is_daily:
        phase4_portfolio(best_config)

    # Phase 5: Param sensitivity
    if is_daily:
        phase5_param_sensitivity(best_config)

    # Final summary
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start_time
    log(f"Total runtime: {elapsed/60:.1f} minutes")
    log(f"\nTop 5 strategies:")
    for i, s in enumerate(rankings[:5]):
        log(f"  #{i+1}: {s['name']:35s} | Avg ret: {s['avg_return']:+.1f}% | "
            f"R/DD: {s['risk_adjusted']:.2f} | Sharpe: {s['avg_sharpe']:.2f} | "
            f"Win: {s['win_pct']:.0f}% | Trades: {s['avg_trades']:.0f}")

    log(f"\nBest strategy: {best_name}")
    log(f"Config: {best_config}")
    log(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
