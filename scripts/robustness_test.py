"""Comprehensive robustness testing suite for momentum strategy.

Tests whether the strategy generalizes to unseen market conditions or is
overfit to the specific 2022-2026 history. Runs 7 independent tests and
produces a clear PASS/FAIL verdict.

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/robustness_test.py" bot
"""

import gc
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import download_ohlcv, load_cached_data
from src.backtest.engine import BacktestEngine, BacktestResult

REPORT_FILE = "data/robustness_report.txt"
INITIAL_CAPITAL = 10000

# Current optimized config from config/momentum_config.yaml
BASELINE_CONFIG = {
    "ema_fast": 20,
    "ema_slow": 50,
    "trailing_stop_atr_multiplier": 3.5,
    "adx_min_strength": 20,
    "rsi_long_threshold": 50,
    "required_signals": ["ema", "adx"],
    "risk_per_trade_pct": 1.0,
}

CORE_SYMBOLS = ["ETH/USDT", "AVAX/USDT", "LINK/USDT"]
EXTRA_SYMBOLS = [
    "DOT/USDT", "DOGE/USDT", "ADA/USDT", "SOL/USDT",
    "UNI/USDT", "ATOM/USDT", "NEAR/USDT",
]


@dataclass
class TestResult:
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    metrics: dict = field(default_factory=dict)


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
    """Load cached data, download if missing."""
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


def run_backtest(df: pd.DataFrame, config: dict | None = None) -> BacktestResult:
    """Run a single momentum backtest with the given config."""
    cfg = BASELINE_CONFIG.copy()
    if config:
        cfg.update(config)
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine.run_momentum_backtest(df, config=cfg, long_only=True)


# ---------------------------------------------------------------
# TEST 1: Parameter Sensitivity Analysis
# ---------------------------------------------------------------
def test_parameter_sensitivity(data: dict[str, pd.DataFrame]) -> TestResult:
    """Verify we're on a performance plateau, not a sharp peak."""
    log("\n  Testing parameter sensitivity (OAT perturbation)...")

    params = {
        "ema_fast": [16, 18, 20, 22, 24],
        "ema_slow": [40, 45, 50, 55, 60],
        "trailing_stop_atr_multiplier": [2.8, 3.15, 3.5, 3.85, 4.2],
        "adx_min_strength": [16, 18, 20, 22, 24],
        "rsi_long_threshold": [40, 45, 50, 55, 60],
    }

    all_returns = []
    param_results = {}  # param_name -> list of (value, avg_return)

    for param_name, values in params.items():
        param_results[param_name] = []
        for val in values:
            cfg_override = {param_name: val}
            returns_for_val = []
            for symbol, df in data.items():
                result = run_backtest(df, cfg_override)
                returns_for_val.append(result.total_return_pct)
            avg_ret = np.mean(returns_for_val)
            param_results[param_name].append((val, avg_ret))
            all_returns.append(avg_ret)

    # ASCII heatmap
    log("\n  Parameter Sensitivity (avg return across symbols):")
    log(f"  {'Parameter':<30} {'-20%':>8} {'-10%':>8} {'BASE':>8} {'+10%':>8} {'+20%':>8}")
    log(f"  {'-' * 70}")
    for param_name, vals in param_results.items():
        base_val = BASELINE_CONFIG[param_name]
        row = f"  {param_name} ({base_val})"
        row = f"{row:<30}"
        for val, ret in vals:
            row += f" {ret:>+7.1f}%"
        log(row)

    # Compute coefficient of variation
    mean_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    cv = abs(std_ret / mean_ret * 100) if mean_ret != 0 else 100

    log(f"\n  Mean return: {mean_ret:+.1f}%, Std: {std_ret:.1f}%, CV: {cv:.1f}%")

    passed = cv < 30
    score = max(0, 1.0 - cv / 30)

    return TestResult(
        name="Parameter Sensitivity",
        passed=passed,
        score=score,
        details=f"CV={cv:.1f}% ({'< 30%' if passed else '>= 30%'})",
        metrics={"cv": cv, "mean_return": mean_ret, "std_return": std_ret},
    )


# ---------------------------------------------------------------
# TEST 2: Rolling Window Walk-Forward
# ---------------------------------------------------------------
def test_rolling_walk_forward(data: dict[str, pd.DataFrame]) -> TestResult:
    """Test fixed params across multiple time windows."""
    log("\n  Running rolling walk-forward (6mo train, 3mo test, slide 3mo)...")

    window_results = []

    for symbol, df in data.items():
        log(f"\n  {symbol}:")
        total_candles = len(df)
        train_size = 4380  # ~6 months of 1h candles
        test_size = 2190   # ~3 months
        slide = 2190       # slide by 3 months

        start = 0
        window_num = 0
        while start + train_size + test_size <= total_candles:
            test_start = start + train_size
            test_end = test_start + test_size

            test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

            if len(test_df) < 100:
                start += slide
                continue

            result = run_backtest(test_df)
            window_num += 1

            # Determine test period dates
            t_start = test_df.iloc[0]["timestamp"].strftime("%Y-%m")
            t_end = test_df.iloc[-1]["timestamp"].strftime("%Y-%m")
            profitable = result.total_return_pct > 0
            trades = result.winning_trades + result.losing_trades

            window_results.append({
                "symbol": symbol,
                "window": window_num,
                "period": f"{t_start} to {t_end}",
                "return_pct": result.total_return_pct,
                "trades": trades,
                "profitable": profitable,
            })

            verdict = "PASS" if profitable else "FAIL"
            log(f"    Window {window_num:>2}: {t_start} to {t_end}  "
                f"ret={result.total_return_pct:>+6.1f}%  trades={trades:>3}  {verdict}")

            start += slide

    total_windows = len(window_results)
    profitable_windows = sum(1 for w in window_results if w["profitable"])
    pct = profitable_windows / total_windows * 100 if total_windows > 0 else 0

    log(f"\n  Overall: {profitable_windows}/{total_windows} windows profitable ({pct:.1f}%)")

    passed = pct > 60
    score = min(1.0, pct / 60)

    return TestResult(
        name="Rolling Walk-Forward",
        passed=passed,
        score=score,
        details=f"{profitable_windows}/{total_windows} windows profitable ({pct:.1f}%)",
        metrics={"pct_profitable": pct, "total_windows": total_windows,
                 "profitable_windows": profitable_windows},
    )


# ---------------------------------------------------------------
# TEST 3: Synthetic Stress Testing
# ---------------------------------------------------------------
def inject_flash_crash(df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    """Inject -30% crash over 24h, then 50% recovery over 7 days."""
    df = df.copy()
    crash_candles = 24
    recovery_candles = 168

    for i in range(crash_candles + recovery_candles):
        idx = start_idx + i
        if idx >= len(df):
            break

        if i < crash_candles:
            # Crash phase: linear drop to -30%
            progress = (i + 1) / crash_candles
            mult = 1.0 - 0.30 * progress
        else:
            # Recovery phase: sqrt recovery back to ~original
            r_progress = (i - crash_candles + 1) / recovery_candles
            crash_bottom = 0.70
            mult = crash_bottom + (1.0 - crash_bottom) * (r_progress ** 0.5)

        for col in ["open", "high", "low", "close"]:
            df.at[df.index[idx], col] = df.iloc[idx][col] * mult

    return df


def inject_prolonged_bear(df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    """Apply -1% daily drift for 90 days (~40% total decline)."""
    df = df.copy()
    duration = 90 * 24  # 2160 candles

    for i in range(duration):
        idx = start_idx + i
        if idx >= len(df):
            break

        day = i // 24
        mult = 0.99 ** day  # compounds daily

        for col in ["open", "high", "low", "close"]:
            df.at[df.index[idx], col] = df.iloc[idx][col] * mult

    return df


def inject_volatility_spike(df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    """Triple the high-low range for 14 days."""
    df = df.copy()
    duration = 336  # 14 days

    for i in range(duration):
        idx = start_idx + i
        if idx >= len(df):
            break

        close = df.iloc[idx]["close"]
        high_diff = df.iloc[idx]["high"] - close
        low_diff = close - df.iloc[idx]["low"]

        df.at[df.index[idx], "high"] = close + high_diff * 3.0
        df.at[df.index[idx], "low"] = close - low_diff * 3.0

    return df


def inject_sideways_chop(df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
    """Flatten prices into +/-2% band for 60 days."""
    df = df.copy()
    duration = 1440  # 60 days
    base_price = df.iloc[start_idx]["close"]
    band = 0.02

    rng = random.Random(42)

    for i in range(duration):
        idx = start_idx + i
        if idx >= len(df):
            break

        # Random walk within band
        noise = rng.uniform(-band, band)
        new_close = base_price * (1 + noise)
        ratio = new_close / df.iloc[idx]["close"] if df.iloc[idx]["close"] != 0 else 1

        for col in ["open", "high", "low", "close"]:
            df.at[df.index[idx], col] = df.iloc[idx][col] * ratio

        # Fix OHLC consistency
        row_idx = df.index[idx]
        vals = [df.at[row_idx, c] for c in ["open", "high", "low", "close"]]
        df.at[row_idx, "high"] = max(vals)
        df.at[row_idx, "low"] = min(vals)

    return df


def test_synthetic_stress(data: dict[str, pd.DataFrame]) -> TestResult:
    """Test against extreme market scenarios."""
    log("\n  Running synthetic stress scenarios...")

    scenarios = [
        ("Flash Crash (-30%/24h)", inject_flash_crash),
        ("Prolonged Bear (-1%/day, 90d)", inject_prolonged_bear),
        ("Volatility Spike (3x, 14d)", inject_volatility_spike),
        ("Sideways Chop (+/-2%, 60d)", inject_sideways_chop),
    ]

    worst_dd = 0
    all_pass = True

    for scenario_name, inject_fn in scenarios:
        scenario_dds = []
        for symbol, df in data.items():
            inject_idx = min(8000, len(df) - 3000)
            stressed_df = inject_fn(df, inject_idx)
            result = run_backtest(stressed_df)
            scenario_dds.append(result.max_drawdown_pct)

        max_dd = max(scenario_dds)
        worst_dd = max(worst_dd, max_dd)
        scenario_pass = max_dd < 25
        if not scenario_pass:
            all_pass = False

        verdict = "PASS" if scenario_pass else "FAIL"
        avg_dd = np.mean(scenario_dds)
        log(f"  {scenario_name:<35} avg DD={avg_dd:.1f}%, max DD={max_dd:.1f}%  {verdict}")

    score = max(0, 1.0 - worst_dd / 25)

    return TestResult(
        name="Synthetic Stress",
        passed=all_pass,
        score=score,
        details=f"Worst DD={worst_dd:.1f}% ({'< 25%' if all_pass else '>= 25%'})",
        metrics={"worst_dd": worst_dd},
    )


# ---------------------------------------------------------------
# TEST 4: Noise Injection (Anti-Overfit)
# ---------------------------------------------------------------
def test_noise_injection(data: dict[str, pd.DataFrame]) -> TestResult:
    """Add random noise to prices and check if strategy survives."""
    log("\n  Running noise injection (100 runs, 0.5% noise)...")

    N_RUNS = 100
    noise_std = 0.005  # 0.5% of price

    all_returns = []
    all_sharpes = []

    for symbol, df in data.items():
        symbol_returns = []
        symbol_sharpes = []

        for seed in range(N_RUNS):
            np.random.seed(seed)
            df_noisy = df.copy()

            for col in ["open", "high", "low", "close"]:
                noise = np.random.normal(0, noise_std, len(df_noisy))
                df_noisy[col] = df_noisy[col] * (1 + noise)

            # Fix OHLC consistency
            df_noisy["high"] = df_noisy[["open", "high", "low", "close"]].max(axis=1)
            df_noisy["low"] = df_noisy[["open", "high", "low", "close"]].min(axis=1)

            result = run_backtest(df_noisy)
            symbol_returns.append(result.total_return_pct)
            symbol_sharpes.append(result.sharpe_ratio)

        profitable = sum(1 for r in symbol_returns if r > 0)
        sharpe_std = np.std(symbol_sharpes)

        log(f"  {symbol}: {profitable}/{N_RUNS} profitable ({profitable}%), "
            f"return mean={np.mean(symbol_returns):+.1f}% std={np.std(symbol_returns):.1f}%, "
            f"Sharpe std={sharpe_std:.2f}")

        all_returns.extend(symbol_returns)
        all_sharpes.extend(symbol_sharpes)

    total_profitable = sum(1 for r in all_returns if r > 0)
    total_runs = len(all_returns)
    pct_profitable = total_profitable / total_runs * 100 if total_runs > 0 else 0
    sharpe_std_overall = np.std(all_sharpes)

    passed = pct_profitable >= 80 and sharpe_std_overall < 0.5
    score = min(1.0, pct_profitable / 80) * min(1.0, 0.5 / max(sharpe_std_overall, 0.01))

    log(f"\n  Overall: {total_profitable}/{total_runs} profitable ({pct_profitable:.1f}%), "
        f"Sharpe std={sharpe_std_overall:.2f}")

    return TestResult(
        name="Noise Injection",
        passed=passed,
        score=min(score, 1.0),
        details=f"{pct_profitable:.1f}% profitable, Sharpe std={sharpe_std_overall:.2f}",
        metrics={"pct_profitable": pct_profitable, "sharpe_std": sharpe_std_overall},
    )


# ---------------------------------------------------------------
# TEST 5: Cross-Asset Generalization
# ---------------------------------------------------------------
def test_cross_asset(data: dict[str, pd.DataFrame]) -> TestResult:
    """Test if momentum works across many pairs, not just the 3 we picked."""
    log("\n  Testing cross-asset generalization...")

    results = []
    for symbol, df in data.items():
        result = run_backtest(df)
        trades = result.winning_trades + result.losing_trades
        profitable = result.total_return_pct > 0

        results.append({
            "symbol": symbol,
            "return_pct": result.total_return_pct,
            "pf": result.profit_factor,
            "max_dd": result.max_drawdown_pct,
            "trades": trades,
            "profitable": profitable,
        })

        is_core = symbol in CORE_SYMBOLS
        tag = " (core)" if is_core else ""
        verdict = "PROFIT" if profitable else "LOSS"
        log(f"  {symbol:<12}{tag:<8} ret={result.total_return_pct:>+6.1f}%  "
            f"PF={result.profit_factor:>5.2f}  DD={result.max_drawdown_pct:>5.1f}%  "
            f"trades={trades:>3}  {verdict}")

    total = len(results)
    profitable_count = sum(1 for r in results if r["profitable"])
    pct = profitable_count / total * 100 if total > 0 else 0

    log(f"\n  Profitable: {profitable_count}/{total} pairs ({pct:.1f}%)")

    passed = pct >= 60
    score = min(1.0, pct / 60)

    return TestResult(
        name="Cross-Asset Generalization",
        passed=passed,
        score=score,
        details=f"{profitable_count}/{total} pairs profitable ({pct:.1f}%)",
        metrics={"pct_profitable": pct, "total_pairs": total,
                 "profitable_pairs": profitable_count},
    )


# ---------------------------------------------------------------
# TEST 6: Minimum Activity Check
# ---------------------------------------------------------------
def test_minimum_activity(data: dict[str, pd.DataFrame]) -> TestResult:
    """Ensure the strategy actually trades, not profiting by being idle."""
    log("\n  Checking trading activity...")

    activity_ok = True
    all_trades_per_year = []
    all_time_in_position = []

    for symbol, df in data.items():
        result = run_backtest(df)

        # Count round trips
        round_trips = [t for t in result.trades if t.pnl != 0]
        total_rt = len(round_trips)

        # Estimate years of data
        if len(df) > 0:
            first_ts = df.iloc[0]["timestamp"]
            last_ts = df.iloc[-1]["timestamp"]
            years = max(0.5, (last_ts - first_ts).days / 365.25)
        else:
            years = 1

        trades_per_year = total_rt / years

        # Estimate time in position from trades
        # Each buy-sell pair represents time in position
        buys = [t for t in result.trades if t.side == "buy"]
        sells = [t for t in result.trades if t.side == "sell"]
        total_candles = len(df)

        # Count candles in position (from equity curve changes)
        # Simpler: estimate from trade count and typical hold time
        # Each round trip is buy + sell, so time between consecutive buy and sell
        time_in_candles = 0
        buy_idx = 0
        for i, trade in enumerate(result.trades):
            if trade.side == "buy":
                buy_idx = i
            elif trade.side == "sell" and buy_idx < i:
                # Find how many candles between buy and sell by timestamp
                buy_ts = result.trades[buy_idx].timestamp
                sell_ts = trade.timestamp
                if hasattr(buy_ts, 'timestamp'):
                    hours = (sell_ts - buy_ts).total_seconds() / 3600
                else:
                    hours = 72  # default estimate
                time_in_candles += hours  # 1h candles

        time_pct = time_in_candles / total_candles * 100 if total_candles > 0 else 0

        all_trades_per_year.append(trades_per_year)
        all_time_in_position.append(time_pct)

        log(f"  {symbol}: {total_rt} round trips over {years:.1f}yr = "
            f"{trades_per_year:.1f}/yr, in-position ~{time_pct:.1f}%")

    avg_trades = np.mean(all_trades_per_year)
    avg_time = np.mean(all_time_in_position)

    trades_ok = avg_trades >= 15
    time_ok = avg_time >= 15

    if not trades_ok:
        log(f"  WARNING: Avg {avg_trades:.1f} trades/yr < 15 threshold")
    if not time_ok:
        log(f"  WARNING: Avg {avg_time:.1f}% time in position < 15% threshold")

    passed = trades_ok and time_ok
    score = min(1.0, avg_trades / 15) * 0.5 + min(1.0, avg_time / 15) * 0.5

    return TestResult(
        name="Minimum Activity",
        passed=passed,
        score=score,
        details=f"Avg {avg_trades:.1f} trades/yr, {avg_time:.1f}% time in position",
        metrics={"avg_trades_per_year": avg_trades, "avg_time_in_position": avg_time},
    )


# ---------------------------------------------------------------
# TEST 7: Bootstrap Confidence Intervals
# ---------------------------------------------------------------
def test_bootstrap_confidence(data: dict[str, pd.DataFrame]) -> TestResult:
    """Estimate statistical confidence via bootstrap resampling."""
    log("\n  Running bootstrap resampling (1000 iterations)...")

    N_BOOTSTRAP = 1000
    all_ci_pass = True

    for symbol, df in data.items():
        result = run_backtest(df)
        pnls = [t.pnl for t in result.trades if t.pnl != 0]

        if len(pnls) < 5:
            log(f"  {symbol}: Too few trades ({len(pnls)}) for bootstrap")
            continue

        boot_returns = []
        boot_pfs = []
        boot_sharpes = []

        for _ in range(N_BOOTSTRAP):
            sample = np.random.choice(pnls, size=len(pnls), replace=True)

            total_ret = sum(sample) / INITIAL_CAPITAL * 100
            gross_profit = sum(p for p in sample if p > 0)
            gross_loss = abs(sum(p for p in sample if p <= 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 10.0
            sharpe = (np.mean(sample) / np.std(sample) * np.sqrt(252)
                      if np.std(sample) > 0 else 0)

            boot_returns.append(total_ret)
            boot_pfs.append(pf)
            boot_sharpes.append(sharpe)

        ret_ci = (np.percentile(boot_returns, 2.5), np.percentile(boot_returns, 97.5))
        pf_ci = (np.percentile(boot_pfs, 2.5), np.percentile(boot_pfs, 97.5))
        sharpe_ci = (np.percentile(boot_sharpes, 2.5), np.percentile(boot_sharpes, 97.5))

        ret_pass = ret_ci[0] > 0
        pf_pass = pf_ci[0] > 1.0

        if not (ret_pass and pf_pass):
            all_ci_pass = False

        log(f"  {symbol} ({len(pnls)} trades):")
        log(f"    Return 95% CI: [{ret_ci[0]:+.1f}%, {ret_ci[1]:+.1f}%]  "
            f"{'PASS' if ret_pass else 'FAIL'}")
        log(f"    PF 95% CI:     [{pf_ci[0]:.2f}, {pf_ci[1]:.2f}]  "
            f"{'PASS' if pf_pass else 'FAIL'}")
        log(f"    Sharpe 95% CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")

    score = 1.0 if all_ci_pass else 0.3

    return TestResult(
        name="Bootstrap Confidence",
        passed=all_ci_pass,
        score=score,
        details="95% CI lower bounds > 0 for return, > 1.0 for PF" if all_ci_pass
                else "Some CIs include zero/below 1.0",
        metrics={},
    )


# ---------------------------------------------------------------
# FINAL VERDICT
# ---------------------------------------------------------------
def compute_final_verdict(results: list[TestResult]) -> tuple[str, str, float]:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    weights = {
        "Parameter Sensitivity": 1.0,
        "Rolling Walk-Forward": 1.5,
        "Synthetic Stress": 1.0,
        "Noise Injection": 1.2,
        "Cross-Asset Generalization": 1.3,
        "Minimum Activity": 0.8,
        "Bootstrap Confidence": 1.2,
    }

    weighted_score = sum(
        r.score * weights.get(r.name, 1.0)
        for r in results
    ) / sum(weights.get(r.name, 1.0) for r in results)

    if passed == total:
        verdict = "STRONG PASS"
        rec = "Strategy passes ALL robustness tests. Safe to deploy."
    elif passed >= total - 2:
        failed = [r.name for r in results if not r.passed]
        verdict = "CONDITIONAL PASS"
        rec = f"Passes most tests. Review failures: {', '.join(failed)}"
    else:
        failed = [r.name for r in results if not r.passed]
        verdict = "FAIL"
        rec = f"Too many failures — strategy may be overfit. Failures: {', '.join(failed)}"

    return verdict, rec, weighted_score


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    start_time = time.time()

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(f"Robustness Test Suite -- {datetime.utcnow().isoformat()}\n")
        f.write(f"{'=' * 70}\n\n")

    log("Robustness test suite started")
    log(f"Baseline config: EMA {BASELINE_CONFIG['ema_fast']}/{BASELINE_CONFIG['ema_slow']}, "
        f"ATR {BASELINE_CONFIG['trailing_stop_atr_multiplier']}x, "
        f"ADX>{BASELINE_CONFIG['adx_min_strength']}")

    # Phase 0: Load all data
    write_section("PHASE 0: DATA LOADING")
    data_core = {}
    data_all = {}

    for symbol in CORE_SYMBOLS:
        df = load_symbol_data(symbol)
        if df is not None:
            data_core[symbol] = df
            data_all[symbol] = df
            log(f"  {symbol}: {len(df)} candles")
        gc.collect()

    for symbol in EXTRA_SYMBOLS:
        df = load_symbol_data(symbol)
        if df is not None:
            data_all[symbol] = df
            log(f"  {symbol}: {len(df)} candles")
        else:
            log(f"  {symbol}: SKIPPED (download failed)")
        gc.collect()

    log(f"\n  Loaded {len(data_core)} core + {len(data_all) - len(data_core)} extra symbols")

    if len(data_core) == 0:
        log("FATAL: No core symbol data. Cannot proceed.")
        return

    results = []

    # Test 1
    write_section("TEST 1: PARAMETER SENSITIVITY")
    try:
        r = test_parameter_sensitivity(data_core)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Parameter Sensitivity", False, 0.0, str(e)))
    gc.collect()

    # Test 2
    write_section("TEST 2: ROLLING WALK-FORWARD")
    try:
        r = test_rolling_walk_forward(data_core)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Rolling Walk-Forward", False, 0.0, str(e)))
    gc.collect()

    # Test 3
    write_section("TEST 3: SYNTHETIC STRESS TESTING")
    try:
        r = test_synthetic_stress(data_core)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Synthetic Stress", False, 0.0, str(e)))
    gc.collect()

    # Test 4
    write_section("TEST 4: NOISE INJECTION (ANTI-OVERFIT)")
    try:
        r = test_noise_injection(data_core)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Noise Injection", False, 0.0, str(e)))
    gc.collect()

    # Test 5
    write_section("TEST 5: CROSS-ASSET GENERALIZATION")
    try:
        r = test_cross_asset(data_all)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Cross-Asset Generalization", False, 0.0, str(e)))
    gc.collect()

    # Test 6
    write_section("TEST 6: MINIMUM ACTIVITY CHECK")
    try:
        r = test_minimum_activity(data_core)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Minimum Activity", False, 0.0, str(e)))
    gc.collect()

    # Test 7
    write_section("TEST 7: BOOTSTRAP CONFIDENCE INTERVALS")
    try:
        r = test_bootstrap_confidence(data_core)
        results.append(r)
        log(f"\n  RESULT: {'PASS' if r.passed else 'FAIL'} — {r.details}")
    except Exception as e:
        log(f"  ERROR: {e}")
        results.append(TestResult("Bootstrap Confidence", False, 0.0, str(e)))
    gc.collect()

    # Final verdict
    write_section("FINAL VERDICT")
    verdict, recommendation, score = compute_final_verdict(results)

    log(f"\n  {'Test':<30} {'Result':<8} {'Score':<8}")
    log(f"  {'-' * 50}")
    for r in results:
        log(f"  {r.name:<30} {'PASS' if r.passed else 'FAIL':<8} {r.score:.2f}")
    log(f"  {'-' * 50}")
    passed = sum(1 for r in results if r.passed)
    log(f"  {'OVERALL':<30} {verdict:<16} {score:.2f}")
    log(f"  Passed: {passed}/{len(results)} tests")
    log(f"\n  {recommendation}")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    log(f"\nRobustness test complete! Total time: {minutes}m {seconds}s")
    log(f"Full report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
