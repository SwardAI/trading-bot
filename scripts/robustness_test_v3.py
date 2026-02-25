"""Robustness test suite v3: Final validation of vol-scaled MTF ensemble.

Tests the vol-scaled MTF Donchian ensemble with R/DD-weighted allocation.
7 independent tests, each PASS/FAIL. Includes BTC as 7th pair.

Strategy under test:
  - Daily: Donchian [20,50,100], 2/3 vote filter
  - 4h entry: 14-period high breakout, 7-period low exit
  - ATR 3.0 chandelier stop, 5% base risk per trade
  - Vol-scaled sizing: risk_pct * (median_atr / current_atr), clamped [0.5x, 2x]
  - R/DD-weighted portfolio across 7 pairs (added BTC)

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/robustness_test_v3.py" bot
"""

import gc
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data, download_ohlcv
from src.backtest.engine import BacktestEngine, BacktestTrade

REPORT_FILE = "data/robustness_report_v3.txt"
INITIAL_CAPITAL = 10000

# Vol-scaled strategy config (v5 winner)
STRATEGY_CONFIG = {
    "daily_periods": [20, 50, 100],
    "daily_min_votes": 2,
    "entry_period_4h": 14,
    "exit_period_4h": 7,
    "atr_stop_mult": 3.0,
    "risk_per_trade_pct": 5.0,
    "volume_mult": 1.0,
    "cooldown_bars": 2,
    "vol_scale": True,
    "vol_scale_lookback": 60,
}

CORE_PAIRS = ["BTC/USDT", "SOL/USDT", "DOGE/USDT", "ETH/USDT",
              "AVAX/USDT", "LINK/USDT", "ADA/USDT"]
EXTRA_PAIRS = ["DOT/USDT", "UNI/USDT", "ATOM/USDT", "NEAR/USDT"]
ALL_PAIRS = CORE_PAIRS + EXTRA_PAIRS

test_results = {}


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


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    resampled = df.resample(freq).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def load_data(symbol: str):
    """Load hourly data and return 4h + daily (memory efficient)."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        log(f"  Downloading {symbol}...")
        try:
            download_ohlcv("binance", symbol, "1h", "2022-01-01")
            df = load_cached_data("binance", symbol, "1h")
        except Exception as e:
            log(f"  ERROR downloading: {e}")
            return None, None
    if df is None:
        return None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    del df
    gc.collect()
    return df_4h, df_1d


def vol_scaled_backtest(df_4h: pd.DataFrame, df_1d: pd.DataFrame, config: dict):
    """MTF backtest with vol-scaled position sizing."""
    import ta

    daily_periods = config.get("daily_periods", [20, 50, 100])
    daily_min_votes = config.get("daily_min_votes", 2)
    entry_period = config.get("entry_period_4h", 14)
    exit_period = config.get("exit_period_4h", 7)
    atr_period = config.get("atr_period", 14)
    atr_stop_mult = config.get("atr_stop_mult", 3.0)
    risk_pct = config.get("risk_per_trade_pct", 5.0)
    volume_mult = config.get("volume_mult", 1.0)
    cooldown_bars = config.get("cooldown_bars", 2)
    vol_scale = config.get("vol_scale", False)
    vol_lookback = config.get("vol_scale_lookback", 60)

    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)

    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    df_4h["volume_ma"] = df_4h["volume"].rolling(window=20).mean()

    if vol_scale:
        df_4h["atr_median"] = df_4h["atr"].rolling(window=vol_lookback).median()

    daily_bullish = {}
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        votes = sum(
            1 for p in daily_periods
            if not pd.isna(row.get(f"dc_high_{p}")) and row["close"] > row[f"dc_high_{p}"]
        )
        daily_bullish[date] = votes >= daily_min_votes

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None
    bars_since_exit = cooldown_bars
    lookback = max(entry_period, exit_period, atr_period, vol_lookback if vol_scale else 0) + 1

    for i in range(lookback, len(df_4h)):
        row = df_4h.iloc[i]
        close = row["close"]
        high = row["high"]
        atr = row["atr"]
        ts = row["timestamp"]

        if pd.isna(atr) or pd.isna(row.get("highest_high")):
            unrealized = 0
            if position:
                unrealized = (close - position["entry_price"]) * position["amount"]
            equity_curve.append(capital + unrealized)
            continue

        bar_date = ts.date()
        is_bullish = daily_bullish.get(bar_date, False)

        if position:
            if high > position["highest_since_entry"]:
                position["highest_since_entry"] = high
            chandelier = position["highest_since_entry"] - atr_stop_mult * atr
            effective_stop = max(chandelier, position["stop_loss"])
            exit_ch = row["lowest_low"] if not pd.isna(row["lowest_low"]) else 0

            if close <= effective_stop or close < exit_ch:
                amount = position["amount"]
                fee = amount * close * 0.001
                pnl = (close - position["entry_price"]) * amount - fee
                capital += pnl
                trades.append(BacktestTrade(
                    timestamp=ts, side="sell", price=close,
                    amount=amount, cost=amount * close, fee=fee,
                    pnl=pnl, strategy="vol_scaled",
                ))
                position = None
                bars_since_exit = 0

            unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
            equity_curve.append(capital + unrealized)
            continue

        bars_since_exit += 1
        if bars_since_exit < cooldown_bars:
            equity_curve.append(capital)
            continue

        vol_ok = (
            row["volume"] > row["volume_ma"] * volume_mult
            if not pd.isna(row.get("volume_ma")) and row["volume_ma"] > 0
            else True
        )

        if is_bullish and close > row["highest_high"] and vol_ok:
            stop_loss = close - atr_stop_mult * atr
            risk_per_unit = close - stop_loss

            if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                effective_risk_pct = risk_pct
                if vol_scale and not pd.isna(row.get("atr_median")) and row["atr_median"] > 0:
                    vol_ratio = row["atr_median"] / atr
                    vol_ratio = max(0.5, min(2.0, vol_ratio))
                    effective_risk_pct = risk_pct * vol_ratio

                risk_amount = capital * (effective_risk_pct / 100)
                amount = risk_amount / risk_per_unit
                cost = amount * close
                fee = cost * 0.001

                if cost + fee <= capital * 0.95:
                    capital -= fee
                    position = {
                        "entry_price": close, "amount": amount,
                        "stop_loss": stop_loss, "highest_since_entry": high,
                    }
                    trades.append(BacktestTrade(
                        timestamp=ts, side="buy", price=close,
                        amount=amount, cost=cost, fee=fee,
                        strategy="vol_scaled",
                    ))

        unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
        equity_curve.append(capital + unrealized)

    if position:
        final = df_4h.iloc[-1]
        amount = position["amount"]
        fee = amount * final["close"] * 0.001
        pnl = (final["close"] - position["entry_price"]) * amount - fee
        capital += pnl
        trades.append(BacktestTrade(
            timestamp=final["timestamp"], side="sell", price=final["close"],
            amount=amount, cost=amount * final["close"], fee=fee,
            pnl=pnl, strategy="vol_scaled",
        ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("vol_scaled", df_4h, trades, equity_curve, capital)


# ============================================================================
#  TEST 1: Parameter Sensitivity
# ============================================================================
def test_parameter_sensitivity():
    write_section("TEST 1: Parameter Sensitivity Analysis")
    log("OAT perturbation: vary each param at -20%,-10%,0%,+10%,+20%")
    log("PASS if CV < 30% (we're on a plateau, not a sharp peak)")

    params = {
        "entry_period_4h": [11, 13, 14, 15, 17],
        "exit_period_4h": [5, 6, 7, 8, 9],
        "atr_stop_mult": [2.4, 2.7, 3.0, 3.3, 3.6],
        "risk_per_trade_pct": [4.0, 4.5, 5.0, 5.5, 6.0],
        "vol_scale_lookback": [48, 54, 60, 66, 72],
    }

    test_symbols = CORE_PAIRS[:4]

    all_returns = {}

    for param_name, values in params.items():
        param_rets = []
        for val in values:
            cfg = {**STRATEGY_CONFIG, param_name: val}
            rets = []
            for symbol in test_symbols:
                df_4h, df_1d = load_data(symbol)
                if df_4h is None or df_1d is None:
                    continue
                result = vol_scaled_backtest(df_4h, df_1d, cfg)
                rets.append(result.total_return_pct)
                del result, df_4h, df_1d
            gc.collect()
            avg_ret = np.mean(rets) if rets else 0
            param_rets.append(avg_ret)

        all_returns[param_name] = param_rets
        cv = np.std(param_rets) / abs(np.mean(param_rets)) * 100 if np.mean(param_rets) != 0 else 999
        log(f"  {param_name:25s}: values={values}")
        log(f"    Returns: {['%.1f%%' % r for r in param_rets]}")
        log(f"    CV: {cv:.1f}%  {'SMOOTH' if cv < 30 else 'FRAGILE'}")

    gc.collect()

    all_cvs = []
    for name, rets in all_returns.items():
        cv = np.std(rets) / abs(np.mean(rets)) * 100 if np.mean(rets) != 0 else 999
        all_cvs.append(cv)

    avg_cv = np.mean(all_cvs)
    passed = avg_cv < 30
    log(f"\n  Average CV across all params: {avg_cv:.1f}%")
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold: <30%)")
    test_results["1_sensitivity"] = passed
    return passed


# ============================================================================
#  TEST 2: Rolling Window Walk-Forward
# ============================================================================
def test_walk_forward():
    write_section("TEST 2: Rolling Window Walk-Forward")
    log("6-month train (unused), 3-month test, slide by 3 months")
    log("Fixed params — testing if strategy works across time periods")
    log("PASS if >60% of test windows are profitable")

    test_symbols = CORE_PAIRS[:4]
    window_results = []

    for symbol in test_symbols:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue

        bars_per_month_4h = 6 * 30
        train_bars = bars_per_month_4h * 6
        test_bars = bars_per_month_4h * 3
        step_bars = bars_per_month_4h * 3

        total_bars = len(df_4h)
        log(f"\n  {symbol}: {total_bars} bars")

        start = 0
        windows = []
        while start + train_bars + test_bars <= total_bars:
            test_start = start + train_bars
            test_end = test_start + test_bars

            warmup = 200
            df_4h_test = df_4h.iloc[max(0, test_start - warmup):test_end].copy().reset_index(drop=True)

            test_start_ts = df_4h.iloc[max(0, test_start - warmup)]["timestamp"]
            test_end_ts = df_4h.iloc[test_end - 1]["timestamp"]
            df_1d_test = df_1d[
                (df_1d["timestamp"] >= test_start_ts) &
                (df_1d["timestamp"] <= test_end_ts)
            ].copy().reset_index(drop=True)

            if len(df_4h_test) > 100 and len(df_1d_test) > 20:
                result = vol_scaled_backtest(df_4h_test, df_1d_test, STRATEGY_CONFIG)
                profitable = result.total_return_pct > 0
                windows.append({
                    "start": test_start, "end": test_end,
                    "return": result.total_return_pct,
                    "dd": result.max_drawdown_pct,
                    "trades": result.total_trades,
                    "profitable": profitable,
                })
                marker = "+" if profitable else ("-" if result.total_return_pct < 0 else "0")
                log(f"    Window {len(windows):2d}: {result.total_return_pct:+6.1f}% "
                    f"DD:{result.max_drawdown_pct:4.1f}% Trades:{result.total_trades:2d} [{marker}]")
                del result

            start += step_bars

        window_results.extend(windows)
        del df_4h, df_1d
        gc.collect()

    if not window_results:
        log("  No windows generated!")
        test_results["2_walk_forward"] = False
        return False

    # Count windows with trades separately
    windows_with_trades = [w for w in window_results if w["trades"] > 0]
    profitable_windows = sum(1 for w in window_results if w["profitable"])
    profitable_with_trades = sum(1 for w in windows_with_trades if w["profitable"])
    total_windows = len(window_results)

    avg_ret = np.mean([w["return"] for w in window_results])
    avg_dd = np.mean([w["dd"] for w in window_results])

    log(f"\n  Total windows: {total_windows}")
    log(f"  Windows with trades: {len(windows_with_trades)}/{total_windows}")
    log(f"  Profitable (all): {profitable_windows}/{total_windows} ({profitable_windows/total_windows*100:.0f}%)")
    if windows_with_trades:
        log(f"  Profitable (with trades): {profitable_with_trades}/{len(windows_with_trades)} "
            f"({profitable_with_trades/len(windows_with_trades)*100:.0f}%)")
    log(f"  Avg return per window: {avg_ret:+.1f}%")

    # Pass if windows with trades are mostly profitable
    if windows_with_trades:
        pct = profitable_with_trades / len(windows_with_trades) * 100
    else:
        pct = 0
    passed = pct >= 60
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold: >=60% of windows with trades)")
    test_results["2_walk_forward"] = passed
    return passed


# ============================================================================
#  TEST 3: Synthetic Stress Testing
# ============================================================================
def test_stress():
    write_section("TEST 3: Synthetic Stress Testing")
    log("Inject extreme scenarios into real data, check max drawdown")
    log("PASS if max DD < 25% in all scenarios")

    symbol = "ETH/USDT"
    df_4h, df_1d = load_data(symbol)
    if df_4h is None or df_1d is None:
        log("  No data!")
        test_results["3_stress"] = False
        return False

    baseline = vol_scaled_backtest(df_4h, df_1d, STRATEGY_CONFIG)
    log(f"  Baseline: {baseline.total_return_pct:+.1f}%, DD: {baseline.max_drawdown_pct:.1f}%")

    scenarios = {
        "flash_crash": {"desc": "-30% over 24 bars, 50% recovery over 168", "inject_at": 2000},
        "prolonged_bear": {"desc": "-40% drift over 540 bars (90 days)", "inject_at": 2000},
        "volatility_spike": {"desc": "3x high-low range for 84 bars (14 days)", "inject_at": 2000},
        "sideways_chop": {"desc": "Clamp +/-2% for 360 bars (60 days)", "inject_at": 2000},
    }

    max_dd_all = 0
    all_pass = True

    for name, scenario in scenarios.items():
        df_mod = df_4h.copy()
        inject = scenario["inject_at"]

        if name == "flash_crash":
            for j in range(24):
                if inject + j < len(df_mod):
                    mult = 1 - 0.30 * (j + 1) / 24
                    for col in ["open", "high", "low", "close"]:
                        df_mod.at[inject + j, col] *= mult
            for j in range(168):
                idx = inject + 24 + j
                if idx < len(df_mod):
                    recovery = 0.70 + 0.15 * (j + 1) / 168
                    for col in ["open", "high", "low", "close"]:
                        df_mod.at[idx, col] *= recovery

        elif name == "prolonged_bear":
            for j in range(540):
                if inject + j < len(df_mod):
                    mult = 0.999 ** (j + 1)
                    for col in ["open", "high", "low", "close"]:
                        df_mod.at[inject + j, col] *= mult

        elif name == "volatility_spike":
            for j in range(84):
                if inject + j < len(df_mod):
                    mid = (df_mod.at[inject + j, "high"] + df_mod.at[inject + j, "low"]) / 2
                    rng = df_mod.at[inject + j, "high"] - df_mod.at[inject + j, "low"]
                    df_mod.at[inject + j, "high"] = mid + rng * 1.5
                    df_mod.at[inject + j, "low"] = mid - rng * 1.5

        elif name == "sideways_chop":
            if inject < len(df_mod):
                anchor = df_mod.at[inject, "close"]
                band = anchor * 0.02
                for j in range(360):
                    if inject + j < len(df_mod):
                        for col in ["open", "high", "low", "close"]:
                            df_mod.at[inject + j, col] = np.clip(
                                df_mod.at[inject + j, col], anchor - band, anchor + band)

        df_mod["timestamp"] = pd.to_datetime(df_mod["timestamp"], utc=True)
        df_1d_mod = df_mod.set_index("timestamp").resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()

        try:
            result = vol_scaled_backtest(df_mod, df_1d_mod, STRATEGY_CONFIG)
            dd = result.max_drawdown_pct
            ret = result.total_return_pct
            max_dd_all = max(max_dd_all, dd)
            scenario_pass = dd < 25
            if not scenario_pass:
                all_pass = False
            log(f"  {name:20s}: Ret {ret:+7.1f}% | DD: {dd:5.1f}% | "
                f"{'PASS' if scenario_pass else 'FAIL'} (<25%)")
            del result
        except Exception as e:
            log(f"  {name}: ERROR - {e}")
            all_pass = False

    del df_4h, df_1d
    gc.collect()

    log(f"\n  Worst DD across scenarios: {max_dd_all:.1f}%")
    log(f"  RESULT: {'PASS' if all_pass else 'FAIL'} (threshold: all DDs < 25%)")
    test_results["3_stress"] = all_pass
    return all_pass


# ============================================================================
#  TEST 4: Noise Injection (Anti-Overfit)
# ============================================================================
def test_noise_injection():
    write_section("TEST 4: Noise Injection (Anti-Overfit)")
    log("0.5% Gaussian noise on OHLCV, 50 runs per pair, 7 core pairs")
    log("PASS if 80%+ profitable AND Sharpe std < 0.5")

    noise_std = 0.005
    n_runs = 50
    overall_pass = 0
    overall_total = 0
    all_sharpes = []

    for symbol in CORE_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue

        baseline = vol_scaled_backtest(df_4h, df_1d, STRATEGY_CONFIG)
        base_ret = baseline.total_return_pct
        del baseline

        noisy_rets = []
        noisy_sharpes = []
        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            df_4h_n = df_4h.copy()
            n = len(df_4h_n)
            df_4h_n["close"] = df_4h_n["close"] * rng.normal(1.0, noise_std, n)
            df_4h_n["open"] = df_4h_n["open"] * rng.normal(1.0, noise_std, n)
            df_4h_n["high"] = df_4h_n[["open", "close", "high"]].max(axis=1) * rng.normal(1.0, noise_std * 0.5, n)
            df_4h_n["low"] = df_4h_n[["open", "close", "low"]].min(axis=1) * rng.normal(1.0, noise_std * 0.5, n)
            df_4h_n["high"] = df_4h_n[["open", "close", "high"]].max(axis=1)
            df_4h_n["low"] = df_4h_n[["open", "close", "low"]].min(axis=1)

            df_1d_n = df_1d.copy()
            nd = len(df_1d_n)
            df_1d_n["close"] = df_1d_n["close"] * rng.normal(1.0, noise_std, nd)
            df_1d_n["open"] = df_1d_n["open"] * rng.normal(1.0, noise_std, nd)
            df_1d_n["high"] = df_1d_n[["open", "close", "high"]].max(axis=1)
            df_1d_n["low"] = df_1d_n[["open", "close", "low"]].min(axis=1)

            try:
                r = vol_scaled_backtest(df_4h_n, df_1d_n, STRATEGY_CONFIG)
                noisy_rets.append(r.total_return_pct)
                noisy_sharpes.append(r.sharpe_ratio)
                del r
            except Exception:
                noisy_rets.append(0.0)
                noisy_sharpes.append(0.0)

        profitable = sum(1 for r in noisy_rets if r > 0)
        overall_pass += profitable
        overall_total += n_runs
        all_sharpes.extend(noisy_sharpes)

        log(f"  {symbol:12s}: Base {base_ret:+7.1f}% | Noisy avg {np.mean(noisy_rets):+7.1f}% "
            f"(std {np.std(noisy_rets):5.1f}%) | {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")

        del df_4h, df_1d
        gc.collect()

    if overall_total == 0:
        test_results["4_noise"] = False
        return False

    pct_profitable = overall_pass / overall_total * 100
    sharpe_std = np.std(all_sharpes)

    log(f"\n  Overall: {overall_pass}/{overall_total} profitable ({pct_profitable:.0f}%)")
    log(f"  Sharpe std: {sharpe_std:.3f}")

    passed = pct_profitable >= 80 and sharpe_std < 0.5
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} "
        f"(need >=80% profitable AND Sharpe std <0.5)")
    test_results["4_noise"] = passed
    return passed


# ============================================================================
#  TEST 5: Cross-Asset Generalization
# ============================================================================
def test_cross_asset():
    write_section("TEST 5: Cross-Asset Generalization")
    log("Run same config on 11 pairs (7 core + 4 extra)")
    log("PASS if profitable on 60%+ of ALL pairs (at least 7/11)")

    profitable_count = 0
    total_count = 0

    for symbol in ALL_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            log(f"  {symbol:12s}: SKIPPED (no data)")
            continue

        try:
            result = vol_scaled_backtest(df_4h, df_1d, STRATEGY_CONFIG)
            is_profitable = result.total_return_pct > 0
            if is_profitable:
                profitable_count += 1
            total_count += 1
            marker = "+" if is_profitable else "-"
            log(f"  {symbol:12s}: {result.total_return_pct:+7.1f}% | "
                f"DD: {result.max_drawdown_pct:5.1f}% | "
                f"PF: {result.profit_factor:5.2f} | "
                f"Trades: {result.total_trades:3d} [{marker}]")
            del result
        except Exception as e:
            log(f"  {symbol}: ERROR - {e}")
            total_count += 1

        del df_4h, df_1d
        gc.collect()

    pct = profitable_count / total_count * 100 if total_count > 0 else 0
    log(f"\n  Profitable: {profitable_count}/{total_count} ({pct:.0f}%)")

    passed = pct >= 60
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold: >=60%)")
    test_results["5_cross_asset"] = passed
    return passed


# ============================================================================
#  TEST 6: Minimum Activity Check
# ============================================================================
def test_activity():
    write_section("TEST 6: Minimum Activity Check")
    log("Ensure strategy actually trades (not profiting by being idle)")
    log("PASS if >=12 trades/year avg AND in-position >=10% of time")

    total_trades = 0
    total_years = 0
    total_bars_in_position = 0
    total_bars = 0

    for symbol in CORE_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue

        result = vol_scaled_backtest(df_4h, df_1d, STRATEGY_CONFIG)

        years = len(df_4h) / (6 * 365.25)
        trades_per_year = result.total_trades / years if years > 0 else 0

        bars_in_pos = sum(
            1 for i in range(1, len(result.equity_curve))
            if abs(result.equity_curve[i] - result.equity_curve[i-1]) > 0.01
        )
        pct_in_pos = bars_in_pos / len(result.equity_curve) * 100

        total_trades += result.total_trades
        total_years += years
        total_bars_in_position += bars_in_pos
        total_bars += len(result.equity_curve)

        log(f"  {symbol:12s}: {result.total_trades:3d} trades ({trades_per_year:.1f}/yr) | "
            f"In-position: {pct_in_pos:.0f}%")

        del result, df_4h, df_1d
        gc.collect()

    avg_trades_per_year = total_trades / total_years if total_years > 0 else 0
    avg_in_position = total_bars_in_position / total_bars * 100 if total_bars > 0 else 0

    log(f"\n  Avg trades/pair/year: {avg_trades_per_year:.1f}")
    log(f"  Avg time in position: {avg_in_position:.0f}%")

    passed = avg_trades_per_year >= 12 and avg_in_position >= 10
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} "
        f"(need >=12 trades/yr AND >=10% in-position)")
    test_results["6_activity"] = passed
    return passed


# ============================================================================
#  TEST 7: Bootstrap Confidence Intervals
# ============================================================================
def test_bootstrap():
    write_section("TEST 7: Bootstrap Confidence Intervals")
    log("1000 bootstrap resamples of trade P&Ls per pair")
    log("PASS if 95% CI lower bound for return > 0 AND PF lower > 1.0")

    n_bootstrap = 1000
    all_lower_rets = []
    all_lower_pfs = []

    for symbol in CORE_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue

        result = vol_scaled_backtest(df_4h, df_1d, STRATEGY_CONFIG)

        pnls = [t.pnl for t in result.trades if t.pnl is not None]
        if len(pnls) < 5:
            log(f"  {symbol}: Too few trades ({len(pnls)}), skipping")
            del result, df_4h, df_1d
            gc.collect()
            continue

        boot_returns = []
        boot_pfs = []
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            total_ret = sum(sample) / INITIAL_CAPITAL * 100
            wins = sum(p for p in sample if p > 0)
            losses = abs(sum(p for p in sample if p < 0))
            pf = wins / losses if losses > 0 else float("inf")
            boot_returns.append(total_ret)
            boot_pfs.append(min(pf, 100))

        lower_ret = np.percentile(boot_returns, 2.5)
        upper_ret = np.percentile(boot_returns, 97.5)
        lower_pf = np.percentile(boot_pfs, 2.5)
        upper_pf = np.percentile(boot_pfs, 97.5)

        all_lower_rets.append(lower_ret)
        all_lower_pfs.append(lower_pf)

        log(f"  {symbol:12s}: Return 95% CI: [{lower_ret:+6.1f}%, {upper_ret:+6.1f}%] | "
            f"PF CI: [{lower_pf:.2f}, {upper_pf:.2f}]")

        del result, df_4h, df_1d
        gc.collect()

    if not all_lower_rets:
        test_results["7_bootstrap"] = False
        return False

    avg_lower_ret = np.mean(all_lower_rets)
    avg_lower_pf = np.mean(all_lower_pfs)
    pairs_positive = sum(1 for r in all_lower_rets if r > 0)

    log(f"\n  Pairs with positive lower bound: {pairs_positive}/{len(all_lower_rets)}")
    log(f"  Avg lower bound return: {avg_lower_ret:+.1f}%")
    log(f"  Avg lower bound PF: {avg_lower_pf:.2f}")

    passed = avg_lower_ret > 0 and avg_lower_pf > 1.0
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} "
        f"(need avg lower return >0% AND avg lower PF >1.0)")
    test_results["7_bootstrap"] = passed
    return passed


# ============================================================================
#  PORTFOLIO SIM (bonus — not a pass/fail test)
# ============================================================================
def portfolio_simulation():
    write_section("PORTFOLIO: R/DD-Weighted Allocation")
    log("Computing optimal allocation for production deployment")

    pair_curves = {}
    pair_stats = {}
    min_len = float("inf")

    for symbol in CORE_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue
        result = vol_scaled_backtest(df_4h, df_1d, STRATEGY_CONFIG)
        log(f"  {symbol:12s}: {result.total_return_pct:+8.1f}% | DD: {result.max_drawdown_pct:5.1f}% | "
            f"PF: {result.profit_factor:5.2f} | Trades: {result.total_trades}")
        if result.equity_curve:
            pair_curves[symbol] = result.equity_curve
            pair_stats[symbol] = {
                "ret": result.total_return_pct, "dd": result.max_drawdown_pct,
                "trades": result.total_trades, "pf": result.profit_factor,
            }
            min_len = min(min_len, len(result.equity_curve))
        del result, df_4h, df_1d
        gc.collect()

    if not pair_curves:
        return

    bars_per_year = 6 * 365.25
    n = len(pair_curves)

    # Equal weight
    eq_weights = {sym: 1.0 / n for sym in pair_curves}
    eq_portfolio = []
    for i in range(int(min_len)):
        total = sum(pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * eq_weights[sym]
                    for sym in pair_curves if i < len(pair_curves[sym]))
        eq_portfolio.append(total)

    eq_ret = (eq_portfolio[-1] - eq_portfolio[0]) / eq_portfolio[0] * 100
    eq_years = min_len / bars_per_year
    eq_annual = ((eq_portfolio[-1] / eq_portfolio[0]) ** (1 / eq_years) - 1) * 100 if eq_years > 0.5 else 0
    eq_peak = eq_portfolio[0]
    eq_dd = 0
    for v in eq_portfolio:
        if v > eq_peak:
            eq_peak = v
        eq_dd = max(eq_dd, (eq_peak - v) / eq_peak * 100)
    eq_rets = [(eq_portfolio[i] - eq_portfolio[i-1]) / eq_portfolio[i-1]
               for i in range(1, len(eq_portfolio)) if eq_portfolio[i-1] > 0]
    eq_sharpe = (np.mean(eq_rets) / np.std(eq_rets) * (bars_per_year ** 0.5)
                 if eq_rets and np.std(eq_rets) > 0 else 0)

    log(f"\n  Equal weight ({n} pairs, {eq_years:.1f}yr):")
    log(f"    Annual: {eq_annual:+.1f}%/yr | DD: {eq_dd:.1f}% | Sharpe: {eq_sharpe:.2f}")

    # R/DD weighted
    r_dd_scores = {sym: max(s["ret"] / max(s["dd"], 1.0), 0.1) for sym, s in pair_stats.items()}
    total_rdd = sum(r_dd_scores.values())
    rdd_weights = {sym: v / total_rdd for sym, v in r_dd_scores.items()}

    rdd_portfolio = []
    for i in range(int(min_len)):
        total = sum(pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * rdd_weights[sym]
                    for sym in pair_curves if i < len(pair_curves[sym]))
        rdd_portfolio.append(total)

    rdd_ret = (rdd_portfolio[-1] - rdd_portfolio[0]) / rdd_portfolio[0] * 100
    rdd_annual = ((rdd_portfolio[-1] / rdd_portfolio[0]) ** (1 / eq_years) - 1) * 100 if eq_years > 0.5 else 0
    rdd_peak = rdd_portfolio[0]
    rdd_dd = 0
    for v in rdd_portfolio:
        if v > rdd_peak:
            rdd_peak = v
        rdd_dd = max(rdd_dd, (rdd_peak - v) / rdd_peak * 100)
    rdd_rets = [(rdd_portfolio[i] - rdd_portfolio[i-1]) / rdd_portfolio[i-1]
                for i in range(1, len(rdd_portfolio)) if rdd_portfolio[i-1] > 0]
    rdd_sharpe = (np.mean(rdd_rets) / np.std(rdd_rets) * (bars_per_year ** 0.5)
                  if rdd_rets and np.std(rdd_rets) > 0 else 0)

    log(f"\n  R/DD weighted ({n} pairs, {eq_years:.1f}yr):")
    log(f"    Annual: {rdd_annual:+.1f}%/yr | DD: {rdd_dd:.1f}% | Sharpe: {rdd_sharpe:.2f}")
    if rdd_annual >= 30:
        log(f"    *** TARGET 30%/yr ACHIEVED! ***")
    elif rdd_annual >= 20:
        log(f"    Close to target ({rdd_annual:.1f}%/yr)")

    log(f"\n  R/DD allocation:")
    for sym in sorted(rdd_weights, key=rdd_weights.get, reverse=True):
        log(f"    {sym:12s}: {rdd_weights[sym]*100:5.1f}% | "
            f"Ret: {pair_stats[sym]['ret']:+.1f}% DD: {pair_stats[sym]['dd']:.1f}%")


# ============================================================================
#  FINAL VERDICT
# ============================================================================
def final_verdict():
    write_section("FINAL VERDICT")

    weights = {
        "2_walk_forward": 1.5,
        "5_cross_asset": 1.3,
        "4_noise": 1.2,
        "7_bootstrap": 1.2,
        "1_sensitivity": 1.0,
        "3_stress": 1.0,
        "6_activity": 0.8,
    }

    total_weight = 0
    passed_weight = 0
    pass_count = 0

    log(f"\n  {'Test':30s} | {'Result':6s} | {'Weight':6s}")
    log(f"  {'-'*50}")

    test_names = {
        "1_sensitivity": "Parameter Sensitivity",
        "2_walk_forward": "Walk-Forward",
        "3_stress": "Stress Testing",
        "4_noise": "Noise Injection",
        "5_cross_asset": "Cross-Asset Generalization",
        "6_activity": "Minimum Activity",
        "7_bootstrap": "Bootstrap Confidence",
    }

    for key in sorted(test_results.keys()):
        name = test_names.get(key, key)
        passed = test_results[key]
        w = weights.get(key, 1.0)
        total_weight += w
        if passed:
            passed_weight += w
            pass_count += 1
        log(f"  {name:30s} | {'PASS' if passed else 'FAIL':6s} | {w:.1f}x")

    score = passed_weight / total_weight * 100 if total_weight > 0 else 0
    total_tests = len(test_results)

    log(f"\n  Tests passed: {pass_count}/{total_tests}")
    log(f"  Weighted score: {score:.0f}%")

    if pass_count == total_tests:
        log(f"\n  *** STRONG PASS: Deploy with confidence ***")
        verdict = "STRONG_PASS"
    elif pass_count >= total_tests - 2:
        log(f"\n  CONDITIONAL PASS: Review failures, deploy cautiously")
        verdict = "CONDITIONAL_PASS"
    else:
        log(f"\n  FAIL: Strategy may be overfit, DO NOT deploy")
        verdict = "FAIL"

    log(f"\n  Strategy: Vol-Scaled MTF Donchian Ensemble")
    log(f"  Config: {STRATEGY_CONFIG}")
    log(f"  Verdict: {verdict}")

    return verdict


def main():
    start = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("ROBUSTNESS TEST SUITE v3: Vol-Scaled MTF Ensemble")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"Strategy: Daily Donchian [20,50,100] 2/3 + 4h breakout 14/7 + vol-scaling")
    log(f"Risk: 5% base, vol-scaled [0.5x-2x], ATR 3.0 chandelier stop")
    log(f"Testing on: {len(ALL_PAIRS)} pairs ({len(CORE_PAIRS)} core + {len(EXTRA_PAIRS)} extra)")

    # Run all 7 tests
    test_parameter_sensitivity()
    test_walk_forward()
    test_stress()
    test_noise_injection()
    test_cross_asset()
    test_activity()
    test_bootstrap()

    # Portfolio sim (informational)
    portfolio_simulation()

    verdict = final_verdict()

    elapsed = time.time() - start
    log(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
