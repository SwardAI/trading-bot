"""Strategy research v10: Combined long+short robustness & optimization.

Validates the combined MTF Donchian system (long in bull, short in bear)
with 8 robustness tests + portfolio allocation optimization.

V9 showed +34.8%/yr avg across 11 pairs — this script tests whether
those numbers are real or overfit.

Usage (local):
    py scripts/strategy_research_v10_combined_robustness.py

Usage (droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v10_combined_robustness.py" bot
"""

import gc
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data

REPORT_FILE = "data/strategy_research_v10_combined_robustness_report.txt"
FUNDING_DATA_DIR = Path("data/historical/funding_rates")
INITIAL_CAPITAL = 10000

CORE_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
              "AVAX/USDT", "LINK/USDT", "ADA/USDT"]
EXTRA_PAIRS = ["DOT/USDT", "ATOM/USDT", "NEAR/USDT", "UNI/USDT"]
ALL_PAIRS = CORE_PAIRS + EXTRA_PAIRS

# Fee models
SPOT_TAKER_FEE = 0.001      # 0.10% per trade (spot)
FUTURES_TAKER_FEE = 0.0004   # 0.04% per trade (futures)

# Baseline config — proven from v3+v9 research
BASELINE_CONFIG = {
    "daily_periods": [20, 50, 100],
    "daily_min_votes": 2,
    "entry_period_4h": 14,
    "exit_period_4h": 7,
    "atr_period": 14,
    "atr_stop_mult": 3.0,
    "risk_per_trade_pct": 5.0,
    "cooldown_bars": 2,
    "vol_scale": True,
    "vol_scale_lookback": 60,
}

test_results = {}


# ============================================================
# Utilities
# ============================================================

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
    """Load 1h data, resample to 4h + daily."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        return None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    del df
    gc.collect()
    return df_4h, df_1d


def load_funding_rates(symbol: str) -> dict:
    """Load cached funding rates → dict mapping date -> avg daily rate."""
    safe_symbol = symbol.replace("/", "_")
    cache_file = FUNDING_DATA_DIR / f"binance_{safe_symbol}_funding.csv"
    if not cache_file.exists():
        return {}
    df = pd.read_csv(cache_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["funding_rate"].mean()
    return daily.to_dict()


# ============================================================
# Combined backtest engine
# ============================================================

def combined_backtest(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    config: dict,
    funding_rates: dict | None = None,
) -> dict:
    """Run long+short on same capital. Long in bull, short in bear, cash otherwise."""
    daily_periods = config.get("daily_periods", [20, 50, 100])
    daily_min_votes = config.get("daily_min_votes", 2)
    entry_period = config.get("entry_period_4h", 14)
    exit_period = config.get("exit_period_4h", 7)
    atr_period = config.get("atr_period", 14)
    atr_stop_mult = config.get("atr_stop_mult", 3.0)
    risk_pct = config.get("risk_per_trade_pct", 5.0)
    cooldown_bars = config.get("cooldown_bars", 2)
    vol_scale = config.get("vol_scale", True)
    vol_lookback = config.get("vol_scale_lookback", 60)

    # Prepare daily indicators
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)
        df_1d[f"dc_low_{p}"] = df_1d["low"].rolling(window=p).min().shift(1)

    # Prepare 4h indicators
    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["lowest_low_entry"] = df_4h["low"].rolling(window=entry_period).min().shift(1)
    df_4h["highest_high_exit"] = df_4h["high"].rolling(window=exit_period).max().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    if vol_scale:
        df_4h["atr_median"] = df_4h["atr"].rolling(window=vol_lookback).median()

    # Daily regime map
    daily_regime = {}
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        bull_votes = sum(
            1 for p in daily_periods
            if not pd.isna(row.get(f"dc_high_{p}")) and row["close"] > row[f"dc_high_{p}"]
        )
        bear_votes = sum(
            1 for p in daily_periods
            if not pd.isna(row.get(f"dc_low_{p}")) and row["close"] < row[f"dc_low_{p}"]
        )
        if bull_votes >= daily_min_votes:
            daily_regime[date] = "bull"
        elif bear_votes >= daily_min_votes:
            daily_regime[date] = "bear"
        else:
            daily_regime[date] = "neutral"

    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0
    equity_curve = [capital]
    position = None
    bars_since_exit = cooldown_bars
    long_pnl = 0
    short_pnl = 0
    trade_pnls = []
    num_trades = 0
    num_long = 0
    num_short = 0
    bars_in_position = 0
    lookback = max(entry_period, exit_period, atr_period, vol_lookback if vol_scale else 0) + 1

    for i in range(lookback, len(df_4h)):
        row = df_4h.iloc[i]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["atr"]
        ts = row["timestamp"]

        if pd.isna(atr):
            unrealized = 0
            if position:
                bars_in_position += 1
                if position["side"] == "long":
                    unrealized = (close - position["entry_price"]) * position["amount"]
                else:
                    unrealized = (position["entry_price"] - close) * position["amount"]
            equity_curve.append(capital + unrealized)
            continue

        bar_date = ts.date()
        regime = daily_regime.get(bar_date, "neutral")

        # Manage existing position
        if position:
            bars_in_position += 1

            if position["side"] == "long":
                if high > position["highest_since_entry"]:
                    position["highest_since_entry"] = high
                chandelier = position["highest_since_entry"] - atr_stop_mult * atr
                effective_stop = max(chandelier, position["stop_loss"])
                exit_ch = row["lowest_low"] if not pd.isna(row["lowest_low"]) else 0

                if close <= effective_stop or close < exit_ch:
                    amount = position["amount"]
                    fee = amount * close * SPOT_TAKER_FEE
                    pnl = (close - position["entry_price"]) * amount - fee
                    capital += pnl
                    long_pnl += pnl
                    trade_pnls.append(pnl)
                    num_trades += 1
                    num_long += 1
                    position = None
                    bars_since_exit = 0

            else:  # short
                if low < position["lowest_since_entry"]:
                    position["lowest_since_entry"] = low
                chandelier = position["lowest_since_entry"] + atr_stop_mult * atr
                effective_stop = min(chandelier, position["stop_loss"])
                exit_ch = row["highest_high_exit"] if not pd.isna(row["highest_high_exit"]) else float("inf")

                # Funding payment (every 2 bars on 4h = every 8h)
                if funding_rates and i % 2 == 0:
                    daily_rate = funding_rates.get(bar_date, 0)
                    payment = position["notional"] * daily_rate / 3
                    capital += payment

                if close >= effective_stop or close > exit_ch:
                    amount = position["amount"]
                    fee = amount * close * FUTURES_TAKER_FEE
                    pnl = (position["entry_price"] - close) * amount - fee
                    capital += pnl
                    short_pnl += pnl
                    trade_pnls.append(pnl)
                    num_trades += 1
                    num_short += 1
                    position = None
                    bars_since_exit = 0

            unrealized = 0
            if position:
                if position["side"] == "long":
                    unrealized = (close - position["entry_price"]) * position["amount"]
                else:
                    unrealized = (position["entry_price"] - close) * position["amount"]
            eq = capital + unrealized
            equity_curve.append(eq)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            continue

        bars_since_exit += 1
        if bars_since_exit < cooldown_bars:
            equity_curve.append(capital)
            if capital > peak:
                peak = capital
            continue

        # Vol-scaled risk
        effective_risk_pct = risk_pct
        if vol_scale and not pd.isna(row.get("atr_median")) and row["atr_median"] > 0:
            vol_ratio = row["atr_median"] / atr
            vol_ratio = max(0.5, min(2.0, vol_ratio))
            effective_risk_pct = risk_pct * vol_ratio

        # LONG entry
        if regime == "bull" and not pd.isna(row["highest_high"]) and close > row["highest_high"]:
            stop_loss = close - atr_stop_mult * atr
            risk_per_unit = close - stop_loss
            if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                risk_amount = capital * (effective_risk_pct / 100)
                amount = risk_amount / risk_per_unit
                cost = amount * close
                fee = cost * SPOT_TAKER_FEE
                if cost + fee <= capital * 0.95:
                    capital -= fee
                    position = {
                        "side": "long", "entry_price": close,
                        "amount": amount, "stop_loss": stop_loss,
                        "highest_since_entry": high,
                    }

        # SHORT entry
        elif regime == "bear" and not pd.isna(row["lowest_low_entry"]) and close < row["lowest_low_entry"]:
            stop_loss = close + atr_stop_mult * atr
            risk_per_unit = stop_loss - close
            if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                risk_amount = capital * (effective_risk_pct / 100)
                amount = risk_amount / risk_per_unit
                notional = amount * close
                fee = notional * FUTURES_TAKER_FEE
                if notional + fee <= capital * 0.95:
                    capital -= fee
                    position = {
                        "side": "short", "entry_price": close,
                        "amount": amount, "stop_loss": stop_loss,
                        "lowest_since_entry": low, "notional": notional,
                    }

        unrealized = 0
        if position:
            if position["side"] == "long":
                unrealized = (close - position["entry_price"]) * position["amount"]
            else:
                unrealized = (position["entry_price"] - close) * position["amount"]
        eq = capital + unrealized
        equity_curve.append(eq)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Close any open position
    if position:
        final = df_4h.iloc[-1]
        if position["side"] == "long":
            pnl = (final["close"] - position["entry_price"]) * position["amount"] - position["amount"] * final["close"] * SPOT_TAKER_FEE
            long_pnl += pnl
            num_long += 1
        else:
            pnl = (position["entry_price"] - final["close"]) * position["amount"] - position["amount"] * final["close"] * FUTURES_TAKER_FEE
            short_pnl += pnl
            num_short += 1
        capital += pnl
        trade_pnls.append(pnl)
        num_trades += 1

    total_bars = len(df_4h) - lookback
    years = total_bars / (6 * 365)
    total_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Sharpe ratio
    sharpe = 0
    if len(equity_curve) > 1:
        rets = [(equity_curve[j] - equity_curve[j - 1]) / equity_curve[j - 1]
                for j in range(1, len(equity_curve)) if equity_curve[j - 1] > 0]
        if rets and np.std(rets) > 0:
            sharpe = np.mean(rets) / np.std(rets) * (365 ** 0.5)

    # Profit factor
    wins = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))
    pf = wins / losses if losses > 0 else float("inf")

    return {
        "total_return_pct": total_return_pct,
        "ann_return_pct": total_return_pct / years if years > 0 else 0,
        "final_capital": capital,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "profit_factor": pf,
        "num_trades": num_trades,
        "num_long": num_long,
        "num_short": num_short,
        "win_rate": sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) * 100 if trade_pnls else 0,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "trade_pnls": trade_pnls,
        "years": years,
        "bars_in_position": bars_in_position,
        "total_bars": total_bars,
        "equity_curve": equity_curve,
    }


# ============================================================
# TEST 1: Parameter Sensitivity (Combined System)
# ============================================================

def test_1_sensitivity():
    write_section("TEST 1: PARAMETER SENSITIVITY (COMBINED)")
    log("OAT perturbation: vary each param at -20%,-10%,0%,+10%,+20%")
    log("PASS if avg CV < 30%\n")

    params = {
        "entry_period_4h": [11, 13, 14, 15, 17],
        "exit_period_4h":  [5,  6,  7,  8,  9],
        "atr_stop_mult":   [2.4, 2.7, 3.0, 3.3, 3.6],
        "risk_per_trade_pct": [4.0, 4.5, 5.0, 5.5, 6.0],
        "vol_scale_lookback": [48, 54, 60, 66, 72],
    }

    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    all_cvs = []

    # Preload data
    data_cache = {}
    for symbol in test_symbols:
        df_4h, df_1d = load_data(symbol)
        if df_4h is not None:
            funding = load_funding_rates(symbol)
            data_cache[symbol] = (df_4h, df_1d, funding)

    for param_name, values in params.items():
        param_rets = []
        for val in values:
            cfg = {**BASELINE_CONFIG, param_name: val}
            rets = []
            for symbol, (df_4h, df_1d, funding) in data_cache.items():
                r = combined_backtest(df_4h, df_1d, cfg, funding)
                rets.append(r["ann_return_pct"])
            avg = np.mean(rets) if rets else 0
            param_rets.append(avg)

        cv = np.std(param_rets) / abs(np.mean(param_rets)) * 100 if np.mean(param_rets) != 0 else 999
        all_cvs.append(cv)
        log(f"  {param_name:22s}: {[f'{r:+.1f}%' for r in param_rets]}")
        log(f"    CV: {cv:.1f}%  {'SMOOTH' if cv < 30 else 'FRAGILE'}")

    # Free cached data
    del data_cache
    gc.collect()

    avg_cv = np.mean(all_cvs)
    passed = avg_cv < 30
    log(f"\n  Avg CV: {avg_cv:.1f}%")
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold: <30%)")
    test_results["1_sensitivity"] = passed
    return passed


# ============================================================
# TEST 2: Walk-Forward (Combined System)
# ============================================================

def test_2_walk_forward():
    write_section("TEST 2: WALK-FORWARD (COMBINED)")
    log("6-month train (unused), 3-month test, slide by 3 months")
    log("Fixed params. PASS if >60% of test windows with trades are profitable\n")

    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    window_results = []

    for symbol in test_symbols:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        bars_per_month = 6 * 30  # approx bars per month on 4h
        train_bars = bars_per_month * 6
        test_bars = bars_per_month * 3
        step_bars = bars_per_month * 3
        warmup = 200

        total = len(df_4h)
        log(f"  {symbol}: {total} bars")

        start = 0
        while start + train_bars + test_bars <= total:
            test_start = start + train_bars
            test_end = test_start + test_bars

            df_4h_w = df_4h.iloc[max(0, test_start - warmup):test_end].copy().reset_index(drop=True)
            ts_start = df_4h.iloc[max(0, test_start - warmup)]["timestamp"]
            ts_end = df_4h.iloc[test_end - 1]["timestamp"]
            df_1d_w = df_1d[
                (df_1d["timestamp"] >= ts_start) & (df_1d["timestamp"] <= ts_end)
            ].copy().reset_index(drop=True)

            if len(df_4h_w) > 100 and len(df_1d_w) > 20:
                r = combined_backtest(df_4h_w, df_1d_w, BASELINE_CONFIG, funding)
                profitable = r["total_return_pct"] > 0
                window_results.append({
                    "symbol": symbol, "return": r["total_return_pct"],
                    "dd": r["max_dd"], "trades": r["num_trades"],
                    "profitable": profitable,
                })
                m = "+" if profitable else "-"
                log(f"    W{len(window_results):2d}: {r['total_return_pct']:+6.1f}% "
                    f"DD:{r['max_dd']:4.1f}% T:{r['num_trades']:2d} [{m}]")

            start += step_bars

        del df_4h, df_1d
        gc.collect()

    if not window_results:
        log("  No windows!")
        test_results["2_walk_forward"] = False
        return False

    with_trades = [w for w in window_results if w["trades"] > 0]
    profitable_wt = sum(1 for w in with_trades if w["profitable"])
    total_w = len(window_results)
    pct = profitable_wt / len(with_trades) * 100 if with_trades else 0

    log(f"\n  Total windows: {total_w}, with trades: {len(with_trades)}")
    log(f"  Profitable (with trades): {profitable_wt}/{len(with_trades)} ({pct:.0f}%)")
    log(f"  Avg return: {np.mean([w['return'] for w in window_results]):+.1f}%")

    passed = pct >= 60
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold: >=60%)")
    test_results["2_walk_forward"] = passed
    return passed


# ============================================================
# TEST 3: Stress Testing (Combined System)
# ============================================================

def test_3_stress():
    write_section("TEST 3: STRESS TESTING (COMBINED)")
    log("Inject extreme scenarios into real data")
    log("PASS if max DD < 30% in all scenarios\n")

    symbol = "ETH/USDT"
    df_4h, df_1d = load_data(symbol)
    if df_4h is None:
        test_results["3_stress"] = False
        return False
    funding = load_funding_rates(symbol)

    baseline = combined_backtest(df_4h, df_1d, BASELINE_CONFIG, funding)
    log(f"  Baseline: {baseline['total_return_pct']:+.1f}%, DD: {baseline['max_dd']:.1f}%")

    inject = 2000
    max_dd_all = 0
    all_pass = True

    scenarios = {
        "flash_crash": "−30% over 24 bars, 50% recovery over 168",
        "prolonged_bear": "−40% drift over 540 bars",
        "volatility_spike": "3x range for 84 bars",
        "sideways_chop": "±2% band for 360 bars",
    }

    for name, desc in scenarios.items():
        df_mod = df_4h.copy()

        if name == "flash_crash":
            for j in range(24):
                if inject + j < len(df_mod):
                    mult = 1 - 0.30 * (j + 1) / 24
                    for col in ["open", "high", "low", "close"]:
                        df_mod.iloc[inject + j, df_mod.columns.get_loc(col)] *= mult
            for j in range(168):
                idx = inject + 24 + j
                if idx < len(df_mod):
                    recovery = 0.70 + 0.15 * (j + 1) / 168
                    for col in ["open", "high", "low", "close"]:
                        df_mod.iloc[idx, df_mod.columns.get_loc(col)] *= recovery

        elif name == "prolonged_bear":
            for j in range(540):
                if inject + j < len(df_mod):
                    mult = 0.999 ** (j + 1)
                    for col in ["open", "high", "low", "close"]:
                        df_mod.iloc[inject + j, df_mod.columns.get_loc(col)] *= mult

        elif name == "volatility_spike":
            for j in range(84):
                if inject + j < len(df_mod):
                    idx = inject + j
                    h = df_mod.iloc[idx]["high"]
                    l = df_mod.iloc[idx]["low"]
                    mid = (h + l) / 2
                    rng = h - l
                    df_mod.iloc[idx, df_mod.columns.get_loc("high")] = mid + rng * 1.5
                    df_mod.iloc[idx, df_mod.columns.get_loc("low")] = mid - rng * 1.5

        elif name == "sideways_chop":
            anchor = df_mod.iloc[inject]["close"]
            band = anchor * 0.02
            for j in range(360):
                if inject + j < len(df_mod):
                    idx = inject + j
                    for col in ["open", "high", "low", "close"]:
                        val = df_mod.iloc[idx][col]
                        df_mod.iloc[idx, df_mod.columns.get_loc(col)] = np.clip(val, anchor - band, anchor + band)

        # Rebuild daily from modified 4h
        df_mod["timestamp"] = pd.to_datetime(df_mod["timestamp"], utc=True)
        df_1d_mod = df_mod.set_index("timestamp").resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()

        try:
            r = combined_backtest(df_mod, df_1d_mod, BASELINE_CONFIG, funding)
            dd = r["max_dd"]
            max_dd_all = max(max_dd_all, dd)
            ok = dd < 30
            if not ok:
                all_pass = False
            log(f"  {name:20s}: Ret {r['total_return_pct']:+7.1f}% DD {dd:5.1f}% {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            log(f"  {name}: ERROR - {e}")
            all_pass = False

    del df_4h, df_1d
    gc.collect()

    log(f"\n  Worst DD: {max_dd_all:.1f}%")
    log(f"  RESULT: {'PASS' if all_pass else 'FAIL'} (threshold: all DD < 30%)")
    test_results["3_stress"] = all_pass
    return all_pass


# ============================================================
# TEST 4: Noise Injection (Combined)
# ============================================================

def test_4_noise():
    write_section("TEST 4: NOISE INJECTION (COMBINED)")
    log("0.5% Gaussian noise on OHLCV, 50 runs per pair")
    log("PASS if >=80% profitable AND Sharpe std < 0.5\n")

    noise_std = 0.005
    n_runs = 50
    overall_pass = 0
    overall_total = 0
    all_sharpes = []

    for symbol in CORE_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        baseline = combined_backtest(df_4h, df_1d, BASELINE_CONFIG, funding)

        noisy_rets = []
        noisy_sharpes = []
        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            df_4h_n = df_4h.copy()
            df_1d_n = df_1d.copy()

            for col in ["open", "high", "low", "close"]:
                df_4h_n[col] = df_4h_n[col] * rng.normal(1.0, noise_std, len(df_4h_n))
            df_4h_n["high"] = df_4h_n[["open", "high", "low", "close"]].max(axis=1)
            df_4h_n["low"] = df_4h_n[["open", "high", "low", "close"]].min(axis=1)

            for col in ["open", "high", "low", "close"]:
                df_1d_n[col] = df_1d_n[col] * rng.normal(1.0, noise_std, len(df_1d_n))
            df_1d_n["high"] = df_1d_n[["open", "high", "low", "close"]].max(axis=1)
            df_1d_n["low"] = df_1d_n[["open", "high", "low", "close"]].min(axis=1)

            try:
                r = combined_backtest(df_4h_n, df_1d_n, BASELINE_CONFIG, funding)
                noisy_rets.append(r["total_return_pct"])
                noisy_sharpes.append(r["sharpe"])
            except Exception:
                noisy_rets.append(0)
                noisy_sharpes.append(0)

        profitable = sum(1 for r in noisy_rets if r > 0)
        overall_pass += profitable
        overall_total += n_runs
        all_sharpes.extend(noisy_sharpes)

        log(f"  {symbol:12s}: Base {baseline['total_return_pct']:+7.1f}% | "
            f"Noisy avg {np.mean(noisy_rets):+7.1f}% (std {np.std(noisy_rets):5.1f}%) | "
            f"{profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")

        del df_4h, df_1d
        gc.collect()

    pct_profitable = overall_pass / overall_total * 100 if overall_total > 0 else 0
    sharpe_std = np.std(all_sharpes) if all_sharpes else 999

    log(f"\n  Overall: {overall_pass}/{overall_total} ({pct_profitable:.0f}%)")
    log(f"  Sharpe std: {sharpe_std:.3f}")

    passed = pct_profitable >= 80 and sharpe_std < 0.5
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (need >=80% AND Sharpe std <0.5)")
    test_results["4_noise"] = passed
    return passed


# ============================================================
# TEST 5: Cross-Asset Generalization (Combined)
# ============================================================

def test_5_cross_asset():
    write_section("TEST 5: CROSS-ASSET GENERALIZATION (COMBINED)")
    log("Same config on all 11 pairs. PASS if >=60% profitable\n")

    profitable_count = 0
    total_count = 0
    results_table = []

    for symbol in ALL_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        r = combined_backtest(df_4h, df_1d, BASELINE_CONFIG, funding)
        is_prof = r["total_return_pct"] > 0
        if is_prof:
            profitable_count += 1
        total_count += 1
        results_table.append((symbol, r))

        m = "+" if is_prof else "-"
        log(f"  {symbol:12s}: {r['ann_return_pct']:+7.1f}%/yr DD:{r['max_dd']:5.1f}% "
            f"PF:{r['profit_factor']:5.2f} S:{r['sharpe']:.2f} "
            f"T:{r['num_trades']:3d} (L:{r['num_long']} S:{r['num_short']}) [{m}]")

        del df_4h, df_1d
        gc.collect()

    pct = profitable_count / total_count * 100 if total_count > 0 else 0
    log(f"\n  Profitable: {profitable_count}/{total_count} ({pct:.0f}%)")

    passed = pct >= 60
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (threshold: >=60%)")
    test_results["5_cross_asset"] = passed
    return passed, results_table


# ============================================================
# TEST 6: Activity Check (Combined)
# ============================================================

def test_6_activity():
    write_section("TEST 6: MINIMUM ACTIVITY (COMBINED)")
    log("PASS if >=12 trades/yr AND >=10% time in-position\n")

    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    trades_per_yr = []
    pct_in_pos = []

    for symbol in test_symbols:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        r = combined_backtest(df_4h, df_1d, BASELINE_CONFIG, funding)
        tpy = r["num_trades"] / r["years"] if r["years"] > 0 else 0
        pip = r["bars_in_position"] / r["total_bars"] * 100 if r["total_bars"] > 0 else 0
        trades_per_yr.append(tpy)
        pct_in_pos.append(pip)

        log(f"  {symbol}: {tpy:.1f} trades/yr, {pip:.1f}% in-position "
            f"({r['num_trades']} trades over {r['years']:.1f}yr)")

        del df_4h, df_1d
        gc.collect()

    avg_tpy = np.mean(trades_per_yr) if trades_per_yr else 0
    avg_pip = np.mean(pct_in_pos) if pct_in_pos else 0

    log(f"\n  Avg trades/yr: {avg_tpy:.1f}")
    log(f"  Avg in-position: {avg_pip:.1f}%")

    # Combined system should trade more than long-only (both sides of market)
    passed = avg_tpy >= 12 and avg_pip >= 10
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (need >=12 t/yr AND >=10% in-pos)")
    test_results["6_activity"] = passed
    return passed


# ============================================================
# TEST 7: Bootstrap Confidence Intervals (Combined)
# ============================================================

def test_7_bootstrap():
    write_section("TEST 7: BOOTSTRAP CONFIDENCE INTERVALS (COMBINED)")
    log("1000 resamples of trade P&Ls. PASS if 95% CI lower bound > 0\n")

    n_bootstrap = 1000
    all_lower_rets = []
    all_lower_pfs = []

    for symbol in CORE_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        r = combined_backtest(df_4h, df_1d, BASELINE_CONFIG, funding)
        pnls = r["trade_pnls"]

        if len(pnls) < 5:
            log(f"  {symbol}: Too few trades ({len(pnls)}), skip")
            del df_4h, df_1d
            gc.collect()
            continue

        rng = np.random.RandomState(42)
        boot_returns = []
        boot_pfs = []

        for _ in range(n_bootstrap):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            total_ret = sum(sample) / INITIAL_CAPITAL * 100
            w = sum(p for p in sample if p > 0)
            l = abs(sum(p for p in sample if p < 0))
            pf = w / l if l > 0 else 100
            boot_returns.append(total_ret)
            boot_pfs.append(min(pf, 100))

        lr = np.percentile(boot_returns, 2.5)
        ur = np.percentile(boot_returns, 97.5)
        lpf = np.percentile(boot_pfs, 2.5)
        upf = np.percentile(boot_pfs, 97.5)

        all_lower_rets.append(lr)
        all_lower_pfs.append(lpf)

        log(f"  {symbol:12s}: Return CI [{lr:+7.1f}%, {ur:+7.1f}%] | PF CI [{lpf:.2f}, {upf:.2f}]")

        del df_4h, df_1d
        gc.collect()

    if not all_lower_rets:
        test_results["7_bootstrap"] = False
        return False

    avg_lr = np.mean(all_lower_rets)
    avg_lpf = np.mean(all_lower_pfs)
    pairs_positive = sum(1 for r in all_lower_rets if r > 0)

    log(f"\n  Pairs with positive lower bound: {pairs_positive}/{len(all_lower_rets)}")
    log(f"  Avg lower return: {avg_lr:+.1f}%")
    log(f"  Avg lower PF: {avg_lpf:.2f}")

    passed = avg_lr > 0 and avg_lpf > 1.0
    log(f"  RESULT: {'PASS' if passed else 'FAIL'} (need avg lower ret >0, PF >1.0)")
    test_results["7_bootstrap"] = passed
    return passed


# ============================================================
# TEST 8: Parameter Optimization Grid Search
# ============================================================

def test_8_optimization():
    write_section("TEST 8: PARAMETER OPTIMIZATION")
    log("Grid search over short-side ATR mult + entry period")
    log("Goal: find if baseline is optimal or if we're leaving money on table\n")

    # The long side is already proven at baseline. Vary only short-related params.
    # Short uses same entry/exit periods and ATR mult, so varying them affects both sides.
    # Also test short-specific ATR mult (tighter/wider stops on shorts).

    atr_mults = [2.0, 2.5, 3.0, 3.5, 4.0]
    entry_periods = [10, 12, 14, 16, 20]
    risk_pcts = [4.0, 5.0, 6.0, 7.0]

    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

    # Load data once
    data_cache = {}
    for symbol in test_symbols:
        df_4h, df_1d = load_data(symbol)
        if df_4h is not None:
            funding = load_funding_rates(symbol)
            data_cache[symbol] = (df_4h, df_1d, funding)

    # Phase 1: ATR mult grid
    log("  ATR multiplier scan:")
    best_atr = 3.0
    best_atr_ret = -999
    for atr in atr_mults:
        cfg = {**BASELINE_CONFIG, "atr_stop_mult": atr}
        rets = []
        for symbol, (df_4h, df_1d, funding) in data_cache.items():
            r = combined_backtest(df_4h, df_1d, cfg, funding)
            rets.append(r["ann_return_pct"])
        avg = np.mean(rets)
        dd_avg = 0  # compute if needed
        log(f"    ATR {atr:.1f}: avg {avg:+.1f}%/yr")
        if avg > best_atr_ret:
            best_atr_ret = avg
            best_atr = atr

    # Phase 2: Entry period grid (using best ATR)
    log(f"\n  Entry period scan (with ATR={best_atr}):")
    best_entry = 14
    best_entry_ret = -999
    for ep in entry_periods:
        cfg = {**BASELINE_CONFIG, "atr_stop_mult": best_atr, "entry_period_4h": ep}
        rets = []
        for symbol, (df_4h, df_1d, funding) in data_cache.items():
            r = combined_backtest(df_4h, df_1d, cfg, funding)
            rets.append(r["ann_return_pct"])
        avg = np.mean(rets)
        log(f"    Entry {ep}: avg {avg:+.1f}%/yr")
        if avg > best_entry_ret:
            best_entry_ret = avg
            best_entry = ep

    # Phase 3: Risk pct grid (using best ATR + entry)
    log(f"\n  Risk % scan (with ATR={best_atr}, entry={best_entry}):")
    best_risk = 5.0
    best_risk_ret = -999
    for rp in risk_pcts:
        cfg = {**BASELINE_CONFIG, "atr_stop_mult": best_atr, "entry_period_4h": best_entry,
               "risk_per_trade_pct": rp}
        rets = []
        dds = []
        for symbol, (df_4h, df_1d, funding) in data_cache.items():
            r = combined_backtest(df_4h, df_1d, cfg, funding)
            rets.append(r["ann_return_pct"])
            dds.append(r["max_dd"])
        avg = np.mean(rets)
        avg_dd = np.mean(dds)
        r_dd = avg / avg_dd if avg_dd > 0 else 0
        log(f"    Risk {rp}%: avg {avg:+.1f}%/yr, DD {avg_dd:.1f}%, R/DD {r_dd:.2f}")
        if avg > best_risk_ret:
            best_risk_ret = avg
            best_risk = rp

    # Phase 4: Exit period scan
    exit_periods = [5, 7, 10, 14]
    log(f"\n  Exit period scan (with ATR={best_atr}, entry={best_entry}, risk={best_risk}%):")
    best_exit = 7
    best_exit_ret = -999
    for ep in exit_periods:
        cfg = {**BASELINE_CONFIG, "atr_stop_mult": best_atr, "entry_period_4h": best_entry,
               "risk_per_trade_pct": best_risk, "exit_period_4h": ep}
        rets = []
        for symbol, (df_4h, df_1d, funding) in data_cache.items():
            r = combined_backtest(df_4h, df_1d, cfg, funding)
            rets.append(r["ann_return_pct"])
        avg = np.mean(rets)
        log(f"    Exit {ep}: avg {avg:+.1f}%/yr")
        if avg > best_exit_ret:
            best_exit_ret = avg
            best_exit = ep

    del data_cache
    gc.collect()

    optimized_config = {
        **BASELINE_CONFIG,
        "atr_stop_mult": best_atr,
        "entry_period_4h": best_entry,
        "risk_per_trade_pct": best_risk,
        "exit_period_4h": best_exit,
    }

    log(f"\n  OPTIMIZED CONFIG:")
    log(f"    ATR mult: {best_atr} (was {BASELINE_CONFIG['atr_stop_mult']})")
    log(f"    Entry period: {best_entry} (was {BASELINE_CONFIG['entry_period_4h']})")
    log(f"    Risk %: {best_risk} (was {BASELINE_CONFIG['risk_per_trade_pct']})")
    log(f"    Exit period: {best_exit} (was {BASELINE_CONFIG['exit_period_4h']})")
    log(f"    Best avg: {best_exit_ret:+.1f}%/yr")

    # This is informational, always "passes"
    test_results["8_optimization"] = True
    return optimized_config


# ============================================================
# PHASE 9: Portfolio Allocation Optimization
# ============================================================

def phase_9_allocation(results_table, optimized_config):
    write_section("PHASE 9: PORTFOLIO ALLOCATION OPTIMIZATION")
    log("Finding optimal pair weights for combined system\n")

    # Run optimized config on all pairs
    pair_metrics = {}
    for symbol in ALL_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        r = combined_backtest(df_4h, df_1d, optimized_config, funding)
        pair_metrics[symbol] = r
        log(f"  {symbol:12s}: {r['ann_return_pct']:+7.1f}%/yr, DD:{r['max_dd']:5.1f}%, "
            f"Sharpe:{r['sharpe']:.2f}, PF:{r['profit_factor']:.2f}")

        del df_4h, df_1d
        gc.collect()

    if not pair_metrics:
        log("  No data!")
        return

    # Method 1: Equal weight
    equal_avg = np.mean([r["ann_return_pct"] for r in pair_metrics.values()])
    equal_dd = np.mean([r["max_dd"] for r in pair_metrics.values()])
    log(f"\n  Equal weight (1/{len(pair_metrics)}): {equal_avg:+.1f}%/yr, DD {equal_dd:.1f}%")

    # Method 2: R/DD weighted (proven approach from long-only)
    r_dd_scores = {}
    for sym, r in pair_metrics.items():
        if r["max_dd"] > 0:
            r_dd_scores[sym] = r["ann_return_pct"] / r["max_dd"]
        else:
            r_dd_scores[sym] = r["ann_return_pct"]

    total_score = sum(max(0, s) for s in r_dd_scores.values())
    r_dd_weights = {}
    for sym, score in r_dd_scores.items():
        r_dd_weights[sym] = max(0, score) / total_score if total_score > 0 else 0

    r_dd_avg = sum(r_dd_weights[s] * pair_metrics[s]["ann_return_pct"] for s in pair_metrics)
    r_dd_dd = sum(r_dd_weights[s] * pair_metrics[s]["max_dd"] for s in pair_metrics)

    log(f"\n  R/DD weighted allocation:")
    for sym in sorted(r_dd_weights, key=lambda x: r_dd_weights[x], reverse=True):
        w = r_dd_weights[sym]
        if w > 0.01:
            log(f"    {sym:12s}: {w*100:5.1f}% -> {pair_metrics[sym]['ann_return_pct']:+.1f}%/yr")
    log(f"  Portfolio: {r_dd_avg:+.1f}%/yr, DD {r_dd_dd:.1f}%")

    # Method 3: Return-weighted (maximize return)
    returns = {sym: max(0, r["ann_return_pct"]) for sym, r in pair_metrics.items()}
    total_ret = sum(returns.values())
    ret_weights = {sym: r / total_ret if total_ret > 0 else 0 for sym, r in returns.items()}

    ret_avg = sum(ret_weights[s] * pair_metrics[s]["ann_return_pct"] for s in pair_metrics)
    ret_dd = sum(ret_weights[s] * pair_metrics[s]["max_dd"] for s in pair_metrics)

    log(f"\n  Return-weighted allocation:")
    for sym in sorted(ret_weights, key=lambda x: ret_weights[x], reverse=True):
        w = ret_weights[sym]
        if w > 0.01:
            log(f"    {sym:12s}: {w*100:5.1f}% -> {pair_metrics[sym]['ann_return_pct']:+.1f}%/yr")
    log(f"  Portfolio: {ret_avg:+.1f}%/yr, DD {ret_dd:.1f}%")

    # Method 4: Sharpe-weighted
    sharpes = {sym: max(0, r["sharpe"]) for sym, r in pair_metrics.items()}
    total_sharpe = sum(sharpes.values())
    sharpe_weights = {sym: s / total_sharpe if total_sharpe > 0 else 0 for sym, s in sharpes.items()}

    sharpe_avg = sum(sharpe_weights[s] * pair_metrics[s]["ann_return_pct"] for s in pair_metrics)
    sharpe_dd = sum(sharpe_weights[s] * pair_metrics[s]["max_dd"] for s in pair_metrics)

    log(f"\n  Sharpe-weighted allocation:")
    for sym in sorted(sharpe_weights, key=lambda x: sharpe_weights[x], reverse=True):
        w = sharpe_weights[sym]
        if w > 0.01:
            log(f"    {sym:12s}: {w*100:5.1f}% -> {pair_metrics[sym]['ann_return_pct']:+.1f}%/yr")
    log(f"  Portfolio: {sharpe_avg:+.1f}%/yr, DD {sharpe_dd:.1f}%")

    # Method 5: Top-N concentration
    for n in [3, 5, 7]:
        sorted_pairs = sorted(pair_metrics.items(), key=lambda x: x[1]["ann_return_pct"], reverse=True)[:n]
        top_avg = np.mean([r["ann_return_pct"] for _, r in sorted_pairs])
        top_dd = np.mean([r["max_dd"] for _, r in sorted_pairs])
        names = [s for s, _ in sorted_pairs]
        log(f"\n  Top-{n} equal: {top_avg:+.1f}%/yr, DD {top_dd:.1f}% ({', '.join(names)})")

    # Best allocation
    best_method = max([
        ("Equal", equal_avg, equal_dd),
        ("R/DD", r_dd_avg, r_dd_dd),
        ("Return", ret_avg, ret_dd),
        ("Sharpe", sharpe_avg, sharpe_dd),
    ], key=lambda x: x[1])

    log(f"\n  BEST METHOD: {best_method[0]} at {best_method[1]:+.1f}%/yr, DD {best_method[2]:.1f}%")

    return pair_metrics


# ============================================================
# FINAL VERDICT
# ============================================================

def final_verdict(optimized_config):
    write_section("FINAL VERDICT")

    weights = {
        "2_walk_forward": 1.5,
        "5_cross_asset": 1.3,
        "4_noise": 1.2,
        "7_bootstrap": 1.2,
        "1_sensitivity": 1.0,
        "3_stress": 1.0,
        "6_activity": 0.8,
        "8_optimization": 0.0,  # informational
    }

    test_names = {
        "1_sensitivity": "Parameter Sensitivity",
        "2_walk_forward": "Walk-Forward",
        "3_stress": "Stress Testing",
        "4_noise": "Noise Injection",
        "5_cross_asset": "Cross-Asset Generalization",
        "6_activity": "Minimum Activity",
        "7_bootstrap": "Bootstrap Confidence",
        "8_optimization": "Parameter Optimization",
    }

    total_weight = 0
    passed_weight = 0
    pass_count = 0

    log(f"\n  {'Test':30s} | {'Result':6s} | {'Weight':6s}")
    log(f"  {'-'*50}")

    for key in sorted(test_results.keys()):
        name = test_names.get(key, key)
        passed = test_results[key]
        w = weights.get(key, 1.0)
        if w > 0:
            total_weight += w
            if passed:
                passed_weight += w
                pass_count += 1
        log(f"  {name:30s} | {'PASS' if passed else 'FAIL':6s} | {w:.1f}x")

    score = passed_weight / total_weight * 100 if total_weight > 0 else 0
    scored_tests = sum(1 for k, w in weights.items() if w > 0 and k in test_results)

    log(f"\n  Tests passed: {pass_count}/{scored_tests}")
    log(f"  Weighted score: {score:.0f}%")

    if pass_count == scored_tests:
        log(f"\n  *** STRONG PASS: Combined system is robust ***")
        verdict = "STRONG_PASS"
    elif pass_count >= scored_tests - 2:
        log(f"\n  CONDITIONAL PASS: Review failures")
        verdict = "CONDITIONAL_PASS"
    else:
        log(f"\n  FAIL: Combined system may be overfit")
        verdict = "FAIL"

    log(f"\n  Optimized config: {optimized_config}")
    log(f"  Verdict: {verdict}")

    return verdict


# ============================================================
# MAIN
# ============================================================

def main():
    start = datetime.now(timezone.utc)

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V10: COMBINED ROBUSTNESS & OPTIMIZATION")
    log(f"Start time: {start.isoformat()}")
    log(f"Capital: ${INITIAL_CAPITAL:,}")
    log(f"Pairs: {len(ALL_PAIRS)}")
    log(f"Baseline config: {BASELINE_CONFIG}")

    # Run all tests
    test_1_sensitivity()
    gc.collect()

    test_2_walk_forward()
    gc.collect()

    test_3_stress()
    gc.collect()

    test_4_noise()
    gc.collect()

    result_5 = test_5_cross_asset()
    results_table = result_5[1] if isinstance(result_5, tuple) else []
    gc.collect()

    test_6_activity()
    gc.collect()

    test_7_bootstrap()
    gc.collect()

    optimized_config = test_8_optimization()
    gc.collect()

    phase_9_allocation(results_table, optimized_config)
    gc.collect()

    verdict = final_verdict(optimized_config)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nFull report: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        with open(REPORT_FILE, "a") as f:
            f.write(f"\nFATAL ERROR: {e}\n")
            traceback.print_exc(file=f)
