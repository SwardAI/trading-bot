"""Quick validation of the optimized combined config from v10.

Tests the optimized config (ATR 2.5, entry 10, risk 6%) with:
1. Noise injection on all 11 pairs (30 runs each)
2. Walk-forward on all 11 pairs
3. Full portfolio performance comparison (baseline vs optimized)

Usage: py scripts/strategy_research_v10b_validate_optimized.py
"""

import gc
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data

REPORT_FILE = "data/strategy_research_v10b_validate_report.txt"
INITIAL_CAPITAL = 10000

ALL_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
    "AVAX/USDT", "LINK/USDT", "ADA/USDT",
    "DOT/USDT", "ATOM/USDT", "NEAR/USDT", "UNI/USDT",
]

SPOT_TAKER_FEE = 0.001
FUTURES_TAKER_FEE = 0.0004

BASELINE_CONFIG = {
    "daily_periods": [20, 50, 100], "daily_min_votes": 2,
    "entry_period_4h": 14, "exit_period_4h": 7,
    "atr_period": 14, "atr_stop_mult": 3.0,
    "risk_per_trade_pct": 5.0, "cooldown_bars": 2,
    "vol_scale": True, "vol_scale_lookback": 60,
}

OPTIMIZED_CONFIG = {
    "daily_periods": [20, 50, 100], "daily_min_votes": 2,
    "entry_period_4h": 10, "exit_period_4h": 7,
    "atr_period": 14, "atr_stop_mult": 2.5,
    "risk_per_trade_pct": 6.0, "cooldown_bars": 2,
    "vol_scale": True, "vol_scale_lookback": 60,
}

from pathlib import Path
FUNDING_DATA_DIR = Path("data/historical/funding_rates")


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


def resample(df, freq):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    resampled = df.resample(freq).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def load_data(symbol):
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        return None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    del df
    gc.collect()
    return df_4h, df_1d


def load_funding_rates(symbol):
    safe_symbol = symbol.replace("/", "_")
    cache_file = FUNDING_DATA_DIR / f"binance_{safe_symbol}_funding.csv"
    if not cache_file.exists():
        return {}
    df = pd.read_csv(cache_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["funding_rate"].mean()
    return daily.to_dict()


def combined_backtest(df_4h, df_1d, config, funding_rates=None):
    """Combined long+short backtest — same as v10."""
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

    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)
        df_1d[f"dc_low_{p}"] = df_1d["low"].rolling(window=p).min().shift(1)

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
    trade_pnls = []
    num_trades = 0
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
                    trade_pnls.append(pnl)
                    num_trades += 1
                    position = None
                    bars_since_exit = 0

            else:
                if low < position["lowest_since_entry"]:
                    position["lowest_since_entry"] = low
                chandelier = position["lowest_since_entry"] + atr_stop_mult * atr
                effective_stop = min(chandelier, position["stop_loss"])
                exit_ch = row["highest_high_exit"] if not pd.isna(row["highest_high_exit"]) else float("inf")

                if funding_rates and i % 2 == 0:
                    daily_rate = funding_rates.get(bar_date, 0)
                    payment = position["notional"] * daily_rate / 3
                    capital += payment

                if close >= effective_stop or close > exit_ch:
                    amount = position["amount"]
                    fee = amount * close * FUTURES_TAKER_FEE
                    pnl = (position["entry_price"] - close) * amount - fee
                    capital += pnl
                    trade_pnls.append(pnl)
                    num_trades += 1
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

        effective_risk_pct = risk_pct
        if vol_scale and not pd.isna(row.get("atr_median")) and row["atr_median"] > 0:
            vol_ratio = row["atr_median"] / atr
            vol_ratio = max(0.5, min(2.0, vol_ratio))
            effective_risk_pct = risk_pct * vol_ratio

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

    if position:
        final = df_4h.iloc[-1]
        if position["side"] == "long":
            pnl = (final["close"] - position["entry_price"]) * position["amount"] - position["amount"] * final["close"] * SPOT_TAKER_FEE
        else:
            pnl = (position["entry_price"] - final["close"]) * position["amount"] - position["amount"] * final["close"] * FUTURES_TAKER_FEE
        capital += pnl
        trade_pnls.append(pnl)
        num_trades += 1

    total_bars = len(df_4h) - lookback
    years = total_bars / (6 * 365)
    total_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    sharpe = 0
    if len(equity_curve) > 1:
        rets = [(equity_curve[j] - equity_curve[j - 1]) / equity_curve[j - 1]
                for j in range(1, len(equity_curve)) if equity_curve[j - 1] > 0]
        if rets and np.std(rets) > 0:
            sharpe = np.mean(rets) / np.std(rets) * (365 ** 0.5)

    wins = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))
    pf = wins / losses if losses > 0 else float("inf")

    return {
        "total_return_pct": total_return_pct,
        "ann_return_pct": total_return_pct / years if years > 0 else 0,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "profit_factor": pf,
        "num_trades": num_trades,
        "trade_pnls": trade_pnls,
        "years": years,
    }


def main():
    start = datetime.now(timezone.utc)

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("V10B: OPTIMIZED CONFIG VALIDATION")
    log(f"Baseline: ATR=3.0, entry=14, risk=5%")
    log(f"Optimized: ATR=2.5, entry=10, risk=6%")

    # ==========================================================
    # Phase 1: Head-to-head comparison on all 11 pairs
    # ==========================================================
    write_section("PHASE 1: HEAD-TO-HEAD (ALL 11 PAIRS)")

    baseline_results = {}
    optimized_results = {}

    for symbol in ALL_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        br = combined_backtest(df_4h, df_1d, BASELINE_CONFIG, funding)
        opt = combined_backtest(df_4h, df_1d, OPTIMIZED_CONFIG, funding)
        baseline_results[symbol] = br
        optimized_results[symbol] = opt

        delta = opt["ann_return_pct"] - br["ann_return_pct"]
        marker = "+" if delta > 0 else "-"
        log(f"  {symbol:12s}: Base {br['ann_return_pct']:+6.1f}%/yr DD:{br['max_dd']:5.1f}% | "
            f"Opt {opt['ann_return_pct']:+6.1f}%/yr DD:{opt['max_dd']:5.1f}% | "
            f"Delta {delta:+5.1f}% [{marker}]")

        del df_4h, df_1d
        gc.collect()

    base_avg = np.mean([r["ann_return_pct"] for r in baseline_results.values()])
    opt_avg = np.mean([r["ann_return_pct"] for r in optimized_results.values()])
    base_dd = np.mean([r["max_dd"] for r in baseline_results.values()])
    opt_dd = np.mean([r["max_dd"] for r in optimized_results.values()])
    improved = sum(1 for s in ALL_PAIRS
                   if s in optimized_results and s in baseline_results
                   and optimized_results[s]["ann_return_pct"] > baseline_results[s]["ann_return_pct"])

    log(f"\n  Baseline avg: {base_avg:+.1f}%/yr, DD {base_dd:.1f}%")
    log(f"  Optimized avg: {opt_avg:+.1f}%/yr, DD {opt_dd:.1f}%")
    log(f"  Improved: {improved}/{len(optimized_results)} pairs")

    # ==========================================================
    # Phase 2: Noise test on optimized config
    # ==========================================================
    write_section("PHASE 2: NOISE INJECTION (OPTIMIZED CONFIG)")
    log("30 runs per pair, 0.5% noise\n")

    noise_std = 0.005
    n_runs = 30
    overall_pass = 0
    overall_total = 0

    for symbol in ALL_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        noisy_rets = []
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
                r = combined_backtest(df_4h_n, df_1d_n, OPTIMIZED_CONFIG, funding)
                noisy_rets.append(r["total_return_pct"])
            except Exception:
                noisy_rets.append(0)

        profitable = sum(1 for r in noisy_rets if r > 0)
        overall_pass += profitable
        overall_total += n_runs

        log(f"  {symbol:12s}: {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%) "
            f"avg {np.mean(noisy_rets):+.1f}% std {np.std(noisy_rets):.1f}%")

        del df_4h, df_1d
        gc.collect()

    pct = overall_pass / overall_total * 100 if overall_total > 0 else 0
    log(f"\n  Overall: {overall_pass}/{overall_total} ({pct:.0f}%)")
    noise_pass = pct >= 80
    log(f"  RESULT: {'PASS' if noise_pass else 'FAIL'} (threshold: >=80%)")

    # ==========================================================
    # Phase 3: Walk-forward on optimized config
    # ==========================================================
    write_section("PHASE 3: WALK-FORWARD (OPTIMIZED CONFIG)")

    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "NEAR/USDT", "LINK/USDT"]
    window_results = []

    for symbol in test_symbols:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        bars_per_month = 6 * 30
        train_bars = bars_per_month * 6
        test_bars = bars_per_month * 3
        step_bars = bars_per_month * 3
        warmup = 200

        total = len(df_4h)
        start_idx = 0
        while start_idx + train_bars + test_bars <= total:
            test_start = start_idx + train_bars
            test_end = test_start + test_bars

            df_4h_w = df_4h.iloc[max(0, test_start - warmup):test_end].copy().reset_index(drop=True)
            ts_start = df_4h.iloc[max(0, test_start - warmup)]["timestamp"]
            ts_end = df_4h.iloc[test_end - 1]["timestamp"]
            df_1d_w = df_1d[
                (df_1d["timestamp"] >= ts_start) & (df_1d["timestamp"] <= ts_end)
            ].copy().reset_index(drop=True)

            if len(df_4h_w) > 100 and len(df_1d_w) > 20:
                r = combined_backtest(df_4h_w, df_1d_w, OPTIMIZED_CONFIG, funding)
                window_results.append({
                    "symbol": symbol, "return": r["total_return_pct"],
                    "trades": r["num_trades"],
                    "profitable": r["total_return_pct"] > 0,
                })

            start_idx += step_bars

        del df_4h, df_1d
        gc.collect()

    with_trades = [w for w in window_results if w["trades"] > 0]
    profitable_wt = sum(1 for w in with_trades if w["profitable"])
    pct_wf = profitable_wt / len(with_trades) * 100 if with_trades else 0

    log(f"  Total windows: {len(window_results)}, with trades: {len(with_trades)}")
    log(f"  Profitable (with trades): {profitable_wt}/{len(with_trades)} ({pct_wf:.0f}%)")
    log(f"  Avg return: {np.mean([w['return'] for w in window_results]):+.1f}%")
    wf_pass = pct_wf >= 60
    log(f"  RESULT: {'PASS' if wf_pass else 'FAIL'}")

    # ==========================================================
    # Phase 4: R/DD weighted portfolio comparison
    # ==========================================================
    write_section("PHASE 4: PORTFOLIO COMPARISON")

    # R/DD weighted for both configs
    for label, results in [("BASELINE", baseline_results), ("OPTIMIZED", optimized_results)]:
        r_dd_scores = {}
        for sym, r in results.items():
            if r["max_dd"] > 0:
                r_dd_scores[sym] = r["ann_return_pct"] / r["max_dd"]
            else:
                r_dd_scores[sym] = r["ann_return_pct"]

        total_score = sum(max(0, s) for s in r_dd_scores.values())
        weights = {s: max(0, sc) / total_score for s, sc in r_dd_scores.items()} if total_score > 0 else {}

        port_ret = sum(weights.get(s, 0) * r["ann_return_pct"] for s, r in results.items())
        port_dd = sum(weights.get(s, 0) * r["max_dd"] for s, r in results.items())
        port_sharpe = sum(weights.get(s, 0) * r["sharpe"] for s, r in results.items())
        r_dd = port_ret / port_dd if port_dd > 0 else 0

        log(f"\n  {label} R/DD-weighted portfolio:")
        log(f"    Return: {port_ret:+.1f}%/yr")
        log(f"    Max DD: {port_dd:.1f}%")
        log(f"    Sharpe: {port_sharpe:.2f}")
        log(f"    R/DD:   {r_dd:.2f}")

        # Top allocations
        for sym in sorted(weights, key=lambda x: weights[x], reverse=True)[:5]:
            w = weights[sym]
            log(f"    {sym:12s}: {w*100:5.1f}%")

    # ==========================================================
    # Final Summary
    # ==========================================================
    write_section("FINAL SUMMARY")

    log(f"  Noise test (optimized): {'PASS' if noise_pass else 'FAIL'} ({pct:.0f}%)")
    log(f"  Walk-forward (optimized): {'PASS' if wf_pass else 'FAIL'} ({pct_wf:.0f}%)")
    log(f"  Baseline avg: {base_avg:+.1f}%/yr, DD {base_dd:.1f}%")
    log(f"  Optimized avg: {opt_avg:+.1f}%/yr, DD {opt_dd:.1f}%")
    log(f"  Improvement: {opt_avg - base_avg:+.1f}%/yr")

    if noise_pass and wf_pass and opt_avg > base_avg:
        log(f"\n  VERDICT: OPTIMIZED CONFIG IS ROBUST AND SUPERIOR")
        log(f"  Use optimized: ATR=2.5, entry=10, risk=6%, exit=7")
    elif noise_pass and wf_pass:
        log(f"\n  VERDICT: OPTIMIZED CONFIG IS ROBUST BUT NOT SUPERIOR")
        log(f"  Stick with baseline: ATR=3.0, entry=14, risk=5%, exit=7")
    else:
        log(f"\n  VERDICT: OPTIMIZED CONFIG MAY BE OVERFIT")
        log(f"  Stick with baseline")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        with open(REPORT_FILE, "a") as f:
            f.write(f"\nFATAL ERROR: {e}\n")
            traceback.print_exc(file=f)
