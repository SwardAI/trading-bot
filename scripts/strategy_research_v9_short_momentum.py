"""Strategy research v9: Short-side momentum backtesting.

Tests the mirror of our proven MTF Donchian strategy for SHORT positions
during bear markets. Uses spot price data as proxy for perpetual futures
(perps track spot closely). Incorporates real funding rate costs from v8.

Key questions:
1. Does short momentum work in bear markets with the same Donchian logic?
2. What are realistic returns accounting for funding costs + futures fees?
3. Which pairs are best for shorting?
4. Combined with long momentum: what's the total system return?

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v9_short_momentum.py" bot
"""

import gc
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data
from src.backtest.engine import BacktestEngine, BacktestTrade

REPORT_FILE = "data/strategy_research_v9_short_momentum_report.txt"
FUNDING_DATA_DIR = Path("data/historical/funding_rates")
INITIAL_CAPITAL = 10000

CORE_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
              "AVAX/USDT", "LINK/USDT", "ADA/USDT"]
EXTRA_PAIRS = ["DOT/USDT", "ATOM/USDT", "NEAR/USDT", "UNI/USDT"]
ALL_PAIRS = CORE_PAIRS + EXTRA_PAIRS

# Futures fee model (Binance USDT-M)
FUTURES_TAKER_FEE = 0.0004   # 0.04%
FUTURES_MAKER_FEE = 0.0002   # 0.02%
# Use taker for momentum (time-sensitive entries)
TRADE_FEE = FUTURES_TAKER_FEE


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
    """Load 1h, 4h, and daily data."""
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
    """Load cached funding rates and return dict mapping date -> avg daily rate.

    Rate is in decimal (0.0001 = 0.01%). Negative = shorts pay.
    """
    safe_symbol = symbol.replace("/", "_")
    cache_file = FUNDING_DATA_DIR / f"binance_{safe_symbol}_funding.csv"
    if not cache_file.exists():
        return {}

    df = pd.read_csv(cache_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["date"] = df["timestamp"].dt.date

    # Average daily rate (3 payments per day)
    daily = df.groupby("date")["funding_rate"].mean()
    return daily.to_dict()


# ============================================================
# Short momentum backtest
# ============================================================

def short_momentum_backtest(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    config: dict,
    funding_rates: dict | None = None,
) -> dict:
    """MTF Donchian backtest for SHORT positions.

    Mirror of the long-side strategy:
    - Daily filter: price BELOW Donchian low channels (2/3 must agree = bearish)
    - 4h entry: price breaks below lowest_low (N periods) → SHORT
    - 4h exit: price breaks above highest_high (M periods) → COVER
    - Chandelier stop: lowest_low_since_entry + K * ATR (trailing stop above)
    - Vol-scaled sizing: same as long side

    Funding rate: when short on perps, you RECEIVE funding when positive
    (bull/sideways) and PAY when negative (some bear periods).
    """
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
    include_funding = config.get("include_funding", True)

    # Daily bearish filter: price below Donchian LOW channels
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_low_{p}"] = df_1d["low"].rolling(window=p).min().shift(1)

    # 4h indicators for shorts (reversed)
    df_4h = df_4h.copy()
    df_4h["lowest_low"] = df_4h["low"].rolling(window=entry_period).min().shift(1)    # entry: break below
    df_4h["highest_high"] = df_4h["high"].rolling(window=exit_period).max().shift(1)   # exit: break above
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )

    if vol_scale:
        df_4h["atr_median"] = df_4h["atr"].rolling(window=vol_lookback).median()

    # Build daily bearish map
    daily_bearish = {}
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        votes = sum(
            1 for p in daily_periods
            if not pd.isna(row.get(f"dc_low_{p}")) and row["close"] < row[f"dc_low_{p}"]
        )
        daily_bearish[date] = votes >= daily_min_votes

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None
    bars_since_exit = cooldown_bars
    lookback = max(entry_period, exit_period, atr_period, vol_lookback if vol_scale else 0) + 1

    for i in range(lookback, len(df_4h)):
        row = df_4h.iloc[i]
        close = row["close"]
        low = row["low"]
        atr = row["atr"]
        ts = row["timestamp"]

        if pd.isna(atr) or pd.isna(row.get("lowest_low")):
            unrealized = 0
            if position:
                unrealized = (position["entry_price"] - close) * position["amount"]
            equity_curve.append(capital + unrealized)
            continue

        bar_date = ts.date()
        is_bearish = daily_bearish.get(bar_date, False)

        if position:
            # Track lowest low since entry (for trailing stop)
            if low < position["lowest_since_entry"]:
                position["lowest_since_entry"] = low

            # Chandelier stop for shorts: trailing ABOVE the lowest low
            chandelier = position["lowest_since_entry"] + atr_stop_mult * atr
            effective_stop = min(chandelier, position["stop_loss"])

            # Exit channel: highest high (if price rallies above, cover)
            exit_ch = row["highest_high"] if not pd.isna(row["highest_high"]) else float("inf")

            # Apply funding rate cost/benefit (every 8h = every 2 bars on 4h)
            if include_funding and funding_rates and i % 2 == 0:
                daily_rate = funding_rates.get(bar_date, 0)
                # When SHORT: positive rate = we RECEIVE, negative = we PAY
                # Rate is per 8h period, so daily_rate * 3 = daily impact
                # But we apply per-funding-period, and there are ~3 per day
                funding_payment = position["notional"] * daily_rate  # already averaged daily
                capital += funding_payment / 3  # one of 3 daily payments

            # Exit conditions (reversed from long)
            if close >= effective_stop or close > exit_ch:
                amount = position["amount"]
                fee = amount * close * TRADE_FEE
                pnl = (position["entry_price"] - close) * amount - fee  # SHORT P&L
                capital += pnl
                trades.append(BacktestTrade(
                    timestamp=ts, side="buy", price=close,
                    amount=amount, cost=amount * close, fee=fee,
                    pnl=pnl, strategy="short_mtf",
                ))
                position = None
                bars_since_exit = 0

            unrealized = (position["entry_price"] - close) * position["amount"] if position else 0
            equity_curve.append(capital + unrealized)
            continue

        bars_since_exit += 1
        if bars_since_exit < cooldown_bars:
            equity_curve.append(capital)
            continue

        # Entry: bearish daily filter + price breaks below lowest low
        if is_bearish and close < row["lowest_low"]:
            stop_loss = close + atr_stop_mult * atr  # stop ABOVE entry
            risk_per_unit = stop_loss - close

            if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                effective_risk_pct = risk_pct
                if vol_scale and not pd.isna(row.get("atr_median")) and row["atr_median"] > 0:
                    vol_ratio = row["atr_median"] / atr
                    vol_ratio = max(0.5, min(2.0, vol_ratio))
                    effective_risk_pct = risk_pct * vol_ratio

                risk_amount = capital * (effective_risk_pct / 100)
                amount = risk_amount / risk_per_unit
                notional = amount * close
                fee = notional * TRADE_FEE

                # Need margin for short: assume 1x leverage (full notional as collateral)
                # Plus keep 5% reserve
                if notional + fee <= capital * 0.95:
                    capital -= fee
                    position = {
                        "entry_price": close, "amount": amount,
                        "stop_loss": stop_loss, "lowest_since_entry": low,
                        "notional": notional,
                    }
                    trades.append(BacktestTrade(
                        timestamp=ts, side="sell", price=close,
                        amount=amount, cost=notional, fee=fee,
                        strategy="short_mtf",
                    ))

        unrealized = (position["entry_price"] - close) * position["amount"] if position else 0
        equity_curve.append(capital + unrealized)

    # Close any open position at end
    if position:
        final = df_4h.iloc[-1]
        amount = position["amount"]
        fee = amount * final["close"] * TRADE_FEE
        pnl = (position["entry_price"] - final["close"]) * amount - fee
        capital += pnl
        trades.append(BacktestTrade(
            timestamp=final["timestamp"], side="buy", price=final["close"],
            amount=amount, cost=amount * final["close"], fee=fee,
            pnl=pnl, strategy="short_mtf",
        ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("short_mtf", df_4h, trades, equity_curve, capital)


# ============================================================
# Long momentum backtest (copy from v3 for comparison)
# ============================================================

def long_momentum_backtest(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    config: dict,
) -> dict:
    """Standard long-side MTF Donchian backtest (proven strategy)."""
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

    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
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
            unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
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
                fee = amount * close * 0.001  # spot fee
                pnl = (close - position["entry_price"]) * amount - fee
                capital += pnl
                trades.append(BacktestTrade(
                    timestamp=ts, side="sell", price=close,
                    amount=amount, cost=amount * close, fee=fee,
                    pnl=pnl, strategy="long_mtf",
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

        if is_bullish and close > row["highest_high"]:
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
                        strategy="long_mtf",
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
            pnl=pnl, strategy="long_mtf",
        ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("long_mtf", df_4h, trades, equity_curve, capital)


# ============================================================
# Main analysis phases
# ============================================================

def main():
    start = datetime.now(timezone.utc)

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V9: SHORT-SIDE MOMENTUM")
    log(f"Start time: {start.isoformat()}")
    log(f"Capital: ${INITIAL_CAPITAL:,}")
    log(f"Pairs: {len(ALL_PAIRS)}")

    base_config = {
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
        "include_funding": True,
    }

    # ========================================================
    # Phase 1: Short momentum on each pair (with funding costs)
    # ========================================================
    write_section("PHASE 1: SHORT MOMENTUM PER PAIR")
    log(f"Config: mirror of long MTF Donchian, vol-scaled, with funding rate costs")
    log(f"Fee model: futures taker {FUTURES_TAKER_FEE*100:.3f}% per trade\n")

    # Store only summary numbers, not full result objects (memory-efficient)
    short_summaries = {}  # {symbol: {ret, ann, dd, trades, wr}}
    long_summaries = {}

    for symbol in ALL_PAIRS:
        log(f"--- {symbol} ---")
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            log(f"  No data for {symbol}, skipping")
            continue

        years = len(df_4h) / (6 * 365)
        funding = load_funding_rates(symbol)
        funding_status = f"({len(funding)} days)" if funding else "(no funding data)"

        # Short backtest
        try:
            sr = short_momentum_backtest(df_4h, df_1d, base_config, funding)
            sell_trades = [t for t in sr.trades if t.side == "sell"]
            all_pnl = [t.pnl for t in sr.trades if t.pnl != 0]
            winners = [p for p in all_pnl if p > 0]
            win_rate = len(winners) / len(all_pnl) * 100 if all_pnl else 0
            ann = sr.total_return_pct / years if years > 0 else 0

            short_summaries[symbol] = {
                "ret": sr.total_return_pct, "ann": ann,
                "trades": len(sell_trades), "wr": win_rate,
            }
            log(f"  SHORT: {sr.total_return_pct:+.1f}% total, "
                f"{ann:+.1f}%/yr, "
                f"{len(sell_trades)} entries, "
                f"WR {win_rate:.0f}% "
                f"{funding_status}")
            del sr
        except Exception as e:
            log(f"  SHORT ERROR: {e}")
            traceback.print_exc()

        # Long backtest (for comparison)
        try:
            lr = long_momentum_backtest(df_4h, df_1d, base_config)
            buy_entries = [t for t in lr.trades if t.side == "buy"]
            all_pnl_l = [t.pnl for t in lr.trades if t.pnl != 0]
            winners_l = [p for p in all_pnl_l if p > 0]
            win_rate_l = len(winners_l) / len(all_pnl_l) * 100 if all_pnl_l else 0
            ann_l = lr.total_return_pct / years if years > 0 else 0

            long_summaries[symbol] = {
                "ret": lr.total_return_pct, "ann": ann_l,
                "trades": len(buy_entries), "wr": win_rate_l,
            }
            log(f"  LONG:  {lr.total_return_pct:+.1f}% total, "
                f"{ann_l:+.1f}%/yr, "
                f"{len(buy_entries)} entries, "
                f"WR {win_rate_l:.0f}%")
            del lr
        except Exception as e:
            log(f"  LONG ERROR: {e}")

        del df_4h, df_1d
        gc.collect()

    # ========================================================
    # Phase 2: Summary comparison
    # ========================================================
    write_section("PHASE 2: SHORT vs LONG COMPARISON")

    log(f"{'Symbol':<12} {'Short%':>8} {'Long%':>8} {'Short Trades':>13} {'Long Trades':>12}")
    log("-" * 58)

    for symbol in ALL_PAIRS:
        ss = short_summaries.get(symbol)
        ls = long_summaries.get(symbol)
        if not ss and not ls:
            continue

        s_ret = ss["ret"] if ss else 0
        l_ret = ls["ret"] if ls else 0
        s_trades = ss["trades"] if ss else 0
        l_trades = ls["trades"] if ls else 0

        log(f"{symbol:<12} {s_ret:>+7.1f}% {l_ret:>+7.1f}% {s_trades:>13} {l_trades:>12}")

    # ========================================================
    # Phase 3: Parameter variations for shorts
    # ========================================================
    write_section("PHASE 3: SHORT PARAMETER VARIATIONS")

    # Test different ATR multipliers and entry periods
    param_configs = {
        "baseline": base_config.copy(),
        "atr_2.5": {**base_config, "atr_stop_mult": 2.5},
        "atr_3.5": {**base_config, "atr_stop_mult": 3.5},
        "atr_4.0": {**base_config, "atr_stop_mult": 4.0},
        "entry_10": {**base_config, "entry_period_4h": 10},
        "entry_20": {**base_config, "entry_period_4h": 20},
        "exit_5": {**base_config, "exit_period_4h": 5},
        "exit_10": {**base_config, "exit_period_4h": 10},
        "risk_3": {**base_config, "risk_per_trade_pct": 3.0},
        "risk_7": {**base_config, "risk_per_trade_pct": 7.0},
        "no_vol_scale": {**base_config, "vol_scale": False},
        "no_funding": {**base_config, "include_funding": False},
    }

    # Test on top pairs
    test_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]

    # Load each pair once, then run all configs (memory-efficient)
    param_results = {name: [] for name in param_configs}
    for symbol in test_pairs:
        log(f"  Loading {symbol}...")
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)
        years = len(df_4h) / (6 * 365)

        for name, cfg in param_configs.items():
            try:
                sr = short_momentum_backtest(df_4h, df_1d, cfg, funding)
                ann = sr.total_return_pct / years if years > 0 else 0
                param_results[name].append(ann)
            except Exception as e:
                log(f"    {name}: ERROR - {e}")

        del df_4h, df_1d
        gc.collect()

    # Compute averages
    param_avgs = {}
    for name, returns in param_results.items():
        param_avgs[name] = np.mean(returns) if returns else 0

    log(f"\n{'Config':<20} {'Avg Ann%':>10} ({'|'.join(test_pairs)})")
    log("-" * 40)
    for name, avg in sorted(param_avgs.items(), key=lambda x: x[1], reverse=True):
        log(f"{name:<20} {avg:>+9.1f}%")

    # ========================================================
    # Phase 4: Combined long + short simulation
    # ========================================================
    write_section("PHASE 4: COMBINED LONG + SHORT SYSTEM")

    log("Running both long and short on same capital, switching based on daily filter.")
    log("Long when daily bullish (2/3 Donchian above), Short when daily bearish (2/3 below).")
    log("Cash when neither.\n")

    combined_results = {}
    for symbol in ALL_PAIRS:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        try:
            result = combined_backtest(df_4h, df_1d, base_config, funding)
            combined_results[symbol] = result
        except Exception as e:
            log(f"  {symbol}: ERROR - {e}")

        del df_4h, df_1d
        gc.collect()

    if combined_results:
        log(f"{'Symbol':<12} {'Total%':>8} {'Ann%':>7} {'DD%':>6} {'Trades':>7} {'WR%':>5} {'Long$':>8} {'Short$':>8}")
        log("-" * 70)

        for symbol in ALL_PAIRS:
            r = combined_results.get(symbol)
            if not r:
                continue

            years = r["years"]
            ann = r["total_return_pct"] / years if years > 0 else 0
            log(f"{symbol:<12} {r['total_return_pct']:>+7.1f}% {ann:>+6.1f}% "
                f"{r['max_dd']:>5.1f}% {r['num_trades']:>7} {r['win_rate']:>4.0f}% "
                f"${r['long_pnl']:>7.0f} ${r['short_pnl']:>7.0f}")

        # Portfolio summary
        all_ann = [r["total_return_pct"] / r["years"] for r in combined_results.values() if r["years"] > 0]
        if all_ann:
            log(f"\nPortfolio avg annual: {np.mean(all_ann):+.1f}%")
            log(f"Best pair:  {max(all_ann):+.1f}%")
            log(f"Worst pair: {min(all_ann):+.1f}%")
            profitable = sum(1 for a in all_ann if a > 0)
            log(f"Profitable: {profitable}/{len(all_ann)}")

    # ========================================================
    # Phase 5: Noise resilience for shorts
    # ========================================================
    write_section("PHASE 5: NOISE RESILIENCE (SHORT STRATEGY)")

    log("Adding 0.5% Gaussian noise to prices, running 30 iterations on BTC+ETH...")

    noise_pairs = ["BTC/USDT", "ETH/USDT"]
    noise_iters = 30
    noise_results = {s: [] for s in noise_pairs}

    for symbol in noise_pairs:
        df_4h, df_1d = load_data(symbol)
        if df_4h is None:
            continue
        funding = load_funding_rates(symbol)

        for seed in range(noise_iters):
            rng = np.random.RandomState(seed)
            df_4h_n = df_4h.copy()
            df_1d_n = df_1d.copy()

            # Add noise to OHLCV
            for col in ["open", "high", "low", "close"]:
                noise = 1 + rng.normal(0, 0.005, len(df_4h_n))
                df_4h_n[col] = df_4h_n[col] * noise
            # Fix OHLC consistency
            df_4h_n["high"] = df_4h_n[["open", "high", "low", "close"]].max(axis=1)
            df_4h_n["low"] = df_4h_n[["open", "high", "low", "close"]].min(axis=1)

            for col in ["open", "high", "low", "close"]:
                noise = 1 + rng.normal(0, 0.005, len(df_1d_n))
                df_1d_n[col] = df_1d_n[col] * noise
            df_1d_n["high"] = df_1d_n[["open", "high", "low", "close"]].max(axis=1)
            df_1d_n["low"] = df_1d_n[["open", "high", "low", "close"]].min(axis=1)

            sr = short_momentum_backtest(df_4h_n, df_1d_n, base_config, funding)
            noise_results[symbol].append(sr.total_return_pct)

        del df_4h, df_1d
        gc.collect()

        profitable = sum(1 for r in noise_results[symbol] if r > 0)
        mean_ret = np.mean(noise_results[symbol])
        std_ret = np.std(noise_results[symbol])
        log(f"  {symbol}: {profitable}/{noise_iters} profitable ({profitable/noise_iters*100:.0f}%), "
            f"mean {mean_ret:+.1f}%, std {std_ret:.1f}%")

    # ========================================================
    # Final verdict
    # ========================================================
    write_section("FINAL VERDICT")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Summarize
    short_profitable = sum(1 for s in short_summaries.values() if s["ret"] > 0)
    short_total = len(short_summaries)
    log(f"\nShort momentum profitable: {short_profitable}/{short_total} pairs")

    if short_summaries:
        avg_short = np.mean([s["ret"] for s in short_summaries.values()])
        avg_ann = np.mean([s["ann"] for s in short_summaries.values()])
        log(f"Avg short return: {avg_short:+.1f}% total ({avg_ann:+.1f}%/yr)")

    if combined_results:
        avg_combined = np.mean([r["total_return_pct"] / r["years"] for r in combined_results.values() if r["years"] > 0])
        log(f"Avg combined (long+short) return: {avg_combined:+.1f}%/yr")

    long_avg = np.mean([s["ret"] for s in long_summaries.values()]) if long_summaries else 0
    long_years = 4.15  # approximate
    log(f"Avg long-only return: {long_avg:+.1f}% total ({long_avg / long_years:+.1f}%/yr)")

    log(f"\nKey question: Does short momentum justify building futures capability?")
    if combined_results:
        avg_c = np.mean([r["total_return_pct"] / r["years"] for r in combined_results.values() if r["years"] > 0])
        if avg_c > long_avg / long_years + 5:
            log(f"VERDICT: YES — combined {avg_c:+.1f}%/yr vs long-only {long_avg/long_years:+.1f}%/yr = +{avg_c - long_avg/long_years:.1f}% improvement")
        elif avg_c > long_avg / long_years:
            log(f"VERDICT: MARGINAL — combined {avg_c:+.1f}%/yr vs long-only {long_avg/long_years:+.1f}%/yr = modest improvement")
        else:
            log(f"VERDICT: NO — combined {avg_c:+.1f}%/yr does not beat long-only {long_avg/long_years:+.1f}%/yr")

    log(f"\nFull report: {REPORT_FILE}")


def combined_backtest(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    config: dict,
    funding_rates: dict | None = None,
) -> dict:
    """Run both long and short strategies on the same capital.

    Long when daily is bullish, short when daily is bearish, cash otherwise.
    Only one position at a time.
    """
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

    # Prepare indicators
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)
        df_1d[f"dc_low_{p}"] = df_1d["low"].rolling(window=p).min().shift(1)

    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    # Also need reversed channels for short entries
    df_4h["lowest_low_entry"] = df_4h["low"].rolling(window=entry_period).min().shift(1)
    df_4h["highest_high_exit"] = df_4h["high"].rolling(window=exit_period).max().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    if vol_scale:
        df_4h["atr_median"] = df_4h["atr"].rolling(window=vol_lookback).median()

    # Daily regime
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
    position = None  # {"side": "long"|"short", ...}
    bars_since_exit = cooldown_bars
    long_pnl = 0
    short_pnl = 0
    trades = []
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
            if position["side"] == "long":
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
                    long_pnl += pnl
                    trades.append({"pnl": pnl, "side": "long"})
                    position = None
                    bars_since_exit = 0

            else:  # short
                if low < position["lowest_since_entry"]:
                    position["lowest_since_entry"] = low
                chandelier = position["lowest_since_entry"] + atr_stop_mult * atr
                effective_stop = min(chandelier, position["stop_loss"])
                exit_ch = row["highest_high_exit"] if not pd.isna(row["highest_high_exit"]) else float("inf")

                # Funding (every 2 bars on 4h ≈ every 8h)
                if funding_rates and i % 2 == 0:
                    daily_rate = funding_rates.get(bar_date, 0)
                    payment = position["notional"] * daily_rate / 3
                    capital += payment

                if close >= effective_stop or close > exit_ch:
                    amount = position["amount"]
                    fee = amount * close * TRADE_FEE
                    pnl = (position["entry_price"] - close) * amount - fee
                    capital += pnl
                    short_pnl += pnl
                    trades.append({"pnl": pnl, "side": "short"})
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
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
            continue

        bars_since_exit += 1
        if bars_since_exit < cooldown_bars:
            equity_curve.append(capital)
            if capital > peak:
                peak = capital
            continue

        # Entry logic
        effective_risk_pct = risk_pct
        if vol_scale and not pd.isna(row.get("atr_median")) and row["atr_median"] > 0:
            vol_ratio = row["atr_median"] / atr
            vol_ratio = max(0.5, min(2.0, vol_ratio))
            effective_risk_pct = risk_pct * vol_ratio

        # LONG entry
        if regime == "bull" and close > row["highest_high"] and not pd.isna(row["highest_high"]):
            stop_loss = close - atr_stop_mult * atr
            risk_per_unit = close - stop_loss
            if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                risk_amount = capital * (effective_risk_pct / 100)
                amount = risk_amount / risk_per_unit
                cost = amount * close
                fee = cost * 0.001
                if cost + fee <= capital * 0.95:
                    capital -= fee
                    position = {
                        "side": "long", "entry_price": close,
                        "amount": amount, "stop_loss": stop_loss,
                        "highest_since_entry": high,
                    }

        # SHORT entry
        elif regime == "bear" and close < row["lowest_low_entry"] and not pd.isna(row["lowest_low_entry"]):
            stop_loss = close + atr_stop_mult * atr
            risk_per_unit = stop_loss - close
            if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                risk_amount = capital * (effective_risk_pct / 100)
                amount = risk_amount / risk_per_unit
                notional = amount * close
                fee = notional * TRADE_FEE
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
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Close any open position
    if position:
        final = df_4h.iloc[-1]
        if position["side"] == "long":
            pnl = (final["close"] - position["entry_price"]) * position["amount"] - position["amount"] * final["close"] * 0.001
            long_pnl += pnl
        else:
            pnl = (position["entry_price"] - final["close"]) * position["amount"] - position["amount"] * final["close"] * TRADE_FEE
            short_pnl += pnl
        capital += pnl
        trades.append({"pnl": pnl, "side": position["side"]})

    winners = [t for t in trades if t["pnl"] > 0]
    years = len(df_4h) / (6 * 365)

    return {
        "total_return_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "final_capital": capital,
        "max_dd": max_dd,
        "num_trades": len(trades),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "years": years,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        with open(REPORT_FILE, "a") as f:
            f.write(f"\nFATAL ERROR: {e}\n")
            traceback.print_exc(file=f)
