"""Strategy research v5: Enhancement stacking on proven MTF ensemble.

V3/V4 established: Daily Donchian [20,50,100] + 4h breakout = 22.1%/yr R/DD-weighted.
Now test modular improvements to find what stacks:

1. VOLATILITY-SCALED SIZING: Risk more in low-vol, less in high-vol
2. PROFIT TARGETS: Lock in gains at 3-4x ATR instead of only trailing stop
3. PARTIAL TAKE-PROFIT: Sell 50% at 2x ATR, let rest ride with tighter stop
4. WEEKLY FILTER + DAILY ENTRY: Higher conviction, fewer trades
5. BTC/USDT: The most liquid pair we haven't tested
6. COMBINED: Stack the winning improvements
7. PORTFOLIO: R/DD-weighted with best combo

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v5.py" bot
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

REPORT_FILE = "data/strategy_research_v5_report.txt"
INITIAL_CAPITAL = 10000

TOP_PAIRS = ["SOL/USDT", "DOGE/USDT", "ETH/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT"]
BTC_PAIR = "BTC/USDT"
ALL_PAIRS = [BTC_PAIR] + TOP_PAIRS

# V3 baseline config (proven best)
BASE = {
    "daily_periods": [20, 50, 100],
    "daily_min_votes": 2,
    "entry_period_4h": 14,
    "exit_period_4h": 7,
    "atr_stop_mult": 3.0,
    "risk_per_trade_pct": 5.0,
    "volume_mult": 1.0,
    "cooldown_bars": 2,
    # v5 enhancements (all off by default)
    "vol_scale": False,
    "vol_scale_lookback": 60,
    "profit_target_atr": 0,       # 0 = disabled
    "partial_tp_atr": 0,          # 0 = disabled
    "partial_tp_pct": 0.5,        # Sell this fraction at partial TP
    "tighten_stop_after_tp": 1.5, # Tighten stop to 1.5x ATR after partial TP
}


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


def load_data(symbol: str, timeframes=("4h", "1D")):
    """Load hourly data and resample to requested timeframes."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        log(f"  Downloading {symbol}...")
        try:
            download_ohlcv("binance", symbol, "1h", "2022-01-01")
            df = load_cached_data("binance", symbol, "1h")
        except Exception as e:
            log(f"  ERROR: {e}")
            return {}
    if df is None:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    result = {"n_1h": len(df)}
    for tf in timeframes:
        result[tf] = resample(df, tf)
    del df
    gc.collect()
    return result


def enhanced_backtest(df_4h: pd.DataFrame, df_1d: pd.DataFrame, config: dict):
    """Enhanced MTF backtest with v5 improvements.

    Supports: vol-scaled sizing, profit targets, partial take-profit,
    and tightening stops after partial TP.
    """
    import ta

    # Core config
    daily_periods = config.get("daily_periods", [20, 50, 100])
    daily_min_votes = config.get("daily_min_votes", 2)
    entry_period = config.get("entry_period_4h", 14)
    exit_period = config.get("exit_period_4h", 7)
    atr_period = config.get("atr_period", 14)
    atr_stop_mult = config.get("atr_stop_mult", 3.0)
    risk_pct = config.get("risk_per_trade_pct", 5.0)
    volume_mult = config.get("volume_mult", 1.0)
    cooldown_bars = config.get("cooldown_bars", 2)

    # V5 enhancements
    vol_scale = config.get("vol_scale", False)
    vol_lookback = config.get("vol_scale_lookback", 60)
    profit_target_atr = config.get("profit_target_atr", 0)
    partial_tp_atr = config.get("partial_tp_atr", 0)
    partial_tp_pct = config.get("partial_tp_pct", 0.5)
    tighten_after_tp = config.get("tighten_stop_after_tp", 1.5)

    # Daily indicators
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)

    # 4h indicators
    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    df_4h["volume_ma"] = df_4h["volume"].rolling(window=20).mean()

    # For vol-scaling: compute median ATR over lookback
    if vol_scale:
        df_4h["atr_median"] = df_4h["atr"].rolling(window=vol_lookback).median()

    # Daily trend lookup
    daily_bullish = {}
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        votes = sum(
            1 for p in daily_periods
            if not pd.isna(row.get(f"dc_high_{p}")) and row["close"] > row[f"dc_high_{p}"]
        )
        daily_bullish[date] = votes >= daily_min_votes

    # Backtest loop
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None  # {entry_price, amount, stop_loss, highest_since_entry, partial_taken}
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

        # --- Position management ---
        if position:
            if high > position["highest_since_entry"]:
                position["highest_since_entry"] = high

            # Determine current stop multiplier
            current_mult = atr_stop_mult
            if position.get("partial_taken"):
                current_mult = tighten_after_tp

            chandelier = position["highest_since_entry"] - current_mult * atr
            effective_stop = max(chandelier, position["stop_loss"])
            exit_ch = row["lowest_low"] if not pd.isna(row["lowest_low"]) else 0

            # Check profit target (full exit)
            hit_profit_target = False
            if profit_target_atr > 0 and not position.get("partial_taken"):
                target_price = position["entry_price"] + profit_target_atr * position.get("entry_atr", atr)
                if close >= target_price:
                    hit_profit_target = True

            # Check partial take-profit
            if partial_tp_atr > 0 and not position.get("partial_taken"):
                partial_target = position["entry_price"] + partial_tp_atr * position.get("entry_atr", atr)
                if close >= partial_target:
                    sell_amount = position["amount"] * partial_tp_pct
                    fee = sell_amount * close * 0.001
                    pnl = (close - position["entry_price"]) * sell_amount - fee
                    capital += pnl
                    position["amount"] -= sell_amount
                    position["partial_taken"] = True
                    # Update stop to breakeven
                    position["stop_loss"] = max(position["stop_loss"], position["entry_price"])
                    trades.append(BacktestTrade(
                        timestamp=ts, side="sell", price=close,
                        amount=sell_amount, cost=sell_amount * close, fee=fee,
                        pnl=pnl, strategy="v5_partial",
                    ))

            # Exit conditions
            should_exit = (
                close <= effective_stop
                or close < exit_ch
                or hit_profit_target
            )

            if should_exit:
                amount = position["amount"]
                fee = amount * close * 0.001
                pnl = (close - position["entry_price"]) * amount - fee
                capital += pnl
                trades.append(BacktestTrade(
                    timestamp=ts, side="sell", price=close,
                    amount=amount, cost=amount * close, fee=fee,
                    pnl=pnl, strategy="v5",
                ))
                position = None
                bars_since_exit = 0

            unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
            equity_curve.append(capital + unrealized)
            continue

        # --- Entry logic ---
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
                # Vol-scaled risk: risk more when vol is low, less when high
                effective_risk_pct = risk_pct
                if vol_scale and not pd.isna(row.get("atr_median")) and row["atr_median"] > 0:
                    vol_ratio = row["atr_median"] / atr
                    # Clamp between 0.5x and 2x
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
                        "entry_atr": atr, "partial_taken": False,
                    }
                    trades.append(BacktestTrade(
                        timestamp=ts, side="buy", price=close,
                        amount=amount, cost=cost, fee=fee,
                        strategy="v5",
                    ))

        unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
        equity_curve.append(capital + unrealized)

    # Close remaining position
    if position:
        final = df_4h.iloc[-1]
        amount = position["amount"]
        fee = amount * final["close"] * 0.001
        pnl = (final["close"] - position["entry_price"]) * amount - fee
        capital += pnl
        trades.append(BacktestTrade(
            timestamp=final["timestamp"], side="sell", price=final["close"],
            amount=amount, cost=amount * final["close"], fee=fee,
            pnl=pnl, strategy="v5",
        ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("v5", df_4h, trades, equity_curve, capital)


def weekly_daily_backtest(df_1d: pd.DataFrame, df_1w: pd.DataFrame, config: dict):
    """Weekly Donchian filter + daily breakout entry."""
    import ta

    weekly_periods = config.get("weekly_periods", [10, 20, 40])
    weekly_min_votes = config.get("weekly_min_votes", 2)
    entry_period = config.get("entry_period_1d", 14)
    exit_period = config.get("exit_period_1d", 7)
    atr_period = config.get("atr_period", 14)
    atr_stop_mult = config.get("atr_stop_mult", 3.0)
    risk_pct = config.get("risk_per_trade_pct", 5.0)
    cooldown_bars = config.get("cooldown_bars", 2)

    # Weekly indicators
    df_1w = df_1w.copy()
    for p in weekly_periods:
        df_1w[f"dc_high_{p}"] = df_1w["high"].rolling(window=p).max().shift(1)

    # Daily indicators
    df_1d = df_1d.copy()
    df_1d["highest_high"] = df_1d["high"].rolling(window=entry_period).max().shift(1)
    df_1d["lowest_low"] = df_1d["low"].rolling(window=exit_period).min().shift(1)
    df_1d["atr"] = ta.volatility.average_true_range(
        df_1d["high"], df_1d["low"], df_1d["close"], window=atr_period,
    )

    # Weekly trend lookup (week start -> is_bullish)
    weekly_bullish = {}
    for _, row in df_1w.iterrows():
        # Use ISO week as key
        wk = row["timestamp"].isocalendar()[:2]
        votes = sum(
            1 for p in weekly_periods
            if not pd.isna(row.get(f"dc_high_{p}")) and row["close"] > row[f"dc_high_{p}"]
        )
        weekly_bullish[wk] = votes >= weekly_min_votes

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None
    bars_since_exit = cooldown_bars
    lookback = max(entry_period, exit_period, atr_period) + 1

    for i in range(lookback, len(df_1d)):
        row = df_1d.iloc[i]
        close = row["close"]
        high = row["high"]
        atr = row["atr"]
        ts = row["timestamp"]

        if pd.isna(atr) or pd.isna(row.get("highest_high")):
            unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
            equity_curve.append(capital + unrealized)
            continue

        wk = ts.isocalendar()[:2]
        is_bullish = weekly_bullish.get(wk, False)

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
                    pnl=pnl, strategy="weekly_daily",
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
                risk_amount = capital * (risk_pct / 100)
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
                        strategy="weekly_daily",
                    ))

        unrealized = (close - position["entry_price"]) * position["amount"] if position else 0
        equity_curve.append(capital + unrealized)

    if position:
        final = df_1d.iloc[-1]
        amount = position["amount"]
        fee = amount * final["close"] * 0.001
        pnl = (final["close"] - position["entry_price"]) * amount - fee
        capital += pnl
        trades.append(BacktestTrade(
            timestamp=final["timestamp"], side="sell", price=final["close"],
            amount=amount, cost=amount * final["close"], fee=fee,
            pnl=pnl, strategy="weekly_daily",
        ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("weekly_daily", df_1d, trades, equity_curve, capital)


# Enhancement configs to test
ENHANCEMENTS = {
    "A_baseline": {},
    "B_vol_scaled": {"vol_scale": True, "vol_scale_lookback": 60},
    "C_profit_target_4atr": {"profit_target_atr": 4.0},
    "D_profit_target_3atr": {"profit_target_atr": 3.0},
    "E_partial_tp_2atr": {"partial_tp_atr": 2.0, "partial_tp_pct": 0.5, "tighten_stop_after_tp": 1.5},
    "F_partial_tp_3atr": {"partial_tp_atr": 3.0, "partial_tp_pct": 0.5, "tighten_stop_after_tp": 1.5},
    "G_vol_plus_partial2": {"vol_scale": True, "partial_tp_atr": 2.0, "partial_tp_pct": 0.5, "tighten_stop_after_tp": 1.5},
    "H_vol_plus_target4": {"vol_scale": True, "profit_target_atr": 4.0},
}


def portfolio_stats(pair_curves: dict, min_len: int, bars_per_year: float,
                    weights: dict = None) -> dict:
    """Compute portfolio stats from per-pair equity curves."""
    n = len(pair_curves)
    if n == 0 or min_len < 10:
        return {}

    if weights is None:
        weights = {sym: 1.0 / n for sym in pair_curves}

    portfolio = []
    for i in range(int(min_len)):
        total = sum(
            pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * weights[sym]
            for sym in pair_curves if i < len(pair_curves[sym])
        )
        portfolio.append(total)

    initial = portfolio[0]
    final = portfolio[-1]
    total_ret = (final - initial) / initial * 100

    peak = portfolio[0]
    max_dd = 0
    for eq in portfolio:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    years = min_len / bars_per_year
    annual = ((final / initial) ** (1 / years) - 1) * 100 if years > 0.5 else 0
    rets = [(portfolio[i] - portfolio[i-1]) / portfolio[i-1]
            for i in range(1, len(portfolio)) if portfolio[i-1] > 0]
    sharpe = (np.mean(rets) / np.std(rets) * (bars_per_year ** 0.5)
              if rets and np.std(rets) > 0 else 0)

    return {
        "total_ret": total_ret, "annual": annual, "max_dd": max_dd,
        "sharpe": sharpe, "r_dd": total_ret / max_dd if max_dd > 0 else 0,
        "years": years,
    }


def main():
    start = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V5: Enhancement Stacking")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("Testing modular improvements on proven MTF ensemble")
    log(f"Enhancements: {len(ENHANCEMENTS)} configs across {len(ALL_PAIRS)} pairs")

    # ================================================================
    # Phase 1: Test each enhancement individually
    # ================================================================
    write_section("PHASE 1: Individual Enhancement Testing")
    all_results = {}  # {enhancement: [(symbol, stats_dict), ...]}

    for symbol in ALL_PAIRS:
        log(f"\n--- {symbol} ---")
        data = load_data(symbol)
        if not data or "4h" not in data or "1D" not in data:
            log(f"  Skipping: no data")
            continue
        df_4h, df_1d = data["4h"], data["1D"]
        if len(df_4h) < 200:
            log(f"  Skipping: insufficient data ({len(df_4h)} bars)")
            del data
            gc.collect()
            continue

        log(f"  Data: {data.get('n_1h', 0)} 1h, {len(df_4h)} 4h, {len(df_1d)} daily")

        for name, overrides in ENHANCEMENTS.items():
            try:
                cfg = {**BASE, **overrides}
                result = enhanced_backtest(df_4h, df_1d, cfg)
                log(f"  {name:25s} | Ret: {result.total_return_pct:+8.1f}% | "
                    f"DD: {result.max_drawdown_pct:5.1f}% | "
                    f"PF: {result.profit_factor:5.2f} | Trades: {result.total_trades:3d}")

                if name not in all_results:
                    all_results[name] = []
                all_results[name].append((symbol, {
                    "ret": result.total_return_pct,
                    "dd": result.max_drawdown_pct,
                    "sharpe": result.sharpe_ratio,
                    "pf": result.profit_factor,
                    "trades": result.total_trades,
                }))
                del result
            except Exception as e:
                log(f"  {name}: ERROR - {e}")

        del data, df_4h, df_1d
        gc.collect()

    # ================================================================
    # Phase 2: Weekly + Daily timeframe
    # ================================================================
    write_section("PHASE 2: Weekly Filter + Daily Entry")
    weekly_results = []

    WEEKLY_CONFIGS = {
        "W10_20_40_D14": {"weekly_periods": [10, 20, 40], "weekly_min_votes": 2,
                          "entry_period_1d": 14, "exit_period_1d": 7},
        "W10_20_40_D20": {"weekly_periods": [10, 20, 40], "weekly_min_votes": 2,
                          "entry_period_1d": 20, "exit_period_1d": 10},
    }

    for symbol in ALL_PAIRS:
        log(f"\n--- {symbol} ---")
        data = load_data(symbol, timeframes=("1D", "W"))
        if not data or "1D" not in data or "W" not in data:
            log(f"  Skipping")
            continue
        df_1d, df_1w = data["1D"], data["W"]

        for wname, wcfg in WEEKLY_CONFIGS.items():
            try:
                cfg = {**wcfg, "atr_stop_mult": 3.0, "risk_per_trade_pct": 5.0, "cooldown_bars": 2}
                result = weekly_daily_backtest(df_1d, df_1w, cfg)
                log(f"  {wname:25s} | Ret: {result.total_return_pct:+8.1f}% | "
                    f"DD: {result.max_drawdown_pct:5.1f}% | "
                    f"PF: {result.profit_factor:5.2f} | Trades: {result.total_trades:3d}")
                weekly_results.append((symbol, wname, {
                    "ret": result.total_return_pct,
                    "dd": result.max_drawdown_pct,
                    "sharpe": result.sharpe_ratio,
                    "pf": result.profit_factor,
                    "trades": result.total_trades,
                }))
                del result
            except Exception as e:
                log(f"  {wname}: ERROR - {e}")

        del data, df_1d, df_1w
        gc.collect()

    # Weekly aggregate
    for wname in WEEKLY_CONFIGS:
        rets = [r["ret"] for _, n, r in weekly_results if n == wname]
        dds = [r["dd"] for _, n, r in weekly_results if n == wname]
        if rets:
            profitable = sum(1 for r in rets if r > 0)
            log(f"\n  {wname}: Avg {np.mean(rets):+.1f}%, DD {np.mean(dds):.1f}%, "
                f"Win {profitable}/{len(rets)}")

    # ================================================================
    # Phase 3: Aggregate ranking of enhancements
    # ================================================================
    write_section("PHASE 3: Enhancement Ranking")

    rankings = []
    log(f"\n{'Enhancement':25s} | {'Avg Ret':>9s} | {'Avg DD':>7s} | {'R/DD':>6s} | "
        f"{'Sharpe':>7s} | {'Win':>5s} | {'Trades':>7s}")
    log("-" * 90)

    for name, pairs in sorted(all_results.items()):
        rets = [r["ret"] for _, r in pairs]
        dds = [r["dd"] for _, r in pairs]
        sharpes = [r["sharpe"] for _, r in pairs]
        trade_counts = [r["trades"] for _, r in pairs]
        profitable = sum(1 for r in rets if r > 0)

        avg_ret = np.mean(rets)
        avg_dd = np.mean(dds)
        avg_sharpe = np.mean(sharpes)
        r_dd = avg_ret / avg_dd if avg_dd > 0 else 0
        win_pct = profitable / len(pairs) * 100

        log(f"{name:25s} | {avg_ret:+8.1f}% | {avg_dd:6.1f}% | {r_dd:5.2f} | "
            f"{avg_sharpe:6.2f} | {win_pct:4.0f}% | {np.mean(trade_counts):6.0f} | "
            f"{profitable}/{len(pairs)}")

        rankings.append({
            "name": name, "avg_ret": avg_ret, "avg_dd": avg_dd,
            "r_dd": r_dd, "avg_sharpe": avg_sharpe, "win_pct": win_pct,
            "pairs": pairs,
        })

    # Rank by multiple criteria
    by_rdd = sorted(rankings, key=lambda x: x["r_dd"], reverse=True)
    by_ret = sorted(rankings, key=lambda x: x["avg_ret"], reverse=True)
    by_sharpe = sorted(rankings, key=lambda x: x["avg_sharpe"], reverse=True)

    log("\n--- RANKED BY R/DD ---")
    for i, s in enumerate(by_rdd):
        log(f"  #{i+1}: {s['name']:25s} R/DD:{s['r_dd']:5.2f} Ret:{s['avg_ret']:+.1f}% DD:{s['avg_dd']:.1f}%")

    log("\n--- RANKED BY RETURN ---")
    for i, s in enumerate(by_ret):
        log(f"  #{i+1}: {s['name']:25s} Ret:{s['avg_ret']:+.1f}% DD:{s['avg_dd']:.1f}% R/DD:{s['r_dd']:.2f}")

    # Identify best enhancement
    best = by_rdd[0]
    best_name = best["name"]
    best_overrides = ENHANCEMENTS[best_name]
    best_config = {**BASE, **best_overrides}

    # ================================================================
    # Phase 4: R/DD-weighted portfolio with best enhancement
    # ================================================================
    write_section("PHASE 4: Portfolio Simulations")

    # Test top 2 by R/DD and top 1 by return
    configs_to_test = [(by_rdd[0]["name"], ENHANCEMENTS[by_rdd[0]["name"]])]
    if by_ret[0]["name"] != by_rdd[0]["name"]:
        configs_to_test.append((by_ret[0]["name"], ENHANCEMENTS[by_ret[0]["name"]]))

    for label, overrides in configs_to_test:
        cfg = {**BASE, **overrides}
        log(f"\n--- {label} ---")

        pair_curves = {}
        pair_stats = {}
        min_len = float("inf")

        for symbol in ALL_PAIRS:
            data = load_data(symbol)
            if not data or "4h" not in data or "1D" not in data:
                continue
            result = enhanced_backtest(data["4h"], data["1D"], cfg)
            log(f"  {symbol:12s}: {result.total_return_pct:+8.1f}% | DD: {result.max_drawdown_pct:5.1f}% | "
                f"PF: {result.profit_factor:5.2f} | Trades: {result.total_trades}")
            if result.equity_curve:
                pair_curves[symbol] = result.equity_curve
                pair_stats[symbol] = {"ret": result.total_return_pct, "dd": result.max_drawdown_pct,
                                       "trades": result.total_trades, "pf": result.profit_factor}
                min_len = min(min_len, len(result.equity_curve))
            del result, data
            gc.collect()

        if not pair_curves:
            continue

        bars_per_year = 6 * 365.25

        # Equal weight
        eq = portfolio_stats(pair_curves, min_len, bars_per_year)
        log(f"\n  Equal weight ({len(pair_curves)} pairs, {eq.get('years', 0):.1f}yr):")
        log(f"    Annual: {eq.get('annual', 0):+.1f}%/yr | DD: {eq.get('max_dd', 0):.1f}% | "
            f"Sharpe: {eq.get('sharpe', 0):.2f} | R/DD: {eq.get('r_dd', 0):.2f}")

        # R/DD weighted
        r_dd_scores = {sym: max(s["ret"] / max(s["dd"], 1.0), 0.1) for sym, s in pair_stats.items()}
        total_rdd = sum(r_dd_scores.values())
        rdd_weights = {sym: v / total_rdd for sym, v in r_dd_scores.items()}

        rdd = portfolio_stats(pair_curves, min_len, bars_per_year, rdd_weights)
        log(f"\n  R/DD weighted:")
        log(f"    Annual: {rdd.get('annual', 0):+.1f}%/yr | DD: {rdd.get('max_dd', 0):.1f}% | "
            f"Sharpe: {rdd.get('sharpe', 0):.2f} | R/DD: {rdd.get('r_dd', 0):.2f}")
        if rdd.get("annual", 0) >= 30:
            log(f"    *** TARGET 30%/yr ACHIEVED! ***")
        elif rdd.get("annual", 0) >= 20:
            log(f"    Close to target ({rdd.get('annual', 0):.1f}%)")

        log(f"\n  R/DD allocation:")
        for sym in sorted(rdd_weights, key=rdd_weights.get, reverse=True):
            log(f"    {sym:12s}: {rdd_weights[sym]*100:5.1f}% | "
                f"Ret: {pair_stats[sym]['ret']:+.1f}% DD: {pair_stats[sym]['dd']:.1f}%")

        del pair_curves, pair_stats
        gc.collect()

    # ================================================================
    # Phase 5: Quick noise test on best
    # ================================================================
    write_section("PHASE 5: Noise Resilience (Quick Check)")
    log(f"Testing: {best_name}")
    log("0.5% noise, 20 runs per pair, 4 pairs")

    noise_std = 0.005
    n_runs = 20
    test_symbols = TOP_PAIRS[:4]
    overall_pass = 0
    overall_total = 0

    for symbol in test_symbols:
        data = load_data(symbol)
        if not data or "4h" not in data:
            continue
        df_4h, df_1d = data["4h"], data["1D"]

        baseline = enhanced_backtest(df_4h, df_1d, best_config)
        base_ret = baseline.total_return_pct
        del baseline

        noisy_rets = []
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
                r = enhanced_backtest(df_4h_n, df_1d_n, best_config)
                noisy_rets.append(r.total_return_pct)
                del r
            except Exception:
                noisy_rets.append(0.0)

        profitable = sum(1 for r in noisy_rets if r > 0)
        overall_pass += profitable
        overall_total += n_runs
        log(f"  {symbol:12s}: Base {base_ret:+7.1f}% | Noisy avg {np.mean(noisy_rets):+7.1f}% "
            f"(std {np.std(noisy_rets):5.1f}%) | {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")

        del data, df_4h, df_1d
        gc.collect()

    if overall_total > 0:
        pct = overall_pass / overall_total * 100
        log(f"\n  OVERALL: {overall_pass}/{overall_total} ({pct:.0f}%)")
        log(f"  {'PASS' if pct >= 70 else 'MARGINAL' if pct >= 50 else 'FAIL'}")

    # Final summary
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start
    log(f"Runtime: {elapsed/60:.1f} minutes")
    log(f"\nBest R/DD enhancement: {by_rdd[0]['name']} (R/DD: {by_rdd[0]['r_dd']:.2f})")
    log(f"Best return enhancement: {by_ret[0]['name']} (Avg: {by_ret[0]['avg_ret']:+.1f}%)")
    log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
