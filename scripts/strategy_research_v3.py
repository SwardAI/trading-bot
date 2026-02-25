"""Strategy research v3: Multi-timeframe ensemble + optimized position sizing.

V2 results showed daily ensemble Donchian is noise-resistant (69% survive)
with R/DD=1.53 — 10x better than hourly EMA. But too few trades (7/pair/4yr)
and only 2.5%/yr.

V3 approach:
1. Daily ensemble as TREND FILTER (are 2/3 Donchian channels bullish?)
2. 4h channel breakout for ENTRY TIMING within daily trend
3. Higher risk per trade (3-5%, proven smooth CV=5% in v2)
4. Test combined signals vs pure daily/4h strategies

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v3.py" bot
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

REPORT_FILE = "data/strategy_research_v3_report.txt"
INITIAL_CAPITAL = 10000

ALL_SYMBOLS = [
    "ETH/USDT", "AVAX/USDT", "LINK/USDT",
    "DOT/USDT", "DOGE/USDT", "ADA/USDT", "SOL/USDT",
    "UNI/USDT", "ATOM/USDT", "NEAR/USDT",
]


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
    """Resample OHLCV to given frequency."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    resampled = df.resample(freq).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def load_data(symbol: str):
    """Load hourly data and return hourly, 4h, and daily DataFrames."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        log(f"  Downloading {symbol}...")
        try:
            download_ohlcv("binance", symbol, "1h", "2022-01-01")
            df = load_cached_data("binance", symbol, "1h")
        except Exception as e:
            log(f"  ERROR: {e}")
            return None, None, None
    if df is None:
        return None, None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_1h = df.reset_index(drop=True)
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    return df_1h, df_4h, df_1d


def multi_tf_backtest(df_4h: pd.DataFrame, df_1d: pd.DataFrame, config: dict):
    """Run multi-timeframe backtest: daily trend filter + 4h entry timing.

    Entry: Daily ensemble says bullish (2/3 channels) AND 4h breaks above
    its N-period high.
    Exit: Chandelier stop (ATR-based) OR 4h closes below exit channel.
    """
    import ta

    # Config
    daily_periods = config.get("daily_periods", [20, 50, 100])
    daily_min_votes = config.get("daily_min_votes", 2)
    entry_period = config.get("entry_period_4h", 20)
    exit_period = config.get("exit_period_4h", 10)
    atr_period = config.get("atr_period", 14)
    atr_stop_mult = config.get("atr_stop_mult", 3.0)
    risk_pct = config.get("risk_per_trade_pct", 3.0)
    volume_mult = config.get("volume_mult", 1.0)
    cooldown_bars = config.get("cooldown_bars", 3)

    # Compute daily indicators
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)
        df_1d[f"dc_low_{p}"] = df_1d["low"].rolling(window=max(p // 2, 5)).min().shift(1)

    # Compute 4h indicators
    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    df_4h["volume_ma"] = df_4h["volume"].rolling(window=20).mean()

    # Build daily trend signal lookup (date -> is_bullish)
    daily_bullish = {}
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        votes = 0
        for p in daily_periods:
            dc_high = row.get(f"dc_high_{p}")
            if not pd.isna(dc_high) and row["close"] > dc_high:
                votes += 1
        daily_bullish[date] = votes >= daily_min_votes

    # Run 4h backtest with daily filter
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None
    bars_since_exit = cooldown_bars
    lookback = max(entry_period, exit_period, atr_period) + 1

    for i in range(lookback, len(df_4h)):
        row = df_4h.iloc[i]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["atr"]
        ts = row["timestamp"]

        if pd.isna(atr) or pd.isna(row.get("highest_high")):
            equity_curve.append(capital)
            continue

        # Check daily trend for this bar's date
        bar_date = ts.date()
        is_daily_bullish = daily_bullish.get(bar_date, False)

        # Manage position
        if position:
            if high > position["highest_since_entry"]:
                position["highest_since_entry"] = high

            chandelier_stop = position["highest_since_entry"] - atr_stop_mult * atr
            effective_stop = max(chandelier_stop, position["stop_loss"])

            # Exit channel
            exit_ch = row["lowest_low"] if not pd.isna(row["lowest_low"]) else 0

            if close <= effective_stop or close < exit_ch:
                amount = position["amount"]
                fee = amount * close * 0.001  # 0.1% taker
                pnl = (close - position["entry_price"]) * amount - fee
                capital += pnl
                trades.append(BacktestTrade(
                    timestamp=ts, side="sell", price=close,
                    amount=amount, cost=amount * close, fee=fee,
                    pnl=pnl, strategy="multi_tf",
                ))
                position = None
                bars_since_exit = 0

            equity = capital + ((close - position["entry_price"]) * position["amount"] if position else 0)
            equity_curve.append(equity)
            continue

        bars_since_exit += 1
        if bars_since_exit < cooldown_bars:
            equity_curve.append(capital)
            continue

        # Entry: daily bullish + 4h breakout
        vol_ok = (
            row["volume"] > row["volume_ma"] * volume_mult
            if not pd.isna(row.get("volume_ma")) and row["volume_ma"] > 0
            else True
        )

        if is_daily_bullish and close > row["highest_high"] and vol_ok:
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
                        "side": "long",
                    }
                    trades.append(BacktestTrade(
                        timestamp=ts, side="buy", price=close,
                        amount=amount, cost=cost, fee=fee,
                        strategy="multi_tf",
                    ))

        equity = capital + ((close - position["entry_price"]) * position["amount"] if position else 0)
        equity_curve.append(equity)

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
            pnl=pnl, strategy="multi_tf",
        ))

    # Build result
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("multi_tf", df_4h, trades, equity_curve, capital)


def run_daily_only(df_1d: pd.DataFrame, config: dict):
    """Run pure daily ensemble for comparison."""
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    daily_config = {
        "signal_mode": "ensemble_donchian",
        "donchian_periods": config.get("daily_periods", [20, 50, 100]),
        "donchian_min_votes": config.get("daily_min_votes", 2),
        "donchian_exit_divisor": 2,
        "atr_stop_mult": config.get("atr_stop_mult", 3.0),
        "risk_per_trade_pct": config.get("risk_per_trade_pct", 3.0),
        "volume_mult": config.get("volume_mult", 1.0),
        "cooldown_bars": 3,
    }
    return engine.run_daily_trend_backtest(df_1d, config=daily_config, long_only=True)


# Strategy configs to test
CONFIGS = {
    # Multi-TF: daily filter + 4h entry (core strategy)
    "MTF_D20_50_100_4H20_3pct": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 20,
        "exit_period_4h": 10,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Higher risk (smooth per v2 scan)
    "MTF_D20_50_100_4H20_5pct": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 20,
        "exit_period_4h": 10,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 5.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # ATR 3.5 (best R/DD in v2 scan)
    "MTF_D20_50_100_4H20_3pct_atr35": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 20,
        "exit_period_4h": 10,
        "atr_stop_mult": 3.5,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # With volume filter on 4h
    "MTF_D20_50_100_4H20_3pct_vol": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 20,
        "exit_period_4h": 10,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.3,
        "cooldown_bars": 3,
    },
    # Faster 4h entry (14-period)
    "MTF_D20_50_100_4H14_3pct": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 14,
        "exit_period_4h": 7,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.0,
        "cooldown_bars": 2,
    },
    # Slower 4h entry (30-period)
    "MTF_D20_50_100_4H30_3pct": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 30,
        "exit_period_4h": 15,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Aggressive: 5% risk + ATR 3.5
    "MTF_aggressive_5pct_atr35": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "entry_period_4h": 20,
        "exit_period_4h": 10,
        "atr_stop_mult": 3.5,
        "risk_per_trade_pct": 5.0,
        "volume_mult": 1.0,
        "cooldown_bars": 3,
    },
    # Pure daily comparison (same risk level)
    "Daily_only_3pct": {
        "daily_periods": [20, 50, 100],
        "daily_min_votes": 2,
        "atr_stop_mult": 3.0,
        "risk_per_trade_pct": 3.0,
        "volume_mult": 1.3,
    },
}


def main():
    start = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V3: Multi-Timeframe Ensemble")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("Daily ensemble filter + 4h breakout timing")
    log(f"Testing {len(CONFIGS)} configs across {len(ALL_SYMBOLS)} pairs")

    # Phase 1: Individual pairs
    write_section("PHASE 1: Individual Pair Performance")
    all_results = {}  # {name: [(symbol, result), ...]}

    for symbol in ALL_SYMBOLS:
        log(f"\n--- {symbol} ---")
        df_1h, df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None or len(df_4h) < 200:
            log(f"  Skipping: insufficient data")
            continue

        log(f"  Data: {len(df_1h)} 1h, {len(df_4h)} 4h, {len(df_1d)} daily candles")

        for name, config in CONFIGS.items():
            try:
                if name.startswith("Daily_only"):
                    result = run_daily_only(df_1d, config)
                else:
                    result = multi_tf_backtest(df_4h, df_1d, config)

                r_dd = result.total_return_pct / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0
                log(f"  {name:40s} | Ret: {result.total_return_pct:+7.1f}% | DD: {result.max_drawdown_pct:5.1f}% | "
                    f"Sharpe: {result.sharpe_ratio:5.2f} | PF: {result.profit_factor:5.2f} | "
                    f"Trades: {result.total_trades:3d}")

                if name not in all_results:
                    all_results[name] = []
                all_results[name].append((symbol, result))
            except Exception as e:
                log(f"  {name}: ERROR - {e}")

        gc.collect()

    # Phase 2: Aggregate ranking
    write_section("PHASE 2: Aggregate Ranking")

    rankings = []
    log(f"\n{'Strategy':40s} | {'Avg Ret':>8s} | {'Avg DD':>7s} | {'Sharpe':>7s} | "
        f"{'R/DD':>6s} | {'Win':>5s} | {'Trades':>7s}")
    log("-" * 100)

    for name, pairs in sorted(all_results.items()):
        rets = [r.total_return_pct for _, r in pairs]
        dds = [r.max_drawdown_pct for _, r in pairs]
        sharpes = [r.sharpe_ratio for _, r in pairs]
        trades = [r.total_trades for _, r in pairs]
        profitable = sum(1 for r in rets if r > 0)

        avg_ret = np.mean(rets)
        avg_dd = np.mean(dds)
        avg_sharpe = np.mean(sharpes)
        r_dd = avg_ret / avg_dd if avg_dd > 0 else 0
        win_pct = profitable / len(pairs) * 100

        log(f"{name:40s} | {avg_ret:+7.1f}% | {avg_dd:6.1f}% | {avg_sharpe:6.2f} | "
            f"{r_dd:5.2f} | {win_pct:4.0f}% | {np.mean(trades):6.0f} | {profitable}/{len(pairs)}")

        rankings.append({
            "name": name, "avg_return": avg_ret, "avg_dd": avg_dd,
            "avg_sharpe": avg_sharpe, "r_dd": r_dd, "win_pct": win_pct,
            "avg_trades": np.mean(trades), "pairs": pairs,
        })

    # Composite score
    rankings.sort(key=lambda x: x["r_dd"], reverse=True)

    log("\n--- RANKED BY RISK-ADJUSTED RETURN (R/DD) ---")
    for i, s in enumerate(rankings):
        marker = " <-- BEST" if i == 0 else ""
        log(f"  #{i+1}: {s['name']:40s} | R/DD: {s['r_dd']:5.2f} | Ret: {s['avg_return']:+.1f}% | "
            f"DD: {s['avg_dd']:.1f}% | Win: {s['win_pct']:.0f}%{marker}")

    best = rankings[0]
    best_name = best["name"]
    best_config = CONFIGS[best_name]
    log(f"\n  BEST: {best_name}")

    # Phase 3: Noise test on best
    write_section("PHASE 3: Noise Resilience Test")
    log("0.5% Gaussian noise, 30 runs per pair, 5 pairs")

    noise_std = 0.005
    n_runs = 30
    test_symbols = ALL_SYMBOLS[:6]
    overall_pass = 0
    overall_total = 0

    for symbol in test_symbols:
        df_1h, df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue

        # Baseline
        if best_name.startswith("Daily_only"):
            baseline = run_daily_only(df_1d, best_config)
        else:
            baseline = multi_tf_backtest(df_4h, df_1d, best_config)
        base_ret = baseline.total_return_pct

        noisy_rets = []
        for seed in range(n_runs):
            rng = np.random.RandomState(seed)

            # Add noise to 4h data
            df_4h_n = df_4h.copy()
            n = len(df_4h_n)
            df_4h_n["close"] = df_4h_n["close"] * rng.normal(1.0, noise_std, n)
            df_4h_n["open"] = df_4h_n["open"] * rng.normal(1.0, noise_std, n)
            df_4h_n["high"] = df_4h_n[["open", "close", "high"]].max(axis=1) * rng.normal(1.0, noise_std * 0.5, n)
            df_4h_n["low"] = df_4h_n[["open", "close", "low"]].min(axis=1) * rng.normal(1.0, noise_std * 0.5, n)
            df_4h_n["high"] = df_4h_n[["open", "close", "high"]].max(axis=1)
            df_4h_n["low"] = df_4h_n[["open", "close", "low"]].min(axis=1)

            # Also noise daily data
            df_1d_n = df_1d.copy()
            nd = len(df_1d_n)
            df_1d_n["close"] = df_1d_n["close"] * rng.normal(1.0, noise_std, nd)
            df_1d_n["open"] = df_1d_n["open"] * rng.normal(1.0, noise_std, nd)
            df_1d_n["high"] = df_1d_n[["open", "close", "high"]].max(axis=1)
            df_1d_n["low"] = df_1d_n[["open", "close", "low"]].min(axis=1)

            try:
                if best_name.startswith("Daily_only"):
                    r = run_daily_only(df_1d_n, best_config)
                else:
                    r = multi_tf_backtest(df_4h_n, df_1d_n, best_config)
                noisy_rets.append(r.total_return_pct)
            except Exception:
                noisy_rets.append(0.0)

        profitable = sum(1 for r in noisy_rets if r > 0)
        overall_pass += profitable
        overall_total += n_runs
        avg_n = np.mean(noisy_rets)
        std_n = np.std(noisy_rets)

        log(f"  {symbol:12s}: Base {base_ret:+6.1f}% | Noisy avg {avg_n:+6.1f}% "
            f"(std {std_n:5.1f}%) | {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")
        gc.collect()

    if overall_total > 0:
        pct = overall_pass / overall_total * 100
        log(f"\n  OVERALL: {overall_pass}/{overall_total} ({pct:.0f}%)")
        if pct >= 70:
            log("  PASS")
        elif pct >= 50:
            log("  MARGINAL")
        else:
            log("  FAIL")

    # Phase 4: Portfolio simulation with best strategy
    write_section("PHASE 4: Portfolio Simulation")
    min_len = float("inf")
    pair_data = {}

    for symbol in ALL_SYMBOLS:
        _, df_4h, df_1d = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue
        if best_name.startswith("Daily_only"):
            result = run_daily_only(df_1d, best_config)
        else:
            result = multi_tf_backtest(df_4h, df_1d, best_config)
        if result.equity_curve:
            pair_data[symbol] = result
            min_len = min(min_len, len(result.equity_curve))

    n_pairs = len(pair_data)
    if n_pairs == 0:
        log("No data for portfolio sim")
    else:
        per_pair = INITIAL_CAPITAL / n_pairs
        portfolio = []
        for i in range(int(min_len)):
            total = sum(r.equity_curve[i] / INITIAL_CAPITAL * per_pair
                        for r in pair_data.values() if i < len(r.equity_curve))
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

        # For 4h data, years = bars / (6 bars/day * 365.25)
        bars_per_year = 6 * 365.25 if not best_name.startswith("Daily") else 365.25
        years = min_len / bars_per_year
        annual = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

        rets = [(portfolio[i] - portfolio[i-1]) / portfolio[i-1]
                for i in range(1, len(portfolio)) if portfolio[i-1] > 0]
        sharpe = np.mean(rets) / np.std(rets) * (365.25 ** 0.5) if rets and np.std(rets) > 0 else 0

        total_trades = sum(r.total_trades for r in pair_data.values())

        log(f"\n  Portfolio ({n_pairs} pairs, {years:.1f} years):")
        log(f"    Total return:      {total_ret:+.1f}%")
        log(f"    Annualized return: {annual:+.1f}%")
        log(f"    Max drawdown:      {max_dd:.1f}%")
        log(f"    Sharpe ratio:      {sharpe:.2f}")
        log(f"    R/DD:              {total_ret / max_dd:.2f}" if max_dd > 0 else "    R/DD: inf")
        log(f"    Total trades:      {total_trades}")
        log(f"    Trades/pair/year:  {total_trades / n_pairs / years:.1f}")

        log(f"\n  TARGET: 30%/year")
        if annual >= 30:
            log(f"    ACHIEVED! {annual:.1f}%/year")
        elif annual >= 15:
            log(f"    PROMISING: {annual:.1f}%/year — close to target")
        else:
            log(f"    {annual:.1f}%/year — below target")

        log(f"\n  Per-pair breakdown:")
        for sym, r in sorted(pair_data.items(), key=lambda x: x[1].total_return_pct, reverse=True):
            log(f"    {sym:12s}: {r.total_return_pct:+7.1f}% | DD: {r.max_drawdown_pct:5.1f}% | "
                f"Trades: {r.total_trades:3d} | PF: {r.profit_factor:.2f}")

    # Final summary
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start
    log(f"Runtime: {elapsed/60:.1f} minutes")
    log(f"\nBest strategy: {best_name}")
    log(f"Config: {best_config}")
    log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
