"""Strategy research v4: Pyramiding + volatility-weighted allocation.

V3 found MTF ensemble with 16.2%/yr, Sharpe 1.65, 100% noise resilience.
The edge is proven — now maximize capital deployment:

1. PYRAMIDING: Add units to winning positions during strong trends (up to max_units)
2. FAST RE-ENTRY: When stopped out but daily trend is 3/3 bullish, skip cooldown
3. VOLATILITY-WEIGHTED portfolio: Allocate more capital to better risk-adjusted pairs
4. Compare v4 pyramid vs v3 single-position

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v4.py" bot
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

REPORT_FILE = "data/strategy_research_v4_report.txt"
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
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    resampled = df.resample(freq).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def load_data(symbol: str):
    """Load hourly data and return 4h and daily DataFrames (memory efficient)."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        log(f"  Downloading {symbol}...")
        try:
            download_ohlcv("binance", symbol, "1h", "2022-01-01")
            df = load_cached_data("binance", symbol, "1h")
        except Exception as e:
            log(f"  ERROR: {e}")
            return None, None, 0
    if df is None:
        return None, None, 0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    n_1h = len(df)
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    del df
    gc.collect()
    return df_4h, df_1d, n_1h


def pyramid_backtest(df_4h: pd.DataFrame, df_1d: pd.DataFrame, config: dict):
    """Multi-TF backtest WITH pyramiding.

    Key differences from v3 single-position:
    - Maintains a list of 'units' (sub-positions), each with own entry/stop
    - New 4h breakout while in a position = ADD a unit (up to max_units)
    - Each unit has its own chandelier stop tracked independently
    - Fast re-entry: skip cooldown when daily trend is unanimous (3/3)
    - Exit all units when composite stop is hit
    """
    import ta

    # Config
    daily_periods = config.get("daily_periods", [20, 50, 100])
    daily_min_votes = config.get("daily_min_votes", 2)
    entry_period = config.get("entry_period_4h", 14)
    exit_period = config.get("exit_period_4h", 7)
    atr_period = config.get("atr_period", 14)
    atr_stop_mult = config.get("atr_stop_mult", 3.0)
    risk_pct = config.get("risk_per_trade_pct", 5.0)
    volume_mult = config.get("volume_mult", 1.0)
    cooldown_bars = config.get("cooldown_bars", 2)
    max_units = config.get("max_units", 3)
    # Minimum price move (in ATRs) before adding a new unit
    add_threshold_atr = config.get("add_threshold_atr", 0.5)
    # Max portfolio exposure as fraction of capital
    max_exposure_pct = config.get("max_exposure_pct", 95.0)

    # Compute daily indicators
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)

    # Compute 4h indicators
    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    df_4h["volume_ma"] = df_4h["volume"].rolling(window=20).mean()

    # Build daily trend signal lookup (date -> vote_count)
    daily_votes = {}
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        votes = 0
        for p in daily_periods:
            dc_high = row.get(f"dc_high_{p}")
            if not pd.isna(dc_high) and row["close"] > dc_high:
                votes += 1
        daily_votes[date] = votes

    # Run 4h backtest with pyramiding
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    units = []  # List of {entry_price, amount, stop_loss, highest_since_entry}
    bars_since_exit = cooldown_bars
    lookback = max(entry_period, exit_period, atr_period) + 1

    for i in range(lookback, len(df_4h)):
        row = df_4h.iloc[i]
        close = row["close"]
        high = row["high"]
        atr = row["atr"]
        ts = row["timestamp"]

        if pd.isna(atr) or pd.isna(row.get("highest_high")):
            unrealized = sum((close - u["entry_price"]) * u["amount"] for u in units)
            equity_curve.append(capital + unrealized)
            continue

        bar_date = ts.date()
        n_votes = daily_votes.get(bar_date, 0)
        is_bullish = n_votes >= daily_min_votes
        is_unanimous = n_votes >= len(daily_periods)

        # --- Manage existing units ---
        if units:
            # Update highest price for all units
            for u in units:
                if high > u["highest_since_entry"]:
                    u["highest_since_entry"] = high

            # Check exit conditions for each unit independently
            exit_ch = row["lowest_low"] if not pd.isna(row["lowest_low"]) else 0
            units_to_close = []

            for idx, u in enumerate(units):
                chandelier = u["highest_since_entry"] - atr_stop_mult * atr
                effective_stop = max(chandelier, u["stop_loss"])

                if close <= effective_stop or close < exit_ch:
                    units_to_close.append(idx)

            # Close stopped-out units (reverse order to preserve indices)
            if units_to_close:
                for idx in reversed(units_to_close):
                    u = units.pop(idx)
                    fee = u["amount"] * close * 0.001
                    pnl = (close - u["entry_price"]) * u["amount"] - fee
                    capital += pnl
                    trades.append(BacktestTrade(
                        timestamp=ts, side="sell", price=close,
                        amount=u["amount"], cost=u["amount"] * close, fee=fee,
                        pnl=pnl, strategy="pyramid",
                    ))

                if not units:
                    bars_since_exit = 0

        # --- Entry / pyramid logic ---
        if not units:
            bars_since_exit += 1

        # Determine cooldown: skip if unanimous daily trend
        effective_cooldown = 0 if is_unanimous else cooldown_bars

        can_enter = (
            is_bullish
            and close > row["highest_high"]
            and len(units) < max_units
        )

        # Volume check
        vol_ok = (
            row["volume"] > row["volume_ma"] * volume_mult
            if not pd.isna(row.get("volume_ma")) and row["volume_ma"] > 0
            else True
        )

        if can_enter and vol_ok:
            # For new entry (no units), check cooldown
            if not units and bars_since_exit < effective_cooldown:
                pass  # Still in cooldown
            else:
                # For pyramid add: price must have moved up from last unit
                should_add = True
                if units:
                    last_entry = units[-1]["entry_price"]
                    min_move = add_threshold_atr * atr
                    if close < last_entry + min_move:
                        should_add = False

                if should_add:
                    # Position sizing: risk a fraction of current equity
                    current_equity = capital + sum(
                        (close - u["entry_price"]) * u["amount"] for u in units
                    )
                    stop_loss = close - atr_stop_mult * atr
                    risk_per_unit = close - stop_loss

                    if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                        risk_amount = current_equity * (risk_pct / 100)
                        amount = risk_amount / risk_per_unit
                        cost = amount * close
                        fee = cost * 0.001

                        # Check total exposure
                        total_cost = sum(u["amount"] * close for u in units) + cost
                        if total_cost + fee <= current_equity * (max_exposure_pct / 100):
                            capital -= fee
                            units.append({
                                "entry_price": close,
                                "amount": amount,
                                "stop_loss": stop_loss,
                                "highest_since_entry": high,
                            })
                            trades.append(BacktestTrade(
                                timestamp=ts, side="buy", price=close,
                                amount=amount, cost=cost, fee=fee,
                                strategy="pyramid",
                            ))

        unrealized = sum((close - u["entry_price"]) * u["amount"] for u in units)
        equity_curve.append(capital + unrealized)

    # Close remaining units
    if units:
        final = df_4h.iloc[-1]
        for u in units:
            fee = u["amount"] * final["close"] * 0.001
            pnl = (final["close"] - u["entry_price"]) * u["amount"] - fee
            capital += pnl
            trades.append(BacktestTrade(
                timestamp=final["timestamp"], side="sell", price=final["close"],
                amount=u["amount"], cost=u["amount"] * final["close"], fee=fee,
                pnl=pnl, strategy="pyramid",
            ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("pyramid", df_4h, trades, equity_curve, capital)


# Configs to test
CONFIGS = {
    # v3 baseline (single position, for comparison)
    "v3_single_5pct": {
        "max_units": 1,
        "risk_per_trade_pct": 5.0,
        "add_threshold_atr": 999,  # Never pyramids
        "max_exposure_pct": 95.0,
    },
    # 2 units, conservative pyramid
    "pyramid_2u_5pct": {
        "max_units": 2,
        "risk_per_trade_pct": 5.0,
        "add_threshold_atr": 1.0,
        "max_exposure_pct": 95.0,
    },
    # 3 units, classic Turtle-style
    "pyramid_3u_5pct": {
        "max_units": 3,
        "risk_per_trade_pct": 5.0,
        "add_threshold_atr": 0.5,
        "max_exposure_pct": 95.0,
    },
    # 3 units, tighter add threshold (add faster)
    "pyramid_3u_5pct_fast": {
        "max_units": 3,
        "risk_per_trade_pct": 5.0,
        "add_threshold_atr": 0.3,
        "max_exposure_pct": 95.0,
    },
    # 4 units, aggressive
    "pyramid_4u_5pct": {
        "max_units": 4,
        "risk_per_trade_pct": 5.0,
        "add_threshold_atr": 0.5,
        "max_exposure_pct": 95.0,
    },
    # 3 units with higher risk per unit
    "pyramid_3u_7pct": {
        "max_units": 3,
        "risk_per_trade_pct": 7.0,
        "add_threshold_atr": 0.5,
        "max_exposure_pct": 95.0,
    },
    # 2 units with ATR 3.5 (wider stops = bigger positions)
    "pyramid_2u_5pct_atr35": {
        "max_units": 2,
        "risk_per_trade_pct": 5.0,
        "add_threshold_atr": 0.5,
        "atr_stop_mult": 3.5,
        "max_exposure_pct": 95.0,
    },
}

# Base config shared by all (overridden by each CONFIGS entry)
BASE_CONFIG = {
    "daily_periods": [20, 50, 100],
    "daily_min_votes": 2,
    "entry_period_4h": 14,
    "exit_period_4h": 7,
    "atr_stop_mult": 3.0,
    "volume_mult": 1.0,
    "cooldown_bars": 2,
    "risk_per_trade_pct": 5.0,
    "max_units": 3,
    "add_threshold_atr": 0.5,
    "max_exposure_pct": 95.0,
}


def main():
    start = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V4: Pyramiding + Optimized Allocation")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("Adding to winners during strong trends")
    log(f"Testing {len(CONFIGS)} configs across {len(ALL_SYMBOLS)} pairs")

    # Phase 1: Individual pair results
    write_section("PHASE 1: Individual Pair Performance")
    all_results = {}

    for symbol in ALL_SYMBOLS:
        log(f"\n--- {symbol} ---")
        df_4h, df_1d, n_1h = load_data(symbol)
        if df_4h is None or df_1d is None or len(df_4h) < 200:
            log(f"  Skipping: insufficient data")
            continue

        log(f"  Data: {n_1h} 1h, {len(df_4h)} 4h, {len(df_1d)} daily candles")

        for name, overrides in CONFIGS.items():
            try:
                cfg = {**BASE_CONFIG, **overrides}
                result = pyramid_backtest(df_4h, df_1d, cfg)

                log(f"  {name:30s} | Ret: {result.total_return_pct:+8.1f}% | "
                    f"DD: {result.max_drawdown_pct:5.1f}% | "
                    f"Sharpe: {result.sharpe_ratio:5.2f} | "
                    f"PF: {result.profit_factor:5.2f} | "
                    f"Trades: {result.total_trades:3d}")

                if name not in all_results:
                    all_results[name] = []
                all_results[name].append((symbol, {
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                }))
                del result
            except Exception as e:
                log(f"  {name}: ERROR - {e}")

        del df_4h, df_1d
        gc.collect()

    # Phase 2: Aggregate ranking
    write_section("PHASE 2: Aggregate Ranking")
    rankings = []
    log(f"\n{'Strategy':30s} | {'Avg Ret':>9s} | {'Avg DD':>7s} | {'Sharpe':>7s} | "
        f"{'R/DD':>6s} | {'Win':>5s} | {'Trades':>7s}")
    log("-" * 95)

    for name, pairs in sorted(all_results.items()):
        rets = [r["total_return_pct"] for _, r in pairs]
        dds = [r["max_drawdown_pct"] for _, r in pairs]
        sharpes = [r["sharpe_ratio"] for _, r in pairs]
        trade_counts = [r["total_trades"] for _, r in pairs]
        profitable = sum(1 for r in rets if r > 0)

        avg_ret = np.mean(rets)
        avg_dd = np.mean(dds)
        avg_sharpe = np.mean(sharpes)
        r_dd = avg_ret / avg_dd if avg_dd > 0 else 0
        win_pct = profitable / len(pairs) * 100

        log(f"{name:30s} | {avg_ret:+8.1f}% | {avg_dd:6.1f}% | {avg_sharpe:6.2f} | "
            f"{r_dd:5.2f} | {win_pct:4.0f}% | {np.mean(trade_counts):6.0f} | "
            f"{profitable}/{len(pairs)}")

        rankings.append({
            "name": name, "avg_return": avg_ret, "avg_dd": avg_dd,
            "avg_sharpe": avg_sharpe, "r_dd": r_dd, "win_pct": win_pct,
            "avg_trades": np.mean(trade_counts), "pairs": pairs,
        })

    rankings.sort(key=lambda x: x["r_dd"], reverse=True)
    log("\n--- RANKED BY R/DD ---")
    for i, s in enumerate(rankings):
        marker = " <-- BEST" if i == 0 else ""
        log(f"  #{i+1}: {s['name']:30s} | R/DD: {s['r_dd']:5.2f} | "
            f"Ret: {s['avg_return']:+.1f}% | DD: {s['avg_dd']:.1f}% | "
            f"Win: {s['win_pct']:.0f}%{marker}")

    # Also rank by absolute return
    by_return = sorted(rankings, key=lambda x: x["avg_return"], reverse=True)
    log("\n--- RANKED BY ABSOLUTE RETURN ---")
    for i, s in enumerate(by_return):
        log(f"  #{i+1}: {s['name']:30s} | Ret: {s['avg_return']:+.1f}% | "
            f"DD: {s['avg_dd']:.1f}% | R/DD: {s['r_dd']:.2f}")

    best = rankings[0]
    best_name = best["name"]
    best_overrides = CONFIGS[best_name]
    best_config = {**BASE_CONFIG, **best_overrides}

    # Phase 3: Noise test on best
    write_section("PHASE 3: Noise Resilience Test")
    log(f"Testing: {best_name}")
    log("0.5% Gaussian noise, 30 runs per pair, 6 pairs")

    noise_std = 0.005
    n_runs = 30
    test_symbols = ALL_SYMBOLS[:6]
    overall_pass = 0
    overall_total = 0

    for symbol in test_symbols:
        df_4h, df_1d, _ = load_data(symbol)
        if df_4h is None or df_1d is None:
            continue

        baseline = pyramid_backtest(df_4h, df_1d, best_config)
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
                r = pyramid_backtest(df_4h_n, df_1d_n, best_config)
                noisy_rets.append(r.total_return_pct)
                del r
            except Exception:
                noisy_rets.append(0.0)

        profitable = sum(1 for r in noisy_rets if r > 0)
        overall_pass += profitable
        overall_total += n_runs
        log(f"  {symbol:12s}: Base {base_ret:+7.1f}% | Noisy avg {np.mean(noisy_rets):+7.1f}% "
            f"(std {np.std(noisy_rets):5.1f}%) | {profitable}/{n_runs} ({profitable/n_runs*100:.0f}%)")

        del df_4h, df_1d
        gc.collect()

    if overall_total > 0:
        pct = overall_pass / overall_total * 100
        log(f"\n  OVERALL: {overall_pass}/{overall_total} ({pct:.0f}%)")
        log(f"  {'PASS' if pct >= 70 else 'MARGINAL' if pct >= 50 else 'FAIL'}")

    # Phase 4: Portfolio simulation — equal weight vs volatility-weighted
    write_section("PHASE 4: Portfolio Simulations")
    TOP_PAIRS = ["SOL/USDT", "DOGE/USDT", "ETH/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT"]

    # Also test the highest-return config for portfolio
    highest_ret_name = by_return[0]["name"]
    highest_ret_config = {**BASE_CONFIG, **CONFIGS[highest_ret_name]}

    configs_to_portfolio = [
        (best_name + " (best R/DD)", best_config),
        (highest_ret_name + " (best return)", highest_ret_config),
    ]
    # Deduplicate if they're the same
    if best_name == highest_ret_name:
        configs_to_portfolio = [(best_name, best_config)]

    for label, cfg in configs_to_portfolio:
        log(f"\n--- {label} ---")

        pair_curves = {}
        pair_stats = {}
        min_len = float("inf")

        for symbol in TOP_PAIRS:
            df_4h, df_1d, _ = load_data(symbol)
            if df_4h is None or df_1d is None:
                continue
            result = pyramid_backtest(df_4h, df_1d, cfg)
            log(f"  {symbol:12s}: {result.total_return_pct:+8.1f}% | "
                f"DD: {result.max_drawdown_pct:5.1f}% | "
                f"PF: {result.profit_factor:5.2f} | Trades: {result.total_trades}")
            if result.equity_curve:
                pair_curves[symbol] = result.equity_curve
                pair_stats[symbol] = {
                    "ret": result.total_return_pct,
                    "dd": result.max_drawdown_pct,
                    "trades": result.total_trades,
                    "pf": result.profit_factor,
                }
                min_len = min(min_len, len(result.equity_curve))
            del result, df_4h, df_1d
            gc.collect()

        if not pair_curves:
            continue

        n = len(pair_curves)
        bars_per_year = 6 * 365.25

        # --- Equal weight portfolio ---
        per_pair = INITIAL_CAPITAL / n
        eq_portfolio = []
        for i in range(int(min_len)):
            total = sum(ec[i] / INITIAL_CAPITAL * per_pair
                        for ec in pair_curves.values() if i < len(ec))
            eq_portfolio.append(total)

        # --- Volatility-weighted portfolio ---
        # Weight by inverse max drawdown (less volatile = more capital)
        inv_dd = {sym: 1 / max(s["dd"], 1.0) for sym, s in pair_stats.items()}
        total_inv_dd = sum(inv_dd.values())
        weights = {sym: v / total_inv_dd for sym, v in inv_dd.items()}

        vw_portfolio = []
        for i in range(int(min_len)):
            total = sum(
                pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * weights[sym]
                for sym in pair_curves if i < len(pair_curves[sym])
            )
            vw_portfolio.append(total)

        # --- R/DD-weighted portfolio ---
        # Weight by return/drawdown ratio
        r_dd_scores = {sym: max(s["ret"] / max(s["dd"], 1.0), 0.1) for sym, s in pair_stats.items()}
        total_rdd = sum(r_dd_scores.values())
        rdd_weights = {sym: v / total_rdd for sym, v in r_dd_scores.items()}

        rdd_portfolio = []
        for i in range(int(min_len)):
            total = sum(
                pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * rdd_weights[sym]
                for sym in pair_curves if i < len(pair_curves[sym])
            )
            rdd_portfolio.append(total)

        for port_name, portfolio in [
            ("Equal weight", eq_portfolio),
            ("Inv-volatility weight", vw_portfolio),
            ("R/DD weight", rdd_portfolio),
        ]:
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
            annual = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

            rets = [(portfolio[i] - portfolio[i-1]) / portfolio[i-1]
                    for i in range(1, len(portfolio)) if portfolio[i-1] > 0]
            sharpe = (np.mean(rets) / np.std(rets) * (bars_per_year ** 0.5)
                      if rets and np.std(rets) > 0 else 0)

            log(f"\n  {port_name} ({n} pairs, {years:.1f}yr):")
            log(f"    Total:     {total_ret:+.1f}%")
            log(f"    Annual:    {annual:+.1f}%/year")
            log(f"    Max DD:    {max_dd:.1f}%")
            log(f"    Sharpe:    {sharpe:.2f}")
            log(f"    R/DD:      {total_ret / max_dd:.2f}" if max_dd > 0 else "    R/DD: inf")
            if annual >= 30:
                log(f"    *** TARGET 30%/yr ACHIEVED! ***")
            elif annual >= 20:
                log(f"    Close to target ({annual:.1f}%)")

        if rdd_weights:
            log(f"\n  R/DD allocation weights:")
            for sym in sorted(rdd_weights, key=rdd_weights.get, reverse=True):
                log(f"    {sym:12s}: {rdd_weights[sym]*100:5.1f}%")

        del pair_curves, pair_stats
        gc.collect()

    # Final summary
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start
    log(f"Runtime: {elapsed/60:.1f} minutes")
    log(f"\nBest R/DD strategy: {best_name}")
    log(f"Best return strategy: {highest_ret_name}")
    log(f"Config (best R/DD): {best_config}")
    log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
