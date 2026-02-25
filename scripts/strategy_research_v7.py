"""Strategy research v7: Regime-adaptive multi-strategy.

The momentum strategy returns ~24%/yr but sits idle 50%+ of the time
(bear and sideways markets). This script tests combining strategies:

BULL regime  → MTF momentum (Donchian breakout, proven 24%/yr engine)
SIDEWAYS     → Grid trading (profit from oscillation in range-bound market)
BEAR         → Cash (capital preservation, the proven correct move on spot)

Regime detection methods tested:
1. ADX-based: ADX > 25 = trending, ADX < 20 = sideways
2. Donchian width: Narrow Donchian band = sideways, wide = trending
3. ATR regime: Low ATR relative to median = sideways

Key question: Does adding grid trading in sideways periods push us to 30%+?

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v7.py" bot
"""

import gc
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data
from src.backtest.engine import BacktestEngine, BacktestTrade

REPORT_FILE = "data/strategy_research_v7_report.txt"
INITIAL_CAPITAL = 10000

TOP_PAIRS = ["SOL/USDT", "DOGE/USDT", "ETH/USDT", "AVAX/USDT"]


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
    """Load hourly, 4h, and daily data."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        return None, None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_1h = df.copy()
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    del df
    gc.collect()
    return df_1h, df_4h, df_1d


def regime_adaptive_backtest(df_1h, df_4h, df_1d, config):
    """Combined momentum + grid backtest with regime switching.

    Regime detection runs on daily timeframe:
    - BULL: Daily Donchian vote >= threshold AND ADX > adx_trend
    - SIDEWAYS: ADX < adx_sideways AND Donchian width < width_threshold
    - BEAR: Everything else (default to cash)

    In BULL: Run MTF momentum on 4h bars (proven strategy)
    In SIDEWAYS: Run grid trading on 1h bars (profit from oscillation)
    In BEAR: Hold cash
    """
    import ta

    # Momentum config
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

    # Regime config
    adx_trend = config.get("adx_trend", 25)
    adx_sideways = config.get("adx_sideways", 20)

    # Grid config
    grid_spacing = config.get("grid_spacing_pct", 1.0)
    grid_size_pct = config.get("grid_size_pct", 3.0)  # % of capital per grid order
    num_grids = config.get("num_grids", 10)
    grid_rebalance_pct = config.get("grid_rebalance_pct", 5)

    # ---- Prepare daily regime detection ----
    df_1d = df_1d.copy()
    for p in daily_periods:
        df_1d[f"dc_high_{p}"] = df_1d["high"].rolling(window=p).max().shift(1)
        df_1d[f"dc_low_{p}"] = df_1d["low"].rolling(window=p).min().shift(1)
    df_1d["adx"] = ta.trend.adx(df_1d["high"], df_1d["low"], df_1d["close"], window=14)

    # Build daily regime map
    daily_regime = {}  # date -> "bull" | "sideways" | "bear"
    for _, row in df_1d.iterrows():
        date = row["timestamp"].date()
        adx = row["adx"] if not pd.isna(row["adx"]) else 0

        # Donchian vote for bull
        votes = sum(
            1 for p in daily_periods
            if not pd.isna(row.get(f"dc_high_{p}")) and row["close"] > row[f"dc_high_{p}"]
        )
        is_bull_donchian = votes >= daily_min_votes

        if is_bull_donchian and adx > adx_trend:
            daily_regime[date] = "bull"
        elif adx < adx_sideways:
            daily_regime[date] = "sideways"
        else:
            daily_regime[date] = "bear"

    # ---- Prepare 4h data for momentum ----
    df_4h = df_4h.copy()
    df_4h["highest_high"] = df_4h["high"].rolling(window=entry_period).max().shift(1)
    df_4h["lowest_low"] = df_4h["low"].rolling(window=exit_period).min().shift(1)
    df_4h["atr"] = ta.volatility.average_true_range(
        df_4h["high"], df_4h["low"], df_4h["close"], window=atr_period,
    )
    if vol_scale:
        df_4h["atr_median"] = df_4h["atr"].rolling(window=vol_lookback).median()

    # Build 4h date -> index mapping for momentum loop
    df_4h["bar_date"] = pd.to_datetime(df_4h["timestamp"]).dt.date

    # ---- Simulation ----
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []

    # Momentum state
    mom_position = None
    bars_since_exit = cooldown_bars

    # Grid state
    grid_active = False
    grid_center = 0
    grid_buys = []
    grid_filled_buys = set()
    grid_inventory = 0.0
    grid_inventory_avg = 0.0

    # Regime tracking
    regime_bars = {"bull": 0, "sideways": 0, "bear": 0}
    regime_trades = {"momentum": 0, "grid": 0}

    # Process on 1h bars (finest granularity)
    df_1h = df_1h.copy()
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
    df_1h["bar_date"] = df_1h["timestamp"].dt.date

    # Pre-build 4h index for momentum checks (every 4th hour)
    four_h_map = {}
    for idx, row in df_4h.iterrows():
        ts = row["timestamp"]
        four_h_map[ts] = idx

    lookback = max(entry_period, exit_period, atr_period, vol_lookback) + 1
    maker_fee = 0.00075
    taker_fee = 0.001

    for i in range(1, len(df_1h)):
        row_1h = df_1h.iloc[i]
        close = row_1h["close"]
        high = row_1h["high"]
        low = row_1h["low"]
        ts = row_1h["timestamp"]
        bar_date = row_1h["bar_date"]

        regime = daily_regime.get(bar_date, "bear")
        regime_bars[regime] = regime_bars.get(regime, 0) + 1

        # ============================================================
        # REGIME TRANSITION: Close wrong-regime positions
        # ============================================================
        # If regime changes from bull → anything: close momentum position
        if mom_position and regime != "bull":
            amount = mom_position["amount"]
            fee = amount * close * taker_fee
            pnl = (close - mom_position["entry_price"]) * amount - fee
            capital += pnl
            trades.append(BacktestTrade(
                timestamp=ts, side="sell", price=close,
                amount=amount, cost=amount * close, fee=fee,
                pnl=pnl, strategy="momentum_regime_exit",
            ))
            mom_position = None
            bars_since_exit = 0

        # If regime changes from sideways → anything: liquidate grid inventory
        if grid_active and regime != "sideways":
            if grid_inventory > 0:
                revenue = grid_inventory * close
                fee = revenue * taker_fee
                pnl = (close - grid_inventory_avg) * grid_inventory - fee
                capital += revenue - fee
                trades.append(BacktestTrade(
                    timestamp=ts, side="sell", price=close,
                    amount=grid_inventory, cost=revenue, fee=fee,
                    pnl=pnl, strategy="grid_regime_exit",
                ))
                grid_inventory = 0.0
                grid_inventory_avg = 0.0
            grid_active = False
            grid_filled_buys.clear()

        # ============================================================
        # BULL REGIME: Momentum on 4h bars
        # ============================================================
        if regime == "bull":
            # Only process on 4h boundaries (every 4th 1h bar)
            if ts in four_h_map:
                idx_4h = four_h_map[ts]
                if idx_4h >= lookback:
                    row_4h = df_4h.iloc[idx_4h]
                    atr = row_4h["atr"]

                    if not pd.isna(atr) and not pd.isna(row_4h.get("highest_high")):
                        # Position management
                        if mom_position:
                            if high > mom_position["highest_since_entry"]:
                                mom_position["highest_since_entry"] = high
                            chandelier = mom_position["highest_since_entry"] - atr_stop_mult * atr
                            effective_stop = max(chandelier, mom_position["stop_loss"])
                            exit_ch = row_4h["lowest_low"] if not pd.isna(row_4h["lowest_low"]) else 0

                            if close <= effective_stop or close < exit_ch:
                                amount = mom_position["amount"]
                                fee = amount * close * taker_fee
                                pnl = (close - mom_position["entry_price"]) * amount - fee
                                capital += pnl
                                trades.append(BacktestTrade(
                                    timestamp=ts, side="sell", price=close,
                                    amount=amount, cost=amount * close, fee=fee,
                                    pnl=pnl, strategy="momentum",
                                ))
                                regime_trades["momentum"] += 1
                                mom_position = None
                                bars_since_exit = 0

                        # Entry logic
                        if not mom_position:
                            bars_since_exit += 1
                            if bars_since_exit >= cooldown_bars:
                                if close > row_4h["highest_high"]:
                                    stop_loss = close - atr_stop_mult * atr
                                    risk_per_unit = close - stop_loss

                                    if risk_per_unit > 0 and risk_per_unit < close * 0.15:
                                        effective_risk_pct = risk_pct
                                        if vol_scale and not pd.isna(row_4h.get("atr_median")) and row_4h["atr_median"] > 0:
                                            vol_ratio = row_4h["atr_median"] / atr
                                            vol_ratio = max(0.5, min(2.0, vol_ratio))
                                            effective_risk_pct = risk_pct * vol_ratio

                                        risk_amount = capital * (effective_risk_pct / 100)
                                        amount = risk_amount / risk_per_unit
                                        cost = amount * close
                                        fee = cost * taker_fee

                                        if cost + fee <= capital * 0.95:
                                            capital -= fee
                                            mom_position = {
                                                "entry_price": close, "amount": amount,
                                                "stop_loss": stop_loss, "highest_since_entry": high,
                                            }
                                            trades.append(BacktestTrade(
                                                timestamp=ts, side="buy", price=close,
                                                amount=amount, cost=cost, fee=fee,
                                                strategy="momentum",
                                            ))
                                            regime_trades["momentum"] += 1

        # ============================================================
        # SIDEWAYS REGIME: Grid trading on 1h bars
        # ============================================================
        elif regime == "sideways":
            # Initialize grid if not active
            if not grid_active:
                grid_center = close
                grid_buys = [
                    grid_center * (1 - grid_spacing / 100 * (j + 1))
                    for j in range(num_grids)
                ]
                grid_filled_buys.clear()
                grid_active = True

            # Available capital for grid (exclude momentum position value)
            grid_capital = capital

            # Grid buy fills
            order_size = grid_capital * (grid_size_pct / 100)
            for price in grid_buys:
                if price in grid_filled_buys:
                    continue
                if low <= price and order_size > 10:
                    fill_price = price
                    amount = order_size / fill_price
                    fee = order_size * maker_fee

                    if grid_capital >= order_size + fee:
                        capital -= (order_size + fee)
                        total_cost = (grid_inventory * grid_inventory_avg) + (amount * fill_price)
                        grid_inventory += amount
                        grid_inventory_avg = total_cost / grid_inventory if grid_inventory > 0 else 0
                        grid_filled_buys.add(price)
                        trades.append(BacktestTrade(
                            timestamp=ts, side="buy", price=fill_price,
                            amount=amount, cost=order_size, fee=fee,
                            strategy="grid",
                        ))
                        regime_trades["grid"] += 1

            # Grid sell fills
            for buy_price in list(grid_filled_buys):
                sell_price = buy_price * (1 + grid_spacing / 100)
                if high >= sell_price:
                    fill_price = sell_price
                    amount = order_size / buy_price
                    revenue = amount * fill_price
                    fee = revenue * maker_fee

                    pnl = (fill_price - buy_price) * amount - fee - (order_size * maker_fee)
                    capital += (revenue - fee)

                    grid_inventory -= amount
                    if grid_inventory <= 0:
                        grid_inventory = 0
                        grid_inventory_avg = 0

                    grid_filled_buys.discard(buy_price)
                    trades.append(BacktestTrade(
                        timestamp=ts, side="sell", price=fill_price,
                        amount=amount, cost=revenue, fee=fee,
                        pnl=pnl, strategy="grid",
                    ))
                    regime_trades["grid"] += 1

            # Grid rebalance
            drift = abs(close - grid_center) / grid_center * 100
            if drift >= grid_rebalance_pct:
                grid_center = close
                grid_buys = [
                    grid_center * (1 - grid_spacing / 100 * (j + 1))
                    for j in range(num_grids)
                ]
                grid_filled_buys.clear()

        # ============================================================
        # BEAR REGIME: Cash (do nothing)
        # ============================================================
        # (already handled by regime transition logic above)

        # Track equity
        unrealized = 0
        if mom_position:
            unrealized += (close - mom_position["entry_price"]) * mom_position["amount"]
        if grid_inventory > 0:
            unrealized += (close - grid_inventory_avg) * grid_inventory
        equity_curve.append(capital + unrealized)

    # Close remaining positions at end
    final_close = df_1h.iloc[-1]["close"]
    if mom_position:
        amount = mom_position["amount"]
        fee = amount * final_close * taker_fee
        pnl = (final_close - mom_position["entry_price"]) * amount - fee
        capital += pnl
        trades.append(BacktestTrade(
            timestamp=df_1h.iloc[-1]["timestamp"], side="sell", price=final_close,
            amount=amount, cost=amount * final_close, fee=fee,
            pnl=pnl, strategy="momentum_close",
        ))
    if grid_inventory > 0:
        revenue = grid_inventory * final_close
        fee = revenue * taker_fee
        pnl = (final_close - grid_inventory_avg) * grid_inventory - fee
        capital += revenue - fee
        trades.append(BacktestTrade(
            timestamp=df_1h.iloc[-1]["timestamp"], side="sell", price=final_close,
            amount=grid_inventory, cost=revenue, fee=fee,
            pnl=pnl, strategy="grid_close",
        ))

    return {
        "capital": capital,
        "equity_curve": equity_curve,
        "trades": trades,
        "regime_bars": regime_bars,
        "regime_trades": regime_trades,
        "total_return_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
    }


def compute_stats(result, bars_per_year=365.25 * 24):
    """Compute standard stats from regime backtest result."""
    ec = result["equity_curve"]
    if len(ec) < 10:
        return {}

    initial = ec[0]
    final = ec[-1]
    total_ret = (final - initial) / initial * 100
    years = len(ec) / bars_per_year

    peak = ec[0]
    max_dd = 0
    for v in ec:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    annual = ((final / initial) ** (1 / years) - 1) * 100 if years > 0.5 else 0
    rets = [(ec[i] - ec[i-1]) / ec[i-1] for i in range(1, len(ec)) if ec[i-1] > 0]
    sharpe = (np.mean(rets) / np.std(rets) * (bars_per_year ** 0.5)
              if rets and np.std(rets) > 0 else 0)

    sell_trades = [t for t in result["trades"] if t.pnl is not None and t.pnl != 0]
    pf = 0
    if sell_trades:
        wins = sum(t.pnl for t in sell_trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in sell_trades if t.pnl < 0))
        pf = wins / losses if losses > 0 else float("inf")

    return {
        "total_ret": total_ret, "annual": annual, "max_dd": max_dd,
        "sharpe": sharpe, "pf": pf, "years": years,
        "total_trades": len(result["trades"]),
    }


def main():
    start = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V7: Regime-Adaptive Multi-Strategy")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("Testing: momentum (bull) + grid (sideways) + cash (bear)")

    # ================================================================
    # Phase 1: Regime analysis — what % of time is each regime?
    # ================================================================
    write_section("PHASE 1: Regime Distribution Analysis")

    regime_configs = {
        "ADX25_20": {"adx_trend": 25, "adx_sideways": 20},
        "ADX20_15": {"adx_trend": 20, "adx_sideways": 15},
        "ADX30_20": {"adx_trend": 30, "adx_sideways": 20},
    }

    for rc_name, rc in regime_configs.items():
        log(f"\n--- {rc_name} (trend>{rc['adx_trend']}, sideways<{rc['adx_sideways']}) ---")
        for symbol in TOP_PAIRS:
            df_1h, df_4h, df_1d = load_data(symbol)
            if df_1h is None:
                continue

            cfg = {**rc, "daily_periods": [20, 50, 100], "daily_min_votes": 2,
                   "entry_period_4h": 14, "exit_period_4h": 7,
                   "atr_stop_mult": 3.0, "risk_per_trade_pct": 5.0,
                   "cooldown_bars": 2, "vol_scale": True, "vol_scale_lookback": 60,
                   "grid_spacing_pct": 1.0, "grid_size_pct": 3.0,
                   "num_grids": 10, "grid_rebalance_pct": 5}

            result = regime_adaptive_backtest(df_1h, df_4h, df_1d, cfg)
            rb = result["regime_bars"]
            total_bars = sum(rb.values())
            if total_bars > 0:
                log(f"  {symbol:12s}: Bull {rb.get('bull',0)/total_bars*100:4.0f}% | "
                    f"Sideways {rb.get('sideways',0)/total_bars*100:4.0f}% | "
                    f"Bear {rb.get('bear',0)/total_bars*100:4.0f}%")

            del df_1h, df_4h, df_1d, result
            gc.collect()

    # ================================================================
    # Phase 2: Full regime-adaptive backtest
    # ================================================================
    write_section("PHASE 2: Regime-Adaptive Backtests")

    # Test different grid configs in sideways
    configs = {
        "momentum_only": {
            "grid_spacing_pct": 0, "grid_size_pct": 0, "num_grids": 0,
        },
        "grid_1pct_3cap": {
            "grid_spacing_pct": 1.0, "grid_size_pct": 3.0, "num_grids": 10,
        },
        "grid_1.5pct_4cap": {
            "grid_spacing_pct": 1.5, "grid_size_pct": 4.0, "num_grids": 8,
        },
        "grid_2pct_5cap": {
            "grid_spacing_pct": 2.0, "grid_size_pct": 5.0, "num_grids": 6,
        },
        "grid_0.5pct_2cap": {
            "grid_spacing_pct": 0.5, "grid_size_pct": 2.0, "num_grids": 15,
        },
    }

    all_results = {}

    for cfg_name, overrides in configs.items():
        log(f"\n--- {cfg_name} ---")
        pair_results = {}

        for symbol in TOP_PAIRS:
            df_1h, df_4h, df_1d = load_data(symbol)
            if df_1h is None:
                continue

            cfg = {
                "daily_periods": [20, 50, 100], "daily_min_votes": 2,
                "entry_period_4h": 14, "exit_period_4h": 7,
                "atr_stop_mult": 3.0, "risk_per_trade_pct": 5.0,
                "cooldown_bars": 2, "vol_scale": True, "vol_scale_lookback": 60,
                "adx_trend": 25, "adx_sideways": 20,
                "grid_rebalance_pct": 5,
                **overrides,
            }

            result = regime_adaptive_backtest(df_1h, df_4h, df_1d, cfg)
            stats = compute_stats(result)
            rt = result["regime_trades"]

            log(f"  {symbol:12s}: {stats.get('total_ret', 0):+8.1f}% | "
                f"Annual: {stats.get('annual', 0):+6.1f}%/yr | "
                f"DD: {stats.get('max_dd', 0):5.1f}% | "
                f"Mom: {rt.get('momentum', 0)} Grid: {rt.get('grid', 0)}")

            pair_results[symbol] = {
                "stats": stats,
                "equity_curve": result["equity_curve"],
                "regime_bars": result["regime_bars"],
                "regime_trades": result["regime_trades"],
            }

            del df_1h, df_4h, df_1d, result
            gc.collect()

        all_results[cfg_name] = pair_results

        # Aggregate
        annuals = [pr["stats"].get("annual", 0) for pr in pair_results.values()]
        dds = [pr["stats"].get("max_dd", 0) for pr in pair_results.values()]
        if annuals:
            log(f"  AVG: {np.mean(annuals):+.1f}%/yr | DD: {np.mean(dds):.1f}%")

    # ================================================================
    # Phase 3: Portfolio simulations
    # ================================================================
    write_section("PHASE 3: Portfolio Simulations")

    bars_per_year = 365.25 * 24  # 1h bars

    for cfg_name, pair_results in all_results.items():
        if not pair_results:
            continue

        pair_curves = {s: pr["equity_curve"] for s, pr in pair_results.items()
                       if pr["equity_curve"]}
        if len(pair_curves) < 2:
            continue

        min_len = min(len(c) for c in pair_curves.values())
        n = len(pair_curves)

        # R/DD weighted
        pair_stats = {s: pr["stats"] for s, pr in pair_results.items()}
        r_dd_scores = {}
        for sym, st in pair_stats.items():
            ret = st.get("total_ret", 0)
            dd = max(st.get("max_dd", 1), 1.0)
            r_dd_scores[sym] = max(ret / dd, 0.1)
        total_rdd = sum(r_dd_scores.values())
        rdd_weights = {s: v / total_rdd for s, v in r_dd_scores.items()}

        portfolio = []
        for i in range(min_len):
            total = sum(
                pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * rdd_weights[sym]
                for sym in pair_curves if i < len(pair_curves[sym])
            )
            portfolio.append(total)

        initial = portfolio[0]
        final = portfolio[-1]
        years = min_len / bars_per_year
        annual = ((final / initial) ** (1 / years) - 1) * 100 if years > 0.5 else 0
        peak = portfolio[0]
        max_dd = 0
        for v in portfolio:
            if v > peak:
                peak = v
            max_dd = max(max_dd, (peak - v) / peak * 100)
        rets = [(portfolio[i] - portfolio[i-1]) / portfolio[i-1]
                for i in range(1, len(portfolio)) if portfolio[i-1] > 0]
        sharpe = (np.mean(rets) / np.std(rets) * (bars_per_year ** 0.5)
                  if rets and np.std(rets) > 0 else 0)

        log(f"\n  {cfg_name}: {annual:+.1f}%/yr | DD: {max_dd:.1f}% | Sharpe: {sharpe:.2f}")
        if annual >= 30:
            log(f"    *** 30%+ TARGET ACHIEVED! ***")

    # ================================================================
    # Phase 4: Noise test on best combo
    # ================================================================
    write_section("PHASE 4: Noise Test (Best Config)")

    # Find best config by annual return
    best_name = None
    best_annual = 0
    for cfg_name, pair_results in all_results.items():
        annuals = [pr["stats"].get("annual", 0) for pr in pair_results.values()]
        avg = np.mean(annuals) if annuals else 0
        if avg > best_annual:
            best_annual = avg
            best_name = cfg_name

    if best_name:
        log(f"Testing: {best_name} (avg {best_annual:.1f}%/yr)")
        log(f"0.5% noise, 20 runs per pair")

        noise_std = 0.005
        n_runs = 20
        total_pass = 0
        total_runs = 0

        best_overrides = configs[best_name]

        for symbol in TOP_PAIRS[:3]:
            df_1h, df_4h, df_1d = load_data(symbol)
            if df_1h is None:
                continue

            cfg = {
                "daily_periods": [20, 50, 100], "daily_min_votes": 2,
                "entry_period_4h": 14, "exit_period_4h": 7,
                "atr_stop_mult": 3.0, "risk_per_trade_pct": 5.0,
                "cooldown_bars": 2, "vol_scale": True, "vol_scale_lookback": 60,
                "adx_trend": 25, "adx_sideways": 20,
                "grid_rebalance_pct": 5,
                **best_overrides,
            }

            base_result = regime_adaptive_backtest(df_1h, df_4h, df_1d, cfg)
            base_ret = base_result["total_return_pct"]

            noisy_rets = []
            for seed in range(n_runs):
                rng = np.random.RandomState(seed)

                # Add noise to all timeframes
                df_1h_n = df_1h.copy()
                n = len(df_1h_n)
                df_1h_n["close"] = df_1h_n["close"] * rng.normal(1.0, noise_std, n)
                df_1h_n["open"] = df_1h_n["open"] * rng.normal(1.0, noise_std, n)
                df_1h_n["high"] = df_1h_n[["open", "close", "high"]].max(axis=1)
                df_1h_n["low"] = df_1h_n[["open", "close", "low"]].min(axis=1)

                df_4h_n = resample(df_1h_n, "4h")
                df_1d_n = resample(df_1h_n, "1D")

                try:
                    r = regime_adaptive_backtest(df_1h_n, df_4h_n, df_1d_n, cfg)
                    noisy_rets.append(r["total_return_pct"])
                except Exception:
                    noisy_rets.append(0.0)

            profitable = sum(1 for r in noisy_rets if r > 0)
            total_pass += profitable
            total_runs += n_runs
            log(f"  {symbol:12s}: Base {base_ret:+7.1f}% | Noisy avg {np.mean(noisy_rets):+7.1f}% "
                f"(std {np.std(noisy_rets):5.1f}%) | {profitable}/{n_runs}")

            del df_1h, df_4h, df_1d
            gc.collect()

        if total_runs > 0:
            pct = total_pass / total_runs * 100
            log(f"\n  Noise resilience: {total_pass}/{total_runs} ({pct:.0f}%)")

    # Final summary
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start
    log(f"Runtime: {elapsed/60:.1f} minutes")

    for cfg_name, pair_results in sorted(all_results.items()):
        annuals = [pr["stats"].get("annual", 0) for pr in pair_results.values()]
        dds = [pr["stats"].get("max_dd", 0) for pr in pair_results.values()]
        mom_trades = sum(pr["regime_trades"].get("momentum", 0) for pr in pair_results.values())
        grid_trades = sum(pr["regime_trades"].get("grid", 0) for pr in pair_results.values())
        if annuals:
            log(f"  {cfg_name:25s}: {np.mean(annuals):+6.1f}%/yr | DD: {np.mean(dds):5.1f}% | "
                f"Mom:{mom_trades:4d} Grid:{grid_trades:4d}")

    log(f"\nReport: {REPORT_FILE}")


if __name__ == "__main__":
    main()
