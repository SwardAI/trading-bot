"""Backtest regime detection filter on 4-year historical data.

Computes weekly BTC regime (bull/bear/sideways) for each week, then replays
grid + momentum backtests with and without regime adjustments. Compares:
  - Total return
  - Max drawdown
  - Per-year breakdown

Usage (on droplet):
    docker-compose run --rm crypto-bot python -u scripts/backtest_regime.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.data_fetcher import load_cached_data, load_cached_data_chunked, count_cached_rows
from src.backtest.engine import BacktestEngine, BacktestResult
from src.data.indicators import add_ema, add_adx

REPORT_FILE = "data/regime_backtest_report.txt"
INITIAL_CAPITAL = 10000

# Regime detection params (must match config/settings.yaml)
EMA_PERIOD = 20
SMA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20

# Regime adjustments (must match config/settings.yaml)
MOMENTUM_SIDEWAYS_SIZE_MULT = 0.7
GRID_BEAR_SIZE_MULT = 0.5
GRID_BEAR_STOP_LOSS_PCT = 5.0

# Strategy configs (must match live configs)
GRID_CONFIGS = [
    {"symbol": "BTC/USDT", "spacing": 0.5, "order_size": 75, "num_grids": 20, "stop_loss": 8},
    {"symbol": "ETH/USDT", "spacing": 1.5, "order_size": 50, "num_grids": 15, "stop_loss": 8},
]

MOMENTUM_CONFIG = {
    "ema_fast": 20, "ema_slow": 50,
    "trailing_stop_atr_multiplier": 3.5,
    "adx_min_strength": 20,
    "rsi_long_threshold": 50,
    "required_signals": ["ema", "adx"],
    "risk_per_trade_pct": 1.0,
}

MOMENTUM_SYMBOLS = ["ETH/USDT", "AVAX/USDT", "LINK/USDT"]


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


def compute_weekly_regimes(btc_1h: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly regime classification from BTC 1h data.

    Resamples to weekly OHLCV, computes EMA(20), SMA(20), ADX(14),
    then classifies each week as bull/bear/sideways.

    Returns:
        DataFrame with columns: week_start, week_end, regime, adx, price, ema, sma
    """
    btc = btc_1h.copy()
    btc = btc.set_index("timestamp").sort_index()

    # Resample to weekly OHLCV
    weekly = btc.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if len(weekly) < EMA_PERIOD + 5:
        log(f"  WARNING: Only {len(weekly)} weekly bars, need {EMA_PERIOD + 5}")
        return pd.DataFrame()

    # Compute indicators on weekly data
    weekly["ema"] = add_ema(weekly, EMA_PERIOD)
    weekly["sma"] = weekly["close"].rolling(window=SMA_PERIOD).mean()
    weekly["adx"] = add_adx(weekly, ADX_PERIOD)

    # Classify each week
    regimes = []
    for idx, row in weekly.iterrows():
        if pd.isna(row["ema"]) or pd.isna(row["sma"]) or pd.isna(row["adx"]):
            regime = "sideways"  # Not enough data yet
            confidence = 0.0
        else:
            price = row["close"]
            ema = row["ema"]
            sma = row["sma"]
            adx = row["adx"]
            above_ema = price > ema
            above_sma = price > sma
            trending = adx > ADX_THRESHOLD

            if above_ema and above_sma and trending:
                regime = "bull"
                confidence = min(adx / 40, 1.0)
            elif not above_ema and not above_sma and trending:
                regime = "bear"
                confidence = min(adx / 40, 1.0)
            else:
                regime = "sideways"
                confidence = 1.0 - min(adx / 40, 1.0)

        regimes.append({
            "week_end": idx,
            "week_start": idx - pd.Timedelta(days=6),
            "regime": regime,
            "adx": row["adx"] if not pd.isna(row["adx"]) else 0,
            "price": row["close"],
            "ema": row["ema"] if not pd.isna(row["ema"]) else 0,
            "sma": row["sma"] if not pd.isna(row["sma"]) else 0,
        })

    return pd.DataFrame(regimes)


def get_regime_for_timestamp(regimes_df: pd.DataFrame, ts: pd.Timestamp) -> str:
    """Look up the regime for a given timestamp."""
    if regimes_df.empty:
        return "sideways"
    # Find the most recent week that started before or at this timestamp
    valid = regimes_df[regimes_df["week_start"] <= ts]
    if valid.empty:
        return "sideways"
    return valid.iloc[-1]["regime"]


def run_grid_backtest_with_regime(
    df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    grid_spacing_pct: float,
    order_size_usd: float,
    num_grids: int,
    stop_loss_pct: float,
    initial_capital: float,
    apply_regime: bool,
) -> dict:
    """Run grid backtest with optional regime-based adjustments.

    In bear regime: halve order size, tighten stop loss to GRID_BEAR_STOP_LOSS_PCT.
    """
    engine_fee = BacktestEngine(initial_capital=initial_capital)
    maker_fee = engine_fee.maker_fee
    slippage = engine_fee.slippage_pct

    capital = initial_capital
    equity_curve = [capital]
    trades = []
    inventory = 0.0
    inventory_avg = 0.0

    center = df.iloc[0]["close"]
    grid_buys = engine_fee._calc_grid_prices(center, grid_spacing_pct, num_grids, "buy")
    filled_buys = set()

    bear_weeks = 0
    bull_weeks = 0
    sideways_weeks = 0
    bear_skipped_buys = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        low = row["low"]
        high = row["high"]
        close = row["close"]
        ts = row["timestamp"]

        # Get current regime
        regime = get_regime_for_timestamp(regimes_df, ts) if apply_regime else "bull"

        # Regime-adjusted parameters
        if apply_regime and regime == "bear":
            eff_order_size = order_size_usd * GRID_BEAR_SIZE_MULT
            eff_stop_loss = GRID_BEAR_STOP_LOSS_PCT
        else:
            eff_order_size = order_size_usd
            eff_stop_loss = stop_loss_pct

        # Check stop loss on inventory
        if inventory > 0 and inventory_avg > 0:
            loss_pct = (inventory_avg - close) / inventory_avg * 100
            if loss_pct >= eff_stop_loss:
                # Emergency sell all inventory
                revenue = inventory * close * (1 - slippage)
                fee = revenue * maker_fee
                capital += (revenue - fee)
                trades.append({"side": "sell", "pnl": revenue - fee - (inventory * inventory_avg), "regime": regime})
                inventory = 0.0
                inventory_avg = 0.0
                filled_buys.clear()
                # Rebalance grid after stop loss
                center = close
                grid_buys = engine_fee._calc_grid_prices(center, grid_spacing_pct, num_grids, "buy")
                equity_curve.append(capital)
                continue

        # Check buy fills
        for price in grid_buys:
            if price in filled_buys:
                continue
            if low <= price:
                fill_price = price * (1 + slippage)
                amount = eff_order_size / fill_price
                fee = eff_order_size * maker_fee

                if capital >= eff_order_size + fee:
                    capital -= (eff_order_size + fee)
                    total_cost = (inventory * inventory_avg) + (amount * fill_price)
                    inventory += amount
                    inventory_avg = total_cost / inventory if inventory > 0 else 0
                    filled_buys.add(price)
                    trades.append({"side": "buy", "pnl": 0, "regime": regime})

        # Check sell fills
        for buy_price in list(filled_buys):
            sell_price = buy_price * (1 + grid_spacing_pct / 100)
            if high >= sell_price:
                fill_price = sell_price * (1 - slippage)
                amount = eff_order_size / buy_price
                revenue = amount * fill_price
                fee = revenue * maker_fee

                capital += (revenue - fee)
                pnl = (fill_price - inventory_avg) * amount - fee - (eff_order_size * maker_fee)

                inventory -= amount
                if inventory <= 0:
                    inventory = 0
                    inventory_avg = 0

                filled_buys.discard(buy_price)
                trades.append({"side": "sell", "pnl": pnl, "regime": regime})

        # Rebalance
        drift = abs(close - center) / center * 100
        if drift >= 8:
            center = close
            grid_buys = engine_fee._calc_grid_prices(center, grid_spacing_pct, num_grids, "buy")
            filled_buys.clear()

        equity = capital + (inventory * close)
        equity_curve.append(equity)

    final_capital = capital + (inventory * df.iloc[-1]["close"])

    # Count regime weeks
    sell_trades = [t for t in trades if t["pnl"] != 0]
    bear_trades = [t for t in sell_trades if t["regime"] == "bear"]
    bull_trades = [t for t in sell_trades if t["regime"] == "bull"]

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "final_capital": final_capital,
        "total_return_pct": (final_capital - initial_capital) / initial_capital * 100,
        "max_drawdown_pct": max_dd,
        "total_trades": len(trades),
        "round_trips": len(sell_trades),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def run_momentum_backtest_with_regime(
    df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    config: dict,
    initial_capital: float,
    apply_regime: bool,
) -> dict:
    """Run momentum backtest with optional regime-based adjustments.

    In bear regime: skip all new entries.
    In sideways regime: reduce position size by MOMENTUM_SIDEWAYS_SIZE_MULT.
    """
    from src.data.indicators import compute_all_indicators

    engine = BacktestEngine(initial_capital=initial_capital)
    taker_fee = engine.taker_fee

    ema_fast_p = config.get("ema_fast", 20)
    ema_slow_p = config.get("ema_slow", 50)
    adx_min = config.get("adx_min_strength", 20)
    atr_mult = config.get("trailing_stop_atr_multiplier", 3.5)
    risk_pct = config.get("risk_per_trade_pct", 1.0)

    df = compute_all_indicators(df, config)

    capital = initial_capital
    equity_curve = [capital]
    trades = []
    position = None
    entries_skipped_bear = 0
    entries_sideways_reduced = 0

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = row["close"]
        ts = row["timestamp"]

        if pd.isna(row["ema_fast"]) or pd.isna(row["adx"]) or pd.isna(row["atr"]):
            equity_curve.append(capital)
            continue

        # Get regime
        regime = get_regime_for_timestamp(regimes_df, ts) if apply_regime else "bull"

        # Manage existing position
        if position:
            if position["side"] == "long" and close <= position["current_stop"]:
                pnl = (close - position["entry_price"]) * position["amount"]
                fee = position["amount"] * close * taker_fee
                pnl -= fee
                capital += pnl
                trades.append({"pnl": pnl, "regime": position.get("entry_regime", "unknown")})
                position = None
            elif position:
                engine._update_backtest_stop(position, close, atr_mult)

            equity = capital + (engine._position_value(position, close) if position else 0)
            equity_curve.append(equity)
            continue

        # Entry signals
        ema_cross_long = row["ema_fast"] > row["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]
        adx_ok = row["adx"] > adx_min if not pd.isna(row["adx"]) else False

        if ema_cross_long and adx_ok:
            # Regime filter
            if apply_regime and regime == "bear":
                entries_skipped_bear += 1
                equity_curve.append(capital)
                continue

            atr = row["atr"]
            stop_loss = close - (atr * atr_mult)
            risk_amount = capital * (risk_pct / 100)

            # Sideways size reduction
            if apply_regime and regime == "sideways":
                risk_amount *= MOMENTUM_SIDEWAYS_SIZE_MULT
                entries_sideways_reduced += 1

            risk_per_unit = close - stop_loss
            if risk_per_unit > 0:
                amount = risk_amount / risk_per_unit
                fee = amount * close * taker_fee
                position = {
                    "side": "long", "entry_price": close,
                    "amount": amount, "stop_loss": stop_loss,
                    "current_stop": stop_loss, "initial_risk": risk_per_unit,
                    "atr": atr, "entry_regime": regime,
                }
                capital -= fee

        equity = capital + (engine._position_value(position, close) if position else 0)
        equity_curve.append(equity)

    # Close remaining position
    if position:
        final_close = df.iloc[-1]["close"]
        pnl = (final_close - position["entry_price"]) * position["amount"]
        fee = position["amount"] * final_close * taker_fee
        pnl -= fee
        capital += pnl
        trades.append({"pnl": pnl, "regime": position.get("entry_regime", "unknown")})

    final_capital = capital

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "final_capital": final_capital,
        "total_return_pct": (final_capital - initial_capital) / initial_capital * 100,
        "max_drawdown_pct": max_dd,
        "total_trades": len(trades),
        "entries_skipped_bear": entries_skipped_bear,
        "entries_sideways_reduced": entries_sideways_reduced,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def yearly_breakdown(equity_curve: list[float], df: pd.DataFrame, initial_capital: float) -> dict:
    """Break equity curve into yearly returns."""
    years = sorted(df["timestamp"].dt.year.unique())
    yearly = {}
    for year in years:
        mask = df["timestamp"].dt.year == year
        indices = df.index[mask]
        if len(indices) < 2:
            continue
        # equity_curve is offset by 1 (starts before first candle)
        start_idx = max(0, indices[0])
        end_idx = min(len(equity_curve) - 1, indices[-1])
        start_eq = equity_curve[start_idx]
        end_eq = equity_curve[end_idx]
        if start_eq > 0:
            yearly[year] = (end_eq - start_eq) / start_eq * 100
    return yearly


def main():
    start_time = time.time()

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(f"Regime Filter Backtest — {datetime.utcnow().isoformat()}\n")
        f.write(f"{'=' * 70}\n\n")

    log(f"Regime filter backtest started at {datetime.utcnow().isoformat()}")

    # ---------------------------------------------------------------
    # Step 1: Compute weekly BTC regimes
    # ---------------------------------------------------------------
    write_section("STEP 1: COMPUTE WEEKLY BTC REGIMES")

    btc_1h = load_cached_data("binance", "BTC/USDT", "1h")
    if btc_1h is None or len(btc_1h) < 500:
        log("ERROR: Need BTC/USDT 1h data. Run deep_analysis.py first.")
        return
    btc_1h["timestamp"] = pd.to_datetime(btc_1h["timestamp"], utc=True)

    regimes_df = compute_weekly_regimes(btc_1h)
    log(f"  Computed {len(regimes_df)} weekly regime classifications")

    # Regime distribution
    regime_counts = regimes_df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(regimes_df) * 100
        log(f"  {regime.upper():>10}: {count} weeks ({pct:.1f}%)")

    # Show regime transitions
    log("\n  Weekly regime timeline (last 20 weeks):")
    for _, row in regimes_df.tail(20).iterrows():
        log(f"    {row['week_end'].strftime('%Y-%m-%d')}: {row['regime'].upper():>8} "
            f"(ADX={row['adx']:.1f}, price=${row['price']:,.0f}, "
            f"EMA=${row['ema']:,.0f}, SMA=${row['sma']:,.0f})")

    # ---------------------------------------------------------------
    # Step 2: Grid backtest WITH vs WITHOUT regime filter
    # ---------------------------------------------------------------
    write_section("STEP 2: GRID BACKTEST — WITH vs WITHOUT REGIME FILTER")

    for cfg in GRID_CONFIGS:
        symbol = cfg["symbol"]
        log(f"\n  {symbol} (spacing={cfg['spacing']}%, size=${cfg['order_size']}, grids={cfg['num_grids']}):")

        # Use 1h data for grid backtest (1m would need year-by-year chunking)
        df = load_cached_data("binance", symbol, "1h")
        if df is None or len(df) < 200:
            log(f"  SKIP {symbol} — no 1h data")
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.reset_index(drop=True)

        # WITHOUT regime filter
        result_no = run_grid_backtest_with_regime(
            df, regimes_df,
            grid_spacing_pct=cfg["spacing"],
            order_size_usd=cfg["order_size"],
            num_grids=cfg["num_grids"],
            stop_loss_pct=cfg["stop_loss"],
            initial_capital=INITIAL_CAPITAL,
            apply_regime=False,
        )

        # WITH regime filter
        result_yes = run_grid_backtest_with_regime(
            df, regimes_df,
            grid_spacing_pct=cfg["spacing"],
            order_size_usd=cfg["order_size"],
            num_grids=cfg["num_grids"],
            stop_loss_pct=cfg["stop_loss"],
            initial_capital=INITIAL_CAPITAL,
            apply_regime=True,
        )

        log(f"    {'':20} {'WITHOUT regime':>16} {'WITH regime':>16} {'Improvement':>14}")
        log(f"    {'':20} {'-'*16} {'-'*16} {'-'*14}")
        log(f"    {'Total Return':20} {result_no['total_return_pct']:>+15.1f}% {result_yes['total_return_pct']:>+15.1f}% {result_yes['total_return_pct'] - result_no['total_return_pct']:>+13.1f}%")
        log(f"    {'Max Drawdown':20} {result_no['max_drawdown_pct']:>15.1f}% {result_yes['max_drawdown_pct']:>15.1f}% {result_no['max_drawdown_pct'] - result_yes['max_drawdown_pct']:>+13.1f}%")
        log(f"    {'Final Capital':20} ${result_no['final_capital']:>14,.0f} ${result_yes['final_capital']:>14,.0f}")
        log(f"    {'Round Trips':20} {result_no['round_trips']:>16} {result_yes['round_trips']:>16}")

        # Yearly breakdown
        yearly_no = yearly_breakdown(result_no["equity_curve"], df, INITIAL_CAPITAL)
        yearly_yes = yearly_breakdown(result_yes["equity_curve"], df, INITIAL_CAPITAL)

        log(f"\n    Yearly breakdown:")
        log(f"    {'Year':>6} {'WITHOUT':>10} {'WITH':>10} {'Delta':>10}")
        for year in sorted(set(list(yearly_no.keys()) + list(yearly_yes.keys()))):
            no_val = yearly_no.get(year, 0)
            yes_val = yearly_yes.get(year, 0)
            log(f"    {year:>6} {no_val:>+9.1f}% {yes_val:>+9.1f}% {yes_val - no_val:>+9.1f}%")

    # ---------------------------------------------------------------
    # Step 3: Momentum backtest WITH vs WITHOUT regime filter
    # ---------------------------------------------------------------
    write_section("STEP 3: MOMENTUM BACKTEST — WITH vs WITHOUT REGIME FILTER")

    for symbol in MOMENTUM_SYMBOLS:
        df = load_cached_data("binance", symbol, "1h")
        if df is None or len(df) < 200:
            log(f"  SKIP {symbol} — no 1h data")
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.reset_index(drop=True)

        log(f"\n  {symbol}:")

        # WITHOUT regime filter
        result_no = run_momentum_backtest_with_regime(
            df, regimes_df, MOMENTUM_CONFIG, INITIAL_CAPITAL, apply_regime=False,
        )

        # WITH regime filter
        result_yes = run_momentum_backtest_with_regime(
            df, regimes_df, MOMENTUM_CONFIG, INITIAL_CAPITAL, apply_regime=True,
        )

        log(f"    {'':20} {'WITHOUT regime':>16} {'WITH regime':>16} {'Improvement':>14}")
        log(f"    {'':20} {'-'*16} {'-'*16} {'-'*14}")
        log(f"    {'Total Return':20} {result_no['total_return_pct']:>+15.1f}% {result_yes['total_return_pct']:>+15.1f}% {result_yes['total_return_pct'] - result_no['total_return_pct']:>+13.1f}%")
        log(f"    {'Max Drawdown':20} {result_no['max_drawdown_pct']:>15.1f}% {result_yes['max_drawdown_pct']:>15.1f}% {result_no['max_drawdown_pct'] - result_yes['max_drawdown_pct']:>+13.1f}%")
        log(f"    {'Final Capital':20} ${result_no['final_capital']:>14,.0f} ${result_yes['final_capital']:>14,.0f}")
        log(f"    {'Trades':20} {result_no['total_trades']:>16} {result_yes['total_trades']:>16}")

        log(f"    Bear entries skipped:     {result_yes['entries_skipped_bear']}")
        log(f"    Sideways entries reduced:  {result_yes['entries_sideways_reduced']}")

        # Yearly
        yearly_no = yearly_breakdown(result_no["equity_curve"], df, INITIAL_CAPITAL)
        yearly_yes = yearly_breakdown(result_yes["equity_curve"], df, INITIAL_CAPITAL)

        log(f"\n    Yearly breakdown:")
        log(f"    {'Year':>6} {'WITHOUT':>10} {'WITH':>10} {'Delta':>10}")
        for year in sorted(set(list(yearly_no.keys()) + list(yearly_yes.keys()))):
            no_val = yearly_no.get(year, 0)
            yes_val = yearly_yes.get(year, 0)
            log(f"    {year:>6} {no_val:>+9.1f}% {yes_val:>+9.1f}% {yes_val - no_val:>+9.1f}%")

    # ---------------------------------------------------------------
    # Step 4: Combined portfolio WITH vs WITHOUT
    # ---------------------------------------------------------------
    write_section("STEP 4: COMBINED PORTFOLIO — WITH vs WITHOUT REGIME FILTER")

    log("  Portfolio: 60% grid (BTC+ETH), 40% momentum (ETH+AVAX+LINK)")
    grid_capital = 3000
    momentum_per_symbol = 4000 / len(MOMENTUM_SYMBOLS)

    for label, apply_regime in [("WITHOUT regime filter", False), ("WITH regime filter", True)]:
        log(f"\n  --- {label} ---")
        total_final = 0
        total_initial = 0

        for cfg in GRID_CONFIGS:
            df = load_cached_data("binance", cfg["symbol"], "1h")
            if df is None:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.reset_index(drop=True)

            result = run_grid_backtest_with_regime(
                df, regimes_df,
                grid_spacing_pct=cfg["spacing"],
                order_size_usd=cfg["order_size"],
                num_grids=cfg["num_grids"],
                stop_loss_pct=cfg["stop_loss"],
                initial_capital=grid_capital,
                apply_regime=apply_regime,
            )
            pnl = result["final_capital"] - grid_capital
            total_final += result["final_capital"]
            total_initial += grid_capital
            log(f"    Grid {cfg['symbol']}: ${grid_capital} -> ${result['final_capital']:,.0f} ({result['total_return_pct']:+.1f}%, DD={result['max_drawdown_pct']:.1f}%)")

        for symbol in MOMENTUM_SYMBOLS:
            df = load_cached_data("binance", symbol, "1h")
            if df is None:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.reset_index(drop=True)

            result = run_momentum_backtest_with_regime(
                df, regimes_df, MOMENTUM_CONFIG, momentum_per_symbol, apply_regime=apply_regime,
            )
            total_final += result["final_capital"]
            total_initial += momentum_per_symbol
            log(f"    Mom  {symbol}: ${momentum_per_symbol:.0f} -> ${result['final_capital']:,.0f} ({result['total_return_pct']:+.1f}%, DD={result['max_drawdown_pct']:.1f}%)")

        total_return = (total_final - total_initial) / total_initial * 100
        log(f"\n    TOTAL: ${total_initial:,.0f} -> ${total_final:,.0f} ({total_return:+.1f}%)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    write_section("SUMMARY")
    log("""
  The regime filter adjusts strategy behavior based on BTC weekly data:
    - BULL:     Full position sizes, all entries allowed
    - SIDEWAYS: Momentum at 70% size, grid normal
    - BEAR:     Momentum skips entries, grid at 50% size + 5% stop loss

  Compare the WITH vs WITHOUT columns above to see the exact impact
  on returns, drawdowns, and trade counts over the 4-year backtest period.

  Key metrics to evaluate:
    1. Does drawdown decrease significantly in bear years (2022)?
    2. Are bull-market returns preserved (2023-2024)?
    3. Is the risk-adjusted return (return / max_dd) better with the filter?
""")

    elapsed = time.time() - start_time
    log(f"\nBacktest complete! Time: {elapsed:.0f}s")
    log(f"Full report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
