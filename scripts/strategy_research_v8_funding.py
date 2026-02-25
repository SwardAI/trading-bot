"""Strategy research v8: Funding rate arbitrage backtesting.

Downloads real historical funding rates from Binance and simulates
the delta-neutral arb strategy (long spot + short perp) to answer:
1. What are realistic annual returns from funding rate arb?
2. How does it perform across different market regimes?
3. Which pairs are best for funding arb?
4. What's the optimal entry/exit threshold?

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v8_funding.py" bot
"""

import csv
import gc
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORT_FILE = "data/strategy_research_v8_funding_report.txt"
FUNDING_DATA_DIR = Path("data/historical/funding_rates")
INITIAL_CAPITAL = 10000

# Pairs to test — our 7 core pairs + extras
CORE_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT"]
EXTRA_PAIRS = ["DOT/USDT", "ATOM/USDT", "NEAR/USDT", "UNI/USDT"]
ALL_PAIRS = CORE_PAIRS + EXTRA_PAIRS

# Fee model
SPOT_TAKER_FEE = 0.001       # 0.1% per trade
FUTURES_TAKER_FEE = 0.0004   # 0.04% per trade (maker rebate on Binance futures)
FUTURES_MAKER_FEE = 0.0002   # 0.02% maker (we'd use limit orders for arb)

# We use maker fees for arb since it's not time-sensitive
ENTRY_FEE = SPOT_TAKER_FEE + FUTURES_MAKER_FEE  # open both legs
EXIT_FEE = SPOT_TAKER_FEE + FUTURES_MAKER_FEE    # close both legs
TOTAL_ROUND_TRIP_FEE = ENTRY_FEE + EXIT_FEE       # ~0.24% total


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


# ============================================================
# Phase 1: Download historical funding rates
# ============================================================

def download_funding_rates(symbol: str, since: str = "2022-01-01") -> pd.DataFrame:
    """Download historical funding rates from Binance.

    Uses ccxt's fetch_funding_rate_history with pagination.
    Funding rates are every 8 hours on Binance.

    Returns DataFrame with columns: [timestamp, funding_rate, symbol]
    """
    import ccxt

    FUNDING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    cache_file = FUNDING_DATA_DIR / f"binance_{safe_symbol}_funding.csv"

    # Check cache
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        if len(df) > 100:
            log(f"  {symbol}: loaded {len(df)} cached funding rates")
            return df

    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
    })
    exchange.load_markets()

    futures_symbol = f"{symbol}:USDT"
    if futures_symbol not in exchange.markets:
        log(f"  {symbol}: no perpetual futures found, skipping")
        return pd.DataFrame()

    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    all_rates = []
    page = 0

    log(f"  {symbol}: downloading funding rates from {since}...")

    while True:
        try:
            rates = exchange.fetch_funding_rate_history(
                futures_symbol, since=since_ts, limit=1000
            )
        except Exception as e:
            log(f"  {symbol}: API error on page {page}: {e}")
            time.sleep(2)
            if page > 0:
                break
            return pd.DataFrame()

        if not rates:
            break

        for r in rates:
            all_rates.append({
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "funding_rate": r["fundingRate"],  # Already decimal (e.g., 0.0001 = 0.01%)
                "symbol": symbol,
            })

        since_ts = rates[-1]["timestamp"] + 1
        page += 1

        if len(rates) < 1000:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_rates:
        log(f"  {symbol}: no funding rate data available")
        return pd.DataFrame()

    df = pd.DataFrame(all_rates)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Save to cache
    df.to_csv(cache_file, index=False)
    log(f"  {symbol}: downloaded {len(df)} funding rates ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")

    return df


# ============================================================
# Phase 2: Funding rate analysis (raw data insights)
# ============================================================

def analyze_funding_rates(all_funding: dict[str, pd.DataFrame]):
    """Analyze raw funding rate patterns across pairs."""
    write_section("PHASE 2: FUNDING RATE ANALYSIS")

    rows = []
    for symbol, df in all_funding.items():
        if df.empty:
            continue

        rates = df["funding_rate"].values
        positive_pct = (rates > 0).sum() / len(rates) * 100
        mean_rate = rates.mean()
        median_rate = np.median(rates)
        std_rate = rates.std()

        # Annualized return from just holding the arb (no fees)
        # 3 funding payments per day * 365 days
        annual_gross = mean_rate * 3 * 365 * 100  # percentage

        # Annual return after round-trip fees (assume hold 7 days avg)
        # Entry/exit fees amortized: TOTAL_ROUND_TRIP_FEE / 7 * 365
        annual_fee_drag = TOTAL_ROUND_TRIP_FEE / 7 * 365 * 100  # percentage
        annual_net = annual_gross - annual_fee_drag

        rows.append({
            "symbol": symbol,
            "count": len(rates),
            "positive_pct": positive_pct,
            "mean_rate": mean_rate * 100,  # to bps-like
            "median_rate": median_rate * 100,
            "std_rate": std_rate * 100,
            "annual_gross": annual_gross,
            "annual_net": annual_net,
            "min_rate": rates.min() * 100,
            "max_rate": rates.max() * 100,
        })

    rows.sort(key=lambda x: x["annual_net"], reverse=True)

    log(f"\n{'Symbol':<12} {'Count':>6} {'Pos%':>6} {'Mean':>8} {'Median':>8} {'Gross%/yr':>10} {'Net%/yr':>9}")
    log("-" * 70)
    for r in rows:
        log(f"{r['symbol']:<12} {r['count']:>6} {r['positive_pct']:>5.1f}% "
            f"{r['mean_rate']:>7.4f}% {r['median_rate']:>7.4f}% "
            f"{r['annual_gross']:>9.1f}% {r['annual_net']:>8.1f}%")

    log(f"\nFee model: spot {SPOT_TAKER_FEE*100:.2f}% + futures {FUTURES_MAKER_FEE*100:.3f}% per leg")
    log(f"Round-trip fee: {TOTAL_ROUND_TRIP_FEE*100:.3f}% (amortized over 7-day avg hold)")

    return rows


# ============================================================
# Phase 3: Simulate funding rate arb strategy
# ============================================================

def simulate_funding_arb(
    funding_df: pd.DataFrame,
    symbol: str,
    capital: float = INITIAL_CAPITAL,
    min_entry_rate: float = 0.0001,    # 0.01% = 1 bps
    exit_rate: float = 0.0,            # Exit when rate goes negative
    exit_confirmations: int = 2,       # Need N consecutive below-exit readings
    max_hold_periods: int = 21 * 3,    # 21 days * 3 per day = 63 periods
    position_size_pct: float = 90.0,   # Use 90% of capital (keep margin buffer)
) -> dict:
    """Simulate funding rate arbitrage on a single pair.

    Strategy:
    - Enter when funding rate > min_entry_rate for 3 consecutive periods
    - Collect funding every 8 hours (3x/day)
    - Exit when: rate < exit_rate for exit_confirmations periods, OR max hold reached
    - Account for entry/exit fees on both spot and futures legs

    Returns dict with performance metrics and trade list.
    """
    if funding_df.empty:
        return {"symbol": symbol, "trades": [], "total_return_pct": 0}

    df = funding_df.sort_values("timestamp").reset_index(drop=True)
    rates = df["funding_rate"].values
    timestamps = df["timestamp"].values

    equity = capital
    peak_equity = capital
    max_dd = 0
    trades = []

    # State
    in_position = False
    entry_idx = 0
    entry_equity = 0
    funding_collected = 0
    entry_confirmations = 0
    exit_confirm_count = 0
    hold_periods = 0

    for i in range(len(rates)):
        rate = rates[i]

        if not in_position:
            # Check entry: N consecutive positive rates above threshold
            if rate >= min_entry_rate:
                entry_confirmations += 1
            else:
                entry_confirmations = 0

            if entry_confirmations >= 3:
                # ENTER: buy spot + short perp
                in_position = True
                position_usd = equity * (position_size_pct / 100)
                entry_fee = position_usd * ENTRY_FEE
                equity -= entry_fee
                entry_equity = equity
                entry_idx = i
                funding_collected = 0
                hold_periods = 0
                exit_confirm_count = 0
                entry_confirmations = 0

        else:
            # IN POSITION: collect funding
            position_usd = entry_equity * (position_size_pct / 100)
            payment = position_usd * rate
            funding_collected += payment
            equity += payment
            hold_periods += 1

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            # 1. Rate below exit threshold
            if rate < exit_rate:
                exit_confirm_count += 1
                if exit_confirm_count >= exit_confirmations:
                    should_exit = True
                    exit_reason = "low_rate"
            else:
                exit_confirm_count = 0

            # 2. Max hold period
            if hold_periods >= max_hold_periods:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                # EXIT: sell spot + close futures short
                exit_fee = position_usd * EXIT_FEE
                equity -= exit_fee

                trade_pnl = funding_collected - (position_usd * ENTRY_FEE) - exit_fee
                hold_days = hold_periods / 3.0

                trades.append({
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[i],
                    "hold_days": hold_days,
                    "funding_collected": funding_collected,
                    "entry_fee": position_usd * ENTRY_FEE,
                    "exit_fee": exit_fee,
                    "net_pnl": trade_pnl,
                    "exit_reason": exit_reason,
                    "avg_rate": np.mean(rates[entry_idx:i+1]),
                })

                in_position = False
                entry_confirmations = 0

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

    # Close any open position at end
    if in_position:
        position_usd = entry_equity * (position_size_pct / 100)
        exit_fee = position_usd * EXIT_FEE
        equity -= exit_fee
        trade_pnl = funding_collected - (position_usd * ENTRY_FEE) - exit_fee
        trades.append({
            "entry_time": timestamps[entry_idx],
            "exit_time": timestamps[-1],
            "hold_days": hold_periods / 3.0,
            "funding_collected": funding_collected,
            "entry_fee": position_usd * ENTRY_FEE,
            "exit_fee": exit_fee,
            "net_pnl": trade_pnl,
            "exit_reason": "end_of_data",
            "avg_rate": np.mean(rates[entry_idx:]),
        })

    total_return_pct = (equity - capital) / capital * 100
    years = len(rates) / (3 * 365)
    annual_return = total_return_pct / years if years > 0 else 0

    # Time in position
    total_hold_periods = sum(t["hold_days"] * 3 for t in trades)
    time_in_position_pct = total_hold_periods / len(rates) * 100 if rates.size > 0 else 0

    winning = [t for t in trades if t["net_pnl"] > 0]
    losing = [t for t in trades if t["net_pnl"] <= 0]

    return {
        "symbol": symbol,
        "total_return_pct": total_return_pct,
        "annual_return_pct": annual_return,
        "max_drawdown_pct": max_dd,
        "final_equity": equity,
        "trades": trades,
        "num_trades": len(trades),
        "win_rate": len(winning) / len(trades) * 100 if trades else 0,
        "avg_hold_days": np.mean([t["hold_days"] for t in trades]) if trades else 0,
        "time_in_position_pct": time_in_position_pct,
        "total_funding": sum(t["funding_collected"] for t in trades),
        "total_fees": sum(t["entry_fee"] + t["exit_fee"] for t in trades),
        "years": years,
    }


# ============================================================
# Phase 4: Parameter sweep
# ============================================================

def parameter_sweep(all_funding: dict[str, pd.DataFrame]):
    """Test different entry thresholds and hold periods."""
    write_section("PHASE 4: PARAMETER SWEEP")

    # Test parameters
    entry_rates = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005]  # 0.5bps to 5bps
    max_holds_days = [7, 14, 21, 30]
    exit_rates = [-0.0001, 0.0, 0.00005]  # negative = only exit on very negative, 0 = exit when negative

    # Use top pairs for sweep (most liquid)
    sweep_pairs = ["ETH/USDT", "BTC/USDT", "SOL/USDT"]

    results = []
    total_configs = len(entry_rates) * len(max_holds_days) * len(exit_rates)
    config_num = 0

    for entry_rate in entry_rates:
        for max_hold_d in max_holds_days:
            for exit_rate in exit_rates:
                config_num += 1
                if config_num % 10 == 0:
                    log(f"  Config {config_num}/{total_configs}...")

                config_returns = []
                config_dds = []

                for symbol in sweep_pairs:
                    if symbol not in all_funding or all_funding[symbol].empty:
                        continue

                    result = simulate_funding_arb(
                        all_funding[symbol], symbol,
                        min_entry_rate=entry_rate,
                        exit_rate=exit_rate,
                        max_hold_periods=max_hold_d * 3,
                    )
                    config_returns.append(result["annual_return_pct"])
                    config_dds.append(result["max_drawdown_pct"])

                if config_returns:
                    avg_return = np.mean(config_returns)
                    avg_dd = np.mean(config_dds) if any(d > 0 for d in config_dds) else 0.1
                    results.append({
                        "entry_bps": entry_rate * 10000,
                        "max_hold_d": max_hold_d,
                        "exit_bps": exit_rate * 10000,
                        "avg_annual": avg_return,
                        "avg_dd": avg_dd,
                        "r_dd": avg_return / avg_dd if avg_dd > 0 else 0,
                    })

    results.sort(key=lambda x: x["avg_annual"], reverse=True)

    log(f"\nTop 15 configs by annual return (avg across {', '.join(sweep_pairs)}):")
    log(f"{'Entry(bps)':>10} {'MaxHold':>8} {'Exit(bps)':>9} {'Ann%':>8} {'DD%':>6} {'R/DD':>6}")
    log("-" * 55)
    for r in results[:15]:
        log(f"{r['entry_bps']:>9.1f} {r['max_hold_d']:>7}d {r['exit_bps']:>8.1f} "
            f"{r['avg_annual']:>7.1f}% {r['avg_dd']:>5.1f}% {r['r_dd']:>5.1f}")

    # Find best config
    best = results[0] if results else None
    if best:
        log(f"\nBest config: entry >= {best['entry_bps']:.1f} bps, "
            f"hold max {best['max_hold_d']}d, exit < {best['exit_bps']:.1f} bps")

    return results


# ============================================================
# Phase 5: Regime-correlated analysis
# ============================================================

def regime_analysis(all_funding: dict[str, pd.DataFrame]):
    """Analyze funding rate behavior during different market regimes.

    Uses BTC price as regime proxy:
    - Bull: BTC above 50-day SMA and SMA rising
    - Bear: BTC below 50-day SMA and SMA falling
    - Sideways: everything else
    """
    write_section("PHASE 5: REGIME ANALYSIS")

    from src.backtest.data_fetcher import load_cached_data

    # Load BTC daily data for regime detection
    btc = load_cached_data("binance", "BTC/USDT", "1h")
    if btc is None:
        log("ERROR: No BTC price data cached. Cannot do regime analysis.")
        return

    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    btc = btc.set_index("timestamp")
    btc_daily = btc["close"].resample("1D").last().dropna()
    sma50 = btc_daily.rolling(50).mean()
    sma50_slope = sma50.diff(5)  # 5-day slope

    # Classify each day
    regime_map = {}
    for date in btc_daily.index:
        if date not in sma50.index or pd.isna(sma50[date]):
            regime_map[date.date()] = "unknown"
            continue

        price = btc_daily[date]
        sma = sma50[date]
        slope = sma50_slope.get(date, 0)
        if pd.isna(slope):
            slope = 0

        if price > sma and slope > 0:
            regime_map[date.date()] = "bull"
        elif price < sma and slope < 0:
            regime_map[date.date()] = "bear"
        else:
            regime_map[date.date()] = "sideways"

    del btc
    gc.collect()

    # Count regime distribution
    regime_counts = {}
    for r in regime_map.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total_days = sum(v for k, v in regime_counts.items() if k != "unknown")
    log(f"\nRegime distribution (BTC SMA50 + slope):")
    for r in ["bull", "sideways", "bear"]:
        count = regime_counts.get(r, 0)
        pct = count / total_days * 100 if total_days > 0 else 0
        log(f"  {r:>10}: {count:>4} days ({pct:.1f}%)")

    # Analyze funding rates by regime
    log(f"\nFunding rates by regime:")
    log(f"{'Symbol':<12} {'Regime':<10} {'Count':>6} {'Mean%':>8} {'Median%':>8} {'Pos%':>6} {'Ann Gross%':>10}")
    log("-" * 70)

    for symbol in CORE_PAIRS:
        if symbol not in all_funding or all_funding[symbol].empty:
            continue

        df = all_funding[symbol].copy()
        df["date"] = df["timestamp"].dt.date
        df["regime"] = df["date"].map(regime_map).fillna("unknown")

        for regime in ["bull", "sideways", "bear"]:
            rdf = df[df["regime"] == regime]
            if len(rdf) < 10:
                continue

            rates = rdf["funding_rate"].values
            mean_r = rates.mean()
            median_r = np.median(rates)
            pos_pct = (rates > 0).sum() / len(rates) * 100
            ann_gross = mean_r * 3 * 365 * 100

            log(f"{symbol:<12} {regime:<10} {len(rdf):>6} {mean_r*100:>7.4f}% "
                f"{median_r*100:>7.4f}% {pos_pct:>5.1f}% {ann_gross:>9.1f}%")

    return regime_map


# ============================================================
# Phase 6: Best strategy simulation with full metrics
# ============================================================

def full_simulation(all_funding: dict[str, pd.DataFrame], best_config: dict | None):
    """Run the best config across all pairs with full reporting."""
    write_section("PHASE 6: FULL SIMULATION (ALL PAIRS)")

    # Default config if sweep didn't find one
    if best_config is None:
        entry_rate = 0.0001
        exit_rate = 0.0
        max_hold_d = 14
    else:
        entry_rate = best_config["entry_bps"] / 10000
        exit_rate = best_config["exit_bps"] / 10000
        max_hold_d = best_config["max_hold_d"]

    log(f"Config: entry >= {entry_rate*10000:.1f} bps, exit < {exit_rate*10000:.1f} bps, max hold {max_hold_d}d")
    log(f"Capital: ${INITIAL_CAPITAL:,}")
    log("")

    all_results = []
    for symbol in ALL_PAIRS:
        if symbol not in all_funding or all_funding[symbol].empty:
            continue

        result = simulate_funding_arb(
            all_funding[symbol], symbol,
            min_entry_rate=entry_rate,
            exit_rate=exit_rate,
            max_hold_periods=max_hold_d * 3,
        )
        all_results.append(result)

    if not all_results:
        log("ERROR: No results to show")
        return

    # Summary table
    log(f"{'Symbol':<12} {'Ann%':>7} {'Tot%':>7} {'DD%':>6} {'Trades':>7} {'WinR%':>6} {'AvgHold':>8} {'InPos%':>7} {'Funding$':>9} {'Fees$':>8}")
    log("-" * 95)

    for r in sorted(all_results, key=lambda x: x["annual_return_pct"], reverse=True):
        log(f"{r['symbol']:<12} {r['annual_return_pct']:>6.1f}% {r['total_return_pct']:>6.1f}% "
            f"{r['max_drawdown_pct']:>5.1f}% {r['num_trades']:>7} {r['win_rate']:>5.1f}% "
            f"{r['avg_hold_days']:>7.1f}d {r['time_in_position_pct']:>6.1f}% "
            f"${r['total_funding']:>8.0f} ${r['total_fees']:>7.0f}")

    # Portfolio simulation: equal weight across profitable pairs
    profitable = [r for r in all_results if r["annual_return_pct"] > 0]
    if profitable:
        log(f"\n--- Portfolio (equal-weight, {len(profitable)} profitable pairs) ---")
        avg_annual = np.mean([r["annual_return_pct"] for r in profitable])
        avg_dd = np.mean([r["max_drawdown_pct"] for r in profitable])
        max_dd = max(r["max_drawdown_pct"] for r in profitable)
        avg_time_in = np.mean([r["time_in_position_pct"] for r in profitable])

        log(f"Avg annual return: {avg_annual:.1f}%")
        log(f"Avg max drawdown:  {avg_dd:.1f}%")
        log(f"Worst pair DD:     {max_dd:.1f}%")
        log(f"Avg time in pos:   {avg_time_in:.1f}%")
        log(f"Profitable pairs:  {len(profitable)}/{len(all_results)}")

        if avg_dd > 0:
            log(f"Return/DD ratio:   {avg_annual / avg_dd:.2f}")

    return all_results


# ============================================================
# Phase 7: Negative funding risk analysis
# ============================================================

def negative_funding_analysis(all_funding: dict[str, pd.DataFrame]):
    """Analyze the risk of sustained negative funding (you'd PAY instead of collect)."""
    write_section("PHASE 7: NEGATIVE FUNDING RISK")

    log("Longest consecutive negative funding streaks:")
    log(f"{'Symbol':<12} {'Max Streak':>10} {'Days':>6} {'Worst Sum%':>10} {'Periods Neg':>12}")
    log("-" * 55)

    for symbol in CORE_PAIRS:
        if symbol not in all_funding or all_funding[symbol].empty:
            continue

        df = all_funding[symbol]
        rates = df["funding_rate"].values

        # Find longest consecutive negative streak
        max_streak = 0
        current_streak = 0
        worst_neg_sum = 0
        current_neg_sum = 0
        total_neg = (rates < 0).sum()

        for r in rates:
            if r < 0:
                current_streak += 1
                current_neg_sum += r
                if current_streak > max_streak:
                    max_streak = current_streak
                    worst_neg_sum = current_neg_sum
            else:
                current_streak = 0
                current_neg_sum = 0

        streak_days = max_streak / 3.0
        log(f"{symbol:<12} {max_streak:>10} {streak_days:>5.1f}d {worst_neg_sum*100:>9.3f}% {total_neg:>12}")


# ============================================================
# Phase 8: Combined strategy estimate (momentum + arb)
# ============================================================

def combined_estimate(all_results: list[dict], regime_map: dict | None):
    """Estimate combined returns: momentum in bull + arb in sideways + cash in bear."""
    write_section("PHASE 8: COMBINED STRATEGY ESTIMATE")

    if not regime_map:
        log("No regime data available, skipping combined estimate")
        return

    # Count regime days
    regime_days = {"bull": 0, "sideways": 0, "bear": 0}
    for r in regime_map.values():
        if r in regime_days:
            regime_days[r] += 1
    total = sum(regime_days.values())

    if total == 0:
        return

    bull_pct = regime_days["bull"] / total
    sideways_pct = regime_days["sideways"] / total
    bear_pct = regime_days["bear"] / total

    log(f"Regime distribution: bull {bull_pct*100:.1f}% | sideways {sideways_pct*100:.1f}% | bear {bear_pct*100:.1f}%")

    # Momentum returns (from our proven strategy)
    # 20.2%/yr but only active in bull markets
    # The strategy already idles in non-bull periods, so its 20.2% is already
    # accounting for the dead time. We don't need to scale it.
    momentum_annual = 20.2

    # Funding arb returns during sideways (conservative estimate)
    arb_results = [r for r in (all_results or []) if r["annual_return_pct"] > 0]
    if arb_results:
        # Average annual return across profitable pairs
        avg_arb_annual = np.mean([r["annual_return_pct"] for r in arb_results])
        # But arb would only run during sideways periods
        arb_contribution = avg_arb_annual * sideways_pct
    else:
        avg_arb_annual = 0
        arb_contribution = 0

    # Bear contribution: 0% (cash) on spot, or short momentum with futures
    # Short momentum estimate: similar to long momentum but in bear periods
    short_momentum_estimate = momentum_annual * 0.7  # Assume 70% as effective as long
    short_contribution = short_momentum_estimate * bear_pct

    log(f"\n--- Estimated Annual Returns by Strategy ---")
    log(f"{'Strategy':<30} {'Full-year%':>10} {'Active':>8} {'Contribution':>12}")
    log("-" * 65)
    log(f"{'MTF Momentum (bull, proven)':<30} {momentum_annual:>9.1f}% {'always':>8} {momentum_annual:>11.1f}%")
    log(f"{'Funding arb (sideways)':<30} {avg_arb_annual:>9.1f}% {sideways_pct*100:>7.0f}% {arb_contribution:>11.1f}%")
    log(f"{'Short momentum (bear, est.)':<30} {short_momentum_estimate:>9.1f}% {bear_pct*100:>7.0f}% {short_contribution:>11.1f}%")
    log(f"{'Cash (chaos/bear)':<30} {'0.0':>9}% {bear_pct*100:>7.0f}% {'0.0':>11}%")

    log(f"\n--- Combined Projections ---")
    spot_only = momentum_annual
    spot_plus_arb = momentum_annual + arb_contribution
    full_system = momentum_annual + arb_contribution + short_contribution

    log(f"Spot only (current):               {spot_only:.1f}%/yr")
    log(f"Spot + funding arb (Phase 2A):     {spot_plus_arb:.1f}%/yr")
    log(f"Full system with futures (Phase 3): {full_system:.1f}%/yr")

    log(f"\nNote: Momentum 20.2%/yr already accounts for idle time in bear/sideways.")
    log(f"Arb contribution is ADDITIVE — it uses capital that's otherwise idle.")
    log(f"Short momentum is speculative — needs backtesting with futures data.")


# ============================================================
# Main
# ============================================================

def main():
    start = datetime.now(timezone.utc)

    # Clear report
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V8: FUNDING RATE ARBITRAGE")
    log(f"Start time: {start.isoformat()}")
    log(f"Capital: ${INITIAL_CAPITAL:,}")
    log(f"Pairs: {len(ALL_PAIRS)} ({len(CORE_PAIRS)} core + {len(EXTRA_PAIRS)} extra)")

    # Phase 1: Download funding rates
    write_section("PHASE 1: DOWNLOADING FUNDING RATE HISTORY")
    all_funding = {}
    for symbol in ALL_PAIRS:
        try:
            df = download_funding_rates(symbol, since="2022-01-01")
            if not df.empty:
                all_funding[symbol] = df
        except Exception as e:
            log(f"  {symbol}: FAILED - {e}")
        gc.collect()

    log(f"\nLoaded funding data for {len(all_funding)}/{len(ALL_PAIRS)} pairs")

    if not all_funding:
        log("FATAL: No funding rate data downloaded. Check internet/API access.")
        return

    # Phase 2: Raw analysis
    rate_analysis = analyze_funding_rates(all_funding)

    # Phase 3: Simulate with default params on all pairs
    write_section("PHASE 3: DEFAULT SIMULATION (ALL PAIRS)")
    log("Config: entry >= 1.0 bps, exit < 0.0 bps, max hold 14d")
    for symbol in ALL_PAIRS:
        if symbol not in all_funding:
            continue
        result = simulate_funding_arb(all_funding[symbol], symbol)
        log(f"  {symbol}: {result['annual_return_pct']:+.1f}%/yr, "
            f"{result['num_trades']} trades, "
            f"DD {result['max_drawdown_pct']:.1f}%, "
            f"in-pos {result['time_in_position_pct']:.0f}%")
        del result
        gc.collect()

    # Phase 4: Parameter sweep
    sweep_results = parameter_sweep(all_funding)
    best_config = sweep_results[0] if sweep_results else None

    # Phase 5: Regime analysis
    regime_map = regime_analysis(all_funding)

    # Phase 6: Full simulation with best config
    full_results = full_simulation(all_funding, best_config)

    # Phase 7: Negative funding risk
    negative_funding_analysis(all_funding)

    # Phase 8: Combined strategy estimate
    combined_estimate(full_results, regime_map)

    # Final summary
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    write_section("FINAL SUMMARY")
    log(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if full_results:
        profitable = [r for r in full_results if r["annual_return_pct"] > 0]
        if profitable:
            avg_ann = np.mean([r["annual_return_pct"] for r in profitable])
            log(f"Profitable pairs: {len(profitable)}/{len(full_results)}")
            log(f"Avg annual return (profitable): {avg_ann:.1f}%")

    log(f"\nKey question: Is funding rate arb worth building?")
    if full_results:
        avg_all = np.mean([r["annual_return_pct"] for r in full_results])
        if avg_all > 3:
            log(f"VERDICT: YES — avg {avg_all:.1f}%/yr across pairs, adds meaningful return")
        elif avg_all > 0:
            log(f"VERDICT: MAYBE — avg {avg_all:.1f}%/yr, marginal benefit")
        else:
            log(f"VERDICT: NO — avg {avg_all:.1f}%/yr, not worth the complexity")

    log(f"\nFull report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
