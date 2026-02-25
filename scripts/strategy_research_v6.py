"""Strategy research v6: Push returns to 30-40%/yr.

The v3/v5 strategy returns 20%/yr with 6% DD across 7 pairs. The bottleneck
is diversification — SOL alone returns 28%/yr. This script tests:

1. HIGHER RISK + VOL-SCALING: 6-8% base risk (vol-scaling prevents blowup)
2. CONCENTRATED PORTFOLIOS: top 3, top 4, top 5 instead of all 7
3. FASTER RE-ENTRY: cooldown 1 vs 2 bars
4. RETURN-WEIGHTED ALLOCATION: weight by raw returns, not R/DD
5. COMBINED: stack the best levers together

Usage (on droplet):
    docker-compose run --rm --entrypoint "python -u scripts/strategy_research_v6.py" bot
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

REPORT_FILE = "data/strategy_research_v6_report.txt"
INITIAL_CAPITAL = 10000

ALL_PAIRS = ["BTC/USDT", "SOL/USDT", "DOGE/USDT", "ETH/USDT",
             "AVAX/USDT", "LINK/USDT", "ADA/USDT"]

# Proven base config
BASE = {
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
    """Load hourly data and return 4h + daily."""
    df = load_cached_data("binance", symbol, "1h")
    if df is None:
        return None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_4h = resample(df, "4h")
    df_1d = resample(df, "1D")
    del df
    gc.collect()
    return df_4h, df_1d


def vol_scaled_backtest(df_4h: pd.DataFrame, df_1d: pd.DataFrame, config: dict):
    """MTF backtest with vol-scaled sizing."""
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
                fee = amount * close * 0.001
                pnl = (close - position["entry_price"]) * amount - fee
                capital += pnl
                trades.append(BacktestTrade(
                    timestamp=ts, side="sell", price=close,
                    amount=amount, cost=amount * close, fee=fee,
                    pnl=pnl, strategy="v6",
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
                        strategy="v6",
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
            pnl=pnl, strategy="v6",
        ))

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    return engine._build_result("v6", df_4h, trades, equity_curve, capital)


def portfolio_sim(pair_curves, pair_stats, weights, bars_per_year):
    """Run portfolio simulation and return stats."""
    min_len = min(len(c) for c in pair_curves.values())
    if min_len < 100:
        return {}

    portfolio = []
    for i in range(min_len):
        total = sum(
            pair_curves[sym][i] / INITIAL_CAPITAL * INITIAL_CAPITAL * weights[sym]
            for sym in pair_curves if i < len(pair_curves[sym])
        )
        portfolio.append(total)

    initial = portfolio[0]
    final = portfolio[-1]
    years = min_len / bars_per_year

    peak = portfolio[0]
    max_dd = 0
    for eq in portfolio:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    annual = ((final / initial) ** (1 / years) - 1) * 100 if years > 0.5 else 0
    rets = [(portfolio[i] - portfolio[i-1]) / portfolio[i-1]
            for i in range(1, len(portfolio)) if portfolio[i-1] > 0]
    sharpe = (np.mean(rets) / np.std(rets) * (bars_per_year ** 0.5)
              if rets and np.std(rets) > 0 else 0)

    return {"annual": annual, "max_dd": max_dd, "sharpe": sharpe, "years": years}


def main():
    start = time.time()
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("")

    write_section("STRATEGY RESEARCH V6: Pushing to 30%+")
    log(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log("Testing aggressive return optimization on proven MTF ensemble")

    # ================================================================
    # Phase 1: Risk sweep with vol-scaling (5%, 6%, 7%, 8%, 10%)
    # ================================================================
    write_section("PHASE 1: Risk Sweep (vol-scaled)")
    log("Testing higher base risk — vol-scaling prevents high-vol blowups")

    risk_levels = [5.0, 6.0, 7.0, 8.0, 10.0]
    cooldowns = [1, 2]

    # Store all results for portfolio sims later
    # {(risk, cooldown): {symbol: {ret, dd, pf, trades, equity_curve}}}
    all_data = {}

    for risk in risk_levels:
        for cd in cooldowns:
            cfg = {**BASE, "risk_per_trade_pct": risk, "cooldown_bars": cd}
            label = f"R{risk:.0f}_CD{cd}"
            log(f"\n--- {label} ---")

            pair_results = {}
            for symbol in ALL_PAIRS:
                df_4h, df_1d = load_data(symbol)
                if df_4h is None:
                    continue
                result = vol_scaled_backtest(df_4h, df_1d, cfg)
                pair_results[symbol] = {
                    "ret": result.total_return_pct,
                    "dd": result.max_drawdown_pct,
                    "pf": result.profit_factor,
                    "trades": result.total_trades,
                    "equity_curve": list(result.equity_curve),
                }
                log(f"  {symbol:12s}: {result.total_return_pct:+8.1f}% | "
                    f"DD: {result.max_drawdown_pct:5.1f}% | Trades: {result.total_trades:3d}")
                del result, df_4h, df_1d
                gc.collect()

            all_data[(risk, cd)] = pair_results

            # Quick aggregate
            rets = [r["ret"] for r in pair_results.values()]
            dds = [r["dd"] for r in pair_results.values()]
            if rets:
                log(f"  Avg: {np.mean(rets):+.1f}% | DD: {np.mean(dds):.1f}% | "
                    f"Win: {sum(1 for r in rets if r > 0)}/{len(rets)}")

    # ================================================================
    # Phase 2: Portfolio simulations — concentration + weighting
    # ================================================================
    write_section("PHASE 2: Portfolio Optimization Grid")
    log("Testing all combinations of risk, cooldown, concentration, weighting")

    bars_per_year = 6 * 365.25

    # Ranked pairs by return (from v5 vol-scaled results)
    pair_rank = ["SOL/USDT", "DOGE/USDT", "ETH/USDT", "AVAX/USDT",
                 "LINK/USDT", "ADA/USDT", "BTC/USDT"]

    portfolio_groups = {
        "top3": pair_rank[:3],
        "top4": pair_rank[:4],
        "top5": pair_rank[:5],
        "all7": pair_rank,
    }

    weighting_methods = ["equal", "r_dd", "return_weighted", "sqrt_return"]

    results = []

    for (risk, cd), pair_results in sorted(all_data.items()):
        for group_name, group_pairs in portfolio_groups.items():
            # Filter to pairs in this group that have data
            active = {s: pair_results[s] for s in group_pairs if s in pair_results}
            if len(active) < 2:
                continue

            pair_curves = {s: d["equity_curve"] for s, d in active.items()}
            pair_stats = {s: {"ret": d["ret"], "dd": d["dd"]} for s, d in active.items()}

            for wmethod in weighting_methods:
                n = len(active)

                if wmethod == "equal":
                    weights = {s: 1.0 / n for s in active}
                elif wmethod == "r_dd":
                    scores = {s: max(d["ret"] / max(d["dd"], 1.0), 0.1) for s, d in pair_stats.items()}
                    total = sum(scores.values())
                    weights = {s: v / total for s, v in scores.items()}
                elif wmethod == "return_weighted":
                    scores = {s: max(d["ret"], 1.0) for s, d in pair_stats.items()}
                    total = sum(scores.values())
                    weights = {s: v / total for s, v in scores.items()}
                elif wmethod == "sqrt_return":
                    scores = {s: max(d["ret"], 1.0) ** 0.5 for s, d in pair_stats.items()}
                    total = sum(scores.values())
                    weights = {s: v / total for s, v in scores.items()}

                stats = portfolio_sim(pair_curves, pair_stats, weights, bars_per_year)
                if not stats:
                    continue

                results.append({
                    "risk": risk, "cd": cd, "group": group_name,
                    "weight": wmethod, "n_pairs": n,
                    "annual": stats["annual"], "dd": stats["max_dd"],
                    "sharpe": stats["sharpe"], "years": stats["years"],
                    "weights": weights,
                })

    # Sort by annual return
    results.sort(key=lambda x: x["annual"], reverse=True)

    log(f"\nTotal portfolio configs tested: {len(results)}")
    log(f"\n{'Rank':>4} | {'Config':30s} | {'Annual':>8} | {'DD':>6} | {'Sharpe':>6} | {'R/DD':>6}")
    log("-" * 85)

    for i, r in enumerate(results[:30]):
        label = f"R{r['risk']:.0f}_CD{r['cd']}_{r['group']}_{r['weight']}"
        r_dd = r["annual"] / r["dd"] if r["dd"] > 0 else 0
        marker = " ***" if r["annual"] >= 30 else ""
        log(f"{i+1:4d} | {label:30s} | {r['annual']:+7.1f}% | {r['dd']:5.1f}% | "
            f"{r['sharpe']:5.2f} | {r_dd:5.2f}{marker}")

    # ================================================================
    # Phase 3: Deep dive on top configs
    # ================================================================
    write_section("PHASE 3: Top Config Analysis")

    # Show top 5 with full allocations
    for i, r in enumerate(results[:5]):
        label = f"R{r['risk']:.0f}_CD{r['cd']}_{r['group']}_{r['weight']}"
        r_dd = r["annual"] / r["dd"] if r["dd"] > 0 else 0
        log(f"\n  #{i+1}: {label}")
        log(f"    Annual: {r['annual']:+.1f}%/yr | DD: {r['dd']:.1f}% | "
            f"Sharpe: {r['sharpe']:.2f} | R/DD: {r_dd:.2f}")
        log(f"    Allocation:")
        for sym in sorted(r["weights"], key=r["weights"].get, reverse=True):
            log(f"      {sym:12s}: {r['weights'][sym]*100:5.1f}%")

    # ================================================================
    # Phase 4: Best config vs 30% target — noise test
    # ================================================================
    write_section("PHASE 4: Noise Test on Best Config")

    if not results:
        log("  No results to test!")
    else:
        best = results[0]
        best_cfg = {**BASE, "risk_per_trade_pct": best["risk"], "cooldown_bars": best["cd"]}
        best_pairs = list(best["weights"].keys())
        best_weights = best["weights"]

        log(f"Testing: R{best['risk']:.0f}_CD{best['cd']}_{best['group']}_{best['weight']}")
        log(f"Pairs: {best_pairs}")
        log(f"50 noise runs per pair, 0.5% noise")

        noise_std = 0.005
        n_runs = 50
        total_pass = 0
        total_runs = 0

        for symbol in best_pairs:
            df_4h, df_1d = load_data(symbol)
            if df_4h is None:
                continue

            baseline = vol_scaled_backtest(df_4h, df_1d, best_cfg)
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
                    r = vol_scaled_backtest(df_4h_n, df_1d_n, best_cfg)
                    noisy_rets.append(r.total_return_pct)
                    del r
                except Exception:
                    noisy_rets.append(0.0)

            profitable = sum(1 for r in noisy_rets if r > 0)
            total_pass += profitable
            total_runs += n_runs
            log(f"  {symbol:12s}: Base {base_ret:+7.1f}% | Noisy avg {np.mean(noisy_rets):+7.1f}% "
                f"(std {np.std(noisy_rets):5.1f}%) | {profitable}/{n_runs}")

            del df_4h, df_1d
            gc.collect()

        if total_runs > 0:
            pct = total_pass / total_runs * 100
            log(f"\n  Noise resilience: {total_pass}/{total_runs} ({pct:.0f}%)")
            log(f"  {'PASS' if pct >= 80 else 'MARGINAL' if pct >= 50 else 'FAIL'}")

    # ================================================================
    # Phase 5: Also test best RISK-ADJUSTED config (for comparison)
    # ================================================================
    write_section("PHASE 5: Best Risk-Adjusted Configs")

    # Sort by R/DD for comparison
    by_rdd = sorted([r for r in results if r["annual"] > 10],
                    key=lambda x: x["annual"] / max(x["dd"], 0.1), reverse=True)

    log(f"\n{'Rank':>4} | {'Config':30s} | {'Annual':>8} | {'DD':>6} | {'Sharpe':>6} | {'R/DD':>6}")
    log("-" * 85)
    for i, r in enumerate(by_rdd[:15]):
        label = f"R{r['risk']:.0f}_CD{r['cd']}_{r['group']}_{r['weight']}"
        r_dd = r["annual"] / r["dd"] if r["dd"] > 0 else 0
        marker = " ***" if r["annual"] >= 30 else ""
        log(f"{i+1:4d} | {label:30s} | {r['annual']:+7.1f}% | {r['dd']:5.1f}% | "
            f"{r['sharpe']:5.2f} | {r_dd:5.2f}{marker}")

    # ================================================================
    # Final Summary
    # ================================================================
    write_section("FINAL SUMMARY")
    elapsed = time.time() - start
    log(f"Runtime: {elapsed/60:.1f} minutes")

    if results:
        best_ret = results[0]
        log(f"\nBest return: R{best_ret['risk']:.0f}_CD{best_ret['cd']}_"
            f"{best_ret['group']}_{best_ret['weight']} = {best_ret['annual']:+.1f}%/yr")

        if by_rdd:
            best_rdd = by_rdd[0]
            log(f"Best R/DD:   R{best_rdd['risk']:.0f}_CD{best_rdd['cd']}_"
                f"{best_rdd['group']}_{best_rdd['weight']} = {best_rdd['annual']:+.1f}%/yr "
                f"(DD: {best_rdd['dd']:.1f}%)")

        targets = [r for r in results if r["annual"] >= 30]
        if targets:
            log(f"\n  *** {len(targets)} configs hit 30%+ target! ***")
            best_30 = min(targets, key=lambda x: x["dd"])
            log(f"  Best 30%+ by DD: R{best_30['risk']:.0f}_CD{best_30['cd']}_"
                f"{best_30['group']}_{best_30['weight']} = {best_30['annual']:+.1f}%/yr, "
                f"DD: {best_30['dd']:.1f}%")
        else:
            log(f"\n  No configs hit 30%. Best: {best_ret['annual']:+.1f}%/yr")

    log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
