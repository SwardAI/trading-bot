# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto quantitative trading bot running 24/7 on a VPS. Targeting 30%+ annual returns through regime-adaptive strategies across multiple pairs.

**Current state**: Spot-only (long positions only). One proven strategy ready to deploy: vol-scaled MTF Donchian ensemble (20%/yr, 6% DD, Sharpe 1.86). Getting to 30% requires adding futures capability for bear-market and sideways-market strategies.

**Design Philosophy**: Survive first/profit second. Regime-aware (idle in bear markets on spot — correct behavior). Everything logged. Modular strategies. Risk-managed by central authority.

## Tech Stack

- **Language**: Python 3.11+
- **Exchange Library**: ccxt (unified API)
- **Data Processing**: pandas, numpy
- **Technical Indicators**: ta (technical-analysis library)
- **Database**: SQLite (start) → PostgreSQL (scale)
- **Scheduling**: APScheduler
- **Alerts**: Telegram Bot API
- **Configuration**: YAML files
- **Deployment**: Docker + docker-compose

## Project Structure

```
crypto-bot/
├── config/                    # YAML configuration files
│   ├── settings.yaml          # Global settings, exchange config, risk params
│   ├── grid_config.yaml       # Grid strategy parameters per pair
│   ├── momentum_config.yaml   # Momentum strategy parameters
│   └── funding_config.yaml    # Funding rate strategy parameters
├── src/
│   ├── core/                  # Bot orchestrator, exchange wrapper, config, logging
│   ├── strategies/            # Strategy implementations (grid, momentum, funding)
│   ├── risk/                  # Risk manager, position tracker, circuit breakers
│   ├── data/                  # Market data, indicators, funding rates
│   ├── execution/             # Order management, smart executor
│   ├── journal/               # Trade logging, performance metrics, reports
│   ├── alerts/                # Telegram notifications
│   └── backtest/              # Backtesting engine
├── tests/                     # Test files
├── scripts/                   # CLI tools (download data, run backtest, reports)
└── data/                      # Local data storage (historical/, bot.db)
```

## Architecture

### Strategy System
- All strategies inherit from `base_strategy.py` abstract class
- Strategies are independent and can be enabled/disabled via YAML config
- Each strategy must check with `risk_manager` before placing any order

### Risk Management (Central Authority)
- **Pre-trade checks**: Every order must pass through `risk_manager.check(order)`
- **Circuit breakers**: Daily (-3%), weekly (-7%), monthly (-12%) loss limits trigger trading halt
- **Recovery mode**: After circuit break, trade at 50% size for 48 hours
- **Exposure limits**: Max 60% total, 20% per pair, 35% correlated assets, 20% cash reserve

### Order Execution Flow
1. Check orderbook depth before placing
2. Place as limit order (maker fee)
3. Adjust price if not filled within 10s
4. Convert to market if not filled after 30s
5. Handle partial fills and retries
6. Reconcile with exchange every 5 minutes

### Database Schema
Key tables: `trades`, `account_snapshots`, `daily_metrics`, `grid_state`, `momentum_positions`, `circuit_breaker_events`

## Strategies

### Vol-Scaled MTF Donchian Ensemble (PRIMARY — deploy this)
- **Regime**: Bull markets only (correctly idles in bear/sideways on spot)
- **Daily filter**: Donchian channels [20, 50, 100] — 2/3 must agree price is above breakout
- **4h entry**: 14-period high breakout. 7-period low exit.
- **Stop loss**: Chandelier exit (highest_high_since_entry - 3.0 * ATR)
- **Position sizing**: `risk_amount = capital * risk_pct / 100; amount = risk_amount / (entry - stop_loss)`
- **Vol-scaling**: `effective_risk = 5% * (median_ATR_60 / current_ATR)`, clamped [0.5x, 2x]
  - Risks more when volatility is low (bigger position), less when high (smaller position)
- **Capital cap**: `cost + fee <= capital * 0.95` (always keep 5% reserve)
- **Fee model**: 0.1% taker per trade
- **Pairs**: 7 with R/DD-weighted allocation: SOL 40%, DOGE 17%, ETH 12%, AVAX 12%, LINK 8%, ADA 6%, BTC 6%
- **Performance (4yr backtest)**: 20.2%/yr, 6.0% DD, Sharpe 1.86
- **Robustness**: 6/7 tests PASS (90% weighted score), 350/350 noise resilient (100%)

### Grid Trading (DISABLED — proven net negative)
- Net negative over full 4yr market cycles in backtests
- Also tested as regime-adaptive (sideways-only): 80-90% drawdown — catastrophic
- Kept in codebase but should NOT be enabled

### EMA Momentum (DISABLED — replaced by MTF Donchian)
- Failed robustness testing (4/7, noise fragile, parameter sensitive)
- The MTF Donchian is strictly better on every metric

### Funding Rate Arbitrage (Phase 2, requires $10K+ and futures)
- Long spot + short perpetual futures = delta neutral
- Collects funding payments every 8 hours
- Activates when funding rate > 0.03% per 8hr period

### Future: 6-Regime System (requires futures)
The long-term architecture is regime-adaptive with 6 market states:
1. Strong Uptrend → MTF momentum (current strategy)
2. Weak Uptrend → Momentum with tighter stops
3. Sideways → Funding rate arbitrage (delta-neutral)
4. Weak Downtrend → Short-side momentum (needs futures)
5. Strong Downtrend → Cash / defensive
6. High Vol Chaos → Reduce size / sit out
On spot-only, states 4-6 all map to "cash". Need futures for full system.

## Configuration

API keys stored in `.env` file (never committed), referenced in YAML as:
```yaml
api_key_env: "BINANCE_API_KEY"
```

Mode is set in `config/settings.yaml`:
```yaml
bot:
  mode: "paper"  # "paper" or "live"
```

## Development Phases

1. **Foundation**: Project setup, exchange connector, market data, database, logging ✅
2. **Grid Strategy**: Grid calculation, order placement, fill detection, rebalancing ✅
3. **Risk Management**: Position tracking, pre-trade checks, circuit breakers ✅
4. **Momentum Strategy**: Indicators, signal generation, trailing stops ✅
5. **Monitoring**: Telegram alerts, daily reports, trade journal ✅
6. **Backtesting**: Historical data, engine with fee/slippage modeling ✅
7. **Deployment**: Docker, VPS, GitHub Actions CI/CD ✅
8. **Production hardening**: Grid restart reconciliation, balance cache TTL, DB transactions, graceful shutdown ✅
9. **Strategy Research**: v1-v7 extensive backtesting, found vol-scaled MTF Donchian ✅
10. **MTF Strategy Implementation**: Implement proven strategy as live code ← CURRENT
11. **Paper Trading**: 2+ weeks validation on Binance testnet
12. **Go Live**: Switch to real trading (spot only, long only)
13. **Futures Capability**: Add perpetual futures for Phase 2 strategies (30%+ target)
14. **Regime System**: Full 6-regime classifier + per-regime strategy router

## Deployment

- **VPS**: DigitalOcean droplet at 134.122.73.180
- **Container**: `crypto-bot` via docker-compose in `~/trading`
- **CI/CD**: Push to `main` triggers GitHub Actions → test → deploy
- **Health check**: `ssh root@134.122.73.180 -i ~/.ssh/digitalocean 'bash ~/trading/scripts/health_check.sh'`
- **Status**: Paper trading (sandbox mode) on Binance testnet

## Architecture Gotchas

- `load_all_configs()` returns a **flat** dict — keys are at top level, NOT nested under "settings"
- `RiskManager(config, db, exchange)` — creates its own `PositionTracker` internally
- `generate_report.py` runs via `docker exec` as a separate process — needs its own exchange connection
- Grid `on_tick()` must place pending orders after checking fills (not just on startup)
- Balance cache has 10s TTL to prevent stale risk checks (reduced from 30s)

## Important Constraints

- Always start in paper/sandbox mode
- Run paper trading minimum 2 weeks before live
- Never store API keys in code
- Disable withdrawal permissions on API keys
- Enable IP whitelisting on exchange API keys
- Backtest fee model: 0.075% maker, 0.1% taker + 0.02-0.05% slippage
