# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto quantitative trading bot running 24/7 on a VPS. Executes multiple uncorrelated strategies (grid trading, momentum, funding rate arbitrage) with centralized risk management.

**Design Philosophy**: Small frequent trades, survive first/profit second, everything logged, modular strategies. Currently spot-only (long positions only).

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

### Grid Trading (Primary)
- Places buy/sell orders at regular price intervals around current price
- Profits from oscillation regardless of direction
- Uses geometric spacing (percentage-based) by default
- Auto-rebalances when price drifts beyond trigger threshold

### Momentum (Secondary)
- Trend-following with 35-50% win rate but large winners
- **Long-only on spot** — short signals are skipped (no futures trading yet)
- Entry signals: EMA crossover + RSI + volume surge + ADX + MACD alignment
- Requires higher timeframe confirmation (1h signal needs 4h trend agreement)
- Dynamic trailing stops based on ATR
- Position sized by risk: `cost_usd = risk_amount` (capital at risk), NOT notional value

### Funding Rate Arbitrage (Phase 2, requires $10K+)
- Long spot + short perpetual futures = delta neutral
- Collects funding payments every 8 hours
- Activates when funding rate > 0.03% per 8hr period

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
6. **Backtesting**: Historical data, engine with fee/slippage modeling (partial)
7. **Deployment**: Docker, VPS, GitHub Actions CI/CD ✅
8. **Production hardening**: Grid restart reconciliation, balance cache TTL, DB transactions, graceful shutdown ✅

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
- Balance cache has 30s TTL to prevent stale risk checks

## Important Constraints

- Always start in paper/sandbox mode
- Run paper trading minimum 2 weeks before live
- Never store API keys in code
- Disable withdrawal permissions on API keys
- Enable IP whitelisting on exchange API keys
- Backtest fee model: 0.075% maker, 0.1% taker + 0.02-0.05% slippage
