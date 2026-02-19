# Crypto Quantitative Trading Bot — Complete Blueprint

## For use with Claude Code: Share this document at project start

---

## 1. Project Overview

Build a fully automated cryptocurrency trading bot that runs 24/7 on a VPS, executes multiple uncorrelated strategies simultaneously, enforces strict risk management, and logs everything for performance analysis.

### Design Philosophy

- **Market-neutral by default** — Profit from volatility and inefficiency, not from predicting direction
- **Small and frequent trades** — High trade count smooths variance and reveals edge faster
- **Survive first, profit second** — Circuit breakers and position limits protect capital above all else
- **Everything is logged** — Every trade, every decision, every metric. Data is how you improve.
- **Modular strategies** — Each strategy is independent and can be enabled/disabled/tuned without affecting others

### Target Performance (Honest Expectations)

| Metric | Conservative | Target | Optimistic |
|--------|-------------|--------|------------|
| Annual Return | 12% | 18–25% | 30%+ |
| Max Monthly Drawdown | -5% | -8% | -12% |
| Win Rate (grid) | 75–85% | 80–90% | 90%+ |
| Win Rate (momentum) | 35–45% | 40–50% | 50%+ |
| Avg Trades/Day | 5–10 | 15–30 | 40+ |
| Sharpe Ratio | 1.0 | 1.5–2.0 | 2.5+ |

Note: Year 1 is a learning year. Expect results closer to "conservative" while tuning. The system improves with data and parameter optimization over time.

---

## 2. Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.11+ | Industry standard for algo trading; best library ecosystem |
| Exchange Library | ccxt | Unified API across 100+ exchanges, battle-tested |
| Data Processing | pandas, numpy | Fast numerical operations, time series handling |
| Technical Indicators | ta (technical-analysis) | Lightweight, pandas-native, no C dependencies |
| Database | SQLite (start) → PostgreSQL (scale) | Zero-config start, easy migration path |
| Task Scheduling | APScheduler | In-process scheduling, no external dependencies |
| Monitoring | Telegram Bot API | Instant alerts to phone, free, simple |
| Backtesting | Custom engine (built in-house) | Tailored to our strategies, no black-box assumptions |
| Configuration | YAML files | Human-readable, easy to version control |
| Logging | Python logging + structured JSON | Queryable logs for debugging and analysis |
| Deployment | Docker + docker-compose | Reproducible, easy VPS deployment |
| VPS | Any Linux VPS ($5–20/month) | DigitalOcean, Hetzner, or Vultr recommended |

### Why NOT these alternatives

- **Not Node.js** — Python's data science ecosystem (pandas, numpy, ta) is unmatched for quant work
- **Not Rust/Go** — We're not doing HFT; Python's speed is fine for trades on second/minute timescales
- **Not a framework like Freqtrade** — Too opinionated, hard to customize for multi-strategy with shared risk management. Building from scratch gives us full control.
- **Not MySQL/Postgres from day 1** — SQLite is simpler to start; we migrate when needed

---

## 3. Project Structure

```
crypto-bot/
├── config/
│   ├── settings.yaml              # Global settings (exchanges, risk params, alerts)
│   ├── grid_config.yaml           # Grid strategy parameters per pair
│   ├── momentum_config.yaml       # Momentum strategy parameters
│   └── funding_config.yaml        # Funding rate strategy parameters
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bot.py                 # Main bot orchestrator — starts/stops strategies
│   │   ├── exchange.py            # Exchange connection manager (ccxt wrapper)
│   │   ├── config_loader.py       # YAML config loading and validation
│   │   └── logger.py              # Structured logging setup
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py       # Abstract base class all strategies inherit
│   │   ├── grid_strategy.py       # Grid/mean-reversion trading
│   │   ├── momentum_strategy.py   # Trend-following with dynamic trailing stops
│   │   └── funding_strategy.py    # Funding rate arbitrage (Phase 2)
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_manager.py        # Central risk engine — all strategies check in here
│   │   ├── position_tracker.py    # Real-time position and exposure tracking
│   │   └── circuit_breaker.py     # Emergency shutdown logic
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_data.py         # OHLCV fetching, orderbook data, ticker data
│   │   ├── indicators.py          # Technical indicator calculations
│   │   └── funding_rates.py       # Funding rate data collection
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_manager.py       # Order placement, tracking, and reconciliation
│   │   └── smart_executor.py      # Retry logic, partial fills, slippage handling
│   ├── journal/
│   │   ├── __init__.py
│   │   ├── trade_logger.py        # Logs every trade to database
│   │   ├── performance.py         # Calculates performance metrics
│   │   └── reporter.py            # Generates daily/weekly performance reports
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── telegram_bot.py        # Telegram notifications
│   └── backtest/
│       ├── __init__.py
│       ├── engine.py              # Backtesting engine with realistic fee/slippage modeling
│       ├── data_fetcher.py        # Historical data download and caching
│       └── analyzer.py            # Backtest result analysis and visualization
├── tests/
│   ├── test_grid_strategy.py
│   ├── test_momentum_strategy.py
│   ├── test_risk_manager.py
│   ├── test_circuit_breaker.py
│   └── test_order_manager.py
├── scripts/
│   ├── download_historical.py     # Download historical data for backtesting
│   ├── run_backtest.py            # CLI for running backtests
│   └── generate_report.py         # CLI for generating performance reports
├── data/                          # Local data storage
│   ├── historical/                # Cached OHLCV data
│   └── bot.db                     # SQLite database (trades, positions, metrics)
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── .env.example                   # API keys template (never committed)
└── README.md
```

---

## 4. Strategy 1: Grid Trading (Primary Strategy)

### Concept

Place a grid of buy and sell orders at regular price intervals around the current price. When price drops, buy orders trigger. When price rises, sell orders trigger. Each buy has a corresponding sell above it. Profit comes from the spread between buy and sell levels.

**This works regardless of market direction as long as price oscillates.**

### How It Works

```
Sell -------- $52,000  ← sell order
Sell -------- $51,500  ← sell order
Sell -------- $51,000  ← sell order
Current Price $50,500
Buy  -------- $50,000  ← buy order
Buy  -------- $49,500  ← buy order
Buy  -------- $49,000  ← buy order
```

1. Price drops to $50,000 → buy order fills
2. Bot immediately places sell order at $50,500 (grid spacing above)
3. Price bounces to $50,500 → sell fills → profit = $500 minus fees
4. Bot replaces the buy at $50,000
5. Repeat forever

### Configuration Parameters

```yaml
# config/grid_config.yaml
grid_strategy:
  enabled: true
  pairs:
    - symbol: "BTC/USDT"
      grid_type: "arithmetic"        # arithmetic (fixed $) or geometric (fixed %)
      num_grids: 20                  # number of grid levels per side
      grid_spacing_pct: 0.5          # 0.5% between each grid level
      order_size_usd: 50             # dollar amount per grid order
      upper_bound_pct: 10            # grid ceiling: +10% from entry
      lower_bound_pct: 10            # grid floor: -10% from entry
      rebalance_trigger_pct: 8       # shift grid when price moves 8% from center
      take_profit_pct: null          # optional: close all if total profit hits X%
      stop_loss_pct: 15              # emergency: close all if position down 15%
    
    - symbol: "ETH/USDT"
      grid_type: "geometric"
      num_grids: 15
      grid_spacing_pct: 0.8
      order_size_usd: 30
      upper_bound_pct: 12
      lower_bound_pct: 12
      rebalance_trigger_pct: 10
      stop_loss_pct: 15

  # Global grid settings
  max_open_orders: 40               # across all pairs
  min_profit_after_fees: 0.15       # skip grids where spacing < fees + 0.15%
  fee_rate: 0.075                   # maker fee (Binance BNB discount)
```

### Grid Type: Arithmetic vs Geometric

- **Arithmetic**: Fixed dollar spacing (e.g., every $500). Better when price is range-bound in a tight channel. Simpler.
- **Geometric**: Fixed percentage spacing (e.g., every 0.5%). Better for volatile assets where moves are proportional. Prevents grids being too tight at high prices or too wide at low prices.

**Default to geometric** — it's more robust across different price levels.

### Grid Rebalancing Logic

When price trends strongly in one direction, one side of the grid gets fully consumed. The bot must detect this and shift the grid center:

1. If price moves beyond `rebalance_trigger_pct` from grid center
2. Cancel all remaining orders on the far side
3. Recalculate grid centered on current price
4. Place new orders
5. Log the rebalance event

**Important**: Rebalancing means you're holding inventory (coins bought during the trend). This is the source of risk in grid trading. The stop_loss_pct prevents this inventory from becoming a catastrophic loss.

### Edge and Expected Returns

- Average profit per grid fill: grid_spacing - (2 × fee_rate) = 0.5% - 0.15% = **0.35% per round trip**
- With 20 grid levels, moderate volatility, and BTC: expect 5–15 round trips per day
- Daily return on deployed capital: 0.35% × 10 avg fills = **~3.5% daily on capital in the grid**
- BUT: Only a portion of total capital is deployed in grids (typically 30–50%)
- Net portfolio contribution: roughly **0.8–1.5% daily, or 15–30% monthly on grid-allocated capital**
- Reality check: some days have 0–2 fills (low vol), some have 20+ (high vol). Monthly averages smooth out.

### Pair Selection Criteria

Not all pairs are good for grid trading. The bot should evaluate pairs on:

1. **Volatility**: Need sufficient movement to trigger grids. 30-day realized vol > 40% annualized.
2. **Liquidity**: Tight bid-ask spreads. Top 20 pairs by volume on the exchange.
3. **Mean-reversion tendency**: Pairs that range-trade rather than trend strongly. Measure with Hurst exponent < 0.5.
4. **Fee efficiency**: Grid spacing must be > 2× the round-trip fee cost.

**Recommended starting pairs**: BTC/USDT, ETH/USDT, SOL/USDT (most liquid, most oscillation).

---

## 5. Strategy 2: Momentum / Trend Following

### Concept

Detect when a strong trend begins, enter in the direction of the trend, ride it with a trailing stop, and exit when momentum fades. This strategy has a LOW win rate (35–50%) but large winners that more than compensate for the small frequent losses.

**This is the counterpart to grid trading** — grid profits from oscillation, momentum profits from strong directional moves. Together they cover both market regimes.

### Signal Generation

Use a combination of indicators (no single indicator is reliable alone):

```
Entry Signal (ALL must align):
1. EMA Crossover: 9 EMA crosses above 21 EMA (bullish) or below (bearish)
2. RSI Confirmation: RSI(14) > 55 for longs, < 45 for shorts
3. Volume Surge: Current volume > 1.5× 20-period average volume
4. ADX Filter: ADX(14) > 25 (confirms a trend exists, not just noise)
5. MACD Histogram: Positive and increasing (longs) or negative and decreasing (shorts)

Exit Signal (ANY triggers exit):
1. Trailing stop hit (see below)
2. EMA crossover reverses
3. RSI divergence (price makes new high but RSI doesn't)
4. Time stop: position open > 72 hours without hitting 2× risk in profit
```

### Trailing Stop Logic

```
Initial stop: 1.5 × ATR(14) from entry price
As price moves in favor:
  - Tighten to 1.2 × ATR when profit > 1× risk
  - Tighten to 1.0 × ATR when profit > 2× risk  
  - Tighten to 0.8 × ATR when profit > 3× risk
Never move stop backward (further from price)
```

### Configuration Parameters

```yaml
# config/momentum_config.yaml
momentum_strategy:
  enabled: true
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
    - "AVAX/USDT"
    - "LINK/USDT"
  
  timeframe: "1h"                    # primary analysis timeframe
  confirmation_timeframe: "4h"       # higher timeframe trend must agree
  
  # Indicator parameters
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  rsi_long_threshold: 55
  rsi_short_threshold: 45
  adx_period: 14
  adx_min_strength: 25
  volume_surge_multiplier: 1.5
  atr_period: 14
  
  # Position management
  risk_per_trade_pct: 1.0            # risk 1% of portfolio per trade
  max_concurrent_positions: 3        # across all pairs
  trailing_stop_atr_multiplier: 1.5  # initial stop distance
  time_stop_hours: 72               # close if not profitable in 72 hours
  
  # Filters
  min_daily_volume_usd: 50000000    # skip illiquid pairs
  avoid_within_minutes_of_funding: 30  # don't enter near funding rate settlements
  
  # Cooldown
  cooldown_after_loss_minutes: 60   # wait 1 hour after a losing trade before re-entering same pair
```

### Timeframe Alignment

Only take trades where the **higher timeframe agrees** with the signal timeframe:

- Signal on 1h chart: BTC trending up
- Check 4h chart: 4h EMA trend must also be bullish
- If 4h is bearish while 1h is bullish → **skip the trade** (counter-trend, lower probability)

### Expected Performance

- Win rate: 35–50% (this is normal and expected for trend following)
- Average winner: 2–4× the average loser
- Profit factor: 1.3–2.0
- Trades per pair per week: 2–5
- Best in: trending markets, high volatility periods
- Worst in: choppy sideways markets (this is when grid trading carries the portfolio)

---

## 6. Strategy 3: Funding Rate Arbitrage (Phase 2 — Add After $10K+ Capital)

### Concept

On perpetual futures exchanges, a "funding rate" is paid between longs and shorts every 8 hours to keep the futures price anchored to spot. When the rate is positive (common in bull markets), longs pay shorts. You can collect this by being short futures while simultaneously being long spot — the funding payments accumulate while your directional risk is hedged.

### How It Works

1. Monitor funding rates across pairs
2. When funding rate > threshold (e.g., > 0.03% per 8 hours = ~0.09%/day):
   - Buy X amount of the coin on spot
   - Short X amount on perpetual futures
   - Net directional exposure: ~zero
3. Collect funding payments every 8 hours
4. When funding rate drops below threshold, unwind both sides
5. Profit = funding collected - trading fees - any slippage on entry/exit

### Configuration

```yaml
# config/funding_config.yaml
funding_strategy:
  enabled: false                     # Enable in Phase 2
  exchanges:
    spot: "binance"
    futures: "binance"               # same exchange to minimize transfer time
  
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
  
  min_funding_rate: 0.03             # minimum rate per 8hr period (0.03% = ~33% APR)
  exit_funding_rate: 0.01            # close when rate drops to this
  position_size_pct: 15              # max % of portfolio per funding position
  max_positions: 2                   # max concurrent funding arb positions
  
  # Risk
  max_basis_divergence_pct: 1.5      # close if spot-futures spread widens beyond this
  rebalance_threshold_pct: 2.0       # rebalance if hedge ratio drifts beyond this
```

### Expected Performance

- Return per position: 20–60% APR (annualized from funding payments)
- Very low variance when hedged properly
- Main risk: brief periods where futures price diverges from spot (basis risk)
- Best when: market sentiment is extremely bullish or bearish (funding rates get extreme)
- Capital requirement: ~$5K minimum per position to overcome fees

---

## 7. Risk Management System (CRITICAL MODULE)

This is the most important part of the entire system. Every strategy must check with the risk manager before placing any order.

### Risk Rules

```yaml
# In config/settings.yaml
risk_management:
  # Per-trade limits
  max_risk_per_trade_pct: 2.0        # max 2% of portfolio at risk per trade
  max_order_size_usd: 500            # hard cap on any single order (adjust with account size)
  
  # Portfolio-level limits
  max_total_exposure_pct: 60         # never have more than 60% of capital in positions
  max_single_pair_exposure_pct: 20   # no more than 20% in any one trading pair
  max_correlated_exposure_pct: 35    # e.g., BTC+ETH combined (highly correlated)
  
  # Drawdown circuit breakers
  daily_loss_limit_pct: 3            # if down 3% today, STOP all trading for 24 hours
  weekly_loss_limit_pct: 7           # if down 7% this week, STOP all trading until manual review
  monthly_loss_limit_pct: 12         # if down 12% this month, STOP everything. Manual restart only.
  
  # Recovery mode
  after_circuit_breaker:
    reduce_position_sizes_by_pct: 50 # trade at half size for recovery_period after circuit break
    recovery_period_hours: 48        # 48 hours of reduced sizing after daily circuit break
  
  # Exchange-level
  max_capital_per_exchange_pct: 60   # never more than 60% on a single exchange
  reserve_cash_pct: 20               # always keep 20% in stablecoins as dry powder
```

### Circuit Breaker Implementation

```
EVERY 60 SECONDS:
  1. Calculate current daily P&L
  2. Calculate current weekly P&L
  3. Calculate current monthly P&L
  
  IF daily_pnl < -daily_loss_limit:
    → Cancel ALL open orders across ALL strategies
    → Close any momentum positions at market
    → Keep grid positions but disable new grid orders
    → Send URGENT Telegram alert
    → Set bot status to DAILY_CIRCUIT_BREAK
    → Resume after 24 hours at 50% sizing
  
  IF weekly_pnl < -weekly_loss_limit:
    → Cancel ALL open orders
    → Close ALL positions at market
    → Send URGENT Telegram alert  
    → Set bot status to WEEKLY_CIRCUIT_BREAK
    → DO NOT auto-resume. Require manual restart.
  
  IF monthly_pnl < -monthly_loss_limit:
    → Full shutdown
    → Send CRITICAL Telegram alert
    → Require manual config change to restart
```

### Pre-Trade Risk Check Flow

```
Strategy wants to place an order:
  → Call risk_manager.check(order)
    1. Would this exceed max_risk_per_trade? → REJECT
    2. Would this exceed max_total_exposure? → REJECT  
    3. Would this exceed max_single_pair_exposure? → REJECT
    4. Would this exceed max_correlated_exposure? → REJECT
    5. Is circuit breaker active? → REJECT
    6. Are we in recovery mode? → Reduce size by 50%, then proceed
    7. Is reserve_cash_pct maintained? → REJECT if it would breach
    → APPROVE order
```

---

## 8. Order Execution Engine

### Smart Executor

Never just fire market orders blindly. The executor handles the messy reality of trading:

```
For each order:
  1. Check current orderbook depth
  2. If order would move price > 0.1%: split into smaller chunks
  3. Place as limit order at best bid/ask (maker fee, not taker)
  4. Wait up to 10 seconds for fill
  5. If not filled: adjust price by 1 tick toward market
  6. If still not filled after 30 seconds: convert to market order (taker fee)
  7. Log fill price, fees, slippage vs expected price
  8. If partial fill: track remaining, retry
  9. If exchange error: retry with exponential backoff (max 3 retries)
  10. If persistent failure: alert via Telegram, mark order as FAILED
```

### Order Reconciliation

Every 5 minutes, reconcile local state with exchange state:

```
1. Fetch all open orders from exchange
2. Fetch all positions from exchange
3. Compare with local database
4. If mismatch found:
   a. Log the discrepancy
   b. Trust exchange state (it's the source of truth)
   c. Update local database
   d. Alert via Telegram if significant
```

---

## 9. Database Schema

```sql
-- Account snapshots (taken every hour)
CREATE TABLE account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    exchange TEXT NOT NULL,
    total_balance_usd REAL NOT NULL,
    free_balance_usd REAL NOT NULL,
    in_positions_usd REAL NOT NULL,
    unrealized_pnl_usd REAL NOT NULL,
    daily_pnl_usd REAL,
    daily_pnl_pct REAL
);

-- Every trade executed
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    strategy TEXT NOT NULL,           -- 'grid', 'momentum', 'funding'
    pair TEXT NOT NULL,               -- 'BTC/USDT'
    side TEXT NOT NULL,               -- 'buy' or 'sell'
    order_type TEXT NOT NULL,         -- 'limit', 'market'
    price REAL NOT NULL,
    amount REAL NOT NULL,
    cost_usd REAL NOT NULL,
    fee_usd REAL NOT NULL,
    fee_currency TEXT,
    exchange_order_id TEXT,
    slippage_pct REAL,               -- actual vs expected price
    linked_trade_id INTEGER,         -- for grid: links buy to its sell
    pnl_usd REAL,                    -- filled in when round trip completes
    notes TEXT
);

-- Strategy-level metrics (aggregated daily)
CREATE TABLE daily_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    strategy TEXT NOT NULL,
    num_trades INTEGER,
    num_wins INTEGER,
    num_losses INTEGER,
    gross_profit_usd REAL,
    gross_loss_usd REAL,
    net_pnl_usd REAL,
    fees_paid_usd REAL,
    max_drawdown_pct REAL,
    sharpe_ratio REAL
);

-- Grid state (persistent across restarts)
CREATE TABLE grid_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    grid_center REAL NOT NULL,
    grid_levels TEXT NOT NULL,        -- JSON array of {price, side, status, order_id}
    inventory_amount REAL DEFAULT 0,  -- coins held from grid fills
    inventory_avg_price REAL,
    total_round_trips INTEGER DEFAULT 0,
    total_profit_usd REAL DEFAULT 0,
    last_rebalance DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker events
CREATE TABLE circuit_breaker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    trigger_type TEXT NOT NULL,       -- 'daily', 'weekly', 'monthly'
    trigger_value REAL NOT NULL,      -- the P&L that triggered it
    positions_closed TEXT,            -- JSON of positions that were closed
    resumed_at DATETIME
);

-- Momentum position tracking
CREATE TABLE momentum_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,               -- 'long' or 'short'
    entry_price REAL NOT NULL,
    entry_time DATETIME NOT NULL,
    amount REAL NOT NULL,
    stop_loss REAL NOT NULL,
    current_stop REAL NOT NULL,       -- trailing stop (updates)
    entry_signals TEXT,               -- JSON of signals that triggered entry
    exit_price REAL,
    exit_time DATETIME,
    exit_reason TEXT,                 -- 'trailing_stop', 'signal_reversal', 'time_stop', 'circuit_breaker'
    pnl_usd REAL,
    pnl_pct REAL,
    status TEXT DEFAULT 'open'        -- 'open' or 'closed'
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_strategy ON trades(strategy);
CREATE INDEX idx_trades_pair ON trades(pair);
CREATE INDEX idx_snapshots_timestamp ON account_snapshots(timestamp);
CREATE INDEX idx_momentum_status ON momentum_positions(status);
```

---

## 10. Telegram Alert System

### Alert Levels

```
📊 INFO (sent in daily digest):
  - Daily P&L summary
  - Strategy performance breakdown  
  - Grid fill count
  - Momentum trade opened/closed

⚠️ WARNING (sent immediately):
  - Position approaching stop loss
  - Exposure near limits
  - Exchange API errors
  - Unusual slippage detected
  - Grid rebalance triggered

🚨 CRITICAL (sent immediately with sound):
  - Circuit breaker triggered
  - Exchange connection lost for > 5 minutes
  - Order reconciliation mismatch
  - Any unhandled exception
```

### Daily Report Format

```
📊 Daily Report — 2026-02-19

💰 Portfolio: $12,450.32 (+$127.50 / +1.03%)

Strategy Breakdown:
  Grid:     +$95.20 (14 round trips)
  Momentum: +$32.30 (1 win, 0 losses)
  Funding:  +$0.00 (not active)

📈 Open Positions:
  Grid BTC: 0.012 BTC inventory ($614 value)
  Grid ETH: 0.8 ETH inventory ($2,100 value)
  Momentum: LONG SOL @ $142.50 (stop: $138.20)

⚡ Risk Status:
  Total exposure: 42% (limit: 60%) ✅
  Daily P&L: +1.03% (limit: -3%) ✅
  Weekly P&L: +2.8% (limit: -7%) ✅
  Reserve cash: 28% (min: 20%) ✅

🔄 24h Stats:
  Total trades: 31
  Avg fill time: 4.2s
  Total fees: $8.40
```

---

## 11. Backtesting Engine

### Requirements

The backtester must model reality accurately. Common backtesting pitfalls to AVOID:

1. **No lookahead bias** — Only use data available at the time of the simulated decision
2. **Realistic fees** — Apply maker/taker fees on every trade (0.075% maker, 0.1% taker)
3. **Slippage modeling** — Add 0.02–0.05% slippage per trade
4. **Partial fills** — Don't assume unlimited liquidity; model based on historical volume
5. **Funding rate simulation** — For funding strategy, use actual historical funding rates
6. **No survivorship bias** — Include delisted pairs if testing historically

### Data Requirements

```
For each trading pair, download:
  - 1-minute OHLCV candles (at least 6 months, ideally 1 year+)
  - Historical funding rates (8-hour intervals)
  - Trading volume data
  
Storage: ~500MB per pair per year at 1-minute resolution
Source: Exchange APIs via ccxt (most exchanges allow historical data download)
```

### Backtest Output

```
For each strategy backtest, output:
  - Total return (% and absolute)
  - Max drawdown (% and duration)
  - Sharpe ratio
  - Sortino ratio
  - Win rate
  - Profit factor
  - Average trade duration
  - Equity curve chart
  - Monthly returns heatmap
  - Drawdown chart
  - Trade distribution histogram
  
Compare against benchmarks:
  - Buy and hold BTC
  - Buy and hold ETH
  - 50/50 BTC/ETH buy and hold
```

---

## 12. Configuration: settings.yaml

```yaml
# config/settings.yaml

bot:
  name: "CryptoQuantBot"
  mode: "paper"                      # "paper" or "live" — START WITH PAPER
  log_level: "INFO"
  timezone: "UTC"

exchanges:
  binance:
    enabled: true
    api_key_env: "BINANCE_API_KEY"       # read from .env
    api_secret_env: "BINANCE_API_SECRET"
    sandbox: true                         # use testnet initially
    rate_limit_ms: 100                    # min ms between API calls
  bybit:
    enabled: false                        # enable as second exchange later
    api_key_env: "BYBIT_API_KEY"
    api_secret_env: "BYBIT_API_SECRET"
    sandbox: true
    rate_limit_ms: 100

risk_management:
  max_risk_per_trade_pct: 2.0
  max_order_size_usd: 500
  max_total_exposure_pct: 60
  max_single_pair_exposure_pct: 20
  max_correlated_exposure_pct: 35
  daily_loss_limit_pct: 3
  weekly_loss_limit_pct: 7
  monthly_loss_limit_pct: 12
  reserve_cash_pct: 20
  after_circuit_breaker:
    reduce_position_sizes_by_pct: 50
    recovery_period_hours: 48

alerts:
  telegram:
    enabled: true
    bot_token_env: "TELEGRAM_BOT_TOKEN"
    chat_id_env: "TELEGRAM_CHAT_ID"
    daily_report_time: "08:00"           # UTC
  
scheduling:
  grid_check_interval_seconds: 5         # check grid fills every 5s
  momentum_check_interval_seconds: 60    # check signals every 60s
  funding_check_interval_seconds: 300    # check funding rates every 5 min
  risk_check_interval_seconds: 60        # risk assessment every 60s
  reconciliation_interval_seconds: 300   # order reconciliation every 5 min
  snapshot_interval_seconds: 3600        # account snapshot every hour
```

---

## 13. Build Order (Phase by Phase)

### Phase 1: Foundation (Week 1–2)
1. Project setup: Python project structure, requirements, Docker
2. Config loader: YAML parsing and validation
3. Exchange connector: ccxt wrapper with authentication, rate limiting, error handling
4. Market data module: fetch OHLCV, tickers, orderbook
5. Database setup: SQLite schema, basic CRUD operations
6. Logging: structured logging to file and console
7. **TEST**: Connect to Binance testnet, fetch data, verify everything works

### Phase 2: Grid Strategy (Week 2–3)
1. Grid calculation: compute grid levels from config
2. Order placement: place grid orders on exchange
3. Fill detection: monitor for filled orders
4. Round-trip tracking: link buys to sells, calculate P&L
5. Grid rebalancing: detect drift, shift grid center
6. Grid state persistence: save/restore grid state across restarts
7. **TEST**: Run grid strategy on testnet for 48+ hours continuously

### Phase 3: Risk Management (Week 3–4)
1. Position tracker: real-time exposure calculation
2. Pre-trade risk checks: implement the approval flow
3. Circuit breakers: daily/weekly/monthly loss limits
4. Order reconciliation: exchange vs local state sync
5. Recovery mode: reduced sizing after circuit breaks
6. **TEST**: Simulate circuit breaker scenarios, verify they trigger correctly

### Phase 4: Momentum Strategy (Week 4–5)
1. Technical indicators: EMA, RSI, ADX, MACD, ATR via ta library
2. Signal generation: combine indicators per config
3. Timeframe alignment: multi-timeframe confirmation
4. Trailing stop: dynamic stop adjustment
5. Position management: entry, stop tracking, exit logic
6. **TEST**: Run on testnet alongside grid strategy

### Phase 5: Monitoring & Alerts (Week 5–6)
1. Telegram bot: setup, message formatting
2. Alert routing: INFO/WARNING/CRITICAL levels
3. Daily report: automated summary generation
4. Trade journal: logging all trades with context
5. Performance metrics: win rate, Sharpe, drawdown calculations
6. **TEST**: Verify alerts fire correctly for all scenarios

### Phase 6: Backtesting (Week 6–7)
1. Historical data downloader
2. Backtest engine with fee/slippage modeling
3. Strategy adapters for backtesting
4. Result analyzer and visualization
5. **RUN**: Backtest both strategies on 6+ months of data
6. **TUNE**: Adjust parameters based on backtest results

### Phase 7: Hardening & Deployment (Week 7–8)
1. Error handling: every external call wrapped in try/catch with retries
2. Graceful shutdown: save state on SIGTERM, resume on restart
3. Docker setup: containerized deployment
4. VPS deployment: docker-compose on DigitalOcean/Hetzner
5. Monitoring: health check endpoint, uptime monitoring
6. **GO LIVE**: Switch from paper to live with minimum capital

---

## 14. Paper Trading Protocol

**DO NOT skip this phase. Run paper trading for minimum 2 weeks before going live.**

```
Week 1 Paper Trading:
  - Run grid + momentum on Binance testnet
  - Verify all orders place and fill correctly
  - Verify circuit breakers work
  - Verify Telegram alerts arrive
  - Verify daily reports are accurate
  - Fix any bugs

Week 2 Paper Trading:
  - Track simulated P&L
  - Compare to backtest expectations
  - Tune grid spacing if fills are too rare or too frequent
  - Tune momentum signals if too many false entries
  - Document any issues

Going Live Checklist:
  □ Paper traded for 2+ weeks with no critical bugs
  □ Circuit breakers tested and verified
  □ Telegram alerts working
  □ P&L tracking matches exchange records
  □ All API keys are production keys (not testnet)
  □ Sandbox mode set to false
  □ Initial capital deposited on exchange
  □ Reserve cash percentage verified
  □ Emergency shutdown procedure documented
  □ Exchange withdrawal address whitelisted (security)
```

---

## 15. Scaling Roadmap

```
$500–$2K:    Grid only (BTC/USDT + ETH/USDT). 2 pairs, conservative spacing.
$2K–$5K:    Add momentum strategy. Add SOL/USDT grid pair.
$5K–$10K:   Add 1–2 more momentum pairs. Tighten grid spacing for more fills.
$10K–$25K:  Enable funding rate strategy. Add second exchange (Bybit).
$25K–$50K:  Full strategy suite. Consider wider grid ranges for larger moves.
$50K+:      Cross-exchange arbitrage becomes viable. Review and optimize.
```

---

## 16. Important Reminders

1. **Start in paper mode.** Always.
2. **Never store API keys in code.** Use .env files, never commit them.
3. **Enable IP whitelisting** on exchange API keys. Only allow your VPS IP.
4. **Disable withdrawal permissions** on API keys. The bot never needs to withdraw.
5. **Keep reserves.** The 20% cash reserve is not optional — it's your buffer for bad days and for re-entering after circuit breaks.
6. **Review weekly.** Spend 15–30 minutes each weekend reviewing the performance report. Are returns matching expectations? Is one strategy underperforming? Adjust configs if needed.
7. **Don't over-optimize.** Backtesting can lead to curve-fitting (parameters that worked perfectly in the past but fail in the future). Keep parameters reasonable and robust.
8. **Expect losing days.** Even a profitable system has losing days and weeks. The circuit breakers exist so that losing periods don't become catastrophic.
9. **The bot is a tool, not a guarantee.** Monitor it, understand it, improve it over time.

---

*This document contains everything needed to build the complete system. Work through the phases in order. Test each module before moving to the next. Good luck.*