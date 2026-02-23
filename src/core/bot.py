import signal
import time
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from src.core.config_loader import load_all_configs
from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.core.logger import setup_logger
from src.data.market_data import MarketDataManager
from src.execution.order_manager import OrderManager
from src.alerts.telegram_bot import AlertLevel, TelegramAlerter
from src.journal.performance import PerformanceTracker
from src.journal.reporter import Reporter
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.strategies.grid_strategy import GridStrategy
from src.strategies.momentum_strategy import MomentumStrategy

logger = setup_logger(__name__)


class Bot:
    """Main bot orchestrator — initializes all components and manages lifecycle.

    Initializes exchange, database, risk management, order execution, and
    strategies. Uses APScheduler for periodic tasks (grid checks, risk checks,
    reconciliation, snapshots).
    """

    def __init__(self):
        logger.info("Initializing CryptoQuantBot...")

        # Load config
        self.config = load_all_configs()
        bot_config = self.config.get("bot", {})
        self.mode = bot_config.get("mode", "paper")
        self.running = False

        # Update log level from config
        log_level = bot_config.get("log_level", "INFO")
        logger.setLevel(log_level)

        # Initialize database
        self.db = Database("data/bot.db")
        self.db.init_db()

        # Initialize exchanges
        self.exchanges: dict[str, ExchangeManager] = {}
        for exchange_id, exchange_config in self.config.get("exchanges", {}).items():
            if exchange_config.get("enabled", False):
                self.exchanges[exchange_id] = ExchangeManager(exchange_id, exchange_config)

        # Get primary exchange
        self.primary_exchange: ExchangeManager | None = None
        if self.exchanges:
            self.primary_exchange = next(iter(self.exchanges.values()))

        # Initialize market data
        self.market_data: MarketDataManager | None = None
        if self.primary_exchange:
            self.market_data = MarketDataManager(self.primary_exchange)

        # Initialize risk manager
        self.risk_manager: RiskManager | None = None
        if self.primary_exchange:
            risk_config = self.config.get("risk_management", {})
            self.risk_manager = RiskManager(risk_config, self.db, self.primary_exchange)

        # Initialize order manager
        self.order_manager: OrderManager | None = None
        if self.primary_exchange:
            self.order_manager = OrderManager(self.primary_exchange, self.db)

        # Alerts
        alert_config = self.config.get("alerts", {}).get("telegram", {})
        self.alerter = TelegramAlerter(alert_config)

        # Performance tracking and reporting
        self.performance = PerformanceTracker(self.db)
        self.reporter = Reporter(self.db, self.performance, self.risk_manager)

        # Strategies
        self.strategies: list[BaseStrategy] = []

        # Scheduler
        self.scheduler = BackgroundScheduler(timezone="UTC")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"Bot initialized: mode={self.mode}, "
            f"exchanges={list(self.exchanges.keys())}"
        )

    def _init_strategies(self):
        """Initialize enabled strategies from config."""
        # Grid strategy — one instance per enabled pair
        grid_config = self.config.get("grid_strategy", {})
        if grid_config.get("enabled") and self.primary_exchange and self.risk_manager and self.order_manager and self.market_data:
            for pair_config in grid_config.get("pairs", []):
                strategy = GridStrategy(
                    pair_config=pair_config,
                    global_config=grid_config,
                    exchange=self.primary_exchange,
                    db=self.db,
                    risk_manager=self.risk_manager,
                    order_manager=self.order_manager,
                    market_data=self.market_data,
                )
                self.strategies.append(strategy)
                logger.info(f"Grid strategy registered for {pair_config['symbol']}")

        # Momentum strategy — single instance covering all pairs
        momentum_config = self.config.get("momentum_strategy", {})
        if momentum_config.get("enabled") and self.primary_exchange and self.risk_manager and self.order_manager and self.market_data:
            strategy = MomentumStrategy(
                config=momentum_config,
                exchange=self.primary_exchange,
                db=self.db,
                risk_manager=self.risk_manager,
                order_manager=self.order_manager,
                market_data=self.market_data,
            )
            self.strategies.append(strategy)
            logger.info(f"Momentum strategy registered for {len(momentum_config.get('pairs', []))} pairs")

        if not self.strategies:
            logger.info("No strategies enabled")

    def _setup_scheduler(self):
        """Configure APScheduler with intervals from config."""
        sched_config = self.config.get("scheduling", {})

        # Grid strategy tick — check fills, rebalance, stop loss
        grid_interval = sched_config.get("grid_check_interval_seconds", 5)
        self.scheduler.add_job(
            self._tick_grid_strategies,
            "interval",
            seconds=grid_interval,
            id="grid_tick",
            name="Grid tick",
            max_instances=1,
            coalesce=True,
        )

        # Momentum strategy tick — separate from grid to avoid being blocked
        momentum_interval = sched_config.get("momentum_check_interval_seconds", 60)
        self.scheduler.add_job(
            self._tick_momentum_strategies,
            "interval",
            seconds=momentum_interval,
            id="momentum_tick",
            name="Momentum tick",
            max_instances=1,
            coalesce=True,
        )

        # Risk check — circuit breaker evaluation
        risk_interval = sched_config.get("risk_check_interval_seconds", 60)
        self.scheduler.add_job(
            self._risk_check,
            "interval",
            seconds=risk_interval,
            id="risk_check",
            name="Risk check",
            max_instances=1,
            coalesce=True,
        )

        # Order reconciliation — sync local state with exchange
        recon_interval = sched_config.get("reconciliation_interval_seconds", 300)
        self.scheduler.add_job(
            self._reconcile,
            "interval",
            seconds=recon_interval,
            id="reconciliation",
            name="Order reconciliation",
            max_instances=1,
            coalesce=True,
        )

        # Account snapshot — hourly balance recording
        snapshot_interval = sched_config.get("snapshot_interval_seconds", 3600)
        self.scheduler.add_job(
            self._take_snapshot,
            "interval",
            seconds=snapshot_interval,
            id="snapshot",
            name="Account snapshot",
            max_instances=1,
            coalesce=True,
        )

        # Periodic Telegram report
        telegram_config = self.config.get("alerts", {}).get("telegram", {})
        report_interval = telegram_config.get("report_interval_seconds", 0)
        if report_interval > 0:
            self.scheduler.add_job(
                self._send_daily_report,
                "interval",
                seconds=report_interval,
                id="periodic_report",
                name="Periodic TG report",
            )
        else:
            # Fall back to daily cron
            report_time = telegram_config.get("daily_report_time", "08:00")
            hour, minute = report_time.split(":")
            self.scheduler.add_job(
                self._send_daily_report,
                "cron",
                hour=int(hour),
                minute=int(minute),
                id="daily_report",
                name="Daily report",
            )

        # Daily metrics aggregation (end of day)
        self.scheduler.add_job(
            self._save_daily_metrics,
            "cron",
            hour=23,
            minute=55,
            id="daily_metrics",
            name="Daily metrics",
        )

        logger.info(
            f"Scheduler configured: grid={grid_interval}s, momentum={momentum_interval}s, "
            f"risk={risk_interval}s, reconcile={recon_interval}s, snapshot={snapshot_interval}s"
        )

    def _tick_strategy(self, strategy: BaseStrategy):
        """Run on_tick() for a single strategy with failure tracking.

        Tracks consecutive failures and stops after 10 in a row to prevent
        blind operation during API outages.
        """
        if not strategy.is_running:
            return

        try:
            strategy.on_tick()
            strategy._consecutive_failures = 0
        except Exception as e:
            strategy._consecutive_failures = getattr(strategy, '_consecutive_failures', 0) + 1
            logger.error(f"Error in {strategy.strategy_name} tick ({strategy._consecutive_failures} consecutive): {e}", exc_info=True)

            if strategy._consecutive_failures >= 10:
                logger.critical(
                    f"{strategy.strategy_name} failed {strategy._consecutive_failures} ticks in a row — "
                    f"stopping to prevent stale orders"
                )
                self.alerter.send_alert(
                    f"Strategy {strategy.strategy_name} stopped after {strategy._consecutive_failures} consecutive errors: {e}",
                    AlertLevel.CRITICAL,
                )
                strategy.stop()

    def _tick_grid_strategies(self):
        """Tick all grid strategies (runs every grid_check_interval_seconds)."""
        for strategy in self.strategies:
            if isinstance(strategy, GridStrategy):
                self._tick_strategy(strategy)

    def _tick_momentum_strategies(self):
        """Tick all momentum strategies (runs every momentum_check_interval_seconds)."""
        for strategy in self.strategies:
            if isinstance(strategy, MomentumStrategy):
                self._tick_strategy(strategy)

    def _risk_check(self):
        """Periodic risk evaluation — check circuit breakers and auto-resume."""
        if not self.risk_manager:
            return

        # Check if a daily circuit breaker should auto-resume
        self.risk_manager.circuit_breaker.check_auto_resume()

        try:
            triggered, level = self.risk_manager.run_circuit_breaker_check()
            if triggered:
                logger.critical(f"Circuit breaker triggered: {level} — stopping strategies")
                pnl = self.risk_manager.circuit_breaker.get_daily_pnl()
                self.alerter.send_circuit_breaker_alert(level, pnl)
                for strategy in self.strategies:
                    if strategy.is_running:
                        strategy.stop()
        except Exception as e:
            logger.error(f"Error in risk check: {e}", exc_info=True)

    def _reconcile(self):
        """Periodic order reconciliation — sync with exchange."""
        if not self.order_manager:
            return

        for strategy in self.strategies:
            if isinstance(strategy, GridStrategy) and strategy.is_running:
                try:
                    self.order_manager.reconcile(strategy.symbol)
                except Exception as e:
                    logger.error(f"Reconciliation error for {strategy.symbol}: {e}")

        # Also sync position tracker
        if self.risk_manager:
            self.risk_manager.position_tracker.sync_with_exchange()

    def _take_snapshot(self):
        """Record account balance snapshot to database.

        Uses the risk manager's capped balance when available so snapshots
        reflect the effective portfolio size, not the raw sandbox balance.
        """
        if not self.primary_exchange:
            return

        try:
            # Use capped balance from risk manager if available
            if self.risk_manager:
                bal = self.risk_manager.position_tracker.get_balance()
                total = bal["total_usd"]
                free = bal["free_usd"]
                used = bal["used_usd"]
            else:
                balance = self.primary_exchange.fetch_balance()
                total = float(balance.get("total", {}).get("USDT", 0))
                free = float(balance.get("free", {}).get("USDT", 0))
                used = total - free

            now = datetime.now(timezone.utc).isoformat()

            # Get daily P&L if risk manager available
            daily_pnl = 0.0
            daily_pnl_pct = 0.0
            if self.risk_manager:
                daily_pnl = self.risk_manager.circuit_breaker.get_daily_pnl()
                daily_pnl_pct = (daily_pnl / total * 100) if total > 0 else 0

            self.db.execute(
                """INSERT INTO account_snapshots
                (timestamp, exchange, total_balance_usd, free_balance_usd,
                 in_positions_usd, unrealized_pnl_usd, daily_pnl_usd, daily_pnl_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, self.primary_exchange.exchange_id, total, free, used, 0.0, daily_pnl, daily_pnl_pct),
            )
            logger.info(f"Snapshot: ${total:.2f} total, ${free:.2f} free, daily P&L: ${daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"Failed to take snapshot: {e}")

    def _send_daily_report(self):
        """Generate and send the daily performance report."""
        try:
            report = self.reporter.generate_daily_report()
            self.alerter.send_daily_report(report)
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")

    def _save_daily_metrics(self):
        """Save aggregated daily metrics at end of day."""
        try:
            self.performance.save_daily_metrics()
        except Exception as e:
            logger.error(f"Failed to save daily metrics: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down...")
        self.stop()

    def start(self):
        """Start the bot — load markets, init strategies, start scheduler."""
        self.running = True
        logger.info(f"Starting bot in {self.mode.upper()} mode...")

        # Load markets for all exchanges
        for name, exchange in self.exchanges.items():
            try:
                exchange.load_markets()
            except Exception as e:
                logger.error(f"Failed to load markets for {name}: {e}")

        # Initialize strategies
        self._init_strategies()

        # Start strategies
        for strategy in self.strategies:
            try:
                strategy.start()
            except Exception as e:
                logger.error(f"Failed to start {strategy.strategy_name}: {e}", exc_info=True)

        # Take initial snapshot so reports have a baseline immediately
        self._take_snapshot()

        # Setup and start scheduler
        self._setup_scheduler()
        self.scheduler.start()

        logger.info(
            f"Bot running: {len(self.strategies)} strategies active, "
            f"scheduler started"
        )

        # Main loop — keep alive while scheduler runs
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the bot gracefully — stop strategies, scheduler, close connections."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping bot...")

        # Stop scheduler first to prevent new ticks
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

        # Stop strategies (cancel orders, save state)
        # Do this BEFORE closing anything else — order cancellation is critical
        for strategy in self.strategies:
            try:
                if strategy.is_running:
                    strategy.stop()
                    logger.info(f"Strategy {strategy.strategy_name} stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping {strategy.strategy_name}: {e}")
                # Last resort: try to cancel all orders directly via exchange
                try:
                    if hasattr(strategy, 'symbol'):
                        self.primary_exchange.exchange.cancel_all_orders(strategy.symbol)
                        logger.info(f"Emergency order cancellation for {strategy.symbol}")
                except Exception as cancel_err:
                    logger.error(f"Emergency cancellation also failed: {cancel_err}")

        # Close database
        self.db.close()

        logger.info("Bot stopped.")

    def run(self):
        """Main entry point — start the bot."""
        self.start()
