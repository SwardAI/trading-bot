"""Phase 5/6/7 tests — Alerts, Backtest Engine, Performance Tracking, Deployment."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.alerts.telegram_bot import AlertLevel, TelegramAlerter
from src.journal.performance import PerformanceTracker
from src.backtest.engine import BacktestEngine, BacktestTrade, BacktestResult


class TestAlertLevel(unittest.TestCase):
    """Test alert level enum."""

    def test_alert_levels_exist(self):
        """AlertLevel should have INFO, WARNING, CRITICAL."""
        self.assertEqual(AlertLevel.INFO.value, "info")
        self.assertEqual(AlertLevel.WARNING.value, "warning")
        self.assertEqual(AlertLevel.CRITICAL.value, "critical")


class TestTelegramAlerter(unittest.TestCase):
    """Test Telegram alerter initialization."""

    def test_disabled_when_no_config(self):
        """Alerter should gracefully handle missing config."""
        alerter = TelegramAlerter({})
        self.assertFalse(alerter.enabled)
        self.assertIsNone(alerter._bot)

    def test_disabled_when_no_env_vars(self):
        """Alerter should be disabled without env vars even if enabled in config."""
        config = {
            "enabled": True,
            "bot_token_env": "NONEXISTENT_TOKEN_12345",
            "chat_id_env": "NONEXISTENT_CHAT_12345",
        }
        alerter = TelegramAlerter(config)
        # Bot won't initialize without valid token
        self.assertIsNone(alerter._bot)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance metric calculations."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.init_db()
        self.tracker = PerformanceTracker(self.db)

    def tearDown(self):
        self.db.close()

    def test_empty_metrics(self):
        """Should return zeroed metrics when no trades exist."""
        metrics = self.tracker.get_strategy_metrics("grid", days=1)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertAlmostEqual(metrics["net_pnl"], 0.0)

    def test_metrics_with_trades(self):
        """Should calculate correct metrics from trades."""
        now = datetime.now(timezone.utc).isoformat()

        # Insert winning trade
        self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd, fee_usd, pnl_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "grid", "BTC/USDT", "sell", "limit", 50100, 0.001, 50.1, 0.0375, 5.0),
        )
        # Insert losing trade
        self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd, fee_usd, pnl_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "grid", "BTC/USDT", "sell", "limit", 49900, 0.001, 49.9, 0.0375, -2.0),
        )

        metrics = self.tracker.get_strategy_metrics("grid", days=1)
        self.assertEqual(metrics["num_trades"], 2)
        self.assertAlmostEqual(metrics["net_pnl"], 3.0)  # 5 - 2
        self.assertEqual(metrics["num_wins"], 1)
        self.assertEqual(metrics["num_losses"], 1)


class TestBacktestEngine(unittest.TestCase):
    """Test backtest engine initialization and data classes."""

    def test_engine_initialization(self):
        """BacktestEngine should initialize with default fee/slippage settings."""
        engine = BacktestEngine()
        self.assertAlmostEqual(engine.initial_capital, 10000)
        self.assertAlmostEqual(engine.maker_fee, 0.00075)  # 0.075%
        self.assertAlmostEqual(engine.taker_fee, 0.001)    # 0.1%
        self.assertAlmostEqual(engine.slippage_pct, 0.0003) # 0.03%

    def test_custom_initialization(self):
        """BacktestEngine should accept custom parameters."""
        engine = BacktestEngine(
            initial_capital=5000,
            maker_fee=0.05,
            taker_fee=0.08,
            slippage_pct=0.02,
        )
        self.assertAlmostEqual(engine.initial_capital, 5000)
        self.assertAlmostEqual(engine.maker_fee, 0.0005)

    def test_backtest_trade_dataclass(self):
        """BacktestTrade should hold trade data."""
        trade = BacktestTrade(
            timestamp=datetime.now(),
            side="buy",
            price=50000.0,
            amount=0.001,
            cost=50.0,
            fee=0.0375,
        )
        self.assertEqual(trade.side, "buy")
        self.assertAlmostEqual(trade.price, 50000.0)
        self.assertAlmostEqual(trade.pnl, 0.0)  # default

    def test_backtest_result_dataclass(self):
        """BacktestResult should hold all result fields."""
        result = BacktestResult(
            strategy="grid",
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2025-01-01",
            end_date="2025-01-31",
            initial_capital=10000,
            final_capital=10500,
            total_return_pct=5.0,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=60.0,
            gross_profit=800.0,
            gross_loss=-300.0,
            net_profit=500.0,
            profit_factor=2.67,
            max_drawdown_pct=3.5,
            sharpe_ratio=1.8,
            avg_trade_pnl=5.0,
        )
        self.assertEqual(result.strategy, "grid")
        self.assertAlmostEqual(result.total_return_pct, 5.0)
        self.assertEqual(result.total_trades, 100)


class TestImports(unittest.TestCase):
    """Verify all critical modules can be imported."""

    def test_core_imports(self):
        from src.core.database import Database
        from src.core.config_loader import load_all_configs
        from src.core.exchange import ExchangeManager
        from src.core.logger import setup_logger

    def test_strategy_imports(self):
        from src.strategies.base_strategy import BaseStrategy
        from src.strategies.grid_strategy import GridStrategy
        from src.strategies.momentum_strategy import MomentumStrategy

    def test_risk_imports(self):
        from src.risk.risk_manager import RiskManager, OrderRequest, RiskDecision
        from src.risk.circuit_breaker import CircuitBreaker
        from src.risk.position_tracker import PositionTracker

    def test_data_imports(self):
        from src.data.market_data import MarketDataManager
        from src.data.indicators import compute_all_indicators

    def test_execution_imports(self):
        from src.execution.order_manager import OrderManager

    def test_journal_imports(self):
        from src.journal.performance import PerformanceTracker
        from src.journal.reporter import Reporter

    def test_backtest_imports(self):
        from src.backtest.engine import BacktestEngine, BacktestResult, BacktestTrade


if __name__ == "__main__":
    unittest.main()
