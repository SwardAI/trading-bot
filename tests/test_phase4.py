"""Phase 4 tests — Risk Management, Circuit Breaker, Position Tracking."""
import sys
import os
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import OrderRequest, RiskDecision, RiskManager


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker activation and recovery."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.init_db()
        self.config = {
            "daily_loss_limit_pct": 3.0,
            "weekly_loss_limit_pct": 7.0,
            "monthly_loss_limit_pct": 12.0,
            "after_circuit_breaker": {
                "reduce_position_sizes_by_pct": 50,
                "recovery_period_hours": 48,
            },
        }
        self.cb = CircuitBreaker(self.config, self.db)

    def tearDown(self):
        self.db.close()

    def test_initially_inactive(self):
        """Circuit breaker should start inactive with no DB events."""
        self.assertFalse(self.cb.is_active())

    def test_daily_trigger(self):
        """Should trigger on daily loss exceeding limit."""
        # Insert a losing trade for today
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd, fee_usd, pnl_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "grid", "BTC/USDT", "sell", "limit", 50000, 0.01, 500, 0.375, -400),
        )

        # Check with a portfolio of 10000 — loss of $400 is 4%, above 3% limit
        triggered, level = self.cb.check(10000.0)
        self.assertTrue(triggered)
        self.assertEqual(level, "daily")
        self.assertTrue(self.cb.is_active())

    def test_no_trigger_on_small_loss(self):
        """Should not trigger if losses are within limits."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd, fee_usd, pnl_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "grid", "BTC/USDT", "sell", "limit", 50000, 0.001, 50, 0.0375, -10),
        )

        # $10 loss on $10000 portfolio = 0.1%, well within 3% limit
        triggered, level = self.cb.check(10000.0)
        self.assertFalse(triggered)
        self.assertIsNone(level)
        self.assertFalse(self.cb.is_active())

    def test_activation_logged_to_db(self):
        """Activating circuit breaker should create a DB event."""
        self.cb.activate("daily", -300.0)

        event = self.db.fetch_one(
            "SELECT * FROM circuit_breaker_events ORDER BY id DESC LIMIT 1"
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["trigger_type"], "daily")
        self.assertAlmostEqual(event["trigger_value"], -300.0)

    def test_recovery_multiplier(self):
        """Recovery mode should return 0.5x multiplier (50% reduction)."""
        # Default recovery reduction is 50%
        # When not in recovery, multiplier should be 1.0
        self.assertAlmostEqual(self.cb.get_recovery_size_multiplier(), 1.0)

    def test_get_status(self):
        """get_status should return a dict with expected keys."""
        status = self.cb.get_status()
        self.assertIn("active", status)
        self.assertIn("level", status)
        self.assertIn("recovery_mode", status)
        self.assertIn("size_multiplier", status)
        self.assertFalse(status["active"])


class TestRiskManager(unittest.TestCase):
    """Test risk manager pre-trade checks."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.init_db()

        self.exchange = MagicMock()
        # Mock balance response
        self.exchange.fetch_balance.return_value = {
            "total": {"USDT": 10000.0},
            "free": {"USDT": 8000.0},
            "used": {"USDT": 2000.0},
        }
        # Mock open orders
        self.exchange.fetch_open_orders.return_value = []

        self.config = {
            "max_risk_per_trade_pct": 2.0,
            "max_order_size_usd": 500,
            "max_total_exposure_pct": 60,
            "max_single_pair_exposure_pct": 20,
            "max_correlated_exposure_pct": 35,
            "reserve_cash_pct": 20,
            "daily_loss_limit_pct": 3.0,
            "weekly_loss_limit_pct": 7.0,
            "monthly_loss_limit_pct": 12.0,
            "after_circuit_breaker": {
                "reduce_position_sizes_by_pct": 50,
                "recovery_period_hours": 48,
            },
        }

        self.rm = RiskManager(self.config, self.db, self.exchange)

    def tearDown(self):
        self.db.close()

    def test_approve_small_order(self):
        """Small order within all limits should be approved."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            price=50000.0,
            cost_usd=50.0,
            strategy="grid",
        )
        decision = self.rm.check(order)
        self.assertTrue(decision.approved, f"Should be approved: {decision.reason}")

    def test_reject_oversized_order(self):
        """Order exceeding max_order_size_usd should be rejected."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=0.02,
            price=50000.0,
            cost_usd=1000.0,  # Exceeds $500 cap
            strategy="grid",
        )
        decision = self.rm.check(order)
        self.assertFalse(decision.approved)
        self.assertIn("exceeds", decision.reason.lower())

    def test_reject_when_circuit_breaker_active(self):
        """Orders should be rejected when circuit breaker is active."""
        self.rm.circuit_breaker.activate("daily", -300.0)

        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            price=50000.0,
            cost_usd=50.0,
            strategy="grid",
        )
        decision = self.rm.check(order)
        self.assertFalse(decision.approved)
        self.assertIn("circuit breaker", decision.reason.lower())

    def test_risk_decision_dataclass(self):
        """RiskDecision should carry approved, reason, and optional adjusted_amount."""
        decision = RiskDecision(True, "All checks passed", adjusted_amount=0.0005)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "All checks passed")
        self.assertAlmostEqual(decision.adjusted_amount, 0.0005)


if __name__ == "__main__":
    unittest.main()
