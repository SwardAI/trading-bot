"""Phase 2 tests — Grid Strategy and Order Management."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.risk.risk_manager import OrderRequest, RiskDecision


class TestGridCalculation(unittest.TestCase):
    """Test grid level calculation logic."""

    def _make_strategy(self, grid_type="geometric", num_grids=10, spacing=0.5,
                       upper=10, lower=10):
        """Create a GridStrategy with mocked dependencies."""
        from src.strategies.grid_strategy import GridStrategy

        pair_config = {
            "symbol": "BTC/USDT",
            "grid_type": grid_type,
            "num_grids": num_grids,
            "grid_spacing_pct": spacing,
            "order_size_usd": 50,
            "upper_bound_pct": upper,
            "lower_bound_pct": lower,
            "rebalance_trigger_pct": 8,
            "stop_loss_pct": 15,
        }
        global_config = {
            "max_open_orders": 40,
            "min_profit_after_fees": 0.15,
            "fee_rate": 0.075,
        }

        exchange = MagicMock()
        db = MagicMock()
        risk_manager = MagicMock()
        risk_manager.check.return_value = RiskDecision(True, "approved")
        order_manager = MagicMock()
        market_data = MagicMock()

        strategy = GridStrategy(
            pair_config=pair_config,
            global_config=global_config,
            exchange=exchange,
            db=db,
            risk_manager=risk_manager,
            order_manager=order_manager,
            market_data=market_data,
        )
        return strategy

    def test_geometric_grid_generates_levels(self):
        """Geometric grid should produce buy levels below and sell levels above center."""
        strategy = self._make_strategy(grid_type="geometric", num_grids=10, spacing=0.5)
        levels = strategy.calculate_grid_levels(50000.0)

        self.assertGreater(len(levels), 0, "Grid should have at least one level")

        buy_levels = [l for l in levels if l["side"] == "buy"]
        sell_levels = [l for l in levels if l["side"] == "sell"]

        self.assertGreater(len(buy_levels), 0, "Should have buy levels")
        self.assertGreater(len(sell_levels), 0, "Should have sell levels")

        # All buys below center, all sells above
        for level in buy_levels:
            self.assertLess(level["price"], 50000.0, "Buy should be below center")
        for level in sell_levels:
            self.assertGreater(level["price"], 50000.0, "Sell should be above center")

    def test_arithmetic_grid_generates_levels(self):
        """Arithmetic grid should produce evenly spaced levels."""
        strategy = self._make_strategy(grid_type="arithmetic", num_grids=5, spacing=1.0)
        levels = strategy.calculate_grid_levels(50000.0)

        self.assertGreater(len(levels), 0)

        buy_levels = sorted([l for l in levels if l["side"] == "buy"], key=lambda x: x["price"])
        sell_levels = sorted([l for l in levels if l["side"] == "sell"], key=lambda x: x["price"])

        self.assertGreater(len(buy_levels), 0)
        self.assertGreater(len(sell_levels), 0)

    def test_levels_respect_bounds(self):
        """Grid levels outside upper/lower bounds should be excluded."""
        strategy = self._make_strategy(num_grids=50, spacing=1.0, upper=5, lower=5)
        levels = strategy.calculate_grid_levels(50000.0)

        lower_bound = 50000.0 * 0.95
        upper_bound = 50000.0 * 1.05

        for level in levels:
            self.assertGreaterEqual(level["price"], lower_bound - 1,
                                    f"Level {level['price']} below lower bound")
            self.assertLessEqual(level["price"], upper_bound + 1,
                                 f"Level {level['price']} above upper bound")

    def test_levels_have_correct_structure(self):
        """Each grid level should have price, side, status, and order_id."""
        strategy = self._make_strategy()
        levels = strategy.calculate_grid_levels(50000.0)

        for level in levels:
            self.assertIn("price", level)
            self.assertIn("side", level)
            self.assertIn("status", level)
            self.assertIn("order_id", level)
            self.assertEqual(level["status"], "pending")
            self.assertIsNone(level["order_id"])
            self.assertIn(level["side"], ("buy", "sell"))


class TestDatabase(unittest.TestCase):
    """Test database initialization and operations."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.init_db()

    def tearDown(self):
        self.db.close()

    def test_tables_created(self):
        """All required tables should exist after init."""
        tables = self.db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [t["name"] for t in tables]

        for expected in ["trades", "account_snapshots", "daily_metrics",
                         "grid_state", "circuit_breaker_events", "momentum_positions"]:
            self.assertIn(expected, table_names, f"Table '{expected}' should exist")

    def test_insert_and_fetch_trade(self):
        """Should be able to insert and retrieve a trade."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd, fee_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "grid", "BTC/USDT", "buy", "limit", 50000.0, 0.001, 50.0, 0.0375),
        )

        trade = self.db.fetch_one("SELECT * FROM trades WHERE pair = 'BTC/USDT'")
        self.assertIsNotNone(trade)
        self.assertEqual(trade["strategy"], "grid")
        self.assertEqual(trade["side"], "buy")
        self.assertAlmostEqual(trade["price"], 50000.0)

    def test_fetch_all_returns_list(self):
        """fetch_all should return a list of dicts."""
        result = self.db.fetch_all("SELECT * FROM trades")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_fetch_one_returns_none_when_empty(self):
        """fetch_one should return None when no rows match."""
        result = self.db.fetch_one("SELECT * FROM trades WHERE id = 999")
        self.assertIsNone(result)


class TestOrderRequest(unittest.TestCase):
    """Test OrderRequest dataclass."""

    def test_create_order_request(self):
        """Should create an OrderRequest with all fields."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            price=50000.0,
            cost_usd=50.0,
            strategy="grid",
        )
        self.assertEqual(order.symbol, "BTC/USDT")
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.cost_usd, 50.0)


if __name__ == "__main__":
    unittest.main()
