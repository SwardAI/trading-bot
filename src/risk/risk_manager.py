from dataclasses import dataclass

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.core.logger import setup_logger
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_tracker import PositionTracker

logger = setup_logger("risk.manager")


@dataclass
class OrderRequest:
    """Represents an order that needs risk approval before placement."""
    symbol: str
    side: str  # "buy" or "sell"
    amount: float  # in base currency
    price: float  # limit price
    cost_usd: float  # total order value in USD
    strategy: str  # "grid", "momentum", "funding"


@dataclass
class RiskDecision:
    """Result of a risk check."""
    approved: bool
    reason: str
    adjusted_amount: float | None = None  # if reduced due to recovery mode


class RiskManager:
    """Central risk authority — every order must pass through check() before placement.

    Enforces:
    - Per-trade risk limits
    - Portfolio exposure limits (total, per-pair, correlated)
    - Circuit breaker state
    - Recovery mode sizing
    - Cash reserve requirements

    Args:
        config: risk_management section from settings.yaml.
        db: Database instance.
        exchange: ExchangeManager instance.
    """

    def __init__(self, config: dict, db: Database, exchange: ExchangeManager):
        self.config = config
        self.db = db

        # Limits
        self.max_risk_per_trade_pct = config.get("max_risk_per_trade_pct", 2.0)
        self.max_order_size_usd = config.get("max_order_size_usd", 500)
        self.max_total_exposure_pct = config.get("max_total_exposure_pct", 60)
        self.max_single_pair_pct = config.get("max_single_pair_exposure_pct", 20)
        self.max_correlated_pct = config.get("max_correlated_exposure_pct", 35)
        self.reserve_cash_pct = config.get("reserve_cash_pct", 20)

        # Portfolio cap (for sandbox accounts with inflated balances)
        self.max_portfolio_usd = config.get("max_portfolio_usd", 0)

        # Components
        self.position_tracker = PositionTracker(db, exchange, self.max_portfolio_usd)
        self.circuit_breaker = CircuitBreaker(config, db)

        cap_msg = f", portfolio_cap=${self.max_portfolio_usd:,.0f}" if self.max_portfolio_usd else ""
        logger.info(
            f"RiskManager initialized: max_exposure={self.max_total_exposure_pct}%, "
            f"max_per_pair={self.max_single_pair_pct}%, "
            f"reserve={self.reserve_cash_pct}%{cap_msg}"
        )

    def check(self, order: OrderRequest) -> RiskDecision:
        """Run the 7-step pre-trade risk check on an order.

        Args:
            order: OrderRequest with order details.

        Returns:
            RiskDecision indicating approval/rejection and reason.
        """
        balance = self.position_tracker.get_balance()
        total_usd = balance["total_usd"]

        if total_usd <= 0:
            return RiskDecision(False, "No balance available")

        # 1. Per-trade risk check
        trade_risk_pct = (order.cost_usd / total_usd) * 100
        if trade_risk_pct > self.max_risk_per_trade_pct:
            return RiskDecision(
                False,
                f"Order risk {trade_risk_pct:.1f}% exceeds max {self.max_risk_per_trade_pct}% per trade"
            )

        # Hard cap on order size
        if order.cost_usd > self.max_order_size_usd:
            return RiskDecision(
                False,
                f"Order size ${order.cost_usd:.2f} exceeds hard cap ${self.max_order_size_usd}"
            )

        # 2. Total exposure check
        current_exposure = self.position_tracker.get_total_exposure_pct()
        new_exposure = current_exposure + (order.cost_usd / total_usd * 100)
        if new_exposure > self.max_total_exposure_pct:
            return RiskDecision(
                False,
                f"Total exposure would be {new_exposure:.1f}%, exceeds limit {self.max_total_exposure_pct}%"
            )

        # 3. Single pair exposure check
        pair_exposure = self.position_tracker.get_pair_exposure_pct(order.symbol)
        new_pair_exposure = pair_exposure + (order.cost_usd / total_usd * 100)
        if new_pair_exposure > self.max_single_pair_pct:
            return RiskDecision(
                False,
                f"{order.symbol} exposure would be {new_pair_exposure:.1f}%, exceeds limit {self.max_single_pair_pct}%"
            )

        # 4. Correlated exposure check
        correlated_exposure = self.position_tracker.get_correlated_exposure_pct(order.symbol)
        new_correlated = correlated_exposure + (order.cost_usd / total_usd * 100)
        if new_correlated > self.max_correlated_pct:
            return RiskDecision(
                False,
                f"Correlated exposure would be {new_correlated:.1f}%, exceeds limit {self.max_correlated_pct}%"
            )

        # 5. Circuit breaker check
        if self.circuit_breaker.is_active():
            status = self.circuit_breaker.get_status()
            return RiskDecision(
                False,
                f"Circuit breaker active: {status['level']} level"
            )

        # 6. Recovery mode — reduce size
        adjusted_amount = None
        if self.circuit_breaker.is_recovery_mode():
            multiplier = self.circuit_breaker.get_recovery_size_multiplier()
            adjusted_amount = order.amount * multiplier
            logger.info(
                f"Recovery mode: reducing {order.symbol} order from "
                f"{order.amount} to {adjusted_amount} ({multiplier:.0%} of normal)"
            )

        # 7. Cash reserve check
        free_after = balance["free_usd"] - order.cost_usd
        min_reserve = total_usd * (self.reserve_cash_pct / 100)
        if free_after < min_reserve:
            return RiskDecision(
                False,
                f"Would breach cash reserve: ${free_after:.2f} free < ${min_reserve:.2f} minimum"
            )

        # All checks passed
        return RiskDecision(
            True,
            "Order approved",
            adjusted_amount=adjusted_amount,
        )

    def run_circuit_breaker_check(self):
        """Periodic check — evaluate P&L against circuit breaker limits.

        Called by the scheduler every 60 seconds.
        Returns the circuit breaker status for alerting.
        """
        balance = self.position_tracker.get_balance()
        triggered, level = self.circuit_breaker.check(balance["total_usd"])

        if triggered:
            logger.critical(f"Circuit breaker triggered: {level}")

        return triggered, level

    def get_status(self) -> dict:
        """Get full risk status for reporting.

        Returns:
            Dict with exposure, circuit breaker, and balance info.
        """
        balance = self.position_tracker.get_balance()
        cb_status = self.circuit_breaker.get_status()

        return {
            "balance": balance,
            "circuit_breaker": cb_status,
            "daily_pnl": self.circuit_breaker.get_daily_pnl(),
            "weekly_pnl": self.circuit_breaker.get_weekly_pnl(),
            "monthly_pnl": self.circuit_breaker.get_monthly_pnl(),
        }
