import json
import threading
from datetime import datetime, timedelta, timezone

from src.core.database import Database
from src.core.logger import setup_logger

logger = setup_logger("risk.circuit_breaker")


class CircuitBreaker:
    """Monitors P&L against daily/weekly/monthly limits and triggers trading halts.

    Circuit breaker levels:
    - DAILY: Triggered at daily_loss_limit_pct. Auto-resumes after 24h at reduced sizing.
    - WEEKLY: Triggered at weekly_loss_limit_pct. Requires manual restart.
    - MONTHLY: Triggered at monthly_loss_limit_pct. Requires manual config change.

    Args:
        config: risk_management section from settings.yaml.
        db: Database instance.
    """

    def __init__(self, config: dict, db: Database):
        self.db = db
        self.daily_limit = config.get("daily_loss_limit_pct", 3.0)
        self.weekly_limit = config.get("weekly_loss_limit_pct", 7.0)
        self.monthly_limit = config.get("monthly_loss_limit_pct", 12.0)

        cb_config = config.get("after_circuit_breaker", {})
        self.recovery_reduction_pct = cb_config.get("reduce_position_sizes_by_pct", 50)
        self.recovery_period_hours = cb_config.get("recovery_period_hours", 48)

        # State
        self._active = False
        self._level: str | None = None  # "daily", "weekly", "monthly"
        self._activated_at: datetime | None = None
        self._resumes_at: datetime | None = None
        self._lock = threading.Lock()  # Protect concurrent check/activate

        # Check for active circuit breaker events in DB on startup
        self._load_state()

    def _load_state(self):
        """Load any active circuit breaker from the database."""
        event = self.db.fetch_one(
            "SELECT * FROM circuit_breaker_events WHERE resumed_at IS NULL ORDER BY timestamp DESC LIMIT 1"
        )
        if event:
            self._active = True
            self._level = event["trigger_type"]
            self._activated_at = datetime.fromisoformat(event["timestamp"])
            logger.warning(f"Active circuit breaker loaded from DB: {self._level} (triggered at {self._activated_at})")

    def get_daily_pnl(self) -> float:
        """Calculate today's realized P&L from trades table.

        Returns:
            Total P&L in USD for today.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl_usd), 0) as total_pnl FROM trades WHERE date(timestamp) = ? AND pnl_usd IS NOT NULL",
            (today,),
        )
        return result["total_pnl"] if result else 0.0

    def get_weekly_pnl(self) -> float:
        """Calculate this week's realized P&L (Monday through now).

        Returns:
            Total P&L in USD for the current week.
        """
        now = datetime.now(timezone.utc)
        monday = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        result = self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl_usd), 0) as total_pnl FROM trades WHERE date(timestamp) >= ? AND pnl_usd IS NOT NULL",
            (monday,),
        )
        return result["total_pnl"] if result else 0.0

    def get_monthly_pnl(self) -> float:
        """Calculate this month's realized P&L.

        Returns:
            Total P&L in USD for the current month.
        """
        month_start = datetime.now(timezone.utc).strftime("%Y-%m-01")
        result = self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl_usd), 0) as total_pnl FROM trades WHERE date(timestamp) >= ? AND pnl_usd IS NOT NULL",
            (month_start,),
        )
        return result["total_pnl"] if result else 0.0

    def check(self, portfolio_value: float) -> tuple[bool, str | None]:
        """Evaluate current P&L against circuit breaker limits.

        Thread-safe: uses lock to prevent multiple concurrent activations.

        Args:
            portfolio_value: Current total portfolio value in USD.

        Returns:
            Tuple of (triggered, level). level is None if not triggered.
        """
        if portfolio_value <= 0:
            return False, None

        with self._lock:
            # Already active — don't re-trigger
            if self._active:
                return False, None

            # Check monthly first (most severe)
            monthly_pnl = self.get_monthly_pnl()
            monthly_pct = (monthly_pnl / portfolio_value) * 100
            if monthly_pct <= -self.monthly_limit:
                self.activate("monthly", monthly_pnl)
                return True, "monthly"

            # Weekly
            weekly_pnl = self.get_weekly_pnl()
            weekly_pct = (weekly_pnl / portfolio_value) * 100
            if weekly_pct <= -self.weekly_limit:
                self.activate("weekly", weekly_pnl)
                return True, "weekly"

            # Daily
            daily_pnl = self.get_daily_pnl()
            daily_pct = (daily_pnl / portfolio_value) * 100
            if daily_pct <= -self.daily_limit:
                self.activate("daily", daily_pnl)
                return True, "daily"

            return False, None

    def activate(self, level: str, pnl_value: float):
        """Activate circuit breaker at the given level.

        Args:
            level: "daily", "weekly", or "monthly".
            pnl_value: The P&L that triggered the breaker.
        """
        if self._active and self._level == level:
            return  # Already active at this level

        now = datetime.now(timezone.utc)
        self._active = True
        self._level = level
        self._activated_at = now

        if level == "daily":
            self._resumes_at = now + timedelta(hours=24)
        else:
            self._resumes_at = None  # Manual restart required

        # Log to database
        self.db.execute(
            "INSERT INTO circuit_breaker_events (timestamp, trigger_type, trigger_value) VALUES (?, ?, ?)",
            (now.isoformat(), level, pnl_value),
        )

        logger.critical(
            f"CIRCUIT BREAKER ACTIVATED: {level.upper()} | P&L: ${pnl_value:.2f} | "
            f"Resumes: {self._resumes_at.isoformat() if self._resumes_at else 'MANUAL RESTART REQUIRED'}"
        )

    def check_auto_resume(self):
        """Check if a daily circuit breaker should auto-resume."""
        if not self._active or self._level != "daily" or not self._resumes_at:
            return

        now = datetime.now(timezone.utc)
        if now >= self._resumes_at:
            logger.info("Daily circuit breaker auto-resuming (entering recovery mode)")
            self.db.execute(
                "UPDATE circuit_breaker_events SET resumed_at = ? WHERE trigger_type = 'daily' AND resumed_at IS NULL",
                (now.isoformat(),),
            )
            self._active = False
            self._level = None
            # Recovery mode is handled by is_recovery_mode()

    def is_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        self.check_auto_resume()
        return self._active

    def is_recovery_mode(self) -> bool:
        """Check if we're in recovery mode (reduced sizing after daily circuit break).

        Recovery mode lasts for recovery_period_hours after a daily circuit breaker resumes.
        """
        if self._active:
            return False  # Still in full stop, not recovery

        # Check if there was a recent daily circuit breaker that resumed
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self.recovery_period_hours)).isoformat()
        event = self.db.fetch_one(
            "SELECT * FROM circuit_breaker_events WHERE trigger_type = 'daily' AND resumed_at IS NOT NULL AND resumed_at > ? ORDER BY resumed_at DESC LIMIT 1",
            (cutoff,),
        )
        return event is not None

    def get_recovery_size_multiplier(self) -> float:
        """Get the position size multiplier for recovery mode.

        Returns:
            1.0 if normal, 0.5 (or configured value) if in recovery mode.
        """
        if self.is_recovery_mode():
            return 1.0 - (self.recovery_reduction_pct / 100)
        return 1.0

    def get_status(self) -> dict:
        """Get current circuit breaker status.

        Returns:
            Dict with active, level, activated_at, resumes_at, recovery_mode.
        """
        return {
            "active": self.is_active(),
            "level": self._level if self._active else None,
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
            "resumes_at": self._resumes_at.isoformat() if self._resumes_at else None,
            "recovery_mode": self.is_recovery_mode(),
            "size_multiplier": self.get_recovery_size_multiplier(),
        }
