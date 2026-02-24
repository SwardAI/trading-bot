import asyncio
import os
from enum import Enum

from src.core.logger import setup_logger

logger = setup_logger("alerts.telegram")


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Prefixes for alert levels
LEVEL_PREFIX = {
    AlertLevel.INFO: "\U0001f4ca",      # 📊
    AlertLevel.WARNING: "\u26a0\ufe0f",  # ⚠️
    AlertLevel.CRITICAL: "\U0001f6a8",   # 🚨
}


class TelegramAlerter:
    """Sends alerts and reports to Telegram.

    Uses python-telegram-bot for async message delivery.
    Falls back to logging if Telegram is not configured.

    Args:
        config: alerts.telegram section from settings.yaml.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.bot_token = os.getenv(config.get("bot_token_env", ""))
        self.chat_id = os.getenv(config.get("chat_id_env", ""))
        self.daily_report_time = config.get("daily_report_time", "08:00")

        if self.enabled and self.bot_token and self.chat_id:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
                logger.info("Telegram alerter initialized")
            except Exception as e:
                logger.warning(f"Telegram init failed, alerts will only be logged: {e}")
                self._bot = None
        else:
            self._bot = None
            if self.enabled:
                logger.warning("Telegram enabled but missing token/chat_id, alerts will only be logged")

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO):
        """Send an alert message.

        Args:
            message: Alert text.
            level: Alert severity level.
        """
        prefix = LEVEL_PREFIX.get(level, "")
        full_message = f"{prefix} {message}"

        # Always log
        if level == AlertLevel.CRITICAL:
            logger.critical(f"TELEGRAM ALERT: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"TELEGRAM ALERT: {message}")
        else:
            logger.info(f"TELEGRAM ALERT: {message}")

        # Send to Telegram — always create a fresh event loop since APScheduler
        # runs callbacks in background threads which have no default event loop.
        # Using asyncio.get_event_loop() is deprecated in Python 3.10+ and
        # raises RuntimeError in 3.12+ when called from non-main threads.
        if self._bot and self.chat_id:
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        self._bot.send_message(
                            chat_id=self.chat_id,
                            text=full_message,
                            parse_mode="HTML",
                        )
                    )
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")

    def send_daily_report(self, report: str):
        """Send the daily performance report.

        Args:
            report: Formatted report string.
        """
        self.send(report, AlertLevel.INFO)

    def send_trade_alert(self, strategy: str, side: str, symbol: str, price: float, pnl: float | None = None):
        """Send a trade execution alert.

        Args:
            strategy: Strategy name.
            side: "buy" or "sell".
            symbol: Trading pair.
            price: Execution price.
            pnl: P&L if closing trade.
        """
        msg = f"<b>{strategy.upper()}</b>: {side.upper()} {symbol} @ ${price:,.2f}"
        if pnl is not None:
            emoji = "\u2705" if pnl >= 0 else "\u274c"  # ✅ or ❌
            msg += f" | P&L: {emoji} ${pnl:,.2f}"
        self.send(msg, AlertLevel.INFO)

    def send_circuit_breaker_alert(self, level: str, pnl: float):
        """Send circuit breaker activation alert.

        Args:
            level: Circuit breaker level (daily, weekly, monthly).
            pnl: P&L that triggered it.
        """
        msg = (
            f"<b>CIRCUIT BREAKER ACTIVATED</b>\n"
            f"Level: {level.upper()}\n"
            f"P&L: ${pnl:,.2f}\n"
            f"Action: {'Auto-resume in 24h' if level == 'daily' else 'MANUAL RESTART REQUIRED'}"
        )
        self.send(msg, AlertLevel.CRITICAL)

    def send_error_alert(self, error: str):
        """Send an error/warning alert.

        Args:
            error: Error description.
        """
        self.send(f"<b>Error:</b> {error}", AlertLevel.WARNING)
