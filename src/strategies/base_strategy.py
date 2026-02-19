from abc import ABC, abstractmethod

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.core.logger import setup_logger


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    All strategies must inherit from this class and implement
    start(), stop(), and on_tick() methods.

    Args:
        name: Strategy identifier (e.g. "grid", "momentum").
        config: Strategy-specific config dict.
        exchange: ExchangeManager instance.
        db: Database instance.
        risk_manager: RiskManager instance (injected to avoid circular imports).
    """

    def __init__(self, name: str, config: dict, exchange: ExchangeManager, db: Database, risk_manager):
        self.strategy_name = name
        self.config = config
        self.exchange = exchange
        self.db = db
        self.risk_manager = risk_manager
        self.logger = setup_logger(f"strategy.{name}")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def is_enabled(self) -> bool:
        """Check if this strategy is enabled in config."""
        return self.config.get("enabled", False)

    @abstractmethod
    def start(self):
        """Initialize and begin strategy execution (place initial orders, etc)."""
        pass

    @abstractmethod
    def stop(self):
        """Gracefully stop — cancel open orders, save state."""
        pass

    @abstractmethod
    def on_tick(self):
        """Called on each check interval — check fills, signals, rebalance, etc."""
        pass
