import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.core.logger import setup_logger

logger = setup_logger(__name__)

SCHEMA_SQL = """
-- Account snapshots (taken every hour)
CREATE TABLE IF NOT EXISTS account_snapshots (
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
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    strategy TEXT NOT NULL,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL NOT NULL,
    amount REAL NOT NULL,
    cost_usd REAL NOT NULL,
    fee_usd REAL NOT NULL,
    fee_currency TEXT,
    exchange_order_id TEXT,
    slippage_pct REAL,
    linked_trade_id INTEGER,
    pnl_usd REAL,
    notes TEXT
);

-- Strategy-level metrics (aggregated daily)
CREATE TABLE IF NOT EXISTS daily_metrics (
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
CREATE TABLE IF NOT EXISTS grid_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    grid_center REAL NOT NULL,
    grid_levels TEXT NOT NULL,
    inventory_amount REAL DEFAULT 0,
    inventory_avg_price REAL,
    total_round_trips INTEGER DEFAULT 0,
    total_profit_usd REAL DEFAULT 0,
    last_rebalance DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Circuit breaker events
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    trigger_type TEXT NOT NULL,
    trigger_value REAL NOT NULL,
    positions_closed TEXT,
    resumed_at DATETIME
);

-- Momentum position tracking
CREATE TABLE IF NOT EXISTS momentum_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_time DATETIME NOT NULL,
    amount REAL NOT NULL,
    stop_loss REAL NOT NULL,
    current_stop REAL NOT NULL,
    entry_signals TEXT,
    exit_price REAL,
    exit_time DATETIME,
    exit_reason TEXT,
    pnl_usd REAL,
    pnl_pct REAL,
    status TEXT DEFAULT 'open'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON account_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_momentum_status ON momentum_positions(status);
"""


class Database:
    """SQLite database wrapper with thread-safe access.

    Usage:
        db = Database("data/bot.db")
        db.init_db()
        db.execute("INSERT INTO trades (...) VALUES (...)", params)
        rows = db.fetch_all("SELECT * FROM trades WHERE strategy = ?", ("grid",))
    """

    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        logger.info(f"Database initialized at {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, timeout=30)
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection.execute("PRAGMA foreign_keys=ON")
        return self._local.connection

    def init_db(self):
        """Create all tables and indexes if they don't exist."""
        conn = self._get_connection()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Database schema initialized")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            sqlite3.Cursor from the execution.
        """
        conn = self._get_connection()
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor

    def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute a query and return the first row as a dict.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            Dict of the first row, or None if no results.
        """
        conn = self._get_connection()
        cursor = conn.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return all rows as a list of dicts.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            List of row dicts.
        """
        conn = self._get_connection()
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions.

        Groups multiple operations into a single atomic commit.
        Rolls back on error.

        Usage:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...", (...))
                conn.execute("UPDATE ...", (...))
        """
        conn = self._get_connection()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        """Close the thread-local database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.info("Database connection closed")
