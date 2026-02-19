from datetime import datetime, timezone

from src.core.database import Database
from src.core.logger import setup_logger

logger = setup_logger("journal.trade_logger")


class TradeLogger:
    """Logs trade events with context for the trading journal.

    Provides higher-level trade logging on top of the raw trades table,
    adding context like signals, market conditions, and notes.

    Args:
        db: Database instance.
    """

    def __init__(self, db: Database):
        self.db = db

    def log_trade(
        self,
        strategy: str,
        pair: str,
        side: str,
        order_type: str,
        price: float,
        amount: float,
        cost_usd: float,
        fee_usd: float,
        exchange_order_id: str | None = None,
        slippage_pct: float = 0.0,
        linked_trade_id: int | None = None,
        pnl_usd: float | None = None,
        notes: str | None = None,
    ) -> int:
        """Log a trade to the database.

        Args:
            strategy: Strategy that placed the trade.
            pair: Trading pair.
            side: "buy" or "sell".
            order_type: "limit" or "market".
            price: Execution price.
            amount: Amount in base currency.
            cost_usd: Total cost in USD.
            fee_usd: Fee paid in USD.
            exchange_order_id: Exchange's order ID.
            slippage_pct: Slippage percentage.
            linked_trade_id: Related trade ID.
            pnl_usd: P&L if closing a position.
            notes: Additional context.

        Returns:
            Trade ID from the database.
        """
        now = datetime.now(timezone.utc).isoformat()

        cursor = self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd,
             fee_usd, exchange_order_id, slippage_pct, linked_trade_id, pnl_usd, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now, strategy, pair, side, order_type, price, amount, cost_usd,
                fee_usd, exchange_order_id, slippage_pct, linked_trade_id, pnl_usd, notes,
            ),
        )

        trade_id = cursor.lastrowid
        logger.info(
            f"Trade #{trade_id}: {strategy} {side} {amount:.6f} {pair} "
            f"@ {price:.2f} (${cost_usd:.2f})"
        )
        return trade_id

    def get_recent_trades(self, limit: int = 20, strategy: str | None = None) -> list[dict]:
        """Get recent trades from the database.

        Args:
            limit: Max number of trades.
            strategy: Filter by strategy name.

        Returns:
            List of trade dicts.
        """
        if strategy:
            return self.db.fetch_all(
                "SELECT * FROM trades WHERE strategy = ? ORDER BY timestamp DESC LIMIT ?",
                (strategy, limit),
            )
        return self.db.fetch_all(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def get_open_grid_round_trips(self, pair: str) -> int:
        """Count grid buys without matching sells (open inventory).

        Args:
            pair: Trading pair.

        Returns:
            Count of unmatched grid buys.
        """
        result = self.db.fetch_one(
            """SELECT COUNT(*) as count FROM trades
            WHERE strategy = 'grid' AND pair = ? AND side = 'buy'
            AND id NOT IN (SELECT COALESCE(linked_trade_id, 0) FROM trades WHERE strategy = 'grid' AND side = 'sell')""",
            (pair,),
        )
        return result["count"] if result else 0
