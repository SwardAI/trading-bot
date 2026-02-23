import time
from datetime import datetime, timezone

import ccxt

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.core.logger import setup_logger

logger = setup_logger("execution.order_manager")

# Smart execution timing
LIMIT_WAIT_SECONDS = 10
ADJUST_WAIT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds, doubles each retry


class OrderManager:
    """Smart order execution with retry logic, partial fills, and reconciliation.

    Execution flow:
    1. Check orderbook depth
    2. Place as limit order (maker fee)
    3. Wait for fill, adjust price if needed
    4. Fall back to market order if still unfilled
    5. Log everything to database

    Args:
        exchange: ExchangeManager instance.
        db: Database instance.
    """

    def __init__(self, exchange: ExchangeManager, db: Database):
        self.exchange = exchange
        self.db = db

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        strategy: str,
        linked_trade_id: int | None = None,
    ) -> dict | None:
        """Place an order with smart execution logic.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT").
            side: "buy" or "sell".
            amount: Amount in base currency.
            price: Target limit price.
            strategy: Strategy name for logging ("grid", "momentum").
            linked_trade_id: Optional ID of related trade (for grid round-trips).

        Returns:
            Trade record dict if successful, None if failed.
        """
        expected_price = price
        cost_usd = amount * price

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Step 1: Place limit order
                logger.info(f"Placing {side} limit {amount} {symbol} @ {price} (attempt {attempt})")
                order = self.exchange.create_order(symbol, "limit", side, amount, price)
                order_id = order["id"]

                # Step 2: Wait for fill
                filled_order = self._wait_for_fill(order_id, symbol, LIMIT_WAIT_SECONDS)

                if filled_order and filled_order.get("status") == "closed":
                    return self._log_trade(filled_order, strategy, expected_price, linked_trade_id)

                # Step 3: Adjust price toward market
                if filled_order and filled_order.get("status") == "open":
                    adjusted_price = self._get_adjusted_price(symbol, side, price)
                    if adjusted_price != price:
                        logger.info(f"Adjusting {order_id} price: {price} -> {adjusted_price}")
                        self.exchange.cancel_order(order_id, symbol)
                        order = self.exchange.create_order(symbol, "limit", side, amount, adjusted_price)
                        order_id = order["id"]

                    # Wait again
                    filled_order = self._wait_for_fill(order_id, symbol, ADJUST_WAIT_SECONDS)

                    if filled_order and filled_order.get("status") == "closed":
                        return self._log_trade(filled_order, strategy, expected_price, linked_trade_id)

                # Step 4: Fall back to market order
                if filled_order and filled_order.get("status") == "open":
                    remaining = amount - filled_order.get("filled", 0)
                    if remaining > 0:
                        logger.warning(f"Limit not filled, converting to market: {remaining} {symbol}")
                        self.exchange.cancel_order(order_id, symbol)

                        # Log partial fill if any
                        if filled_order.get("filled", 0) > 0:
                            self._log_trade(filled_order, strategy, expected_price, linked_trade_id)

                        market_order = self.exchange.create_order(symbol, "market", side, remaining)
                        market_filled = self._wait_for_fill(market_order["id"], symbol, 10)
                        if market_filled:
                            return self._log_trade(market_filled, strategy, expected_price, linked_trade_id)

                return None

            except ccxt.InsufficientFunds as e:
                logger.error(f"Insufficient funds for {side} {amount} {symbol}: {e}")
                return None

            except ccxt.BaseError as e:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Exchange error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {delay}s")
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                else:
                    logger.error(f"Order failed after {MAX_RETRIES} attempts: {side} {amount} {symbol}")
                    return None

        return None

    def _wait_for_fill(self, order_id: str, symbol: str, timeout: int) -> dict | None:
        """Poll exchange for order status until filled or timeout.

        Args:
            order_id: Exchange order ID.
            symbol: Trading pair.
            timeout: Max seconds to wait.

        Returns:
            Order status dict, or None on error.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                order = self.exchange.exchange.fetch_order(order_id, symbol)
                if order["status"] in ("closed", "canceled", "cancelled"):
                    return order
                time.sleep(1)
            except ccxt.BaseError as e:
                logger.warning(f"Error polling order {order_id}: {e}")
                time.sleep(2)
        # Return last known state
        try:
            return self.exchange.exchange.fetch_order(order_id, symbol)
        except ccxt.BaseError:
            return None

    def _get_adjusted_price(self, symbol: str, side: str, current_price: float) -> float:
        """Calculate adjusted price one tick toward market.

        Args:
            symbol: Trading pair.
            side: "buy" or "sell".
            current_price: Current limit price.

        Returns:
            Adjusted price closer to market.
        """
        try:
            ob = self.exchange.fetch_orderbook(symbol, limit=5)
            if side == "buy":
                best_ask = ob["asks"][0][0] if ob.get("asks") else current_price
                # Move buy price up toward ask (but not above it)
                tick = current_price * 0.001  # 0.1% tick
                return min(current_price + tick, best_ask)
            else:
                best_bid = ob["bids"][0][0] if ob.get("bids") else current_price
                tick = current_price * 0.001
                return max(current_price - tick, best_bid)
        except ccxt.BaseError:
            return current_price

    def _log_trade(
        self,
        order: dict,
        strategy: str,
        expected_price: float,
        linked_trade_id: int | None = None,
    ) -> dict:
        """Log a completed trade to the database.

        Args:
            order: Exchange order response dict.
            strategy: Strategy name.
            expected_price: The intended price for slippage calculation.
            linked_trade_id: Related trade ID for round-trip pairing.

        Returns:
            The trade record as a dict.
        """
        fill_price = order.get("average") or order.get("price", 0)
        amount = order.get("filled", order.get("amount", 0))
        cost = order.get("cost", fill_price * amount)

        fee_info = order.get("fee") or {}
        fee_usd = fee_info.get("cost", 0) or 0

        slippage_pct = 0.0
        if expected_price and fill_price:
            slippage_pct = abs(fill_price - expected_price) / expected_price * 100

        now = datetime.now(timezone.utc).isoformat()

        # Check if this order was already logged (prevent duplicates from retries/reconciliation)
        exchange_order_id = order.get("id")
        if exchange_order_id:
            existing = self.db.fetch_one(
                "SELECT id FROM trades WHERE exchange_order_id = ?",
                (exchange_order_id,),
            )
            if existing:
                logger.debug(f"Trade already logged for order {exchange_order_id}, skipping duplicate")
                return {
                    "id": existing["id"],
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "price": fill_price,
                    "amount": amount,
                    "cost_usd": cost,
                    "fee_usd": fee_usd,
                    "slippage_pct": slippage_pct,
                    "exchange_order_id": exchange_order_id,
                }

        cursor = self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd,
             fee_usd, fee_currency, exchange_order_id, slippage_pct, linked_trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                strategy,
                order.get("symbol", ""),
                order.get("side", ""),
                order.get("type", ""),
                fill_price,
                amount,
                cost,
                fee_usd,
                fee_info.get("currency"),
                exchange_order_id,
                slippage_pct,
                linked_trade_id,
            ),
        )

        trade_id = cursor.lastrowid

        logger.info(
            f"Trade logged: #{trade_id} {order.get('side')} {amount} {order.get('symbol')} "
            f"@ {fill_price} (slippage: {slippage_pct:.3f}%, fee: ${fee_usd:.4f})"
        )

        return {
            "id": trade_id,
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "price": fill_price,
            "amount": amount,
            "cost_usd": cost,
            "fee_usd": fee_usd,
            "slippage_pct": slippage_pct,
            "exchange_order_id": order.get("id"),
        }

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a single order with error handling.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except ccxt.OrderNotFound:
            logger.debug(f"Order {order_id} already cancelled or filled")
            return True
        except ccxt.BaseError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol.

        Returns:
            Number of orders cancelled.
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            cancelled = 0
            for order in orders:
                if self.cancel_order(order["id"], symbol):
                    cancelled += 1
            logger.info(f"Cancelled {cancelled}/{len(orders)} orders for {symbol}")
            return cancelled
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch open orders for {symbol}: {e}")
            return 0

    def reconcile(self, symbol: str):
        """Compare local order state with exchange state. Exchange is source of truth.

        Args:
            symbol: Trading pair to reconcile.
        """
        try:
            exchange_orders = self.exchange.fetch_open_orders(symbol)
            exchange_ids = {o["id"] for o in exchange_orders}

            logger.debug(f"Reconciliation for {symbol}: {len(exchange_orders)} open orders on exchange")

            # Check for fills we might have missed
            recent_trades = self.exchange.fetch_my_trades(symbol, limit=20)
            for trade in recent_trades:
                existing = self.db.fetch_one(
                    "SELECT id FROM trades WHERE exchange_order_id = ?",
                    (trade.get("order"),),
                )
                if not existing and trade.get("order"):
                    logger.warning(f"Reconciliation: found untracked trade {trade.get('order')} for {symbol}")
                    self._log_trade(
                        {
                            "id": trade.get("order"),
                            "symbol": symbol,
                            "side": trade.get("side"),
                            "type": trade.get("type") or "limit",
                            "average": trade.get("price"),
                            "filled": trade.get("amount"),
                            "cost": trade.get("cost"),
                            "fee": trade.get("fee"),
                        },
                        strategy="reconciled",
                        expected_price=trade.get("price", 0),
                    )

        except ccxt.BaseError as e:
            logger.error(f"Reconciliation failed for {symbol}: {e}")
