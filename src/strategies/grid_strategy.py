import json
from datetime import datetime, timezone

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.data.market_data import MarketDataManager
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import OrderRequest, RiskManager
from src.strategies.base_strategy import BaseStrategy


class GridStrategy(BaseStrategy):
    """Grid trading strategy — places buy/sell orders at regular intervals around price.

    Profits from price oscillation: buys low, sells high within the grid range.
    Each buy fill triggers a sell order above it, and vice versa.

    One instance per trading pair.

    Args:
        pair_config: Per-pair config dict from grid_config.yaml.
        global_config: Global grid settings (max_open_orders, fee_rate, etc).
        exchange: ExchangeManager instance.
        db: Database instance.
        risk_manager: RiskManager instance.
        order_manager: OrderManager instance.
        market_data: MarketDataManager instance.
    """

    def __init__(
        self,
        pair_config: dict,
        global_config: dict,
        exchange: ExchangeManager,
        db: Database,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        market_data: MarketDataManager,
    ):
        super().__init__("grid", pair_config, exchange, db, risk_manager)
        self.symbol = pair_config["symbol"]
        self.pair_config = pair_config
        self.global_config = global_config
        self.order_manager = order_manager
        self.market_data = market_data

        # Grid parameters
        self.grid_type = pair_config.get("grid_type", "geometric")
        self.num_grids = pair_config.get("num_grids", 20)
        self.grid_spacing_pct = pair_config.get("grid_spacing_pct", 0.5)
        self.order_size_usd = pair_config.get("order_size_usd", 50)
        self.upper_bound_pct = pair_config.get("upper_bound_pct", 10)
        self.lower_bound_pct = pair_config.get("lower_bound_pct", 10)
        self.rebalance_trigger_pct = pair_config.get("rebalance_trigger_pct", 8)
        self.stop_loss_pct = pair_config.get("stop_loss_pct", 15)
        self.take_profit_pct = pair_config.get("take_profit_pct")

        # Global limits
        self.max_open_orders = global_config.get("max_open_orders", 40)
        self.min_profit_after_fees = global_config.get("min_profit_after_fees", 0.15)
        self.fee_rate = global_config.get("fee_rate", 0.075)

        # State
        self.grid_center: float = 0.0
        self.grid_levels: list[dict] = []  # [{price, side, status, order_id}]
        self.inventory_amount: float = 0.0
        self.inventory_avg_price: float = 0.0
        self.total_round_trips: int = 0
        self.total_profit_usd: float = 0.0

        self.logger.info(f"GridStrategy initialized for {self.symbol} ({self.grid_type}, {self.num_grids} levels)")

    def start(self):
        """Initialize grid: load saved state or calculate fresh grid from current price."""
        self._running = True

        # Try to load saved state
        if self._load_state():
            # Reconcile saved state with actual exchange holdings
            self._reconcile_inventory()

            # Check if saved grid is stale (price moved significantly since last run)
            try:
                ticker = self.market_data.get_ticker(self.symbol)
                current_price = ticker["last"]
                drift_pct = abs(current_price - self.grid_center) / self.grid_center * 100

                if drift_pct >= self.rebalance_trigger_pct:
                    self.logger.warning(
                        f"Stale grid detected: price drifted {drift_pct:.1f}% from center {self.grid_center}. "
                        f"Rebalancing to {current_price}."
                    )
                    self.grid_center = current_price
                    self.grid_levels = self.calculate_grid_levels(current_price)
                else:
                    self.logger.info(f"Resumed grid for {self.symbol} from saved state (center: {self.grid_center})")
            except Exception as e:
                self.logger.error(f"Failed to check grid staleness: {e}, using saved state")
        else:
            # Fresh grid from current price
            ticker = self.market_data.get_ticker(self.symbol)
            current_price = ticker["last"]
            self.grid_center = current_price
            self.grid_levels = self.calculate_grid_levels(current_price)
            self.logger.info(f"New grid for {self.symbol}: center={current_price}, {len(self.grid_levels)} levels")

        self._place_grid_orders()
        self._save_state()

    def _reconcile_inventory(self):
        """Sync saved grid inventory with actual exchange holdings on startup."""
        try:
            balance = self.exchange.fetch_balance()
            base_asset = self.symbol.split("/")[0]
            actual_free = float(balance.get("free", {}).get(base_asset, 0))
            actual_total = float(balance.get("total", {}).get(base_asset, 0))

            if abs(self.inventory_amount - actual_total) > 1e-8:
                self.logger.warning(
                    f"Inventory mismatch on restart for {self.symbol}: "
                    f"saved={self.inventory_amount:.8f}, exchange={actual_total:.8f} {base_asset}. "
                    f"Adjusting to exchange holdings."
                )
                self.inventory_amount = actual_total

            # Remove pending sell orders if we don't have enough inventory to cover them
            pending_sell_amount = sum(
                self.order_size_usd / level["price"]
                for level in self.grid_levels
                if level["side"] == "sell" and level["status"] == "pending"
            )
            if pending_sell_amount > actual_free and actual_free < 1e-8:
                removed = sum(1 for l in self.grid_levels if l["side"] == "sell" and l["status"] == "pending")
                self.grid_levels = [
                    l for l in self.grid_levels
                    if not (l["side"] == "sell" and l["status"] == "pending")
                ]
                self.logger.warning(
                    f"Removed {removed} pending sell orders: insufficient {base_asset} holdings "
                    f"(have {actual_free:.8f}, need {pending_sell_amount:.8f})"
                )
        except Exception as e:
            self.logger.error(f"Failed to reconcile inventory with exchange: {e}")

    def stop(self):
        """Cancel all grid orders and save state."""
        self._running = False
        self.logger.info(f"Stopping grid for {self.symbol}...")
        self.order_manager.cancel_all_orders(self.symbol)
        self._save_state()
        self.logger.info(f"Grid stopped for {self.symbol}")

    def on_tick(self):
        """Called every grid_check_interval — check fills, place pending orders, rebalance, stop loss."""
        if not self._running:
            return

        self._check_fills()

        # Place any pending orders (e.g. from restart reconciliation or cancelled orders)
        pending_count = sum(1 for l in self.grid_levels if l["status"] == "pending")
        if pending_count > 0:
            self._place_grid_orders()
            self._save_state()

        self._check_rebalance()
        self._check_stop_loss()

    # --- Grid calculation ---

    def calculate_grid_levels(self, center_price: float) -> list[dict]:
        """Calculate grid price levels around a center price.

        Supports arithmetic (fixed $) and geometric (fixed %) spacing.

        Args:
            center_price: The center price for the grid.

        Returns:
            List of grid level dicts: {price, side, status, order_id}.
        """
        levels = []

        for i in range(1, self.num_grids + 1):
            if self.grid_type == "geometric":
                # Percentage-based spacing
                multiplier = (1 + self.grid_spacing_pct / 100) ** i
                buy_price = center_price / multiplier
                sell_price = center_price * multiplier
            else:
                # Arithmetic (fixed dollar) spacing
                spacing = center_price * (self.grid_spacing_pct / 100)
                buy_price = center_price - (spacing * i)
                sell_price = center_price + (spacing * i)

            # Check bounds
            lower_bound = center_price * (1 - self.lower_bound_pct / 100)
            upper_bound = center_price * (1 + self.upper_bound_pct / 100)

            if buy_price >= lower_bound:
                levels.append({
                    "price": round(buy_price, 2),
                    "side": "buy",
                    "status": "pending",  # pending, open, filled
                    "order_id": None,
                })

            if sell_price <= upper_bound:
                levels.append({
                    "price": round(sell_price, 2),
                    "side": "sell",
                    "status": "pending",
                    "order_id": None,
                })

        # Sort: buys descending (highest buy first), sells ascending
        levels.sort(key=lambda x: (-x["price"] if x["side"] == "buy" else x["price"]))

        # Check minimum profitability
        min_spacing = (2 * self.fee_rate) + self.min_profit_after_fees
        if self.grid_spacing_pct < min_spacing:
            self.logger.warning(
                f"Grid spacing {self.grid_spacing_pct}% is below minimum profitable "
                f"spacing {min_spacing}% (2×fee + min_profit)"
            )

        return levels

    # --- Order placement ---

    def _count_exchange_open_orders(self) -> int:
        """Count currently open orders on the exchange for this symbol."""
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            return len(open_orders)
        except Exception:
            return self.max_open_orders  # assume full on error to be safe

    def _place_grid_orders(self):
        """Place all pending grid orders, each through the risk manager."""
        placed = 0
        skipped_sells = 0
        current_open = self._count_exchange_open_orders()
        max_to_place = max(0, self.max_open_orders - current_open)

        for level in self.grid_levels:
            if level["status"] != "pending":
                continue

            # Skip sell orders when we have no inventory to sell
            if level["side"] == "sell" and self.inventory_amount <= 0:
                skipped_sells += 1
                continue

            if placed >= max_to_place:
                self.logger.warning(f"Open order limit reached ({current_open + placed}/{self.max_open_orders}), skipping remaining")
                break

            amount = self.order_size_usd / level["price"]
            cost_usd = self.order_size_usd

            # Check exchange minimum order size
            try:
                market = self.exchange.exchange.markets.get(self.symbol, {})
                min_amount = market.get("limits", {}).get("amount", {}).get("min", 0) or 0
                if amount < min_amount:
                    self.logger.warning(
                        f"Order amount {amount:.8f} below exchange minimum {min_amount} for {self.symbol}, skipping"
                    )
                    continue
            except Exception:
                pass  # If we can't check, proceed anyway

            # Risk check
            order_request = OrderRequest(
                symbol=self.symbol,
                side=level["side"],
                amount=amount,
                price=level["price"],
                cost_usd=cost_usd,
                strategy="grid",
            )
            decision = self.risk_manager.check(order_request)

            if not decision.approved:
                self.logger.debug(f"Risk rejected grid order {level['side']} @ {level['price']}: {decision.reason}")
                continue

            # Use adjusted amount if in recovery mode
            if decision.adjusted_amount:
                amount = decision.adjusted_amount

            # Place the order
            try:
                order = self.exchange.create_order(
                    self.symbol, "limit", level["side"], amount, level["price"]
                )
                level["status"] = "open"
                level["order_id"] = order["id"]
                placed += 1
                self.logger.debug(f"Grid order placed: {level['side']} {amount:.6f} @ {level['price']}")
            except Exception as e:
                self.logger.error(f"Failed to place grid order {level['side']} @ {level['price']}: {e}")

        msg = f"Placed {placed} grid orders for {self.symbol}"
        if skipped_sells:
            msg += f" (skipped {skipped_sells} sells, no inventory)"
        self.logger.info(msg)

    # --- Fill detection ---

    def _check_fills(self):
        """Check for filled grid orders by verifying each order's status directly."""
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            exchange_open_ids = {o["id"] for o in open_orders}
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders for fill check: {e}")
            return

        for level in self.grid_levels:
            if level["status"] != "open" or not level["order_id"]:
                continue

            # If order is still on exchange, skip
            if level["order_id"] in exchange_open_ids:
                continue

            # Order disappeared — verify it was actually filled (not cancelled/expired)
            try:
                order = self.exchange.exchange.fetch_order(level["order_id"], self.symbol)
                if order["status"] == "closed":
                    self._handle_fill(level)
                elif order["status"] in ("canceled", "expired", "rejected"):
                    if order.get("filled", 0) > 0:
                        self._handle_fill(level)
                    else:
                        self.logger.info(f"Order {level['order_id']} {order['status']}, resetting to pending")
                        level["status"] = "pending"
                        level["order_id"] = None
            except Exception as e:
                self.logger.error(f"Failed to verify order {level['order_id']}: {e}, skipping")

    def _handle_fill(self, level: dict):
        """Handle a filled grid order — create opposite order and update P&L.

        When a buy fills: place sell above it (grid_spacing above).
        When a sell fills: place buy below it (grid_spacing below).
        """
        fill_side = level["side"]
        fill_price = level["price"]
        amount = self.order_size_usd / fill_price

        self.logger.info(f"Grid fill detected: {fill_side} @ {fill_price} ({self.symbol})")

        # Log the fill and update inventory atomically
        now = datetime.now(timezone.utc).isoformat()
        fee_usd = self.order_size_usd * (self.fee_rate / 100)

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                (timestamp, strategy, pair, side, order_type, price, amount, cost_usd,
                 fee_usd, exchange_order_id)
                VALUES (?, 'grid', ?, ?, 'limit', ?, ?, ?, ?, ?)""",
                (now, self.symbol, fill_side, fill_price, amount, self.order_size_usd, fee_usd, level["order_id"]),
            )
            fill_trade_id = cursor.lastrowid

        # Update inventory
        if fill_side == "buy":
            self._update_inventory(amount, fill_price)
        else:
            self._update_inventory(-amount, fill_price)

        # Mark level as filled
        level["status"] = "filled"
        level["order_id"] = None

        # Place opposite order for round-trip
        if fill_side == "buy":
            # Place sell above
            if self.grid_type == "geometric":
                opposite_price = round(fill_price * (1 + self.grid_spacing_pct / 100), 2)
            else:
                spacing = self.grid_center * (self.grid_spacing_pct / 100)
                opposite_price = round(fill_price + spacing, 2)
            opposite_side = "sell"
        else:
            # Place buy below
            if self.grid_type == "geometric":
                opposite_price = round(fill_price / (1 + self.grid_spacing_pct / 100), 2)
            else:
                spacing = self.grid_center * (self.grid_spacing_pct / 100)
                opposite_price = round(fill_price - spacing, 2)
            opposite_side = "buy"

        # Calculate P&L for completed round-trip (sell fills)
        pnl_usd = None
        if fill_side == "sell" and self.inventory_avg_price > 0:
            pnl_usd = (fill_price - self.inventory_avg_price) * amount - (2 * fee_usd)
            self.total_profit_usd += pnl_usd
            self.total_round_trips += 1
            # Update the trade record with P&L
            self.db.execute(
                "UPDATE trades SET pnl_usd = ?, linked_trade_id = ? WHERE id = ?",
                (pnl_usd, fill_trade_id, fill_trade_id),
            )
            self.logger.info(f"Round-trip completed: P&L ${pnl_usd:.2f} (total: ${self.total_profit_usd:.2f})")

        # Risk check for opposite order
        opp_amount = self.order_size_usd / opposite_price
        order_request = OrderRequest(
            symbol=self.symbol,
            side=opposite_side,
            amount=opp_amount,
            price=opposite_price,
            cost_usd=self.order_size_usd,
            strategy="grid",
        )
        decision = self.risk_manager.check(order_request)

        if decision.approved:
            # Check exchange order limit before placing
            current_open = self._count_exchange_open_orders()
            if current_open >= self.max_open_orders:
                self.logger.warning(f"Skipping opposite order: at exchange limit ({current_open}/{self.max_open_orders})")
            else:
                if decision.adjusted_amount:
                    opp_amount = decision.adjusted_amount
                try:
                    order = self.exchange.create_order(
                        self.symbol, "limit", opposite_side, opp_amount, opposite_price
                    )
                    # Add new level to grid
                    self.grid_levels.append({
                        "price": opposite_price,
                        "side": opposite_side,
                        "status": "open",
                        "order_id": order["id"],
                    })
                    self.logger.debug(f"Opposite order placed: {opposite_side} @ {opposite_price}")
                except Exception as e:
                    self.logger.error(f"Failed to place opposite order: {e}")
        else:
            self.logger.warning(f"Risk rejected opposite order: {decision.reason}")

        # Clean up filled levels to prevent unbounded list growth
        self.grid_levels = [l for l in self.grid_levels if l["status"] in ("open", "pending")]

        self._save_state()

    def _update_inventory(self, amount_delta: float, price: float):
        """Update inventory tracking after a fill.

        Args:
            amount_delta: Positive for buys, negative for sells.
            price: Fill price.
        """
        if amount_delta > 0:  # Buy
            total_cost = (self.inventory_amount * self.inventory_avg_price) + (amount_delta * price)
            self.inventory_amount += amount_delta
            self.inventory_avg_price = total_cost / self.inventory_amount if self.inventory_amount > 0 else 0
        else:  # Sell
            self.inventory_amount += amount_delta  # amount_delta is negative
            if self.inventory_amount <= 0:
                self.inventory_amount = 0
                self.inventory_avg_price = 0

    # --- Rebalancing ---

    def _check_rebalance(self):
        """Check if price has drifted enough to trigger a grid rebalance."""
        if self.grid_center <= 0:
            return

        try:
            ticker = self.market_data.get_ticker(self.symbol)
            current_price = ticker["last"]
        except Exception as e:
            self.logger.error(f"Failed to get ticker for rebalance check: {e}")
            return

        drift_pct = abs(current_price - self.grid_center) / self.grid_center * 100

        if drift_pct >= self.rebalance_trigger_pct:
            self.logger.info(
                f"Rebalance triggered for {self.symbol}: "
                f"price {current_price} drifted {drift_pct:.1f}% from center {self.grid_center}"
            )
            self._rebalance(current_price)

    def _rebalance(self, new_center: float):
        """Shift the grid center to a new price and recalculate levels.

        Args:
            new_center: New center price for the grid.
        """
        # Cancel all existing grid orders
        self.order_manager.cancel_all_orders(self.symbol)

        old_center = self.grid_center
        self.grid_center = new_center
        self.grid_levels = self.calculate_grid_levels(new_center)

        self.logger.info(f"Grid rebalanced: {old_center} -> {new_center} ({len(self.grid_levels)} new levels)")

        # Place new orders
        self._place_grid_orders()
        self._save_state()

    # --- Stop loss ---

    def _check_stop_loss(self):
        """Check if inventory loss exceeds stop_loss_pct — emergency close."""
        if self.inventory_amount <= 0 or not self.stop_loss_pct:
            return

        try:
            ticker = self.market_data.get_ticker(self.symbol)
            current_price = ticker["last"]
        except Exception:
            return

        if self.inventory_avg_price <= 0:
            return

        loss_pct = (self.inventory_avg_price - current_price) / self.inventory_avg_price * 100

        if loss_pct >= self.stop_loss_pct:
            self.logger.critical(
                f"STOP LOSS triggered for {self.symbol}: "
                f"inventory loss {loss_pct:.1f}% exceeds limit {self.stop_loss_pct}%"
            )
            # Cancel all orders and sell inventory at market
            self.order_manager.cancel_all_orders(self.symbol)
            if self.inventory_amount > 0:
                self.order_manager.place_order(
                    self.symbol, "sell", self.inventory_amount,
                    current_price, "grid",
                )
            self.inventory_amount = 0
            self.inventory_avg_price = 0
            self._save_state()

    # --- State persistence ---

    def _save_state(self):
        """Save current grid state to database for restart recovery."""
        now = datetime.now(timezone.utc).isoformat()
        # Clean grid_levels for JSON serialization (only keep active levels)
        active_levels = [
            {"price": l["price"], "side": l["side"], "status": l["status"], "order_id": l["order_id"]}
            for l in self.grid_levels
            if l["status"] in ("open", "pending")
        ]
        levels_json = json.dumps(active_levels)

        # Check if state exists
        existing = self.db.fetch_one("SELECT id FROM grid_state WHERE pair = ?", (self.symbol,))

        if existing:
            self.db.execute(
                """UPDATE grid_state SET
                    grid_center = ?, grid_levels = ?, inventory_amount = ?,
                    inventory_avg_price = ?, total_round_trips = ?,
                    total_profit_usd = ?, updated_at = ?
                WHERE pair = ?""",
                (
                    self.grid_center, levels_json, self.inventory_amount,
                    self.inventory_avg_price, self.total_round_trips,
                    self.total_profit_usd, now, self.symbol,
                ),
            )
        else:
            self.db.execute(
                """INSERT INTO grid_state
                (pair, grid_center, grid_levels, inventory_amount, inventory_avg_price,
                 total_round_trips, total_profit_usd, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.symbol, self.grid_center, levels_json, self.inventory_amount,
                    self.inventory_avg_price, self.total_round_trips,
                    self.total_profit_usd, now, now,
                ),
            )

    def _load_state(self) -> bool:
        """Load grid state from database.

        Returns:
            True if state was loaded, False if no saved state exists.
        """
        state = self.db.fetch_one(
            "SELECT * FROM grid_state WHERE pair = ? ORDER BY updated_at DESC LIMIT 1",
            (self.symbol,),
        )
        if not state:
            return False

        self.grid_center = state["grid_center"]
        self.grid_levels = json.loads(state["grid_levels"])
        self.inventory_amount = state["inventory_amount"] or 0
        self.inventory_avg_price = state["inventory_avg_price"] or 0
        self.total_round_trips = state["total_round_trips"] or 0
        self.total_profit_usd = state["total_profit_usd"] or 0

        return True
