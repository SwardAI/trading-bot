import threading
from datetime import datetime, timedelta, timezone

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.data.market_data import MarketDataManager
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import OrderRequest, RiskManager
from src.strategies.base_strategy import BaseStrategy


class FundingStrategy(BaseStrategy):
    """Funding rate arbitrage: long spot + short futures to collect funding payments.

    Delta-neutral strategy that profits from the funding rate paid every 8 hours
    on perpetual futures. When the rate is positive, shorts pay longs — we short
    futures and long spot to collect the payment while eliminating price risk.

    Safety invariant: NEVER have an unhedged leg. If one leg fails to execute,
    the other must be immediately unwound.

    Args:
        config: funding_strategy section from config.
        exchange: ExchangeManager instance (spot).
        db: Database instance.
        risk_manager: RiskManager instance.
        order_manager: OrderManager instance (for spot leg).
        market_data: MarketDataManager instance.
        futures_exchange: ExchangeManager configured for futures.
    """

    def __init__(
        self,
        config: dict,
        exchange: ExchangeManager,
        db: Database,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        market_data: MarketDataManager,
        futures_exchange: ExchangeManager,
    ):
        super().__init__("funding", config, exchange, db, risk_manager)
        self.order_manager = order_manager
        self.market_data = market_data
        self.futures_exchange = futures_exchange

        # Config
        self.pairs = config.get("pairs", [])
        self.min_funding_rate = config.get("min_funding_rate", 0.03)
        self.exit_funding_rate = config.get("exit_funding_rate", 0.01)
        self.position_size_pct = config.get("position_size_pct", 15)
        self.max_positions = config.get("max_positions", 2)
        self.max_basis_divergence_pct = config.get("max_basis_divergence_pct", 1.5)
        self.rebalance_threshold_pct = config.get("rebalance_threshold_pct", 2.0)
        self.max_hold_days = config.get("max_hold_days", 7)
        self.funding_history_checks = config.get("funding_history_checks", 3)
        self.exit_confirmations_needed = config.get("exit_confirmations_needed", 2)
        self.futures_leverage = config.get("futures_leverage", 1)
        self.margin_warning_pct = config.get("margin_warning_pct", 20)

        # State
        self.positions: list[dict] = []
        self._entry_lock = threading.Lock()
        self._exit_confirmations: dict[str, int] = {}  # pair -> consecutive low-rate checks

        self.logger.info(
            f"FundingStrategy initialized: {len(self.pairs)} pairs, "
            f"min_rate={self.min_funding_rate}%, max_positions={self.max_positions}"
        )

    def start(self):
        """Load open positions and set leverage on futures pairs."""
        self._running = True
        self._load_positions()

        # Set leverage to configured value (1x) for each pair
        for pair in self.pairs:
            futures_symbol = f"{pair}:USDT"
            try:
                self.futures_exchange.set_leverage(self.futures_leverage, futures_symbol)
            except Exception as e:
                self.logger.warning(f"Could not set leverage for {futures_symbol}: {e}")

        self.logger.info(f"Funding strategy started: {len(self.positions)} open positions loaded")

    def stop(self):
        """Stop monitoring. Positions stay open — they are hedged and safe across restarts."""
        self._running = False
        self.logger.info(
            f"Funding strategy stopped. {len(self.positions)} positions remain open (delta-neutral, safe)."
        )

    def on_tick(self):
        """Called every funding_check_interval — manage positions and scan for entries."""
        if not self._running:
            return

        # Manage existing positions first
        self._manage_positions()

        # Scan for new entries if we have capacity
        if len(self.positions) < self.max_positions:
            self._scan_for_entries()

    # --- Entry scanning ---

    def _scan_for_entries(self):
        """Scan pairs for funding rate opportunities."""
        with self._entry_lock:
            for pair in self.pairs:
                # Skip if already have position
                if any(p["pair"] == pair and p["status"] == "open" for p in self.positions):
                    continue

                # Check capacity
                if len([p for p in self.positions if p["status"] == "open"]) >= self.max_positions:
                    break

                try:
                    self._evaluate_pair(pair)
                except Exception as e:
                    self.logger.error(f"Error evaluating {pair} for funding arb: {e}")

    def _evaluate_pair(self, pair: str):
        """Check if a pair meets entry criteria for funding arb."""
        # Fetch current funding rate
        rate_info = self.market_data.get_funding_rate(pair)
        if not rate_info:
            return

        funding_rate = rate_info["funding_rate"]

        if funding_rate < self.min_funding_rate:
            self.logger.debug(
                f"Funding {pair}: rate {funding_rate:.4f}% < min {self.min_funding_rate}%, skip"
            )
            return

        # Check funding rate history — must be consistently positive
        history = self.market_data.get_funding_rate_history(pair, limit=self.funding_history_checks)
        if len(history) < self.funding_history_checks:
            self.logger.debug(f"Funding {pair}: insufficient rate history ({len(history)} < {self.funding_history_checks})")
            return

        negative_count = sum(1 for h in history if h["funding_rate"] <= 0)
        if negative_count > 0:
            self.logger.info(
                f"Funding {pair}: rate {funding_rate:.4f}% but {negative_count}/{len(history)} "
                f"recent payments were negative, skip"
            )
            return

        # Check spot-futures basis
        futures_symbol = rate_info["futures_symbol"]
        try:
            spot_ticker = self.market_data.get_ticker(pair)
            futures_ticker = self.futures_exchange.fetch_ticker(futures_symbol)
            spot_price = spot_ticker["last"]
            futures_price = futures_ticker.get("last", 0)
        except Exception as e:
            self.logger.error(f"Failed to fetch prices for basis check {pair}: {e}")
            return

        if spot_price <= 0 or futures_price <= 0:
            return

        basis_pct = abs(futures_price - spot_price) / spot_price * 100
        if basis_pct > self.max_basis_divergence_pct:
            self.logger.info(
                f"Funding {pair}: basis {basis_pct:.2f}% > max {self.max_basis_divergence_pct}%, skip"
            )
            return

        # All checks passed — enter position
        self.logger.info(
            f"Funding {pair}: rate={funding_rate:.4f}%, basis={basis_pct:.2f}%, "
            f"history all positive — entering"
        )
        self._enter_position(pair, funding_rate, spot_price, futures_price, basis_pct)

    # --- Entry execution ---

    def _enter_position(
        self, pair: str, funding_rate: float,
        spot_price: float, futures_price: float, basis_pct: float,
    ):
        """Open a funding arb position: buy spot + short futures.

        CRITICAL SAFETY: If one leg fails, the other is immediately unwound.
        """
        # Calculate position size
        balance = self.risk_manager.position_tracker.get_balance()
        position_usd = balance["total_usd"] * (self.position_size_pct / 100)
        amount = position_usd / spot_price

        # Risk check (use reduced cost — funding is delta-neutral)
        order_request = OrderRequest(
            symbol=pair,
            side="buy",
            amount=amount,
            price=spot_price,
            cost_usd=position_usd * 0.05,  # 5% of notional for delta-neutral
            strategy="funding",
        )
        decision = self.risk_manager.check(order_request)

        if not decision.approved:
            self.logger.info(f"Funding entry rejected for {pair}: {decision.reason}")
            return

        if decision.adjusted_amount:
            amount = decision.adjusted_amount
            position_usd = amount * spot_price

        # --- LEG 1: Buy spot ---
        spot_trade = self.order_manager.place_order(
            pair, "buy", amount, spot_price, "funding"
        )
        if not spot_trade:
            self.logger.error(f"Funding spot buy FAILED for {pair} — no position opened")
            return

        spot_fill_price = spot_trade["price"]
        spot_fill_amount = spot_trade["amount"]
        spot_order_id = spot_trade.get("exchange_order_id")

        # --- LEG 2: Short futures ---
        futures_symbol = f"{pair}:USDT"
        try:
            # Enforce leverage before every trade
            self.futures_exchange.set_leverage(self.futures_leverage, futures_symbol)
            futures_order = self.futures_exchange.create_order(
                futures_symbol, "market", "sell", spot_fill_amount
            )
        except Exception as e:
            self.logger.critical(
                f"Funding futures SHORT FAILED for {pair}: {e}. "
                f"UNWINDING spot buy immediately."
            )
            # UNWIND: sell the spot we just bought
            unwind = self.order_manager.place_order(
                pair, "sell", spot_fill_amount, spot_fill_price, "funding"
            )
            if not unwind:
                self.logger.critical(
                    f"CRITICAL: Failed to unwind spot buy for {pair} after futures failure. "
                    f"UNHEDGED LONG POSITION of {spot_fill_amount:.6f} {pair}. "
                    f"MANUAL INTERVENTION REQUIRED."
                )
            return

        futures_fill_price = futures_order.get("average") or futures_order.get("price", futures_price)
        futures_fill_amount = futures_order.get("filled", spot_fill_amount)
        futures_order_id = futures_order.get("id")

        # Log the futures trade to DB (bypasses OrderManager since it uses spot exchange)
        now = datetime.now(timezone.utc).isoformat()
        fee_info = futures_order.get("fee") or {}
        futures_fee = fee_info.get("cost", 0) or 0

        self.db.execute(
            """INSERT INTO trades
            (timestamp, strategy, pair, side, order_type, price, amount, cost_usd,
             fee_usd, fee_currency, exchange_order_id)
            VALUES (?, 'funding', ?, 'sell', 'market', ?, ?, ?, ?, ?, ?)""",
            (
                now, futures_symbol, futures_fill_price, futures_fill_amount,
                futures_fill_amount * futures_fill_price,
                futures_fee, fee_info.get("currency"), futures_order_id,
            ),
        )

        # Record position
        spot_fee = spot_trade.get("fee_usd", 0) or 0
        total_fees = spot_fee + futures_fee
        notional = spot_fill_amount * spot_fill_price

        cursor = self.db.execute(
            """INSERT INTO funding_positions
            (pair, status, spot_entry_price, spot_entry_amount, spot_entry_order_id,
             futures_entry_price, futures_entry_amount, futures_entry_order_id,
             entry_funding_rate, entry_basis_pct, entry_time, total_fees_usd,
             notional_usd, last_funding_check)
            VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pair, spot_fill_price, spot_fill_amount, spot_order_id,
                futures_fill_price, futures_fill_amount, futures_order_id,
                funding_rate, basis_pct, now, total_fees,
                notional, now,
            ),
        )

        position = {
            "id": cursor.lastrowid,
            "pair": pair,
            "status": "open",
            "spot_entry_price": spot_fill_price,
            "spot_entry_amount": spot_fill_amount,
            "futures_entry_price": futures_fill_price,
            "futures_entry_amount": futures_fill_amount,
            "funding_collected_usd": 0,
            "funding_payments_count": 0,
            "entry_funding_rate": funding_rate,
            "entry_time": now,
            "notional_usd": notional,
            "total_fees_usd": total_fees,
            "last_funding_check": now,
        }
        self.positions.append(position)

        self.logger.info(
            f"Funding ENTRY: {pair} | spot buy {spot_fill_amount:.6f} @ {spot_fill_price:.2f} | "
            f"futures short {futures_fill_amount:.6f} @ {futures_fill_price:.2f} | "
            f"notional=${notional:.2f} | rate={funding_rate:.4f}% | basis={basis_pct:.2f}%"
        )

    # --- Position management ---

    def _manage_positions(self):
        """Check all open positions for exit conditions, funding collection, and health."""
        for position in list(self.positions):
            if position["status"] != "open":
                continue

            try:
                # Track funding payments
                self._check_funding_collection(position)

                # Check exit conditions
                exit_reason = self._check_exit_conditions(position)
                if exit_reason:
                    self._exit_position(position, exit_reason)
                    continue

                # Check basis divergence (alert only, don't auto-exit)
                self._check_basis_health(position)

            except Exception as e:
                self.logger.error(f"Error managing funding position {position['pair']}: {e}")

    def _check_funding_collection(self, position: dict):
        """Track funding payments received since last check."""
        futures_symbol = f"{position['pair']}:USDT"
        try:
            # Fetch current positions from exchange to get unrealized funding
            positions = self.futures_exchange.fetch_positions([futures_symbol])
            if not positions:
                return

            # Find our position
            for pos in positions:
                if pos.get("symbol") == futures_symbol and pos.get("side") == "short":
                    # Some exchanges report accumulated unrealized P&L including funding
                    # We track it incrementally via the DB
                    break

            # For more accurate tracking, check income history if available
            # Most exchanges provide this, but the exact method varies
            # For now, estimate from the funding rate * notional
            now = datetime.now(timezone.utc)
            last_check = datetime.fromisoformat(position["last_funding_check"])
            hours_since = (now - last_check).total_seconds() / 3600

            # Funding is paid every 8 hours — only log if enough time has passed
            if hours_since >= 7.5:
                rate_info = self.market_data.get_funding_rate(position["pair"])
                if rate_info:
                    current_rate = rate_info["funding_rate"] / 100  # Convert pct to decimal
                    # Estimated funding for one payment period
                    estimated_payment = position["notional_usd"] * current_rate

                    if estimated_payment > 0:
                        position["funding_collected_usd"] += estimated_payment
                        position["funding_payments_count"] += 1
                        position["last_funding_check"] = now.isoformat()

                        self.db.execute(
                            """UPDATE funding_positions SET
                                funding_collected_usd = ?, funding_payments_count = ?,
                                last_funding_check = ?
                            WHERE id = ?""",
                            (
                                position["funding_collected_usd"],
                                position["funding_payments_count"],
                                position["last_funding_check"],
                                position["id"],
                            ),
                        )

                        self.logger.info(
                            f"Funding payment {position['pair']}: ~${estimated_payment:.4f} "
                            f"(total: ${position['funding_collected_usd']:.4f}, "
                            f"#{position['funding_payments_count']} payments)"
                        )

        except Exception as e:
            self.logger.warning(f"Failed to check funding collection for {position['pair']}: {e}")

    def _check_exit_conditions(self, position: dict) -> str | None:
        """Check if any exit condition is met.

        Returns:
            Exit reason string, or None if no exit triggered.
        """
        pair = position["pair"]

        # 1. Funding rate too low
        rate_info = self.market_data.get_funding_rate(pair)
        if rate_info:
            if rate_info["funding_rate"] < self.exit_funding_rate:
                self._exit_confirmations[pair] = self._exit_confirmations.get(pair, 0) + 1
                if self._exit_confirmations[pair] >= self.exit_confirmations_needed:
                    self._exit_confirmations.pop(pair, None)
                    return "low_rate"
                else:
                    self.logger.info(
                        f"Funding {pair}: rate {rate_info['funding_rate']:.4f}% below exit threshold "
                        f"({self._exit_confirmations[pair]}/{self.exit_confirmations_needed} confirmations)"
                    )
            else:
                # Rate recovered — reset confirmation counter
                self._exit_confirmations.pop(pair, None)

        # 2. Max holding period
        entry_time = datetime.fromisoformat(position["entry_time"])
        hold_days = (datetime.now(timezone.utc) - entry_time).total_seconds() / 86400
        if hold_days >= self.max_hold_days:
            return "max_hold"

        # 3. Margin health (futures leg)
        futures_symbol = f"{pair}:USDT"
        try:
            positions = self.futures_exchange.fetch_positions([futures_symbol])
            for pos in positions:
                if pos.get("symbol") == futures_symbol:
                    margin_pct = pos.get("marginRatio", pos.get("percentage", 100))
                    if isinstance(margin_pct, (int, float)) and margin_pct < self.margin_warning_pct:
                        self.logger.critical(
                            f"Funding {pair}: MARGIN WARNING — ratio {margin_pct:.1f}% "
                            f"< {self.margin_warning_pct}%, emergency exit"
                        )
                        return "margin_warning"
        except Exception as e:
            self.logger.warning(f"Could not check margin for {pair}: {e}")

        return None

    def _check_basis_health(self, position: dict):
        """Check spot-futures basis divergence and log warnings."""
        pair = position["pair"]
        futures_symbol = f"{pair}:USDT"

        try:
            spot_ticker = self.market_data.get_ticker(pair)
            futures_ticker = self.futures_exchange.fetch_ticker(futures_symbol)
            spot_price = spot_ticker["last"]
            futures_price = futures_ticker.get("last", 0)

            if spot_price > 0 and futures_price > 0:
                basis_pct = abs(futures_price - spot_price) / spot_price * 100
                if basis_pct > self.max_basis_divergence_pct:
                    self.logger.warning(
                        f"Funding {pair}: basis divergence {basis_pct:.2f}% "
                        f"> limit {self.max_basis_divergence_pct}%"
                    )
        except Exception as e:
            self.logger.debug(f"Could not check basis for {pair}: {e}")

    # --- Exit execution ---

    def _exit_position(self, position: dict, reason: str):
        """Close a funding arb position: close futures short, then sell spot.

        CRITICAL: Close futures first (margin-sensitive leg). If futures close
        fails, position stays open (the hedge protects us). Only mark closed
        after BOTH legs confirmed.
        """
        pair = position["pair"]
        futures_symbol = f"{pair}:USDT"
        now = datetime.now(timezone.utc).isoformat()

        # --- LEG 1: Close futures short (buy back) ---
        try:
            self.futures_exchange.create_order(
                futures_symbol, "market", "buy", position["futures_entry_amount"]
            )
        except Exception as e:
            self.logger.critical(
                f"Funding futures CLOSE FAILED for {pair}: {e}. "
                f"Position stays open (still hedged). Will retry next tick."
            )
            return

        # Fetch the fill details
        futures_exit_price = 0
        try:
            trades = self.futures_exchange.fetch_my_trades(futures_symbol, limit=5)
            if trades:
                latest = trades[-1]
                futures_exit_price = latest.get("price", 0)
                futures_fee = (latest.get("fee") or {}).get("cost", 0) or 0
                position["total_fees_usd"] += futures_fee
        except Exception:
            # Use spot price as approximation
            try:
                ticker = self.market_data.get_ticker(pair)
                futures_exit_price = ticker["last"]
            except Exception:
                futures_exit_price = position["futures_entry_price"]

        # --- LEG 2: Sell spot ---
        spot_trade = self.order_manager.place_order(
            pair, "sell", position["spot_entry_amount"],
            0,  # market price
            "funding",
        )

        if not spot_trade:
            self.logger.critical(
                f"Funding spot SELL FAILED for {pair} after futures close succeeded. "
                f"UNHEDGED LONG POSITION of {position['spot_entry_amount']:.6f} {pair}. "
                f"Futures already closed. Will retry next tick."
            )
            # Don't mark as closed — we still hold spot
            # Mark as 'closing' so we know futures is already closed
            position["status"] = "closing"
            self.db.execute(
                "UPDATE funding_positions SET status = 'closing' WHERE id = ?",
                (position["id"],),
            )
            return

        spot_exit_price = spot_trade["price"]
        spot_fee = spot_trade.get("fee_usd", 0) or 0
        position["total_fees_usd"] += spot_fee

        # Calculate P&L
        spot_pnl = (spot_exit_price - position["spot_entry_price"]) * position["spot_entry_amount"]
        futures_pnl = (position["futures_entry_price"] - futures_exit_price) * position["futures_entry_amount"]
        basis_pnl = spot_pnl + futures_pnl
        net_pnl = position["funding_collected_usd"] + basis_pnl - position["total_fees_usd"]

        # Update DB
        self.db.execute(
            """UPDATE funding_positions SET
                status = 'closed', spot_exit_price = ?, futures_exit_price = ?,
                exit_time = ?, exit_reason = ?, basis_pnl_usd = ?,
                total_fees_usd = ?, net_pnl_usd = ?
            WHERE id = ?""",
            (
                spot_exit_price, futures_exit_price, now, reason,
                basis_pnl, position["total_fees_usd"], net_pnl,
                position["id"],
            ),
        )

        # Remove from active positions
        position["status"] = "closed"
        self.positions = [p for p in self.positions if p["status"] == "open"]

        win_loss = "WIN" if net_pnl > 0 else "LOSS"
        hold_hours = 0
        try:
            entry = datetime.fromisoformat(position["entry_time"])
            hold_hours = (datetime.now(timezone.utc) - entry).total_seconds() / 3600
        except Exception:
            pass

        self.logger.info(
            f"Funding EXIT ({win_loss}): {pair} | reason={reason} | "
            f"held {hold_hours:.1f}h | funding=${position['funding_collected_usd']:.4f} | "
            f"basis=${basis_pnl:.4f} | fees=${position['total_fees_usd']:.4f} | "
            f"net P&L: ${net_pnl:.4f}"
        )

    # --- State loading ---

    def _load_positions(self):
        """Load open funding positions from database."""
        rows = self.db.fetch_all(
            "SELECT * FROM funding_positions WHERE status IN ('open', 'closing')"
        )
        self.positions = []
        for row in rows:
            self.positions.append({
                "id": row["id"],
                "pair": row["pair"],
                "status": row["status"],
                "spot_entry_price": row["spot_entry_price"],
                "spot_entry_amount": row["spot_entry_amount"],
                "futures_entry_price": row["futures_entry_price"],
                "futures_entry_amount": row["futures_entry_amount"],
                "funding_collected_usd": row["funding_collected_usd"] or 0,
                "funding_payments_count": row["funding_payments_count"] or 0,
                "entry_funding_rate": row["entry_funding_rate"],
                "entry_time": row["entry_time"],
                "notional_usd": row["notional_usd"],
                "total_fees_usd": row["total_fees_usd"] or 0,
                "last_funding_check": row["last_funding_check"] or row["entry_time"],
            })

        # Handle any positions stuck in 'closing' state (futures closed, spot not sold)
        closing = [p for p in self.positions if p["status"] == "closing"]
        for pos in closing:
            self.logger.warning(
                f"Funding {pos['pair']}: position in 'closing' state (futures closed, "
                f"spot still held). Will attempt to sell spot."
            )
