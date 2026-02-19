import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.data.indicators import compute_all_indicators
from src.data.market_data import MarketDataManager
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import OrderRequest, RiskManager
from src.strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Trend-following strategy with multi-indicator entry signals and dynamic trailing stops.

    Entry requires ALL signals to align:
    1. EMA crossover (fast above slow for longs, below for shorts)
    2. RSI confirmation (>55 for longs, <45 for shorts)
    3. Volume surge (current > 1.5x 20-period average)
    4. ADX filter (>25 confirms a real trend)
    5. MACD histogram positive & increasing (longs) or negative & decreasing (shorts)

    Higher timeframe must agree with signal direction.

    Exit on ANY of:
    1. Trailing stop hit
    2. EMA crossover reversal
    3. Time stop (72h without reaching 2x risk in profit)

    Args:
        config: momentum_strategy section from config.
        exchange: ExchangeManager instance.
        db: Database instance.
        risk_manager: RiskManager instance.
        order_manager: OrderManager instance.
        market_data: MarketDataManager instance.
    """

    def __init__(
        self,
        config: dict,
        exchange: ExchangeManager,
        db: Database,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        market_data: MarketDataManager,
    ):
        super().__init__("momentum", config, exchange, db, risk_manager)
        self.order_manager = order_manager
        self.market_data = market_data

        # Pairs to trade
        self.pairs = config.get("pairs", [])

        # Timeframes
        self.timeframe = config.get("timeframe", "1h")
        self.confirmation_timeframe = config.get("confirmation_timeframe", "4h")

        # Indicator parameters (passed through to compute_all_indicators)
        self.indicator_config = {
            "ema_fast": config.get("ema_fast", 9),
            "ema_slow": config.get("ema_slow", 21),
            "rsi_period": config.get("rsi_period", 14),
            "adx_period": config.get("adx_period", 14),
            "atr_period": config.get("atr_period", 14),
        }

        # Signal thresholds
        self.rsi_long_threshold = config.get("rsi_long_threshold", 55)
        self.rsi_short_threshold = config.get("rsi_short_threshold", 45)
        self.adx_min_strength = config.get("adx_min_strength", 25)
        self.volume_surge_multiplier = config.get("volume_surge_multiplier", 1.5)

        # Position management
        self.risk_per_trade_pct = config.get("risk_per_trade_pct", 1.0)
        self.max_concurrent_positions = config.get("max_concurrent_positions", 3)
        self.trailing_stop_atr_mult = config.get("trailing_stop_atr_multiplier", 1.5)
        self.time_stop_hours = config.get("time_stop_hours", 72)

        # Cooldown
        self.cooldown_minutes = config.get("cooldown_after_loss_minutes", 60)

        # Active positions (loaded from DB)
        self.positions: list[dict] = []

        self.logger.info(
            f"MomentumStrategy initialized: {len(self.pairs)} pairs, "
            f"timeframe={self.timeframe}, confirmation={self.confirmation_timeframe}"
        )

    def start(self):
        """Load open positions from DB and begin monitoring."""
        self._running = True
        self._load_positions()
        self.logger.info(f"Momentum strategy started: {len(self.positions)} open positions loaded")

    def stop(self):
        """Stop monitoring — positions remain open (managed by risk manager)."""
        self._running = False
        self.logger.info("Momentum strategy stopped")

    def on_tick(self):
        """Called every momentum_check_interval — manage positions and scan for entries."""
        if not self._running:
            return

        # Manage existing positions first (trailing stops, time stops)
        self._manage_positions()

        # Scan for new entry signals if we have capacity
        open_count = len(self.positions)
        if open_count < self.max_concurrent_positions:
            self._scan_for_entries()

    # --- Signal generation ---

    def _scan_for_entries(self):
        """Scan all pairs for entry signals."""
        for symbol in self.pairs:
            # Skip if already have position in this pair
            if any(p["pair"] == symbol and p["status"] == "open" for p in self.positions):
                continue

            # Check cooldown
            if self._is_in_cooldown(symbol):
                continue

            try:
                signal = self._check_entry_signal(symbol)
                if signal:
                    self._enter_position(symbol, signal)
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")

    def _check_entry_signal(self, symbol: str) -> dict | None:
        """Check if all entry conditions are met for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            Signal dict with side and details, or None if no signal.
        """
        # Fetch primary timeframe data
        try:
            df = self.market_data.get_ohlcv(symbol, self.timeframe, limit=100)
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None

        if len(df) < 30:
            return None

        # Compute indicators
        df = compute_all_indicators(df, self.indicator_config)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Check for NaN in key indicators
        if pd.isna(latest["ema_fast"]) or pd.isna(latest["adx"]) or pd.isna(latest["atr"]):
            return None

        signals = {}

        # 1. EMA crossover
        ema_cross_long = latest["ema_fast"] > latest["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]
        ema_cross_short = latest["ema_fast"] < latest["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]
        ema_bullish = latest["ema_fast"] > latest["ema_slow"]
        ema_bearish = latest["ema_fast"] < latest["ema_slow"]

        # 2. RSI confirmation
        rsi_long = latest["rsi"] > self.rsi_long_threshold if not pd.isna(latest["rsi"]) else False
        rsi_short = latest["rsi"] < self.rsi_short_threshold if not pd.isna(latest["rsi"]) else False

        # 3. Volume surge
        volume_surge = False
        if not pd.isna(latest["volume_sma"]) and latest["volume_sma"] > 0:
            volume_surge = latest["volume"] > (latest["volume_sma"] * self.volume_surge_multiplier)

        # 4. ADX filter (trend strength)
        adx_strong = latest["adx"] > self.adx_min_strength if not pd.isna(latest["adx"]) else False

        # 5. MACD histogram
        macd_long = False
        macd_short = False
        if not pd.isna(latest["macd_histogram"]) and not pd.isna(prev["macd_histogram"]):
            macd_long = latest["macd_histogram"] > 0 and latest["macd_histogram"] > prev["macd_histogram"]
            macd_short = latest["macd_histogram"] < 0 and latest["macd_histogram"] < prev["macd_histogram"]

        # Check LONG signal (all must align)
        if (ema_cross_long or ema_bullish) and rsi_long and volume_surge and adx_strong and macd_long:
            # Higher timeframe confirmation
            if self._confirm_higher_timeframe(symbol, "long"):
                return {
                    "side": "long",
                    "entry_price": latest["close"],
                    "atr": latest["atr"],
                    "signals": {
                        "ema_cross": ema_cross_long,
                        "ema_bullish": ema_bullish,
                        "rsi": float(latest["rsi"]),
                        "adx": float(latest["adx"]),
                        "volume_ratio": float(latest["volume"] / latest["volume_sma"]) if latest["volume_sma"] > 0 else 0,
                        "macd_hist": float(latest["macd_histogram"]),
                    },
                }

        # Check SHORT signal (all must align)
        if (ema_cross_short or ema_bearish) and rsi_short and volume_surge and adx_strong and macd_short:
            if self._confirm_higher_timeframe(symbol, "short"):
                return {
                    "side": "short",
                    "entry_price": latest["close"],
                    "atr": latest["atr"],
                    "signals": {
                        "ema_cross": ema_cross_short,
                        "ema_bearish": ema_bearish,
                        "rsi": float(latest["rsi"]),
                        "adx": float(latest["adx"]),
                        "volume_ratio": float(latest["volume"] / latest["volume_sma"]) if latest["volume_sma"] > 0 else 0,
                        "macd_hist": float(latest["macd_histogram"]),
                    },
                }

        return None

    def _confirm_higher_timeframe(self, symbol: str, direction: str) -> bool:
        """Check that the higher timeframe trend agrees with the signal.

        Args:
            symbol: Trading pair.
            direction: "long" or "short".

        Returns:
            True if higher timeframe confirms.
        """
        try:
            df = self.market_data.get_ohlcv(symbol, self.confirmation_timeframe, limit=50)
            if len(df) < 25:
                return False

            df = compute_all_indicators(df, self.indicator_config)
            latest = df.iloc[-1]

            if pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]):
                return False

            if direction == "long":
                return latest["ema_fast"] > latest["ema_slow"]
            else:
                return latest["ema_fast"] < latest["ema_slow"]

        except Exception as e:
            self.logger.error(f"Higher timeframe check failed for {symbol}: {e}")
            return False

    # --- Position entry ---

    def _enter_position(self, symbol: str, signal: dict):
        """Open a new momentum position.

        Position size is calculated from risk_per_trade_pct and ATR-based stop distance.

        Args:
            symbol: Trading pair.
            signal: Signal dict from _check_entry_signal.
        """
        entry_price = signal["entry_price"]
        atr = signal["atr"]
        side = signal["side"]

        # Calculate stop loss
        stop_distance = atr * self.trailing_stop_atr_mult
        if side == "long":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Calculate position size from risk
        balance = self.risk_manager.position_tracker.get_balance()
        risk_amount = balance["total_usd"] * (self.risk_per_trade_pct / 100)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            self.logger.warning(f"Invalid stop distance for {symbol}, skipping")
            return

        amount = risk_amount / risk_per_unit
        cost_usd = amount * entry_price

        # Risk check
        order_request = OrderRequest(
            symbol=symbol,
            side="buy" if side == "long" else "sell",
            amount=amount,
            price=entry_price,
            cost_usd=cost_usd,
            strategy="momentum",
        )
        decision = self.risk_manager.check(order_request)

        if not decision.approved:
            self.logger.info(f"Momentum entry rejected for {symbol}: {decision.reason}")
            return

        if decision.adjusted_amount:
            amount = decision.adjusted_amount
            cost_usd = amount * entry_price

        # Place entry order
        order_side = "buy" if side == "long" else "sell"
        trade = self.order_manager.place_order(
            symbol, order_side, amount, entry_price, "momentum"
        )

        if not trade:
            self.logger.error(f"Failed to enter momentum position for {symbol}")
            return

        # Record position in DB
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.db.execute(
            """INSERT INTO momentum_positions
            (pair, side, entry_price, entry_time, amount, stop_loss, current_stop,
             entry_signals, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (
                symbol, side, trade["price"], now, trade["amount"],
                stop_loss, stop_loss, json.dumps(signal["signals"]),
            ),
        )

        position = {
            "id": cursor.lastrowid,
            "pair": symbol,
            "side": side,
            "entry_price": trade["price"],
            "entry_time": now,
            "amount": trade["amount"],
            "stop_loss": stop_loss,
            "current_stop": stop_loss,
            "status": "open",
        }
        self.positions.append(position)

        self.logger.info(
            f"Momentum ENTRY: {side.upper()} {trade['amount']:.6f} {symbol} "
            f"@ {trade['price']}, stop={stop_loss:.2f}, risk=${risk_amount:.2f}"
        )

    # --- Position management ---

    def _manage_positions(self):
        """Check trailing stops, time stops, and signal reversals for open positions."""
        for position in list(self.positions):
            if position["status"] != "open":
                continue

            try:
                ticker = self.market_data.get_ticker(position["pair"])
                current_price = ticker["last"]
            except Exception as e:
                self.logger.error(f"Failed to get ticker for {position['pair']}: {e}")
                continue

            # Check trailing stop
            if self._check_trailing_stop(position, current_price):
                self._exit_position(position, current_price, "trailing_stop")
                continue

            # Update trailing stop (tighten as profit grows)
            self._update_trailing_stop(position, current_price)

            # Check time stop
            if self._check_time_stop(position, current_price):
                self._exit_position(position, current_price, "time_stop")
                continue

            # Check EMA reversal
            if self._check_signal_reversal(position):
                self._exit_position(position, current_price, "signal_reversal")
                continue

    def _check_trailing_stop(self, position: dict, current_price: float) -> bool:
        """Check if trailing stop has been hit.

        Returns:
            True if stop hit, position should be closed.
        """
        if position["side"] == "long":
            return current_price <= position["current_stop"]
        else:
            return current_price >= position["current_stop"]

    def _update_trailing_stop(self, position: dict, current_price: float):
        """Tighten trailing stop as profit grows.

        Stop tightening schedule:
        - Profit > 1x risk: tighten to 1.2 ATR
        - Profit > 2x risk: tighten to 1.0 ATR
        - Profit > 3x risk: tighten to 0.8 ATR
        """
        entry = position["entry_price"]
        initial_risk = abs(entry - position["stop_loss"])

        if initial_risk <= 0:
            return

        if position["side"] == "long":
            profit = current_price - entry
            profit_multiple = profit / initial_risk

            if profit_multiple >= 3:
                atr_mult = 0.8
            elif profit_multiple >= 2:
                atr_mult = 1.0
            elif profit_multiple >= 1:
                atr_mult = 1.2
            else:
                return  # No tightening yet

            # Estimate ATR from initial stop distance
            estimated_atr = initial_risk / self.trailing_stop_atr_mult
            new_stop = current_price - (estimated_atr * atr_mult)

            # Never move stop backward
            if new_stop > position["current_stop"]:
                position["current_stop"] = new_stop
                self.db.execute(
                    "UPDATE momentum_positions SET current_stop = ? WHERE id = ?",
                    (new_stop, position["id"]),
                )
                self.logger.debug(
                    f"Trailing stop tightened: {position['pair']} stop={new_stop:.2f} "
                    f"(profit {profit_multiple:.1f}x risk)"
                )
        else:
            # Short position
            profit = entry - current_price
            profit_multiple = profit / initial_risk

            if profit_multiple >= 3:
                atr_mult = 0.8
            elif profit_multiple >= 2:
                atr_mult = 1.0
            elif profit_multiple >= 1:
                atr_mult = 1.2
            else:
                return

            estimated_atr = initial_risk / self.trailing_stop_atr_mult
            new_stop = current_price + (estimated_atr * atr_mult)

            # Never move stop backward (for shorts, backward means higher)
            if new_stop < position["current_stop"]:
                position["current_stop"] = new_stop
                self.db.execute(
                    "UPDATE momentum_positions SET current_stop = ? WHERE id = ?",
                    (new_stop, position["id"]),
                )

    def _check_time_stop(self, position: dict, current_price: float) -> bool:
        """Check if position has exceeded time limit without sufficient profit.

        Close if open > time_stop_hours without reaching 2x risk in profit.
        """
        entry_time = datetime.fromisoformat(position["entry_time"])
        elapsed = datetime.now(timezone.utc) - entry_time

        if elapsed < timedelta(hours=self.time_stop_hours):
            return False

        # Check if profit is at least 2x risk
        entry = position["entry_price"]
        initial_risk = abs(entry - position["stop_loss"])

        if position["side"] == "long":
            profit = current_price - entry
        else:
            profit = entry - current_price

        if profit < 2 * initial_risk:
            self.logger.info(
                f"Time stop for {position['pair']}: {elapsed.total_seconds()/3600:.1f}h, "
                f"profit {profit:.2f} < 2x risk {2*initial_risk:.2f}"
            )
            return True

        return False

    def _check_signal_reversal(self, position: dict) -> bool:
        """Check if EMA crossover has reversed (exit signal).

        Returns:
            True if signal reversed.
        """
        try:
            df = self.market_data.get_ohlcv(position["pair"], self.timeframe, limit=50)
            if len(df) < 25:
                return False

            df = compute_all_indicators(df, self.indicator_config)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]):
                return False

            if position["side"] == "long":
                # Exit long if fast EMA crosses below slow
                return latest["ema_fast"] < latest["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]
            else:
                # Exit short if fast EMA crosses above slow
                return latest["ema_fast"] > latest["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]

        except Exception:
            return False

    # --- Position exit ---

    def _exit_position(self, position: dict, exit_price: float, reason: str):
        """Close a momentum position and record results.

        Args:
            position: Position dict.
            exit_price: Price to exit at.
            reason: Exit reason (trailing_stop, time_stop, signal_reversal, circuit_breaker).
        """
        # Place exit order
        exit_side = "sell" if position["side"] == "long" else "buy"
        trade = self.order_manager.place_order(
            position["pair"], exit_side, position["amount"],
            exit_price, "momentum",
        )

        actual_exit_price = trade["price"] if trade else exit_price

        # Calculate P&L
        if position["side"] == "long":
            pnl_usd = (actual_exit_price - position["entry_price"]) * position["amount"]
        else:
            pnl_usd = (position["entry_price"] - actual_exit_price) * position["amount"]

        # Subtract fees (estimated)
        fee_pct = 0.075 / 100  # maker fee
        entry_fee = position["entry_price"] * position["amount"] * fee_pct
        exit_fee = actual_exit_price * position["amount"] * fee_pct
        pnl_usd -= (entry_fee + exit_fee)

        pnl_pct = (pnl_usd / (position["entry_price"] * position["amount"])) * 100

        # Update DB
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """UPDATE momentum_positions SET
                exit_price = ?, exit_time = ?, exit_reason = ?,
                pnl_usd = ?, pnl_pct = ?, status = 'closed'
            WHERE id = ?""",
            (actual_exit_price, now, reason, pnl_usd, pnl_pct, position["id"]),
        )

        # Update trade record with P&L
        if trade:
            self.db.execute(
                "UPDATE trades SET pnl_usd = ? WHERE id = ?",
                (pnl_usd, trade["id"]),
            )

        # Remove from active positions
        position["status"] = "closed"
        self.positions = [p for p in self.positions if p["status"] == "open"]

        win_loss = "WIN" if pnl_usd > 0 else "LOSS"
        self.logger.info(
            f"Momentum EXIT ({win_loss}): {position['side'].upper()} {position['pair']} "
            f"@ {actual_exit_price} | reason={reason} | P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)"
        )

    # --- Cooldown ---

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if a pair is in cooldown after a recent loss.

        Args:
            symbol: Trading pair.

        Returns:
            True if in cooldown period.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=self.cooldown_minutes)).isoformat()
        recent_loss = self.db.fetch_one(
            """SELECT id FROM momentum_positions
            WHERE pair = ? AND status = 'closed' AND pnl_usd < 0 AND exit_time > ?
            ORDER BY exit_time DESC LIMIT 1""",
            (symbol, cutoff),
        )
        return recent_loss is not None

    # --- State loading ---

    def _load_positions(self):
        """Load open momentum positions from database."""
        rows = self.db.fetch_all(
            "SELECT * FROM momentum_positions WHERE status = 'open'"
        )
        self.positions = []
        for row in rows:
            self.positions.append({
                "id": row["id"],
                "pair": row["pair"],
                "side": row["side"],
                "entry_price": row["entry_price"],
                "entry_time": row["entry_time"],
                "amount": row["amount"],
                "stop_loss": row["stop_loss"],
                "current_stop": row["current_stop"],
                "status": row["status"],
            })
