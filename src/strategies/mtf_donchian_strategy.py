import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.core.database import Database
from src.core.exchange import ExchangeManager
from src.data.indicators import add_donchian_high, add_donchian_low, add_atr, compute_mtf_donchian_indicators
from src.data.market_data import MarketDataManager
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import OrderRequest, RiskManager
from src.strategies.base_strategy import BaseStrategy


class MtfDonchianStrategy(BaseStrategy):
    """Vol-scaled multi-timeframe Donchian breakout strategy.

    Uses daily Donchian channels [20, 50, 100] as trend filter (2/3 vote)
    and 4h Donchian breakout for entry/exit timing.

    Signal logic (matches backtest exactly):
    - Daily filter uses .shift(1) — previous day's completed channel
    - 4h channels use .shift(1) — previous bar's completed channel
    - Long entry: daily bullish (2/3 vote) AND close > 4h highest_high
    - Short entry: daily bearish (2/3 vote) AND close < 4h lowest_low
    - Chandelier stop: highest_since_entry - atr_mult * current_atr (long)
    - Exit channel: close < 4h lowest_low (7-period) for longs

    Vol-scaling: effective_risk = base_risk * clamp(atr_median / current_atr, 0.5, 2.0)
    Position sizing: amount = (capital * allocation_pct * effective_risk / 100) / risk_per_unit
    Capital cap: cost + fee <= available * 0.95
    """

    def __init__(
        self,
        config: dict,
        exchange: ExchangeManager,
        db: Database,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        market_data: MarketDataManager,
        regime_detector=None,
        futures_exchange: ExchangeManager | None = None,
    ):
        super().__init__("mtf_donchian", config, exchange, db, risk_manager)
        self.order_manager = order_manager
        self.market_data = market_data
        self.regime_detector = regime_detector
        self.futures_exchange = futures_exchange

        # Pairs with allocation percentages
        self.pairs_config = config.get("pairs", [])
        self.pairs = [p["symbol"] for p in self.pairs_config]
        self._allocation = {p["symbol"]: p["allocation_pct"] for p in self.pairs_config}

        # Daily filter parameters
        self.daily_periods = config.get("daily_periods", [20, 50, 100])
        self.daily_min_votes = config.get("daily_min_votes", 2)

        # 4h entry/exit parameters
        self.entry_period_4h = config.get("entry_period_4h", 14)
        self.exit_period_4h = config.get("exit_period_4h", 7)
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_mult = config.get("atr_stop_mult", 3.0)

        # Risk parameters
        self.risk_per_trade_pct = config.get("risk_per_trade_pct", 5.0)
        self.fee_rate = config.get("fee_rate", 0.001)  # 0.1% taker for spot
        self.futures_fee_rate = config.get("futures_fee_rate", 0.0004)  # 0.04% for futures

        # Vol-scaling
        self.vol_scale = config.get("vol_scale", True)
        self.vol_scale_lookback = config.get("vol_scale_lookback", 60)
        self.vol_scale_min = config.get("vol_scale_min", 0.5)
        self.vol_scale_max = config.get("vol_scale_max", 2.0)

        # Shorts toggle
        self.enable_shorts = config.get("enable_shorts", False)

        # Cooldown (in 4h bars, converted to hours)
        self.cooldown_bars = config.get("cooldown_bars", 2)
        self.cooldown_hours = self.cooldown_bars * 4

        # Active positions (loaded from DB)
        self.positions: list[dict] = []
        self._entry_lock = threading.Lock()

        # Caches for daily regime and 4h data
        self._daily_cache: dict[str, tuple[str, float]] = {}  # symbol -> (regime, timestamp)
        self._daily_cache_ttl = 4 * 3600  # 4 hours

        # Indicator config for compute_mtf_donchian_indicators
        self._indicator_config = {
            "entry_period_4h": self.entry_period_4h,
            "exit_period_4h": self.exit_period_4h,
            "atr_period": self.atr_period,
            "vol_scale_lookback": self.vol_scale_lookback,
        }

        # Track scan count for periodic logging
        self._scan_count = 0

        self.logger.info(
            f"MtfDonchianStrategy initialized: {len(self.pairs)} pairs, "
            f"daily_filter={self.daily_periods}, 4h_entry={self.entry_period_4h}, "
            f"4h_exit={self.exit_period_4h}, atr_stop={self.atr_stop_mult}, "
            f"risk={self.risk_per_trade_pct}%, vol_scale={self.vol_scale}, "
            f"shorts={'ON' if self.enable_shorts else 'OFF'}"
        )

    def start(self):
        """Load open positions from DB and begin monitoring."""
        self._running = True
        self._load_positions()
        self.logger.info(f"MTF Donchian strategy started: {len(self.positions)} open positions loaded")

    def stop(self):
        """Stop monitoring — positions remain open (managed by risk manager)."""
        self._running = False
        self.logger.info("MTF Donchian strategy stopped")

    def on_tick(self):
        """Called every tick interval — manage positions then scan for entries."""
        if not self._running:
            return

        self._manage_positions()
        self._scan_for_entries()

    # --- Daily regime filter ---

    def _fetch_daily_regime(self, symbol: str) -> str:
        """Determine daily trend regime using Donchian channel vote.

        Fetches 1d candles, computes Donchian high/low for each period in
        daily_periods, and votes: bullish if close > channel high, bearish
        if close < channel low. Returns "bullish", "bearish", or "neutral".

        Results are cached for 4h since daily data only changes once per day.
        """
        now = time.monotonic()
        cached = self._daily_cache.get(symbol)
        if cached and (now - cached[1]) < self._daily_cache_ttl:
            return cached[0]

        try:
            df = self.market_data.get_ohlcv(symbol, "1d", limit=150)
        except Exception as e:
            self.logger.error(f"Failed to fetch daily data for {symbol}: {e}")
            return "neutral"

        if len(df) < max(self.daily_periods) + 5:
            return "neutral"

        bullish_votes = 0
        bearish_votes = 0

        for period in self.daily_periods:
            high_channel = add_donchian_high(df, period)
            low_channel = add_donchian_low(df, period)

            # Use the last completed bar's channel values
            last_close = df["close"].iloc[-1]
            ch_high = high_channel.iloc[-1]
            ch_low = low_channel.iloc[-1]

            if pd.notna(ch_high) and last_close > ch_high:
                bullish_votes += 1
            if pd.notna(ch_low) and last_close < ch_low:
                bearish_votes += 1

        total = len(self.daily_periods)

        if bullish_votes >= self.daily_min_votes:
            regime = "bullish"
        elif bearish_votes >= self.daily_min_votes:
            regime = "bearish"
        else:
            regime = "neutral"

        self._daily_cache[symbol] = (regime, now)
        return regime

    # --- 4h data ---

    def _fetch_4h_data(self, symbol: str) -> pd.DataFrame | None:
        """Fetch 4h candles and compute Donchian indicators."""
        try:
            df = self.market_data.get_ohlcv(symbol, "4h", limit=200)
        except Exception as e:
            self.logger.error(f"Failed to fetch 4h data for {symbol}: {e}")
            return None

        if len(df) < self.vol_scale_lookback + 10:
            return None

        return compute_mtf_donchian_indicators(df, self._indicator_config)

    # --- Entry scanning ---

    def _scan_for_entries(self):
        """Scan all pairs for entry signals."""
        self._scan_count += 1
        log_summary = (self._scan_count % 5 == 0)

        with self._entry_lock:
            for symbol in self.pairs:
                # Skip if already have position in this pair
                if any(p["pair"] == symbol and p["status"] == "open" for p in self.positions):
                    continue

                # Check cooldown
                if self._is_in_cooldown(symbol):
                    continue

                try:
                    self._check_and_enter(symbol, log_summary)
                except Exception as e:
                    self.logger.error(f"Error scanning {symbol}: {e}", exc_info=True)

    def _check_and_enter(self, symbol: str, log: bool):
        """Check daily regime + 4h breakout for a single pair."""
        # Daily regime filter
        regime = self._fetch_daily_regime(symbol)

        if regime == "neutral":
            if log:
                self.logger.debug(f"MTF {symbol}: daily NEUTRAL, skip")
            return

        # Determine direction
        if regime == "bullish":
            side = "long"
        elif regime == "bearish":
            if not self.enable_shorts:
                if log:
                    self.logger.debug(f"MTF {symbol}: daily BEARISH but shorts disabled")
                return
            side = "short"
        else:
            return

        # Fetch 4h data with indicators
        df = self._fetch_4h_data(symbol)
        if df is None:
            return

        latest = df.iloc[-1]
        close = latest["close"]
        highest_high = latest["highest_high"]
        lowest_low = latest["lowest_low"]
        atr = latest["atr"]
        atr_median = latest["atr_median"]

        if pd.isna(atr) or pd.isna(atr_median) or atr <= 0:
            return

        # Check 4h breakout
        if side == "long":
            if pd.isna(highest_high) or close <= highest_high:
                if log:
                    hh_str = f"{highest_high:.4f}" if pd.notna(highest_high) else "NaN"
                    self.logger.debug(f"MTF {symbol}: daily BULL but no 4h breakout (close={close:.4f} <= HH={hh_str})")
                return
        else:  # short
            if pd.isna(lowest_low) or close >= lowest_low:
                if log:
                    ll_str = f"{lowest_low:.4f}" if pd.notna(lowest_low) else "NaN"
                    self.logger.debug(f"MTF {symbol}: daily BEAR but no 4h breakdown (close={close:.4f} >= LL={ll_str})")
                return

        # Signal confirmed — enter position
        self.logger.info(
            f"MTF SIGNAL: {side.upper()} {symbol} | daily={regime}, "
            f"close={close:.4f}, {'HH' if side == 'long' else 'LL'}="
            f"{highest_high if side == 'long' else lowest_low:.4f}, "
            f"ATR={atr:.4f}, ATR_median={atr_median:.4f}"
        )
        self._enter_position(symbol, side, close, atr, atr_median)

    # --- Position entry ---

    def _enter_position(self, symbol: str, side: str, close: float, atr: float, atr_median: float):
        """Open a new MTF Donchian position with vol-scaled sizing."""
        # Double-check DB for existing open position
        existing = self.db.fetch_one(
            "SELECT id FROM mtf_donchian_positions WHERE pair = ? AND status = 'open'",
            (symbol,),
        )
        if existing:
            self.logger.warning(f"Skipping entry for {symbol}: open position already exists in DB (id={existing['id']})")
            return

        # Vol-scaling
        if self.vol_scale and atr_median > 0 and atr > 0:
            vol_factor = max(self.vol_scale_min, min(self.vol_scale_max, atr_median / atr))
        else:
            vol_factor = 1.0

        effective_risk_pct = self.risk_per_trade_pct * vol_factor
        allocation_pct = self._allocation.get(symbol, 10.0) / 100.0

        # Calculate stop loss
        stop_distance = atr * self.atr_stop_mult
        if side == "long":
            stop_loss = close - stop_distance
        else:
            stop_loss = close + stop_distance

        risk_per_unit = abs(close - stop_loss)
        if risk_per_unit <= 0:
            self.logger.warning(f"Invalid stop distance for {symbol}, skipping")
            return

        # Position sizing
        balance = self.risk_manager.position_tracker.get_balance()
        capital = balance["total_usd"]
        available = balance["free_usd"]

        risk_amount = capital * allocation_pct * effective_risk_pct / 100.0
        amount = risk_amount / risk_per_unit

        # Capital cap: cost + fee <= available * 0.95
        fee_rate = self.futures_fee_rate if (side == "short" and self.futures_exchange) else self.fee_rate
        cost = amount * close
        fee = cost * fee_rate
        cap = available * 0.95

        if cost + fee > cap:
            amount = cap / (close * (1 + fee_rate))
            cost = amount * close
            fee = cost * fee_rate
            self.logger.info(f"MTF {symbol}: position capped to {amount:.6f} (available=${available:.2f})")

        if amount <= 0 or cost < 1.0:
            self.logger.info(f"MTF {symbol}: position too small (cost=${cost:.2f}), skip")
            return

        # Risk check — pass risk_amount as cost_usd (matches momentum pattern)
        order_request = OrderRequest(
            symbol=symbol,
            side="buy" if side == "long" else "sell",
            amount=amount,
            price=close,
            cost_usd=min(risk_amount, cost),
            strategy="mtf_donchian",
        )
        decision = self.risk_manager.check(order_request)

        if not decision.approved:
            self.logger.info(f"MTF entry rejected for {symbol}: {decision.reason}")
            return

        if decision.adjusted_amount:
            amount = decision.adjusted_amount

        # Place order
        if side == "short" and self.futures_exchange:
            trade = self._place_futures_order(symbol, "sell", amount, close)
        else:
            order_side = "buy" if side == "long" else "sell"
            trade = self.order_manager.place_order(symbol, order_side, amount, close, "mtf_donchian")

        if not trade:
            self.logger.error(f"Failed to enter MTF Donchian position for {symbol}")
            return

        # Record position in DB
        now = datetime.now(timezone.utc).isoformat()
        market_type = "futures" if (side == "short" and self.futures_exchange) else "spot"
        signals = {
            "side": side,
            "close": close,
            "atr": atr,
            "atr_median": atr_median,
            "vol_factor": vol_factor,
            "effective_risk_pct": effective_risk_pct,
        }

        cursor = self.db.execute(
            """INSERT INTO mtf_donchian_positions
            (pair, side, market_type, entry_price, entry_time, amount,
             stop_loss, current_stop, highest_since_entry, lowest_since_entry,
             entry_atr, vol_scale_factor, allocation_pct, entry_signals, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (
                symbol, side, market_type, trade["price"], now, trade["amount"],
                stop_loss, stop_loss,
                trade["price"] if side == "long" else None,
                trade["price"] if side == "short" else None,
                atr, vol_factor, allocation_pct * 100,
                json.dumps(signals),
            ),
        )

        position = {
            "id": cursor.lastrowid,
            "pair": symbol,
            "side": side,
            "market_type": market_type,
            "entry_price": trade["price"],
            "entry_time": now,
            "amount": trade["amount"],
            "stop_loss": stop_loss,
            "current_stop": stop_loss,
            "highest_since_entry": trade["price"] if side == "long" else None,
            "lowest_since_entry": trade["price"] if side == "short" else None,
            "entry_atr": atr,
            "status": "open",
        }
        self.positions.append(position)

        self.logger.info(
            f"MTF ENTRY: {side.upper()} {trade['amount']:.6f} {symbol} "
            f"@ {trade['price']:.4f} ({market_type}) | "
            f"stop={stop_loss:.4f}, vol_factor={vol_factor:.2f}, "
            f"risk=${risk_amount:.2f}, alloc={allocation_pct*100:.0f}%"
        )

    def _place_futures_order(self, symbol: str, side: str, amount: float, price: float) -> dict | None:
        """Place a futures order using the futures exchange."""
        try:
            order = self.futures_exchange.create_order(symbol, "market", side, amount)
            filled_price = order.get("average", order.get("price", price))
            filled_amount = order.get("filled", amount)
            fee_cost = 0.0
            fee_info = order.get("fee")
            if fee_info and isinstance(fee_info, dict):
                fee_cost = float(fee_info.get("cost", 0))

            # Record trade in DB
            now = datetime.now(timezone.utc).isoformat()
            cursor = self.db.execute(
                """INSERT INTO trades
                (timestamp, strategy, pair, side, order_type, price, amount,
                 cost_usd, fee_usd, exchange_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now, "mtf_donchian", symbol, side, "market",
                    filled_price, filled_amount,
                    filled_price * filled_amount, fee_cost,
                    order.get("id", ""),
                ),
            )

            return {
                "id": cursor.lastrowid,
                "price": filled_price,
                "amount": filled_amount,
                "fee": fee_cost,
            }
        except Exception as e:
            self.logger.error(f"Futures order failed for {symbol}: {e}", exc_info=True)
            return None

    # --- Position management ---

    def _manage_positions(self):
        """Update chandelier stops, check exit channels for open positions."""
        for position in list(self.positions):
            if position["status"] != "open":
                continue

            try:
                ticker = self.market_data.get_ticker(position["pair"])
                current_price = ticker["last"]
            except Exception as e:
                self.logger.error(f"Failed to get ticker for {position['pair']}: {e}")
                continue

            # Get current ATR for chandelier stop
            df = self._fetch_4h_data(position["pair"])

            # Update high/low tracking
            if position["side"] == "long":
                if position["highest_since_entry"] is None or current_price > position["highest_since_entry"]:
                    position["highest_since_entry"] = current_price
                    self.db.execute(
                        "UPDATE mtf_donchian_positions SET highest_since_entry = ? WHERE id = ?",
                        (current_price, position["id"]),
                    )
            else:
                if position["lowest_since_entry"] is None or current_price < position["lowest_since_entry"]:
                    position["lowest_since_entry"] = current_price
                    self.db.execute(
                        "UPDATE mtf_donchian_positions SET lowest_since_entry = ? WHERE id = ?",
                        (current_price, position["id"]),
                    )

            # Update chandelier stop if we have current ATR
            if df is not None and len(df) > 0:
                current_atr = df["atr"].iloc[-1]
                if pd.notna(current_atr) and current_atr > 0:
                    self._update_chandelier_stop(position, current_atr)

                # Check exit channel
                latest_ll = df["lowest_low"].iloc[-1]
                latest_hh = df["highest_high"].iloc[-1]

                if position["side"] == "long" and pd.notna(latest_ll):
                    if current_price < latest_ll:
                        self.logger.info(
                            f"MTF EXIT SIGNAL (channel): {position['pair']} LONG "
                            f"price={current_price:.4f} < LL={latest_ll:.4f}"
                        )
                        self._exit_position(position, current_price, "exit_channel")
                        continue
                elif position["side"] == "short" and pd.notna(latest_hh):
                    if current_price > latest_hh:
                        self.logger.info(
                            f"MTF EXIT SIGNAL (channel): {position['pair']} SHORT "
                            f"price={current_price:.4f} > HH={latest_hh:.4f}"
                        )
                        self._exit_position(position, current_price, "exit_channel")
                        continue

            # Check chandelier stop
            if self._check_stop(position, current_price):
                self._exit_position(position, current_price, "chandelier_stop")
                continue

    def _update_chandelier_stop(self, position: dict, current_atr: float):
        """Update trailing chandelier stop: never moves backward."""
        if position["side"] == "long":
            if position["highest_since_entry"] is None:
                return
            new_stop = position["highest_since_entry"] - self.atr_stop_mult * current_atr
            # Never move stop backward, and never below initial stop
            new_stop = max(new_stop, position["stop_loss"])
            if new_stop > position["current_stop"]:
                position["current_stop"] = new_stop
                self.db.execute(
                    "UPDATE mtf_donchian_positions SET current_stop = ? WHERE id = ?",
                    (new_stop, position["id"]),
                )
        else:
            if position["lowest_since_entry"] is None:
                return
            new_stop = position["lowest_since_entry"] + self.atr_stop_mult * current_atr
            # Never move stop backward (for shorts, backward means higher)
            new_stop = min(new_stop, position["stop_loss"])
            if new_stop < position["current_stop"]:
                position["current_stop"] = new_stop
                self.db.execute(
                    "UPDATE mtf_donchian_positions SET current_stop = ? WHERE id = ?",
                    (new_stop, position["id"]),
                )

    def _check_stop(self, position: dict, current_price: float) -> bool:
        """Check if chandelier stop has been hit."""
        if position["side"] == "long":
            return current_price <= position["current_stop"]
        else:
            return current_price >= position["current_stop"]

    # --- Position exit ---

    def _exit_position(self, position: dict, exit_price: float, reason: str):
        """Close a position and record results.

        Only marks position as closed if the exit order actually executes.
        If the order fails, position stays open and will retry next tick.
        """
        if position["side"] == "short" and position.get("market_type") == "futures" and self.futures_exchange:
            trade = self._place_futures_order(position["pair"], "buy", position["amount"], exit_price)
        else:
            exit_side = "sell" if position["side"] == "long" else "buy"
            trade = self.order_manager.place_order(
                position["pair"], exit_side, position["amount"],
                exit_price, "mtf_donchian",
            )

        if not trade:
            self.logger.critical(
                f"FAILED to exit MTF position {position['pair']} "
                f"({position['side']} {position['amount']:.6f} @ entry {position['entry_price']:.4f}). "
                f"Reason was: {reason}. Position remains OPEN — will retry next tick."
            )
            return

        actual_exit_price = trade["price"]

        # Calculate P&L
        if position["side"] == "long":
            pnl_usd = (actual_exit_price - position["entry_price"]) * position["amount"]
        else:
            pnl_usd = (position["entry_price"] - actual_exit_price) * position["amount"]

        # Subtract fees
        fee_rate = self.futures_fee_rate if position.get("market_type") == "futures" else self.fee_rate
        entry_fee = position["entry_price"] * position["amount"] * fee_rate
        exit_fee = actual_exit_price * position["amount"] * fee_rate
        pnl_usd -= (entry_fee + exit_fee)

        notional = position["entry_price"] * position["amount"]
        pnl_pct = (pnl_usd / notional * 100) if notional > 0 else 0.0

        # Update DB
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """UPDATE mtf_donchian_positions SET
                exit_price = ?, exit_time = ?, exit_reason = ?,
                pnl_usd = ?, pnl_pct = ?, status = 'closed'
            WHERE id = ?""",
            (actual_exit_price, now, reason, pnl_usd, pnl_pct, position["id"]),
        )

        # Update trade record with P&L
        self.db.execute(
            "UPDATE trades SET pnl_usd = ? WHERE id = ?",
            (pnl_usd, trade["id"]),
        )

        # Remove from active positions
        position["status"] = "closed"
        self.positions = [p for p in self.positions if p["status"] == "open"]

        win_loss = "WIN" if pnl_usd > 0 else "LOSS"
        self.logger.info(
            f"MTF EXIT ({win_loss}): {position['side'].upper()} {position['pair']} "
            f"@ {actual_exit_price:.4f} | reason={reason} | "
            f"P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)"
        )

    # --- Cooldown ---

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if a pair is in cooldown after a recent loss."""
        recent_loss = self.db.fetch_one(
            """SELECT exit_time FROM mtf_donchian_positions
            WHERE pair = ? AND status = 'closed' AND pnl_usd < 0
            ORDER BY exit_time DESC LIMIT 1""",
            (symbol,),
        )
        if not recent_loss or not recent_loss["exit_time"]:
            return False

        loss_time = datetime.fromisoformat(recent_loss["exit_time"])
        cooldown_expires = loss_time + timedelta(hours=self.cooldown_hours)
        return datetime.now(timezone.utc) < cooldown_expires

    # --- State loading ---

    def _load_positions(self):
        """Load open MTF Donchian positions from database."""
        rows = self.db.fetch_all(
            "SELECT * FROM mtf_donchian_positions WHERE status = 'open'"
        )
        self.positions = []
        for row in rows:
            self.positions.append({
                "id": row["id"],
                "pair": row["pair"],
                "side": row["side"],
                "market_type": row["market_type"],
                "entry_price": row["entry_price"],
                "entry_time": row["entry_time"],
                "amount": row["amount"],
                "stop_loss": row["stop_loss"],
                "current_stop": row["current_stop"],
                "highest_since_entry": row["highest_since_entry"],
                "lowest_since_entry": row["lowest_since_entry"],
                "entry_atr": row["entry_atr"],
                "status": row["status"],
            })
