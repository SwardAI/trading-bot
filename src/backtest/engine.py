from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.core.logger import setup_logger
from src.data.indicators import compute_all_indicators

logger = setup_logger("backtest.engine")


@dataclass
class BacktestTrade:
    """A single trade in the backtest."""
    timestamp: datetime
    side: str  # "buy" or "sell"
    price: float
    amount: float
    cost: float
    fee: float
    pnl: float = 0.0
    strategy: str = ""


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_pnl: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """Backtesting engine with realistic fee and slippage modeling.

    Simulates strategy execution on historical data, applying:
    - Maker/taker fees per trade
    - Configurable slippage
    - Position tracking

    Args:
        initial_capital: Starting capital in USD.
        maker_fee: Maker fee percentage (default 0.075%).
        taker_fee: Taker fee percentage (default 0.1%).
        slippage_pct: Slippage percentage per trade (default 0.03%).
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        maker_fee: float = 0.075,
        taker_fee: float = 0.1,
        slippage_pct: float = 0.03,
    ):
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee / 100
        self.taker_fee = taker_fee / 100
        self.slippage_pct = slippage_pct / 100

    def run_grid_backtest(
        self,
        df: pd.DataFrame,
        grid_spacing_pct: float = 0.5,
        order_size_usd: float = 50,
        num_grids: int = 20,
        rebalance_trigger_pct: float = 8,
    ) -> BacktestResult:
        """Backtest the grid strategy on historical data.

        Args:
            df: OHLCV DataFrame.
            grid_spacing_pct: Grid spacing percentage.
            order_size_usd: USD per grid order.
            num_grids: Number of grid levels per side.
            rebalance_trigger_pct: Grid rebalance trigger.

        Returns:
            BacktestResult with full statistics.
        """
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        inventory = 0.0
        inventory_avg = 0.0

        # Initial grid center
        center = df.iloc[0]["close"]
        grid_buys = self._calc_grid_prices(center, grid_spacing_pct, num_grids, "buy")
        grid_sells = self._calc_grid_prices(center, grid_spacing_pct, num_grids, "sell")

        filled_buys = set()  # Prices where we have inventory

        for i in range(1, len(df)):
            row = df.iloc[i]
            low = row["low"]
            high = row["high"]
            close = row["close"]

            # Check buy fills (price dropped to grid level)
            for price in grid_buys:
                if price in filled_buys:
                    continue
                if low <= price:
                    # Simulate fill with slippage
                    fill_price = price * (1 + self.slippage_pct)
                    amount = order_size_usd / fill_price
                    fee = order_size_usd * self.maker_fee

                    if capital >= order_size_usd + fee:
                        capital -= (order_size_usd + fee)
                        total_cost = (inventory * inventory_avg) + (amount * fill_price)
                        inventory += amount
                        inventory_avg = total_cost / inventory if inventory > 0 else 0
                        filled_buys.add(price)

                        trades.append(BacktestTrade(
                            timestamp=row["timestamp"], side="buy",
                            price=fill_price, amount=amount,
                            cost=order_size_usd, fee=fee, strategy="grid",
                        ))

            # Check sell fills (price rose to grid level above a filled buy)
            for buy_price in list(filled_buys):
                sell_price = buy_price * (1 + grid_spacing_pct / 100)
                if high >= sell_price:
                    fill_price = sell_price * (1 - self.slippage_pct)
                    amount = order_size_usd / buy_price  # Same amount as the buy
                    revenue = amount * fill_price
                    fee = revenue * self.maker_fee

                    capital += (revenue - fee)
                    pnl = (fill_price - inventory_avg) * amount - fee - (order_size_usd * self.maker_fee)

                    inventory -= amount
                    if inventory <= 0:
                        inventory = 0
                        inventory_avg = 0

                    filled_buys.discard(buy_price)

                    trades.append(BacktestTrade(
                        timestamp=row["timestamp"], side="sell",
                        price=fill_price, amount=amount,
                        cost=revenue, fee=fee, pnl=pnl, strategy="grid",
                    ))

            # Rebalance check
            drift = abs(close - center) / center * 100
            if drift >= rebalance_trigger_pct:
                center = close
                grid_buys = self._calc_grid_prices(center, grid_spacing_pct, num_grids, "buy")
                grid_sells = self._calc_grid_prices(center, grid_spacing_pct, num_grids, "sell")
                filled_buys.clear()

            # Track equity (capital + inventory value)
            equity = capital + (inventory * close)
            equity_curve.append(equity)

        # Final equity
        final_close = df.iloc[-1]["close"]
        final_capital = capital + (inventory * final_close)

        return self._build_result(
            "grid", df, trades, equity_curve, final_capital,
        )

    def run_momentum_backtest(
        self,
        df: pd.DataFrame,
        config: dict | None = None,
        long_only: bool = True,
    ) -> BacktestResult:
        """Backtest the momentum strategy on historical data.

        Args:
            df: OHLCV DataFrame (should be at least 100 rows).
            config: Momentum config dict (uses defaults if None).
            long_only: If True, skip short signals (matches spot-only live bot).

        Returns:
            BacktestResult with full statistics.
        """
        if config is None:
            config = {}

        ema_fast = config.get("ema_fast", 9)
        ema_slow = config.get("ema_slow", 21)
        rsi_long = config.get("rsi_long_threshold", 55)
        rsi_short = config.get("rsi_short_threshold", 45)
        adx_min = config.get("adx_min_strength", 25)
        vol_mult = config.get("volume_surge_multiplier", 1.5)
        atr_mult = config.get("trailing_stop_atr_multiplier", 1.5)
        risk_pct = config.get("risk_per_trade_pct", 1.0)

        # Configurable signal requirements — which signals must be present to enter
        # Default: all signals required (original behavior)
        required = set(config.get("required_signals", ["ema", "rsi", "adx", "volume", "macd"]))

        # Compute indicators
        df = compute_all_indicators(df, config)

        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        position = None  # {side, entry_price, amount, stop_loss, current_stop}

        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            close = row["close"]

            # Skip if indicators not ready
            if pd.isna(row["ema_fast"]) or pd.isna(row["adx"]) or pd.isna(row["atr"]):
                equity_curve.append(capital)
                continue

            # Manage existing position
            if position:
                # Check trailing stop
                if position["side"] == "long" and close <= position["current_stop"]:
                    pnl = self._close_position(position, close, trades, row, capital)
                    capital += pnl
                    position = None
                elif position["side"] == "short" and close >= position["current_stop"]:
                    pnl = self._close_position(position, close, trades, row, capital)
                    capital += pnl
                    position = None
                else:
                    # Update trailing stop
                    self._update_backtest_stop(position, close, atr_mult)

                equity = capital + (self._position_value(position, close) if position else 0)
                equity_curve.append(equity)
                continue

            # Check entry signals — each can be required or optional based on config
            ema_cross_long = row["ema_fast"] > row["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]
            ema_cross_short = row["ema_fast"] < row["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]
            rsi_ok_long = row["rsi"] > rsi_long if not pd.isna(row["rsi"]) else False
            rsi_ok_short = row["rsi"] < rsi_short if not pd.isna(row["rsi"]) else False
            adx_ok = row["adx"] > adx_min if not pd.isna(row["adx"]) else False
            vol_ok = row["volume"] > (row["volume_sma"] * vol_mult) if not pd.isna(row["volume_sma"]) and row["volume_sma"] > 0 else False
            macd_long = not pd.isna(row["macd_histogram"]) and not pd.isna(prev["macd_histogram"]) and row["macd_histogram"] > 0 and row["macd_histogram"] > prev["macd_histogram"]
            macd_short = not pd.isna(row["macd_histogram"]) and not pd.isna(prev["macd_histogram"]) and row["macd_histogram"] < 0 and row["macd_histogram"] < prev["macd_histogram"]

            # Build long entry check based on required signals
            long_ok = (ema_cross_long if "ema" in required else True)
            long_ok = long_ok and (rsi_ok_long if "rsi" in required else True)
            long_ok = long_ok and (adx_ok if "adx" in required else True)
            long_ok = long_ok and (vol_ok if "volume" in required else True)
            long_ok = long_ok and (macd_long if "macd" in required else True)

            # Must have at least EMA cross to trigger entry (always required as base signal)
            if long_ok and ema_cross_long:
                # Enter long
                atr = row["atr"]
                stop_loss = close - (atr * atr_mult)
                risk_amount = capital * (risk_pct / 100)
                risk_per_unit = close - stop_loss
                if risk_per_unit > 0:
                    amount = risk_amount / risk_per_unit
                    fee = amount * close * self.taker_fee
                    position = {
                        "side": "long", "entry_price": close,
                        "amount": amount, "stop_loss": stop_loss,
                        "current_stop": stop_loss, "initial_risk": risk_per_unit,
                        "atr": atr,
                    }
                    capital -= fee
                    trades.append(BacktestTrade(
                        timestamp=row["timestamp"], side="buy",
                        price=close, amount=amount,
                        cost=amount * close, fee=fee, strategy="momentum",
                    ))

            elif not long_only and ema_cross_short:
                short_ok = (rsi_ok_short if "rsi" in required else True)
                short_ok = short_ok and (adx_ok if "adx" in required else True)
                short_ok = short_ok and (vol_ok if "volume" in required else True)
                short_ok = short_ok and (macd_short if "macd" in required else True)
                if not short_ok:
                    equity = capital + (self._position_value(position, close) if position else 0)
                    equity_curve.append(equity)
                    continue
                # Enter short (skipped in long_only mode)
                atr = row["atr"]
                stop_loss = close + (atr * atr_mult)
                risk_amount = capital * (risk_pct / 100)
                risk_per_unit = stop_loss - close
                if risk_per_unit > 0:
                    amount = risk_amount / risk_per_unit
                    fee = amount * close * self.taker_fee
                    position = {
                        "side": "short", "entry_price": close,
                        "amount": amount, "stop_loss": stop_loss,
                        "current_stop": stop_loss, "initial_risk": risk_per_unit,
                        "atr": atr,
                    }
                    capital -= fee
                    trades.append(BacktestTrade(
                        timestamp=row["timestamp"], side="sell",
                        price=close, amount=amount,
                        cost=amount * close, fee=fee, strategy="momentum",
                    ))

            equity = capital + (self._position_value(position, close) if position else 0)
            equity_curve.append(equity)

        # Close any remaining position
        if position:
            final_close = df.iloc[-1]["close"]
            pnl = self._close_position(position, final_close, trades, df.iloc[-1], capital)
            capital += pnl

        final_capital = capital

        return self._build_result(
            "momentum", df, trades, equity_curve, final_capital,
        )

    def run_breakout_backtest(
        self,
        df: pd.DataFrame,
        config: dict | None = None,
        long_only: bool = True,
    ) -> BacktestResult:
        """Backtest a Donchian breakout strategy on historical data.

        Noise-resistant trend following: enters when price breaks above the
        N-period highest high, exits via chandelier stop (highest high - K*ATR)
        or when price drops below M-period lowest low.

        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume.
            config: Strategy config dict. Keys:
                - breakout_period: Lookback for entry channel (default 20)
                - exit_period: Lookback for exit channel (default 10)
                - atr_period: ATR calculation period (default 14)
                - atr_stop_mult: Chandelier stop multiplier (default 3.0)
                - adx_period: ADX period (default 14)
                - adx_min: Minimum ADX for trend confirmation (default 20)
                - volume_period: Volume MA period (default 20)
                - volume_mult: Volume surge multiplier (default 1.0, 1.0=no filter)
                - risk_per_trade_pct: % of capital risked per trade (default 1.0)
                - cooldown_bars: Bars to wait after exit before re-entry (default 5)
            long_only: If True, skip short signals.

        Returns:
            BacktestResult with full statistics.
        """
        import numpy as np

        if config is None:
            config = {}

        breakout_period = config.get("breakout_period", 20)
        exit_period = config.get("exit_period", 10)
        atr_period = config.get("atr_period", 14)
        atr_stop_mult = config.get("atr_stop_mult", 3.0)
        adx_period = config.get("adx_period", 14)
        adx_min = config.get("adx_min", 20)
        volume_period = config.get("volume_period", 20)
        volume_mult = config.get("volume_mult", 1.0)
        risk_pct = config.get("risk_per_trade_pct", 1.0)
        cooldown_bars = config.get("cooldown_bars", 5)

        # Pre-compute indicators using vectorized operations
        import ta

        df = df.copy()
        df["highest_high"] = df["high"].rolling(window=breakout_period).max()
        df["lowest_low"] = df["low"].rolling(window=exit_period).min()
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=atr_period,
        )
        df["adx"] = ta.trend.adx(
            df["high"], df["low"], df["close"], window=adx_period,
        )
        df["volume_ma"] = df["volume"].rolling(window=volume_period).mean()

        # Also compute the previous bar's highest high for breakout detection
        df["prev_highest"] = df["highest_high"].shift(1)
        df["prev_lowest"] = df["lowest_low"].shift(1)

        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        position = None  # {side, entry_price, amount, stop_loss, highest_since_entry}
        bars_since_exit = cooldown_bars  # Start ready to trade

        lookback = max(breakout_period, exit_period, atr_period, adx_period, volume_period)

        for i in range(lookback + 1, len(df)):
            row = df.iloc[i]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr = row["atr"]

            # Skip if indicators not ready
            if pd.isna(atr) or pd.isna(row["adx"]) or pd.isna(row["prev_highest"]):
                equity_curve.append(capital)
                continue

            # Manage existing position
            if position:
                # Update highest price since entry (for chandelier stop)
                if position["side"] == "long":
                    if high > position["highest_since_entry"]:
                        position["highest_since_entry"] = high
                    # Chandelier stop: highest high since entry - K * ATR
                    chandelier_stop = position["highest_since_entry"] - atr_stop_mult * atr
                    # Use the better (higher) of chandelier or initial stop
                    effective_stop = max(chandelier_stop, position["stop_loss"])
                    # Also check exit channel (close below exit_period low)
                    exit_channel = row["prev_lowest"] if not pd.isna(row["prev_lowest"]) else 0

                    if close <= effective_stop or close < exit_channel:
                        # Exit
                        exit_price = close
                        amount = position["amount"]
                        fee = amount * exit_price * self.taker_fee
                        pnl = (exit_price - position["entry_price"]) * amount - fee
                        capital += pnl
                        trades.append(BacktestTrade(
                            timestamp=row["timestamp"], side="sell",
                            price=exit_price, amount=amount,
                            cost=amount * exit_price, fee=fee, pnl=pnl,
                            strategy="breakout",
                        ))
                        position = None
                        bars_since_exit = 0

                elif position["side"] == "short":
                    if low < position["lowest_since_entry"]:
                        position["lowest_since_entry"] = low
                    chandelier_stop = position["lowest_since_entry"] + atr_stop_mult * atr
                    effective_stop = min(chandelier_stop, position["stop_loss"])
                    exit_channel = row["prev_highest"] if not pd.isna(row["prev_highest"]) else float("inf")

                    if close >= effective_stop or close > exit_channel:
                        exit_price = close
                        amount = position["amount"]
                        fee = amount * exit_price * self.taker_fee
                        pnl = (position["entry_price"] - exit_price) * amount - fee
                        capital += pnl
                        trades.append(BacktestTrade(
                            timestamp=row["timestamp"], side="buy",
                            price=exit_price, amount=amount,
                            cost=amount * exit_price, fee=fee, pnl=pnl,
                            strategy="breakout",
                        ))
                        position = None
                        bars_since_exit = 0

                equity = capital + (self._position_value(position, close) if position else 0)
                equity_curve.append(equity)
                continue

            # Cooldown after exit
            bars_since_exit += 1
            if bars_since_exit < cooldown_bars:
                equity_curve.append(capital)
                continue

            # Check entry signals
            adx_ok = row["adx"] > adx_min
            vol_ok = (
                row["volume"] > row["volume_ma"] * volume_mult
                if not pd.isna(row["volume_ma"]) and row["volume_ma"] > 0
                else True
            )

            # Long breakout: close breaks above previous period's highest high
            prev_high = row["prev_highest"]
            if close > prev_high and adx_ok and vol_ok:
                # Enter long
                stop_loss = close - atr_stop_mult * atr
                risk_per_unit = close - stop_loss
                if risk_per_unit > 0:
                    risk_amount = capital * (risk_pct / 100)
                    amount = risk_amount / risk_per_unit
                    cost = amount * close
                    fee = cost * self.taker_fee
                    if cost + fee <= capital:
                        capital -= fee
                        position = {
                            "side": "long",
                            "entry_price": close,
                            "amount": amount,
                            "stop_loss": stop_loss,
                            "highest_since_entry": high,
                            "atr": atr,
                        }
                        trades.append(BacktestTrade(
                            timestamp=row["timestamp"], side="buy",
                            price=close, amount=amount,
                            cost=cost, fee=fee, strategy="breakout",
                        ))

            # Short breakout: close breaks below previous period's lowest low
            elif not long_only:
                prev_low = row["prev_lowest"]
                if not pd.isna(prev_low) and close < prev_low and adx_ok and vol_ok:
                    stop_loss = close + atr_stop_mult * atr
                    risk_per_unit = stop_loss - close
                    if risk_per_unit > 0:
                        risk_amount = capital * (risk_pct / 100)
                        amount = risk_amount / risk_per_unit
                        cost = amount * close
                        fee = cost * self.taker_fee
                        if cost + fee <= capital:
                            capital -= fee
                            position = {
                                "side": "short",
                                "entry_price": close,
                                "amount": amount,
                                "stop_loss": stop_loss,
                                "lowest_since_entry": low,
                                "atr": atr,
                            }
                            trades.append(BacktestTrade(
                                timestamp=row["timestamp"], side="sell",
                                price=close, amount=amount,
                                cost=cost, fee=fee, strategy="breakout",
                            ))

            equity = capital + (self._position_value(position, close) if position else 0)
            equity_curve.append(equity)

        # Close any remaining position
        if position:
            final_close = df.iloc[-1]["close"]
            amount = position["amount"]
            fee = amount * final_close * self.taker_fee
            if position["side"] == "long":
                pnl = (final_close - position["entry_price"]) * amount - fee
            else:
                pnl = (position["entry_price"] - final_close) * amount - fee
            capital += pnl
            trades.append(BacktestTrade(
                timestamp=df.iloc[-1]["timestamp"],
                side="sell" if position["side"] == "long" else "buy",
                price=final_close, amount=amount,
                cost=amount * final_close, fee=fee, pnl=pnl,
                strategy="breakout",
            ))

        final_capital = capital

        return self._build_result(
            "breakout", df, trades, equity_curve, final_capital,
        )

    def _calc_grid_prices(self, center: float, spacing_pct: float, num: int, side: str) -> list[float]:
        """Calculate grid price levels."""
        prices = []
        for i in range(1, num + 1):
            if side == "buy":
                prices.append(round(center / ((1 + spacing_pct / 100) ** i), 2))
            else:
                prices.append(round(center * ((1 + spacing_pct / 100) ** i), 2))
        return prices

    def _close_position(self, position: dict, price: float, trades: list, row, capital: float) -> float:
        """Close a position and return the P&L (including fees)."""
        amount = position["amount"]
        fee = amount * price * self.taker_fee

        if position["side"] == "long":
            pnl = (price - position["entry_price"]) * amount - fee
        else:
            pnl = (position["entry_price"] - price) * amount - fee

        exit_side = "sell" if position["side"] == "long" else "buy"
        trades.append(BacktestTrade(
            timestamp=row["timestamp"], side=exit_side,
            price=price, amount=amount,
            cost=amount * price, fee=fee, pnl=pnl, strategy="momentum",
        ))
        return pnl

    def _position_value(self, position: dict | None, current_price: float) -> float:
        """Calculate unrealized P&L of a position."""
        if not position:
            return 0
        if position["side"] == "long":
            return (current_price - position["entry_price"]) * position["amount"]
        else:
            return (position["entry_price"] - current_price) * position["amount"]

    def _update_backtest_stop(self, position: dict, current_price: float, atr_mult: float):
        """Update trailing stop in backtest."""
        entry = position["entry_price"]
        initial_risk = position["initial_risk"]
        atr = position.get("atr", initial_risk / atr_mult)

        if position["side"] == "long":
            profit = current_price - entry
            mult = profit / initial_risk if initial_risk > 0 else 0
            if mult >= 3:
                new_stop = current_price - atr * 0.8
            elif mult >= 2:
                new_stop = current_price - atr * 1.0
            elif mult >= 1:
                new_stop = current_price - atr * 1.2
            else:
                return
            if new_stop > position["current_stop"]:
                position["current_stop"] = new_stop
        else:
            profit = entry - current_price
            mult = profit / initial_risk if initial_risk > 0 else 0
            if mult >= 3:
                new_stop = current_price + atr * 0.8
            elif mult >= 2:
                new_stop = current_price + atr * 1.0
            elif mult >= 1:
                new_stop = current_price + atr * 1.2
            else:
                return
            if new_stop < position["current_stop"]:
                position["current_stop"] = new_stop

    def _build_result(
        self,
        strategy: str,
        df: pd.DataFrame,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        final_capital: float,
    ) -> BacktestResult:
        """Build BacktestResult from trade data."""
        sell_trades = [t for t in trades if t.pnl != 0]
        wins = [t for t in sell_trades if t.pnl > 0]
        losses = [t for t in sell_trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                       for i in range(1, len(equity_curve)) if equity_curve[i - 1] > 0]
            if returns:
                import numpy as np
                avg_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = (avg_ret / std_ret * (365 ** 0.5)) if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        return BacktestResult(
            strategy=strategy,
            symbol=df.iloc[0].get("symbol", "unknown") if "symbol" in df.columns else "unknown",
            timeframe="unknown",
            start_date=str(df.iloc[0]["timestamp"]),
            end_date=str(df.iloc[-1]["timestamp"]),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(sell_trades) * 100 if sell_trades else 0,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=gross_profit - gross_loss,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            avg_trade_pnl=(gross_profit - gross_loss) / len(sell_trades) if sell_trades else 0,
            trades=trades,
            equity_curve=equity_curve,
        )
