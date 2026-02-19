import os
import time
from typing import Any

import ccxt
import pandas as pd

from src.core.logger import setup_logger

logger = setup_logger(__name__)


class ExchangeManager:
    """Wrapper around ccxt for unified exchange access with rate limiting and error handling.

    Args:
        exchange_id: Exchange name (e.g. "binance").
        config: Exchange config dict from settings.yaml (api_key_env, sandbox, rate_limit_ms, etc).
    """

    def __init__(self, exchange_id: str, config: dict):
        self.exchange_id = exchange_id
        self.config = config

        # Resolve API keys from env
        api_key = os.getenv(config.get("api_key_env", ""))
        api_secret = os.getenv(config.get("api_secret_env", ""))

        # Initialize ccxt exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": config.get("sandbox", True),
            "enableRateLimit": True,
            "rateLimit": config.get("rate_limit_ms", 100),
            "options": {"defaultType": "spot"},
        })

        sandbox_label = "SANDBOX" if config.get("sandbox", True) else "LIVE"
        logger.info(f"Exchange initialized: {exchange_id} ({sandbox_label})")

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data for a symbol.

        Returns:
            Dict with keys: bid, ask, last, baseVolume, quoteVolume, etc.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candle data as a pandas DataFrame.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT").
            timeframe: Candle timeframe (e.g. "1m", "5m", "1h", "1d").
            limit: Number of candles to fetch.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch OHLCV for {symbol} {timeframe}: {e}")
            raise

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Fetch orderbook data.

        Returns:
            Dict with keys: bids, asks (each a list of [price, amount]).
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise

    def fetch_balance(self) -> dict:
        """Fetch account balance.

        Returns:
            ccxt balance structure with 'total', 'free', 'used' dicts.
        """
        try:
            balance = self.exchange.fetch_balance()
            logger.debug(f"Balance fetched: {balance.get('total', {}).get('USDT', 0)} USDT total")
            return balance
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        """Place an order on the exchange.

        Args:
            symbol: Trading pair.
            order_type: "limit" or "market".
            side: "buy" or "sell".
            amount: Order amount in base currency.
            price: Limit price (required for limit orders).

        Returns:
            Exchange order response dict.
        """
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            logger.info(
                f"Order placed: {side} {amount} {symbol} @ {price or 'market'} "
                f"(type={order_type}, id={order['id']})"
            )
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for {side} {amount} {symbol}: {e}")
            raise
        except ccxt.BaseError as e:
            logger.error(f"Failed to create order {side} {amount} {symbol}: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order.

        Args:
            order_id: Exchange order ID.
            symbol: Trading pair.

        Returns:
            Exchange cancellation response.
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id} ({symbol})")
            return result
        except ccxt.OrderNotFound:
            logger.warning(f"Order not found for cancellation: {order_id} ({symbol})")
            raise
        except ccxt.BaseError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch all open orders, optionally filtered by symbol.

        Returns:
            List of open order dicts.
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logger.debug(f"Open orders fetched: {len(orders)} for {symbol or 'all'}")
            return orders
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch open orders for {symbol}: {e}")
            raise

    def fetch_my_trades(self, symbol: str, since: int | None = None, limit: int = 50) -> list[dict]:
        """Fetch recent trades for the account.

        Args:
            symbol: Trading pair.
            since: Timestamp in ms to fetch trades from.
            limit: Max number of trades to return.

        Returns:
            List of trade dicts.
        """
        try:
            trades = self.exchange.fetch_my_trades(symbol, since=since, limit=limit)
            return trades
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch trades for {symbol}: {e}")
            raise

    def load_markets(self):
        """Load market data (call once on startup to populate symbol info)."""
        try:
            self.exchange.load_markets()
            logger.info(f"Markets loaded: {len(self.exchange.markets)} symbols available")
        except ccxt.BaseError as e:
            logger.error(f"Failed to load markets: {e}")
            raise
