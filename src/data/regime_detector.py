"""Market regime detection using BTC weekly data.

Classifies the crypto market as bull, bear, or sideways based on
price position relative to weekly EMA/SMA and trend strength (ADX).
Strategies query the regime to adjust position sizing and entry rules.
"""

import time

import pandas as pd

from src.core.logger import setup_logger
from src.data.indicators import add_adx, add_ema

logger = setup_logger("data.regime_detector")


class RegimeDetector:
    """Detect market regime from BTC weekly price data.

    Args:
        market_data: MarketDataManager for fetching OHLCV.
        config: regime_detection section from settings.yaml.
    """

    def __init__(self, market_data, config: dict):
        self.market_data = market_data
        self.reference_symbol = config.get("reference_symbol", "BTC/USDT")
        self.timeframe = config.get("timeframe", "1w")
        self.ema_period = config.get("ema_period", 20)
        self.sma_period = config.get("sma_period", 20)
        self.adx_period = config.get("adx_period", 14)
        self.adx_trending_threshold = config.get("adx_trending_threshold", 20)
        self.cache_ttl = config.get("cache_ttl_seconds", 3600)

        # Strategy behavior multipliers
        self.momentum_sideways_size_mult = config.get("momentum_sideways_size_mult", 0.7)
        self.grid_bear_size_mult = config.get("grid_bear_size_mult", 0.5)
        self.grid_bear_stop_loss_pct = config.get("grid_bear_stop_loss_pct", 5.0)

        # Cache
        self._cached_regime = None
        self._cache_time = 0

        logger.info(
            f"RegimeDetector initialized: {self.reference_symbol} {self.timeframe}, "
            f"EMA={self.ema_period}, SMA={self.sma_period}, ADX threshold={self.adx_trending_threshold}"
        )

    def get_regime(self) -> dict:
        """Get current market regime. Results are cached for cache_ttl seconds.

        Returns:
            Dict with keys: regime ('bull'|'bear'|'sideways'), confidence (0-1),
            price, ema, sma, adx.
        """
        now = time.time()
        if self._cached_regime and (now - self._cache_time) < self.cache_ttl:
            return self._cached_regime

        try:
            regime = self._detect()
            self._cached_regime = regime
            self._cache_time = now
            logger.info(
                f"Market regime: {regime['regime'].upper()} "
                f"(ADX={regime['adx']:.1f}, price=${regime['price']:,.0f}, "
                f"EMA20=${regime['ema']:,.0f}, SMA20=${regime['sma']:,.0f})"
            )
            return regime
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            # Default to sideways (safest fallback — no extreme actions)
            return {
                "regime": "sideways",
                "confidence": 0.0,
                "price": 0, "ema": 0, "sma": 0, "adx": 0,
            }

    def _detect(self) -> dict:
        """Run regime detection on weekly BTC data."""
        df = self.market_data.get_ohlcv(
            self.reference_symbol, self.timeframe, limit=max(self.ema_period + 10, 30)
        )

        if len(df) < self.ema_period + 5:
            logger.warning(f"Not enough weekly data ({len(df)} bars) for regime detection")
            return {"regime": "sideways", "confidence": 0.0, "price": 0, "ema": 0, "sma": 0, "adx": 0}

        # Compute indicators
        df["ema"] = add_ema(df, self.ema_period)
        df["sma"] = df["close"].rolling(window=self.sma_period).mean()
        df["adx"] = add_adx(df, self.adx_period)

        # Use the latest completed bar
        latest = df.iloc[-1]
        price = float(latest["close"])
        ema = float(latest["ema"])
        sma = float(latest["sma"])
        adx = float(latest["adx"]) if pd.notna(latest["adx"]) else 0.0

        above_ema = price > ema
        above_sma = price > sma
        trending = adx > self.adx_trending_threshold

        if above_ema and above_sma and trending:
            regime = "bull"
            confidence = min(adx / 40, 1.0)  # Higher ADX = more confident
        elif not above_ema and not above_sma and trending:
            regime = "bear"
            confidence = min(adx / 40, 1.0)
        else:
            regime = "sideways"
            confidence = 1.0 - min(adx / 40, 1.0)  # Low ADX = more confident sideways

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "price": price,
            "ema": ema,
            "sma": sma,
            "adx": adx,
        }
