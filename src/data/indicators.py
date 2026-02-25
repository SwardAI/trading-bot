import pandas as pd
import ta

from src.core.logger import setup_logger

logger = setup_logger("data.indicators")


def add_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        df: DataFrame with OHLCV data.
        period: EMA period.
        column: Column to calculate EMA on.

    Returns:
        Series with EMA values.
    """
    return ta.trend.ema_indicator(df[column], window=period)


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        df: DataFrame with OHLCV data.
        period: RSI period.
        column: Column to calculate RSI on.

    Returns:
        Series with RSI values (0-100).
    """
    return ta.momentum.rsi(df[column], window=period)


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (trend strength).

    Args:
        df: DataFrame with OHLCV data (needs high, low, close).
        period: ADX period.

    Returns:
        Series with ADX values.
    """
    return ta.trend.adx(df["high"], df["low"], df["close"], window=period)


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, pd.Series]:
    """Calculate MACD line, signal line, and histogram.

    Args:
        df: DataFrame with OHLCV data.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        Dict with 'macd', 'signal', 'histogram' Series.
    """
    macd_obj = ta.trend.MACD(df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
    return {
        "macd": macd_obj.macd(),
        "signal": macd_obj.macd_signal(),
        "histogram": macd_obj.macd_diff(),
    }


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (volatility measure).

    Args:
        df: DataFrame with OHLCV data (needs high, low, close).
        period: ATR period.

    Returns:
        Series with ATR values.
    """
    return ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)


def add_donchian_high(df: pd.DataFrame, period: int) -> pd.Series:
    """Highest high over the lookback period, shifted by 1 bar (uses completed bars only)."""
    return df["high"].rolling(period).max().shift(1)


def add_donchian_low(df: pd.DataFrame, period: int) -> pd.Series:
    """Lowest low over the lookback period, shifted by 1 bar (uses completed bars only)."""
    return df["low"].rolling(period).min().shift(1)


def compute_mtf_donchian_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add MTF Donchian strategy indicators to a 4h DataFrame.

    Adds: highest_high (entry channel), lowest_low (exit channel), atr, atr_median.

    Args:
        df: DataFrame with OHLCV data (4h timeframe).
        config: Dict with entry_period_4h, exit_period_4h, atr_period, vol_scale_lookback.

    Returns:
        DataFrame with indicator columns added.
    """
    df = df.copy()

    entry_period = config.get("entry_period_4h", 14)
    exit_period = config.get("exit_period_4h", 7)
    atr_period = config.get("atr_period", 14)
    vol_lookback = config.get("vol_scale_lookback", 60)

    df["highest_high"] = add_donchian_high(df, entry_period)
    df["lowest_low"] = add_donchian_low(df, exit_period)
    df["atr"] = add_atr(df, atr_period)
    df["atr_median"] = df["atr"].rolling(vol_lookback).median()

    return df


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average of volume.

    Args:
        df: DataFrame with OHLCV data.
        period: SMA period.

    Returns:
        Series with volume SMA values.
    """
    return df["volume"].rolling(window=period).mean()


def compute_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all momentum strategy indicators to a DataFrame.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume.
        config: Momentum strategy config dict with indicator parameters.

    Returns:
        DataFrame with all indicator columns added.
    """
    df = df.copy()

    # EMAs
    ema_fast = config.get("ema_fast", 9)
    ema_slow = config.get("ema_slow", 21)
    df["ema_fast"] = add_ema(df, ema_fast)
    df["ema_slow"] = add_ema(df, ema_slow)

    # RSI
    rsi_period = config.get("rsi_period", 14)
    df["rsi"] = add_rsi(df, rsi_period)

    # ADX
    adx_period = config.get("adx_period", 14)
    df["adx"] = add_adx(df, adx_period)

    # MACD
    macd = add_macd(df)
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_histogram"] = macd["histogram"]

    # ATR
    atr_period = config.get("atr_period", 14)
    df["atr"] = add_atr(df, atr_period)

    # Volume SMA
    df["volume_sma"] = add_volume_sma(df, 20)

    return df
