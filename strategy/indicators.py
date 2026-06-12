import pandas as pd
import numpy as np
from config import StrategyConfig


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Add MA20, MA40, and CCI columns to an OHLCV DataFrame (in-place copy)."""
    df = df.copy()

    if cfg.ma_type == "EMA":
        df["ma_fast"] = df["close"].ewm(span=cfg.ma_fast, adjust=False).mean()
        df["ma_slow"] = df["close"].ewm(span=cfg.ma_slow, adjust=False).mean()
    else:
        df["ma_fast"] = df["close"].rolling(cfg.ma_fast).mean()
        df["ma_slow"] = df["close"].rolling(cfg.ma_slow).mean()

    df["cci"] = _cci(df, cfg.cci_period)
    return df


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical.rolling(period).mean()
    # Mean absolute deviation (pandas' .mad() is deprecated; compute manually)
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    # Guard against zero MAD to avoid divide-by-zero on flat bars
    cci = (typical - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return cci
