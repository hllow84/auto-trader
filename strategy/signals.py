import pandas as pd
from config import StrategyConfig


def add_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Adds boolean columns:
      long_signal  — §3.1 all five long conditions met on this bar
      short_signal — §3.3 all five short conditions met on this bar
    """
    df = df.copy()
    k = cfg.slope_lookback

    ma_f = df["ma_fast"]
    ma_s = df["ma_slow"]
    cci  = df["cci"]

    # Long setup (§3.1)
    long_trend      = ma_f > ma_s
    long_ma_rising  = (ma_f > ma_f.shift(k)) & (ma_s > ma_s.shift(k))
    long_cci        = cci < cfg.cci_long_threshold
    long_touch_ma   = df["low"] <= ma_f
    long_held_trend = df["close"] > ma_s

    df["long_signal"] = (
        long_trend & long_ma_rising & long_cci & long_touch_ma & long_held_trend
    )

    # Short setup (§3.3) — mirror
    short_trend      = ma_f < ma_s
    short_ma_falling = (ma_f < ma_f.shift(k)) & (ma_s < ma_s.shift(k))
    short_cci        = cci > cfg.cci_short_threshold
    short_touch_ma   = df["high"] >= ma_f
    short_held_trend = df["close"] < ma_s

    df["short_signal"] = (
        short_trend & short_ma_falling & short_cci & short_touch_ma & short_held_trend
    )

    return df
