"""
Market regime gate (§5.1).

Fetches the 8-index basket, counts how many closed green that day,
and classifies each date as 'bullish', 'bearish', or 'neutral'.

bullish  = green_fraction >  threshold  → longs enabled, shorts disabled
bearish  = green_fraction <  threshold  → shorts enabled, longs disabled
neutral  = green_fraction == threshold  → neither enabled (e.g. 4/8 = exactly 50%)

International indices have different holiday calendars; missing dates are
forward-filled so every US trading day has a reading.
"""

import os
import pandas as pd
import yfinance as yf
from config import StrategyConfig


def build_regime_series(
    start: str,
    end: str,
    cfg: StrategyConfig,
    cache_dir: str = "data",
) -> pd.Series:
    """
    Returns a Series[str] indexed by date ('bullish' | 'bearish' | 'neutral').
    """
    tickers = cfg.regime_indices
    cache_path = os.path.join(cache_dir, f"regime_{start}_{end}.parquet")

    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)["regime"]

    print(f"  Downloading {len(tickers)} index series for regime gate...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance multi-ticker download: columns are (field, ticker) MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        # Single ticker returned as flat DataFrame — shouldn't happen with 8, but guard anyway
        closes = raw[["Close"]]

    # Drop entirely-empty ticker columns (index not available on yfinance)
    closes = closes.dropna(axis=1, how="all")
    available = closes.columns.tolist()
    if len(available) < len(tickers):
        missing = set(tickers) - set(available)
        print(f"  Warning: {missing} unavailable; regime computed from {len(available)} indices.")

    # green[date, ticker] = True if close > previous close, NaN if no data
    green = (closes > closes.shift(1)).astype(float)
    green[closes.isna()] = float("nan")

    # Forward-fill within each column to handle non-US holiday gaps
    green = green.ffill()

    # Fraction of available indices that closed green on each date
    green_count = green.sum(axis=1)
    total_count = green.notna().sum(axis=1).replace(0, float("nan"))
    frac = green_count / total_count

    threshold = cfg.regime_threshold
    regime = pd.Series("neutral", index=frac.index, name="regime", dtype=str)
    regime[frac > threshold] = "bullish"
    regime[frac < threshold] = "bearish"

    os.makedirs(cache_dir, exist_ok=True)
    regime.to_frame().to_parquet(cache_path)
    return regime


def join_regime(df: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """
    Left-joins regime onto df by date index.
    Missing dates (df has data, regime doesn't) are forward-filled then
    defaulted to 'bullish' so a missing index day never silently kills all signals.
    """
    df = df.copy()
    df["regime"] = regime.reindex(df.index).ffill().fillna("bullish")
    return df
