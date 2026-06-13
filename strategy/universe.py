"""
Universe filter (§5.2).

Adds a boolean 'passes_filter' column to a daily OHLCV DataFrame.
For single-stock backtests this gates signal generation bar-by-bar.
For multi-stock universe sweeps it pre-screens candidates.

Conditions (§5.2):
  - Last close > min_price  (default $0.50)
  - 20-day average daily volume > min_avg_volume  (default 500,000)
"""

import pandas as pd
from config import StrategyConfig


def add_universe_filter(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    price_ok = df["close"] > cfg.min_price
    # min_periods=1 so early bars aren't all-NaN; they'll have a short average
    vol_ok = df["volume"].rolling(20, min_periods=1).mean() > cfg.min_avg_volume
    df["passes_filter"] = price_ok & vol_ok
    return df
