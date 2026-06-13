"""
Relative strength filter (§5.3).

RS = stock_return(lookback) − benchmark_return(lookback)

Longs require RS > 0 (spec §5.3 only specifies this for longs).
In a single-stock test, RS > 0 means the stock outperformed the
benchmark over the lookback window.

For a multi-stock universe sweep, callers compute RS for every
candidate and rank them; only the top rs_top_n are eligible.
"""

import pandas as pd
from config import StrategyConfig


def add_rs(df: pd.DataFrame, benchmark_df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Adds an 'rs' column to df.
    benchmark_df must have a 'close' column indexed by the same dates (or a superset).
    """
    df = df.copy()
    lookback = cfg.rs_lookback

    stock_ret = df["close"].pct_change(lookback)
    bench_close = benchmark_df["close"].reindex(df.index).ffill()
    bench_ret = bench_close.pct_change(lookback)

    df["rs"] = stock_ret - bench_ret
    return df
