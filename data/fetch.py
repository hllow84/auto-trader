"""Download and cache OHLCV data via yfinance."""

import os
import pandas as pd
import yfinance as yf


def load_ohlcv(ticker: str, start: str, end: str, cache_dir: str = "data") -> pd.DataFrame:
    path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)
    return df.dropna()
