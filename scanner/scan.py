"""
Multi-stock universe scanner — core logic.

For each ticker: load data, apply optional filter layers,
compute indicators + signals, and return candidates that
fired a signal on the last available bar.
"""

import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import StrategyConfig, BacktestConfig
from data.fetch import load_ohlcv
from strategy.indicators import add_indicators
from strategy.signals import add_signals
from strategy.universe import add_universe_filter
from strategy.rs import add_rs
from strategy.regime import build_regime_series, join_regime


def _scan_one(
    ticker: str,
    start: str,
    end: str,
    strat: StrategyConfig,
    bt: BacktestConfig,
    bench_df: Optional[pd.DataFrame],
    regime_series: Optional[pd.Series],
) -> dict | None:
    """
    Scan a single ticker.  Returns a signal dict or None (no signal / no data).
    Exceptions propagate so the thread pool can count them as errors.
    """
    df = load_ohlcv(ticker, start, end, bt.data_dir)

    if len(df) < strat.ma_slow + 5:
        return None

    if strat.use_universe_filter:
        df = add_universe_filter(df, strat)
    else:
        df["passes_filter"] = True

    if regime_series is not None:
        df = join_regime(df, regime_series)
    else:
        df["regime"] = "bullish"

    # Always compute RS for display / ranking even when filter is off
    if bench_df is not None:
        df = add_rs(df, bench_df, strat)
    else:
        df["rs"] = float("nan")

    df = add_indicators(df, strat)
    df = add_signals(df, strat)
    df["avg_vol_20"] = df["volume"].rolling(20, min_periods=1).mean()

    row = df.iloc[-1]

    if pd.isna(row["ma_fast"]) or pd.isna(row["cci"]):
        return None

    passes = bool(row["passes_filter"])
    regime = str(row["regime"])
    rs     = float(row["rs"]) if not pd.isna(row["rs"]) else float("nan")

    tick = 0.01 if row["close"] >= 1.0 else 0.001

    long_ok = (
        passes
        and (not strat.use_regime_gate or regime != "bearish")
        and (not strat.use_rs_filter   or (not math.isnan(rs) and rs > 0))
    )
    short_ok = (
        passes
        and (not strat.use_regime_gate or regime != "bullish")
    )

    direction = entry_stop = initial_stop = None

    if row["long_signal"] and long_ok:
        direction    = "long"
        entry_stop   = round(float(row["high"]) + tick, 2)
        initial_stop = round(float(row["low"])  - tick, 2)
    elif row["short_signal"] and short_ok:
        direction    = "short"
        entry_stop   = round(float(row["low"])  - tick, 2)
        initial_stop = round(float(row["high"]) + tick, 2)

    if direction is None:
        return None

    idx = df.index[-1]
    scan_date = str(idx.date()) if hasattr(idx, "date") else str(idx)

    return {
        "ticker":       ticker,
        "direction":    direction,
        "entry_stop":   entry_stop,
        "initial_stop": initial_stop,
        "rs":           rs,
        "close":        round(float(row["close"]), 2),
        "avg_vol_m":    float(row["avg_vol_20"]) / 1_000_000,
        "regime":       regime,
        "scan_date":    scan_date,
    }


def scan_universe(
    tickers: list[str],
    start: str,
    end: str,
    strat: StrategyConfig,
    bt: BacktestConfig,
    workers: int = 8,
) -> tuple[list[dict], dict]:
    """
    Scan all tickers in parallel.

    Returns:
        signals  — list of signal dicts (one per ticker that fired)
        meta     — {"scanned": int, "errors": int, "regime_label": str | None}
    """
    regime_series = None
    regime_label  = None
    if strat.use_regime_gate:
        print("  Building regime series...", flush=True)
        regime_series = build_regime_series(start, end, strat, bt.data_dir)
        if regime_series is not None and len(regime_series) > 0:
            regime_label = str(regime_series.iloc[-1]).upper()

    bench_df = None
    try:
        bench_df = load_ohlcv(bt.benchmark, start, end, bt.data_dir)
    except Exception as e:
        print(f"  [WARN] Could not load benchmark {bt.benchmark}: {e}")

    scanned = 0
    errors  = 0
    signals: list[dict] = []

    print(f"  Scanning {len(tickers)} tickers ({workers} threads)...", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scan_one, t, start, end, strat, bt, bench_df, regime_series): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                scanned += 1
                if result is not None:
                    signals.append(result)
            except Exception as exc:
                errors += 1
                print(f"  [ERR] {ticker}: {exc}")

    meta: dict = {"scanned": scanned, "errors": errors}
    if regime_label:
        meta["regime_label"] = regime_label

    return signals, meta
