"""
Portfolio backtest — run MA-CCI strategy across a universe of stocks and
report aggregate IS/OOS statistics pooled across all tickers.

Usage:
    python run_portfolio.py                                       # default 50-stock universe
    python run_portfolio.py --tickers-file data/sp500.txt        # wider universe
    python run_portfolio.py --start 2015-01-01 --end 2026-06-13
    python run_portfolio.py --regime --universe --rs --scaling
    python run_portfolio.py --tickers-file data/sp500.txt --universe --workers 16
"""

import argparse
import math
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from config import StrategyConfig, BacktestConfig, STRATEGY, BACKTEST
from data.fetch import load_ohlcv
from backtest.engine import run_backtest
from backtest.report import build_bh_return
from backtest.portfolio import Trade
from strategy.regime import build_regime_series, join_regime
from strategy.universe import add_universe_filter
from strategy.rs import add_rs

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "UNH",  "JPM",  "XOM",  "JNJ",  "V",     "PG",   "MA",
    "AVGO", "HD",   "CVX",  "MRK",  "ABBV",  "COST", "PEP",
    "KO",   "ADBE", "CSCO", "MCD",  "CRM",   "ACN",  "WMT",
    "BAC",  "LIN",  "TMO",  "NFLX", "AMD",   "DHR",  "NKE",
    "TXN",  "NEE",  "PM",   "HON",  "ORCL",  "UPS",  "IBM",
    "LOW",  "AMGN", "INTC", "QCOM", "INTU",  "CAT",  "GS",
    "SPGI",
]


def parse_args():
    p = argparse.ArgumentParser(description="MA-CCI portfolio backtest across a stock universe")
    p.add_argument("--tickers",      nargs="+", default=None)
    p.add_argument("--tickers-file", default=None, help="File with one ticker per line")
    p.add_argument("--start",        default="2015-01-01")
    p.add_argument("--end",          default=str(date.today()))
    p.add_argument("--benchmark",    default=BACKTEST.benchmark)
    p.add_argument("--cci",          type=int,   default=STRATEGY.cci_period)
    p.add_argument("--ema",          action="store_true")
    p.add_argument("--shorts",       action="store_true", help="Include short trades")
    p.add_argument("--target",       type=float, default=None, help="Profit target in R")
    p.add_argument("--regime",       action="store_true")
    p.add_argument("--universe",     action="store_true")
    p.add_argument("--rs",           action="store_true")
    p.add_argument("--scaling",      action="store_true")
    p.add_argument("--stop-atr",     type=float, default=None,
                   help="ATR multiplier for initial stop (e.g. 1.5); default=bar low/high")
    p.add_argument("--top",          type=int, default=15, help="Top N tickers in detail table")
    p.add_argument("--workers",      type=int, default=8)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-ticker worker                                                            #
# --------------------------------------------------------------------------- #

def _backtest_ticker(
    ticker, start, end, strat, bt, bench_df, regime_series
) -> dict | None:
    df = load_ohlcv(ticker, start, end, bt.data_dir)

    if len(df) < strat.ma_slow + 20:
        return None

    if strat.use_universe_filter:
        df = add_universe_filter(df, strat)
    else:
        df["passes_filter"] = True

    if regime_series is not None:
        df = join_regime(df, regime_series)
    else:
        df["regime"] = "bullish"

    if bench_df is not None:
        df = add_rs(df, bench_df, strat)
    else:
        df["rs"] = 0.0

    split = int(len(df) * strat.in_sample_pct)
    df_is  = df.iloc[:split].copy()
    df_oos = df.iloc[split:].copy()

    port_is  = run_backtest(df_is,  strat, bt)
    port_oos = run_backtest(df_oos, strat, bt)

    return {
        "ticker":     ticker,
        "is_trades":  port_is.trades,
        "oos_trades": port_oos.trades,
        "bh_is":      build_bh_return(df_is),
        "bh_oos":     build_bh_return(df_oos),
        "is_start":   str(df_is.index[0].date()  if hasattr(df_is.index[0], "date")  else df_is.index[0]),
        "is_end":     str(df_is.index[-1].date() if hasattr(df_is.index[-1], "date") else df_is.index[-1]),
        "oos_start":  str(df_oos.index[0].date() if hasattr(df_oos.index[0], "date") else df_oos.index[0]),
        "oos_end":    str(df_oos.index[-1].date()if hasattr(df_oos.index[-1],"date") else df_oos.index[-1]),
    }


# --------------------------------------------------------------------------- #
# Aggregate stats                                                              #
# --------------------------------------------------------------------------- #

def _agg(trades: list[Trade], label: str, bh_values: list[float]) -> dict:
    if not trades:
        return {"label": label, "n": 0}

    pnls      = [t.pnl for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_los = abs(sum(losses))

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return {
        "label":         label,
        "n":             len(trades),
        "win_rate":      len(wins) / len(trades),
        "avg_pnl":       sum(pnls) / len(trades),
        "profit_factor": gross_win / gross_los if gross_los > 0 else float("inf"),
        "avg_hold":      sum(t.days_held for t in trades) / len(trades),
        "exits":         exit_counts,
        "avg_bh":        sum(bh_values) / len(bh_values) if bh_values else float("nan"),
        "long_n":        sum(1 for t in trades if t.direction == "long"),
        "short_n":       sum(1 for t in trades if t.direction == "short"),
    }


def _print_agg(agg: dict):
    if agg["n"] == 0:
        print(f"  [{agg['label']}]  No trades.")
        return

    exits = agg["exits"]
    total = agg["n"]
    exit_str = "  ".join(
        f"{r}: {c} ({c/total:.0%})" for r, c in sorted(exits.items(), key=lambda x: -x[1])
    )
    pf = agg["profit_factor"]
    pf_str = f"{pf:.2f}" if not math.isinf(pf) else "inf"

    print(f"\n  [{agg['label']}]")
    print(f"    Trades       : {agg['n']:,}   (long: {agg['long_n']}  short: {agg['short_n']})")
    print(f"    Win rate     : {agg['win_rate']:.1%}")
    print(f"    Avg P&L/trade: ${agg['avg_pnl']:,.2f}")
    print(f"    Profit factor: {pf_str}")
    print(f"    Avg hold     : {agg['avg_hold']:.1f} days")
    print(f"    Exits        : {exit_str}")
    print(f"    Avg ticker BH: {agg['avg_bh']:.1%}")


def _print_ticker_table(results: list[dict], top: int):
    # Sort by OOS trade count descending
    ranked = sorted(results, key=lambda r: len(r["oos_trades"]), reverse=True)[:top]

    print(f"\n  TOP {top} TICKERS BY OOS TRADE COUNT:")
    print(f"  {'Ticker':<7}  {'IS trades':>9}  {'IS win%':>7}  {'OOS trades':>10}  {'OOS win%':>8}  {'IS BH':>7}  {'OOS BH':>7}")
    print("  " + "-" * 68)
    for r in ranked:
        it = r["is_trades"]
        ot = r["oos_trades"]
        is_wr  = (sum(1 for t in it if t.pnl > 0) / len(it))  if it else float("nan")
        oos_wr = (sum(1 for t in ot if t.pnl > 0) / len(ot)) if ot else float("nan")
        is_wr_s  = f"{is_wr:.0%}"  if not math.isnan(is_wr)  else "  n/a"
        oos_wr_s = f"{oos_wr:.0%}" if not math.isnan(oos_wr) else "  n/a"
        print(
            f"  {r['ticker']:<7}  {len(it):>9,}  {is_wr_s:>7}  {len(ot):>10,}  {oos_wr_s:>8}"
            f"  {r['bh_is']:>6.0%}  {r['bh_oos']:>6.0%}"
        )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    else:
        tickers = args.tickers or DEFAULT_UNIVERSE

    strat = StrategyConfig(
        ma_type="EMA" if args.ema else "SMA",
        cci_period=args.cci,
        profit_target_r=args.target,
        use_regime_gate=args.regime,
        use_universe_filter=args.universe,
        use_rs_filter=args.rs,
        use_scaling=args.scaling,
        stop_atr_mult=args.stop_atr,
    )
    bt = BacktestConfig(
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
    )

    active = [f for f, on in [("regime", args.regime), ("universe", args.universe),
                               ("RS>0", args.rs), ("scaling", args.scaling)] if on]
    if args.shorts:    active.append("shorts")
    if args.stop_atr:  active.append(f"stop={args.stop_atr}×ATR")

    print(f"\nMA-CCI Portfolio Backtest")
    print(f"  Universe : {len(tickers)} tickers")
    print(f"  Period   : {args.start} -> {args.end}  (IS: {strat.in_sample_pct:.0%} / OOS: {1-strat.in_sample_pct:.0%})")
    print(f"  MA       : {'EMA' if args.ema else 'SMA'}  CCI: {args.cci}")
    print(f"  Filters  : {', '.join(active) if active else 'none'}")
    print()

    # Build shared objects once
    regime_series = None
    if strat.use_regime_gate:
        print("  Building regime series...", flush=True)
        regime_series = build_regime_series(args.start, args.end, strat, bt.data_dir)

    bench_df = None
    try:
        bench_df = load_ohlcv(bt.benchmark, args.start, args.end, bt.data_dir)
    except Exception as e:
        print(f"  [WARN] Could not load benchmark: {e}")

    print(f"  Running {len(tickers)} backtests ({args.workers} threads)...", flush=True)

    results = []
    errors  = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_backtest_ticker, t, args.start, args.end, strat, bt, bench_df, regime_series): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as exc:
                errors += 1
                print(f"  [ERR] {ticker}: {exc}")

    # Pool trades
    all_is_trades  = [t for r in results for t in r["is_trades"]]
    all_oos_trades = [t for r in results for t in r["oos_trades"]]
    bh_is_vals  = [r["bh_is"]  for r in results]
    bh_oos_vals = [r["bh_oos"] for r in results]

    # Filter direction if shorts disabled
    if not args.shorts:
        all_is_trades  = [t for t in all_is_trades  if t.direction == "long"]
        all_oos_trades = [t for t in all_oos_trades if t.direction == "long"]

    agg_is  = _agg(all_is_trades,  "IN-SAMPLE",      bh_is_vals)
    agg_oos = _agg(all_oos_trades, "OUT-OF-SAMPLE",  bh_oos_vals)

    # Date range from first result
    if results:
        is_window  = f"{results[0]['is_start']}  to  {results[0]['is_end']}"
        oos_window = f"{results[0]['oos_start']}  to  {results[0]['oos_end']}"
    else:
        is_window = oos_window = "n/a"

    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS  ({len(results)} tickers run  |  {errors} errors)")
    print(f"{'='*60}")
    print(f"\n  IS  period : {is_window}")
    print(f"  OOS period : {oos_window}")

    _print_agg(agg_is)
    _print_agg(agg_oos)

    # IS/OOS consistency check
    if agg_is["n"] > 0 and agg_oos["n"] > 0:
        wr_gap  = agg_is["win_rate"] - agg_oos["win_rate"]
        pnl_gap = agg_is["avg_pnl"]  - agg_oos["avg_pnl"]
        print(f"\n  IS/OOS win rate gap : {wr_gap:+.1%}", end="")
        if abs(wr_gap) > 0.10:
            print("  *** LARGE — possible overfit ***", end="")
        print()
        print(f"  IS/OOS avg P&L gap  : ${pnl_gap:+,.2f}")

    if results:
        _print_ticker_table(results, args.top)

    print()


if __name__ == "__main__":
    main()
