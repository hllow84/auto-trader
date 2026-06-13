"""
Multi-stock universe scanner — MA-CCI signals across a stock universe.

Usage:
    python run_scan.py                              # scan built-in 50-stock universe, today
    python run_scan.py --end 2024-06-01            # scan as of a historical date
    python run_scan.py --tickers AAPL MSFT NVDA   # custom ticker list
    python run_scan.py --top 5                     # show top 5 signals per direction
    python run_scan.py --regime --universe --rs    # enable all filter layers
    python run_scan.py --workers 16                # more parallel download threads
"""

import argparse
import math
import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from config import StrategyConfig, BacktestConfig, STRATEGY, BACKTEST
from scanner.scan import scan_universe

# Default 50-stock universe (S&P 500 large caps, liquid, diverse sectors)
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "UNH",  "JPM",  "XOM",  "JNJ",  "V",     "PG",   "MA",
    "AVGO", "HD",   "CVX",  "MRK",  "ABBV",  "COST", "PEP",
    "KO",   "ADBE", "CSCO", "MCD",  "CRM",   "ACN",  "WMT",
    "BAC",  "LIN",  "TMO",  "NFLX", "AMD",   "DHR",  "NKE",
    "TXN",  "NEE",  "PM",   "HON",  "ORCL",  "UPS",  "IBM",
    "LOW",  "AMGN", "INTC", "QCOM", "INTU",  "CAT",  "GS",
    "SPGI",
]


def parse_args():
    p = argparse.ArgumentParser(description="MA-CCI universe scanner")
    p.add_argument("--tickers",   nargs="+", default=None,
                   help="Tickers to scan (default: built-in 50-stock universe)")
    p.add_argument("--start",     default="2023-01-01",
                   help="History start date for indicator warmup")
    p.add_argument("--end",       default=str(date.today()),
                   help="Scan-as-of date (default: today)")
    p.add_argument("--benchmark", default=BACKTEST.benchmark,
                   help="Benchmark ticker for RS calculation")
    p.add_argument("--cci",       type=int, default=STRATEGY.cci_period)
    p.add_argument("--ema",       action="store_true", help="Use EMA instead of SMA")
    p.add_argument("--regime",    action="store_true", help="Enable market regime gate (§5.1)")
    p.add_argument("--universe",  action="store_true", help="Enable universe filter: price>$0.50, vol>500k (§5.2)")
    p.add_argument("--rs",        action="store_true", help="Enable RS>0 filter vs benchmark (§5.3)")
    p.add_argument("--top",       type=int, default=None,
                   help="Show only top N signals per direction (by RS)")
    p.add_argument("--workers",   type=int, default=8,
                   help="Parallel download threads (default: 8)")
    return p.parse_args()


def _rs_key_long(s: dict) -> tuple:
    """Sort key: longs ranked descending by RS, NaN last."""
    rs = s["rs"]
    if math.isnan(rs):
        return (True, 0.0)
    return (False, -rs)


def _rs_key_short(s: dict) -> tuple:
    """Sort key: shorts ranked ascending by RS (most negative first), NaN last."""
    rs = s["rs"]
    if math.isnan(rs):
        return (True, 0.0)
    return (False, rs)


def _fmt_rs(v: float) -> str:
    return f"{v:+.2%}" if not math.isnan(v) else "    n/a"


def _fmt_vol(v: float) -> str:
    return f"{v:,.1f}" if not math.isnan(v) else "  n/a"


def _print_signals(signals: list[dict], direction: str, top: int | None):
    subset = [s for s in signals if s["direction"] == direction]

    if direction == "long":
        subset.sort(key=_rs_key_long)
        dir_header = "LONG SIGNALS - best RS (strongest relative strength)"
    else:
        subset.sort(key=_rs_key_short)
        dir_header = "SHORT SIGNALS - worst RS (most underperforming)"

    if top:
        subset = subset[:top]

    count_label = f"{len(subset)} found" if not top else f"top {len(subset)}"
    print(f"\n{dir_header}  ({count_label}):")

    if not subset:
        print("  (none)")
        return

    print(f"  {'#':>3}  {'Ticker':<6} {'Close':>8}  {'Entry@':>8}  {'Stop@':>8}  {'RS':>8}  {'Vol(M)':>8}  Regime")
    print("  " + "-" * 72)
    for i, s in enumerate(subset, 1):
        print(
            f"  {i:>3}  {s['ticker']:<6} "
            f"${s['close']:>7.2f}  "
            f"${s['entry_stop']:>7.2f}  "
            f"${s['initial_stop']:>7.2f}  "
            f"{_fmt_rs(s['rs']):>8}  "
            f"{_fmt_vol(s['avg_vol_m']):>8}  "
            f"{s['regime']}"
        )


def main():
    args = parse_args()
    tickers = args.tickers or DEFAULT_UNIVERSE

    strat = StrategyConfig(
        ma_type="EMA" if args.ema else "SMA",
        cci_period=args.cci,
        use_regime_gate=args.regime,
        use_universe_filter=args.universe,
        use_rs_filter=args.rs,
    )
    bt = BacktestConfig(
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
    )

    active_filters = [
        f for f, on in [("regime", args.regime), ("universe", args.universe), ("RS>0", args.rs)]
        if on
    ]

    print(f"\nMA-CCI Universe Scanner")
    print(f"  Universe : {len(tickers)} tickers")
    print(f"  Period   : {args.start} -> {args.end}")
    print(f"  MA       : {'EMA' if args.ema else 'SMA'}  CCI: {args.cci}")
    print(f"  Filters  : {', '.join(active_filters) if active_filters else 'none'}")
    print()

    signals, meta = scan_universe(
        tickers, args.start, args.end, strat, bt, workers=args.workers
    )

    scan_date = signals[0]["scan_date"] if signals else args.end
    longs  = sum(1 for s in signals if s["direction"] == "long")
    shorts = sum(1 for s in signals if s["direction"] == "short")

    print(f"\n  Scan date : {scan_date}")
    print(f"  Scanned   : {meta['scanned']}  |  Errors: {meta['errors']}")
    if "regime_label" in meta:
        print(f"  Regime    : {meta['regime_label']}")
    print(f"  Signals   : {longs} long  |  {shorts} short")

    if not signals:
        print("\nNo signals found.\n")
        return

    _print_signals(signals, "long",  args.top)
    _print_signals(signals, "short", args.top)
    print()


if __name__ == "__main__":
    main()
