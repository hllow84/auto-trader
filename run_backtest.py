"""
Entry point — MA-CCI Swing Strategy backtest.

Usage:
    python run_backtest.py                      # defaults from config.py
    python run_backtest.py --ticker MSFT
    python run_backtest.py --ticker AAPL --start 2015-01-01 --end 2023-12-31
    python run_backtest.py --target 2.0         # enable 2R profit target
    python run_backtest.py --ema                # use EMA instead of SMA
"""

import argparse
import sys
import os

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from config import StrategyConfig, BacktestConfig, STRATEGY, BACKTEST
from data.fetch import load_ohlcv
from backtest.engine import run_backtest
from backtest.report import build_report, build_bh_return, print_report, compare_is_oos


def parse_args():
    p = argparse.ArgumentParser(description="MA-CCI Swing Strategy backtest")
    p.add_argument("--ticker",    default=BACKTEST.ticker)
    p.add_argument("--benchmark", default=BACKTEST.benchmark)
    p.add_argument("--start",     default=BACKTEST.start_date)
    p.add_argument("--end",       default=BACKTEST.end_date)
    p.add_argument("--capital",   type=float, default=BACKTEST.initial_capital)
    p.add_argument("--target",    type=float, default=None, help="Profit target in R (e.g. 2.0)")
    p.add_argument("--ema",       action="store_true", help="Use EMA instead of SMA")
    p.add_argument("--shorts",    action="store_true", help="Enable short trades (default off)")
    p.add_argument("--cci",       type=int, default=STRATEGY.cci_period, help="CCI period")
    p.add_argument("--risk",      type=float, default=STRATEGY.max_risk_pct, help="Risk per trade (fraction)")
    return p.parse_args()


def main():
    args = parse_args()

    strat = StrategyConfig(
        ma_type="EMA" if args.ema else "SMA",
        cci_period=args.cci,
        max_risk_pct=args.risk,
        profit_target_r=args.target,
    )

    bt = BacktestConfig(
        ticker=args.ticker,
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    print(f"\nLoading data: {bt.ticker} {bt.start_date} to {bt.end_date}")
    df = load_ohlcv(bt.ticker, bt.start_date, bt.end_date)
    print(f"  {len(df)} trading days loaded.")

    # In-sample / out-of-sample split
    split = int(len(df) * strat.in_sample_pct)
    df_is  = df.iloc[:split].copy()
    df_oos = df.iloc[split:].copy()
    print(f"  IS: {len(df_is)} bars  |  OOS: {len(df_oos)} bars")

    # Run IS
    print("\nRunning in-sample backtest...")
    port_is = run_backtest(df_is, strat, bt)
    report_is = build_report(port_is, df_is, label="IN-SAMPLE")
    bh_is = build_bh_return(df_is)
    print_report(report_is, bh_is)

    # Run OOS
    print("Running out-of-sample backtest...")
    port_oos = run_backtest(df_oos, strat, bt)
    report_oos = build_report(port_oos, df_oos, label="OUT-OF-SAMPLE")
    bh_oos = build_bh_return(df_oos)
    print_report(report_oos, bh_oos)

    # Side-by-side IS vs OOS
    compare_is_oos(report_is, report_oos, bh_is, bh_oos)

    # Benchmark comparison (full period)
    try:
        df_bench = load_ohlcv(bt.benchmark, bt.start_date, bt.end_date)
        bh_full = build_bh_return(df_bench)
        bh_strategy_full = build_bh_return(df)
        print(f"  Full-period buy-and-hold ({bt.benchmark}): {bh_full:.2%}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
