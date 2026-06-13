"""
Entry point — MA-CCI Swing Strategy backtest.

Usage:
    python run_backtest.py                      # defaults from config.py
    python run_backtest.py --ticker MSFT
    python run_backtest.py --ticker AAPL --start 2015-01-01 --end 2023-12-31
    python run_backtest.py --target 2.0         # enable 2R profit target
    python run_backtest.py --ema                # use EMA instead of SMA
    python run_backtest.py --regime             # enable market regime gate (§5.1)
    python run_backtest.py --universe           # enable universe filter (§5.2)
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
from strategy.regime import build_regime_series, join_regime
from strategy.universe import add_universe_filter


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
    p.add_argument("--regime",    action="store_true", help="Enable market regime gate (§5.1)")
    p.add_argument("--universe",  action="store_true", help="Enable universe filter: price>$0.50, vol>500k (§5.2)")
    return p.parse_args()


def main():
    args = parse_args()

    strat = StrategyConfig(
        ma_type="EMA" if args.ema else "SMA",
        cci_period=args.cci,
        max_risk_pct=args.risk,
        profit_target_r=args.target,
        use_regime_gate=args.regime,
        use_universe_filter=args.universe,
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

    # Optional layers — applied before IS/OOS split so both halves share the same columns
    if strat.use_universe_filter:
        df = add_universe_filter(df, strat)
        pct = df["passes_filter"].mean()
        print(f"  Universe filter: {pct:.1%} of bars pass (price>${strat.min_price}, vol>{strat.min_avg_volume:,})")

    if strat.use_regime_gate:
        regime = build_regime_series(bt.start_date, bt.end_date, strat, bt.data_dir)
        df = join_regime(df, regime)
        counts = df["regime"].value_counts()
        print(f"  Regime gate: {counts.to_dict()}")

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
