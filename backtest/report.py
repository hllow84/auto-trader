"""Performance reporting — §6 requirements."""

import math
import pandas as pd
import numpy as np
from backtest.portfolio import Portfolio, Trade


def build_report(port: Portfolio, df: pd.DataFrame, label: str = "") -> dict:
    trades = port.trades
    if not trades:
        return {"label": label, "error": "no trades"}

    # Equity curve
    equity_df = pd.DataFrame(port.equity_curve, columns=["bar", "equity"])
    equity_df = equity_df.set_index("bar")

    final_equity = equity_df["equity"].iloc[-1]
    initial = port.initial_capital
    net_return = (final_equity - initial) / initial

    # Drawdown
    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Daily returns for Sharpe
    daily_ret = equity_df["equity"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(252)) if daily_ret.std() > 0 else 0.0

    # Trade stats
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(trades)
    avg_days = sum(t.days_held for t in trades) / len(trades)
    total_commission = sum(t.commission for t in trades)
    total_slippage = sum(t.slippage for t in trades)

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Overfit flags (§6)
    flags = []
    if len(trades) < 20:
        flags.append(f"FEW_TRADES ({len(trades)})")
    if net_return > 3.0:
        flags.append(f"SUSPICIOUSLY_HIGH_RETURN ({net_return:.1%})")

    return {
        "label":            label,
        "net_return":       net_return,
        "final_equity":     final_equity,
        "max_drawdown":     max_drawdown,
        "sharpe":           sharpe,
        "win_rate":         win_rate,
        "num_trades":       len(trades),
        "avg_days_held":    avg_days,
        "total_commission": total_commission,
        "total_slippage":   total_slippage,
        "exit_breakdown":   exit_counts,
        "overfit_flags":    flags,
        "equity_df":        equity_df,
    }


def build_bh_return(df: pd.DataFrame) -> float:
    """Simple buy-and-hold return over the same window."""
    return (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]


def print_report(report: dict, bh_return: float | None = None):
    if "error" in report:
        print(f"[{report['label']}] ERROR: {report['error']}")
        return

    label = f"[{report['label']}] " if report["label"] else ""
    print(f"\n{'='*60}")
    print(f"{label}BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Net return (strategy):   {report['net_return']:>10.2%}")
    if bh_return is not None:
        print(f"  Net return (buy & hold): {bh_return:>10.2%}")
    print(f"  Final equity:            ${report['final_equity']:>12,.2f}")
    print(f"  Max drawdown:            {report['max_drawdown']:>10.2%}")
    print(f"  Sharpe ratio (ann.):     {report['sharpe']:>10.3f}")
    print(f"  Win rate:                {report['win_rate']:>10.1%}")
    print(f"  # trades:                {report['num_trades']:>10d}")
    print(f"  Avg days held:           {report['avg_days_held']:>10.1f}")
    print(f"  Total commission:        ${report['total_commission']:>11,.2f}")
    print(f"  Total slippage:          ${report['total_slippage']:>11,.2f}")
    print(f"  Exit breakdown:          {report['exit_breakdown']}")
    if report["overfit_flags"]:
        print(f"\n  *** OVERFIT FLAGS: {', '.join(report['overfit_flags'])} ***")
    print(f"{'='*60}\n")


def compare_is_oos(is_report: dict, oos_report: dict, bh_is: float, bh_oos: float):
    print("\n" + "="*60)
    print("  IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("="*60)
    metrics = ["net_return", "max_drawdown", "sharpe", "win_rate", "num_trades"]
    for m in metrics:
        v_is  = is_report.get(m, float("nan"))
        v_oos = oos_report.get(m, float("nan"))
        fmt = ".2%" if "return" in m or "rate" in m or "drawdown" in m else ".3f" if m == "sharpe" else "d"
        print(f"  {m:<20} IS: {format(v_is, fmt):>10}   OOS: {format(v_oos, fmt):>10}")

    if "net_return" in is_report and "net_return" in oos_report:
        gap = is_report["net_return"] - oos_report["net_return"]
        print(f"\n  IS/OOS return gap: {gap:.2%}", end="")
        if gap > 0.30:
            print("  *** LARGE — possible overfit ***", end="")
        print()

    print(f"\n  Buy-and-hold IS:  {bh_is:.2%}  |  OOS: {bh_oos:.2%}")
    print("="*60 + "\n")
