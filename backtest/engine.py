"""
MA-CCI Swing Strategy — single-instrument backtesting engine.
Build order §7: Core long + short, full-size entry, stops.
"""

import pandas as pd
import numpy as np
from config import StrategyConfig, BacktestConfig
from backtest.portfolio import Portfolio
from strategy.indicators import add_indicators
from strategy.signals import add_signals


def _tick(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.001


def run_backtest(df_raw: pd.DataFrame, strat: StrategyConfig, bt: BacktestConfig) -> Portfolio:
    """
    df_raw must have columns: open, high, low, close, volume (lowercase).
    Returns a Portfolio with all trades and equity_curve populated.
    """
    df = add_indicators(df_raw, strat)
    df = add_signals(df, strat)
    df = df.reset_index(drop=True)

    port = Portfolio(bt.initial_capital)

    # State for pending entry orders
    pending_long: dict  | None = None   # {"stop": float, "initial_stop": float}
    pending_short: dict | None = None

    for i, row in df.iterrows():
        # Skip bars where indicators aren't ready
        if pd.isna(row.get("ma_fast")) or pd.isna(row.get("ma_slow")) or pd.isna(row.get("cci")):
            port.mark_equity(i, row["close"])
            continue

        tick = _tick(row["close"])

        # ------------------------------------------------------------------ #
        # 1. Try to fill a pending entry order (today's bar)                  #
        # ------------------------------------------------------------------ #
        position_opened_this_bar = False

        if port.position is None:
            if pending_long is not None:
                entry_stop = pending_long["stop"]
                if row["high"] >= entry_stop:
                    fill = entry_stop + strat.slippage_ticks * tick
                    initial_stop = pending_long["initial_stop"]
                    shares = port.size_position(
                        fill, initial_stop, strat.max_risk_pct, strat.max_position_pct
                    )
                    if shares > 0:
                        comm = fill * shares * strat.commission_pct
                        port.open_position("long", fill, shares, initial_stop, i, comm)
                        position_opened_this_bar = True
                    pending_long = None

            elif pending_short is not None:
                entry_stop = pending_short["stop"]
                if row["low"] <= entry_stop:
                    fill = entry_stop - strat.slippage_ticks * tick
                    initial_stop = pending_short["initial_stop"]
                    shares = port.size_position(
                        fill, initial_stop, strat.max_risk_pct, strat.max_position_pct
                    )
                    if shares > 0:
                        comm = fill * shares * strat.commission_pct
                        port.open_position("short", fill, shares, initial_stop, i, comm)
                        position_opened_this_bar = True
                    pending_short = None

        # ------------------------------------------------------------------ #
        # 2. Manage open position (stops, time stop, profit target)           #
        # ------------------------------------------------------------------ #
        if port.position is not None and not position_opened_this_bar:
            pos = port.position
            days_held = i - pos.entry_bar
            exit_reason = None
            exit_price = None

            if pos.direction == "long":
                # Breakeven ratchet
                if not pos.breakeven_set and row["close"] >= pos.entry_price * strat.breakeven_trigger:
                    pos.current_stop = max(pos.current_stop, pos.entry_price)
                    pos.breakeven_set = True

                # Trailing stop: ratchet to prior-day low - 1 tick (never loosen)
                if i > 0:
                    new_trail = df.at[i - 1, "low"] - tick
                    pos.current_stop = max(pos.current_stop, new_trail)

                # Profit target (2R or configured R)
                if strat.profit_target_r is not None:
                    target = pos.entry_price + strat.profit_target_r * abs(pos.entry_price - pos.initial_stop)
                    if row["high"] >= target:
                        exit_price = target
                        exit_reason = "profit_target"

                # Stop hit (use open if gap below stop)
                if exit_reason is None and row["low"] <= pos.current_stop:
                    exit_price = min(pos.current_stop - strat.slippage_ticks * tick, row["open"])
                    exit_reason = "stop"

                # Time stop: close of day 5 → exit at next open (we exit at today's open if days_held >= 5)
                if exit_reason is None and days_held >= strat.time_stop_days:
                    exit_price = row["open"]
                    exit_reason = "time_stop"

            else:  # short
                if not pos.breakeven_set and row["close"] <= pos.entry_price * (2 - strat.breakeven_trigger):
                    pos.current_stop = min(pos.current_stop, pos.entry_price)
                    pos.breakeven_set = True

                if i > 0:
                    new_trail = df.at[i - 1, "high"] + tick
                    pos.current_stop = min(pos.current_stop, new_trail)

                if strat.profit_target_r is not None:
                    target = pos.entry_price - strat.profit_target_r * abs(pos.initial_stop - pos.entry_price)
                    if row["low"] <= target:
                        exit_price = target
                        exit_reason = "profit_target"

                if exit_reason is None and row["high"] >= pos.current_stop:
                    exit_price = max(pos.current_stop + strat.slippage_ticks * tick, row["open"])
                    exit_reason = "stop"

                if exit_reason is None and days_held >= strat.time_stop_days:
                    exit_price = row["open"]
                    exit_reason = "time_stop"

            if exit_reason:
                comm = exit_price * pos.shares * strat.commission_pct
                slippage_cost = abs(exit_price - pos.current_stop) * pos.shares if exit_reason == "stop" else 0
                port.close_position(exit_price, i, exit_reason, comm, slippage_cost)

        # ------------------------------------------------------------------ #
        # 3. Set up new pending entry orders for the NEXT bar                 #
        # ------------------------------------------------------------------ #
        if port.position is None:
            # Cancel stale pending orders if trend condition broke
            if pending_long is not None and not (row["ma_fast"] > row["ma_slow"]):
                pending_long = None
            if pending_short is not None and not (row["ma_fast"] < row["ma_slow"]):
                pending_short = None

            if row["long_signal"]:
                new_buy_stop = row["high"] + tick
                initial_stop = row["low"] - tick
                # Trailing: update if already have a pending order
                pending_long = {"stop": new_buy_stop, "initial_stop": initial_stop}
                pending_short = None

            elif row["short_signal"]:
                new_sell_stop = row["low"] - tick
                initial_stop = row["high"] + tick
                pending_short = {"stop": new_sell_stop, "initial_stop": initial_stop}
                pending_long = None

        port.mark_equity(i, row["close"])

    return port
