"""
MA-CCI Swing Strategy — single-instrument backtesting engine.
Build order §7: core + shorts + regime + universe + RS + tiered scale-in.
"""

import math
import pandas as pd
from config import StrategyConfig, BacktestConfig
from backtest.portfolio import Portfolio
from strategy.indicators import add_indicators
from strategy.signals import add_signals


def _tick(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.001


def _tranche_shares(port: Portfolio, fill: float, stop: float,
                    pct: float, strat: StrategyConfig) -> int:
    """Full risk-based S × tranche_pct, capped by position and cash limits."""
    full_s = port.size_position(fill, stop, strat.max_risk_pct, strat.max_position_pct)
    return max(math.floor(pct * full_s), 0)


def run_backtest(df_raw: pd.DataFrame, strat: StrategyConfig, bt: BacktestConfig) -> Portfolio:
    """
    df_raw must have columns: open, high, low, close, volume (lowercase).
    Optional pre-computed columns (joined before calling):
      regime         — 'bullish' | 'bearish' | 'neutral'
      passes_filter  — bool
      rs             — float (stock return minus benchmark return)
    Returns a Portfolio with all trades and equity_curve populated.
    """
    df = add_indicators(df_raw, strat)
    df = add_signals(df, strat)

    # Defaults so the loop never needs to branch on whether layers are active
    if "regime" not in df.columns:
        df["regime"] = "bullish"
    if "passes_filter" not in df.columns:
        df["passes_filter"] = True
    if "rs" not in df.columns:
        df["rs"] = 0.0

    df = df.reset_index(drop=True)

    port = Portfolio(bt.initial_capital)

    pending_long:     dict | None = None   # {"stop": float, "initial_stop": float}
    pending_short:    dict | None = None
    pending_scale_in: dict | None = None   # {"stop": float, "shares": int}

    for i, row in df.iterrows():
        # Skip bars where indicators aren't ready
        if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]) or pd.isna(row["cci"]):
            port.mark_equity(i, row["close"])
            continue

        tick = _tick(row["close"])

        # ------------------------------------------------------------------ #
        # 1a. Fill pending initial entry (only when flat)                     #
        # ------------------------------------------------------------------ #
        position_opened_this_bar = False

        if port.position is None:
            if pending_long is not None:
                entry_stop = pending_long["stop"]
                if row["high"] >= entry_stop:
                    fill = entry_stop + strat.slippage_ticks * tick
                    istop = pending_long["initial_stop"]
                    if strat.use_scaling:
                        shares = _tranche_shares(port, fill, istop, strat.tranche_sizes[0], strat)
                    else:
                        shares = port.size_position(fill, istop, strat.max_risk_pct, strat.max_position_pct)
                    if shares > 0:
                        comm = fill * shares * strat.commission_pct
                        port.open_position("long", fill, shares, istop, i, comm)
                        position_opened_this_bar = True
                    pending_long = None

            elif pending_short is not None:
                entry_stop = pending_short["stop"]
                if row["low"] <= entry_stop:
                    fill = entry_stop - strat.slippage_ticks * tick
                    istop = pending_short["initial_stop"]
                    if strat.use_scaling:
                        shares = _tranche_shares(port, fill, istop, strat.tranche_sizes[0], strat)
                    else:
                        shares = port.size_position(fill, istop, strat.max_risk_pct, strat.max_position_pct)
                    if shares > 0:
                        comm = fill * shares * strat.commission_pct
                        port.open_position("short", fill, shares, istop, i, comm)
                        position_opened_this_bar = True
                    pending_short = None

        # ------------------------------------------------------------------ #
        # 1b. Fill pending scale-in (only when a position is open)            #
        # ------------------------------------------------------------------ #
        if (port.position is not None and not position_opened_this_bar
                and strat.use_scaling and pending_scale_in is not None):
            pos = port.position
            entry_stop = pending_scale_in["stop"]
            triggered = (
                (pos.direction == "long"  and row["high"] >= entry_stop) or
                (pos.direction == "short" and row["low"]  <= entry_stop)
            )
            if triggered:
                if pos.direction == "long":
                    fill = entry_stop + strat.slippage_ticks * tick
                    price_ok = fill > pos.avg_entry_price
                else:
                    fill = entry_stop - strat.slippage_ticks * tick
                    price_ok = fill < pos.avg_entry_price

                if price_ok:
                    add_shares = pending_scale_in["shares"]
                    if add_shares > 0:
                        comm = fill * add_shares * strat.commission_pct
                        port.add_tranche(fill, add_shares, comm)
            pending_scale_in = None

        # ------------------------------------------------------------------ #
        # 2. Manage open position — stops, time stop, profit target           #
        # ------------------------------------------------------------------ #
        if port.position is not None and not position_opened_this_bar:
            pos = port.position
            days_held = i - pos.entry_bar
            exit_reason = None
            exit_price  = None

            if pos.direction == "long":
                # Breakeven ratchet
                if not pos.breakeven_set and row["close"] >= pos.entry_price * strat.breakeven_trigger:
                    pos.current_stop = max(pos.current_stop, pos.entry_price)
                    pos.breakeven_set = True

                # Trailing stop: ratchet to prior-day low - 1 tick (never loosen)
                if i > 0:
                    new_trail = df.at[i - 1, "low"] - tick
                    pos.current_stop = max(pos.current_stop, new_trail)

                if strat.profit_target_r is not None:
                    target = pos.entry_price + strat.profit_target_r * abs(pos.entry_price - pos.initial_stop)
                    if row["high"] >= target:
                        exit_price  = target
                        exit_reason = "profit_target"

                if exit_reason is None and row["low"] <= pos.current_stop:
                    exit_price  = min(pos.current_stop - strat.slippage_ticks * tick, row["open"])
                    exit_reason = "stop"

                if exit_reason is None and days_held >= strat.time_stop_days:
                    exit_price  = row["open"]
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
                        exit_price  = target
                        exit_reason = "profit_target"

                if exit_reason is None and row["high"] >= pos.current_stop:
                    exit_price  = max(pos.current_stop + strat.slippage_ticks * tick, row["open"])
                    exit_reason = "stop"

                if exit_reason is None and days_held >= strat.time_stop_days:
                    exit_price  = row["open"]
                    exit_reason = "time_stop"

            if exit_reason:
                comm = exit_price * pos.shares * strat.commission_pct
                slippage_cost = (
                    abs(exit_price - pos.current_stop) * pos.shares
                    if exit_reason == "stop" else 0
                )
                port.close_position(exit_price, i, exit_reason, comm, slippage_cost)
                pending_scale_in = None   # position gone; clear any pending add

        # ------------------------------------------------------------------ #
        # 3a. Set up new initial-entry pending orders for the NEXT bar        #
        # ------------------------------------------------------------------ #
        if port.position is None:
            regime = row["regime"]
            passes = row["passes_filter"]
            rs     = row["rs"]

            # Cancel stale pending orders if trend breaks OR (gate active) regime flips
            if pending_long is not None:
                trend_broke = not (row["ma_fast"] > row["ma_slow"])
                regime_flip = strat.use_regime_gate and regime == "bearish"
                if trend_broke or regime_flip:
                    pending_long = None
            if pending_short is not None:
                trend_broke = not (row["ma_fast"] < row["ma_slow"])
                regime_flip = strat.use_regime_gate and regime == "bullish"
                if trend_broke or regime_flip:
                    pending_short = None

            long_allowed  = (
                passes
                and (not strat.use_regime_gate or regime != "bearish")
                and (not strat.use_rs_filter   or (not pd.isna(rs) and rs > 0))
            )
            short_allowed = (
                passes
                and (not strat.use_regime_gate or regime != "bullish")
            )

            if row["long_signal"] and long_allowed:
                entry = row["high"] + tick
                if strat.stop_atr_mult is not None and not pd.isna(row.get("atr", float("nan"))):
                    istop = entry - strat.stop_atr_mult * row["atr"]
                else:
                    istop = row["low"] - tick
                pending_long  = {"stop": entry, "initial_stop": istop}
                pending_short = None

            elif row["short_signal"] and short_allowed:
                entry = row["low"] - tick
                if strat.stop_atr_mult is not None and not pd.isna(row.get("atr", float("nan"))):
                    istop = entry + strat.stop_atr_mult * row["atr"]
                else:
                    istop = row["high"] + tick
                pending_short = {"stop": entry, "initial_stop": istop}
                pending_long  = None

        # ------------------------------------------------------------------ #
        # 3b. Set up / update scale-in pending order for the NEXT bar        #
        # ------------------------------------------------------------------ #
        if (port.position is not None and strat.use_scaling):
            pos = port.position
            max_tranches = len(strat.tranche_sizes)
            if pos.tranches < max_tranches:
                signal_ok = (
                    (pos.direction == "long"  and row["long_signal"])  or
                    (pos.direction == "short" and row["short_signal"])
                )
                price_ok = (
                    (pos.direction == "long"  and row["close"] > pos.avg_entry_price) or
                    (pos.direction == "short" and row["close"] < pos.avg_entry_price)
                )
                if signal_ok and price_ok:
                    tranche_pct = strat.tranche_sizes[pos.tranches]
                    if pos.direction == "long":
                        sc_stop = row["high"] + tick
                        sc_istop = row["low"] - tick
                    else:
                        sc_stop  = row["low"] - tick
                        sc_istop = row["high"] + tick
                    add_shares = _tranche_shares(port, sc_stop, sc_istop, tranche_pct, strat)
                    if add_shares > 0:
                        pending_scale_in = {"stop": sc_stop, "shares": add_shares}

        port.mark_equity(i, row["close"])

    return port
