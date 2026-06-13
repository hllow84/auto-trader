from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class Position:
    direction: str          # "long" or "short"
    entry_price: float      # first-tranche fill price (kept for initial_stop reference)
    shares: int             # total shares across all tranches
    initial_stop: float
    current_stop: float
    entry_bar: int          # bar index at entry
    breakeven_set: bool = False
    tranches: int = 1       # tranches fired so far (1 after initial entry)
    cost_basis: float = 0.0 # sum(fill_i * shares_i) across all tranches

    @property
    def avg_entry_price(self) -> float:
        return self.cost_basis / self.shares if self.shares > 0 else self.entry_price


@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    entry_bar: int
    exit_bar: int
    exit_reason: str
    commission: float
    slippage: float

    @property
    def pnl(self) -> float:
        gross = (self.exit_price - self.entry_price) * self.shares
        if self.direction == "short":
            gross = -gross
        return gross - self.commission - self.slippage

    @property
    def days_held(self) -> int:
        return self.exit_bar - self.entry_bar


@dataclass
class Portfolio:
    initial_capital: float
    cash: float = field(init=False)
    position: Optional[Position] = field(default=None, init=False)
    trades: list = field(default_factory=list, init=False)
    equity_curve: list = field(default_factory=list, init=False)  # (bar_idx, equity)

    def __post_init__(self):
        self.cash = self.initial_capital

    @property
    def equity(self) -> float:
        return self.cash

    def mark_equity(self, bar_idx: int, price: float):
        if self.position:
            if self.position.direction == "long":
                pos_value = self.position.shares * price
            else:
                pos_value = self.position.shares * (2 * self.position.entry_price - price)
            self.equity_curve.append((bar_idx, self.cash + pos_value))
        else:
            self.equity_curve.append((bar_idx, self.cash))

    def open_position(self, direction: str, fill_price: float, shares: int,
                      initial_stop: float, bar_idx: int, commission: float):
        cost = fill_price * shares
        self.cash -= cost + commission
        self.position = Position(
            direction=direction,
            entry_price=fill_price,
            shares=shares,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            entry_bar=bar_idx,
            cost_basis=cost,
        )

    def add_tranche(self, fill_price: float, add_shares: int, commission: float):
        cost = fill_price * add_shares
        self.cash -= cost + commission
        pos = self.position
        pos.cost_basis += cost
        pos.shares += add_shares
        pos.tranches += 1

    def close_position(self, fill_price: float, bar_idx: int,
                       exit_reason: str, commission: float, slippage_cost: float):
        pos = self.position
        proceeds = fill_price * pos.shares
        self.cash += proceeds - commission

        trade = Trade(
            direction=pos.direction,
            entry_price=pos.avg_entry_price,
            exit_price=fill_price,
            shares=pos.shares,
            entry_bar=pos.entry_bar,
            exit_bar=bar_idx,
            exit_reason=exit_reason,
            commission=commission,
            slippage=slippage_cost,
        )
        self.trades.append(trade)
        self.position = None
        return trade

    def size_position(self, entry: float, stop: float, max_risk_pct: float,
                      max_pos_pct: float) -> int:
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        equity = self.equity
        raw = (equity * max_risk_pct) / risk_per_share
        max_by_size = (equity * max_pos_pct) / entry
        max_by_cash = self.cash / entry
        shares = math.floor(min(raw, max_by_size, max_by_cash))
        return max(shares, 0)
