# MA-CCI Swing Strategy — all [DEFAULT] parameters in one place
# Change any value here and re-run; do not scatter magic numbers in engine code.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyConfig:
    # --- Indicators ---
    ma_fast: int = 20          # MA20
    ma_slow: int = 40          # MA40
    cci_period: int = 20       # CCI period (14 is common alt)
    slope_lookback: int = 3    # bars used to judge MA direction
    ma_type: str = "SMA"       # "SMA" or "EMA"

    # --- Entry ---
    cci_long_threshold: float = -100.0
    cci_short_threshold: float = 100.0

    # --- Costs ---
    commission_pct: float = 0.0005   # 0.05% per side (IBKR retail approx)
    slippage_ticks: int = 1          # ticks added on stop fills

    # --- Risk / Sizing ---
    max_risk_pct: float = 0.01       # 1% of portfolio equity per trade
    max_position_pct: float = 0.20   # cap position at 20% of portfolio value

    # --- Stops ---
    breakeven_trigger: float = 1.05  # raise stop to entry once price >= entry * 1.05
    time_stop_days: int = 5          # exit at next open if still open on day 5

    # --- Profit target ---
    # None = no fixed target (exits governed by stops only)
    # Set to e.g. 2.0 to take profit at 2R
    profit_target_r: Optional[float] = None

    # --- Tiered scale-in (§4.2) — leave OFF for core tests ---
    use_scaling: bool = False
    tranche_sizes: list = field(default_factory=lambda: [0.50, 0.30, 0.20])

    # --- Phase 1: Market regime gate ---
    use_regime_gate: bool = False
    regime_threshold: float = 0.50   # >50% green → bullish
    regime_indices: list = field(default_factory=lambda: [
        "^GSPC",   # S&P 500
        "^NDX",    # Nasdaq 100
        "^DJI",    # Dow Jones
        "^FTSE",   # FTSE 100
        "^GDAXI",  # DAX
        "^N225",   # Nikkei 225
        "^HSI",    # Hang Seng
        "^STI",    # STI
    ])

    # --- Phase 2: Universe filter ---
    use_universe_filter: bool = False
    min_price: float = 0.50
    min_avg_volume: int = 500_000

    # --- Phase 1.3: Relative strength ---
    use_rs_filter: bool = False
    rs_lookback: int = 20
    rs_top_n: int = 5

    # --- Backtest split ---
    # Fraction of data used for in-sample development
    in_sample_pct: float = 0.70


@dataclass
class BacktestConfig:
    ticker: str = "AAPL"
    benchmark: str = "SPY"
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100_000.0
    data_dir: str = "data"


# Ready-to-use defaults
STRATEGY = StrategyConfig()
BACKTEST = BacktestConfig()
