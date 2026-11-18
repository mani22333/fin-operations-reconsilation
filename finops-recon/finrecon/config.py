"""Central configuration for reconciliation tolerances and control thresholds."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReconConfig:
    # Matching windows
    date_window_days: int = 4            # max |bank value date - ledger posting date|
    amount_tolerance_abs: float = 0.01   # treated as an exact match below this (rounding)
    ref_match_max_rel_diff: float = 0.10 # same reference: accept pair up to 10% apart, then classify
    noref_match_max_rel_diff: float = 0.05

    # Controls
    fx_variance_threshold: float = 0.01  # cross-border settlement drift above 1% is flagged
    duplicate_window_days: int = 7
    bank_charge_max_abs: float = 2_500.0 # small unmatched bank debits with charge keywords

    # Reporting
    base_currency: str = "INR"
    aging_buckets: tuple = (0, 30, 60, 90)
    charge_keywords: tuple = field(default=("CHG", "CHARGES", "FEE", "GST ON", "SMS"))


DEFAULT = ReconConfig()
