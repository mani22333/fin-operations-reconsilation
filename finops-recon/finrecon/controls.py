"""Internal financial controls: turns match results + SQL checks into a
prioritised exception register."""
from __future__ import annotations

import pandas as pd

from .config import DEFAULT, ReconConfig
from .db import SQL_DUPLICATE_CANDIDATES, SQL_ORPHAN_REFERENCES, Store

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
EXC_COLS = ["kind", "severity", "account", "entry_id", "bank_txn_id", "reference", "amount", "details"]


def _rows(df, kind, severity, details_fn, entry="entry_id", bank="bank_txn_id", amount="amount"):
    if df is None or df.empty:
        return pd.DataFrame(columns=EXC_COLS)
    return pd.DataFrame({
        "kind": kind, "severity": severity,
        "account": df["account"].values,
        "entry_id": df[entry].values if entry in df else None,
        "bank_txn_id": df[bank].values if bank in df else None,
        "reference": df["reference"].values if "reference" in df else None,
        "amount": df[amount].round(2).values,
        "details": [details_fn(r) for r in df.itertuples()],
    })


def run_controls(store: Store, matches: pd.DataFrame, unmatched_l: pd.DataFrame,
                 unmatched_b: pd.DataFrame, cfg: ReconConfig = DEFAULT) -> pd.DataFrame:
    parts = []

    # --- 1. Duplicate payments (SQL set-based check)
    dups = store.query(SQL_DUPLICATE_CANDIDATES, window=cfg.duplicate_window_days)
    parts.append(_rows(dups, "DUPLICATE_PAYMENT", "CRITICAL",
                       lambda r: f"{r.n} payments vs invoice {r.reference}; paid {r.paid_orig:,.2f} "
                                 f"> invoiced {r.invoice_amount:,.2f}. Recover or reverse."))
    dup_ids = set(dups["entry_id"])

    # --- 2. Orphan references (paid against an invoice that does not exist)
    orph = store.query(SQL_ORPHAN_REFERENCES)
    parts.append(_rows(orph, "ORPHAN_REFERENCE", "HIGH",
                       lambda r: f"Reference {r.reference} not found in invoice master."))

    # --- 3. Variances on matched pairs
    var = matches[(matches["amt_diff"] > cfg.amount_tolerance_abs) & (matches["match_type"] != "BATCH")].copy()
    var["amount"] = var["amount_b"] - var["amount_l"]
    fx = var[var["currency_orig"] != cfg.base_currency]
    fx_flag = fx[fx["rel_diff"] > cfg.fx_variance_threshold]
    parts.append(_rows(fx_flag, "FX_VARIANCE", "MEDIUM",
                       lambda r: f"{r.currency_orig} settlement off booking by {r.rel_diff:.2%} "
                                 f"(ledger {r.amount_l:,.2f} vs bank {r.amount_b:,.2f})."))
    dom = var[var["currency_orig"] == cfg.base_currency]
    parts.append(_rows(dom, "AMOUNT_MISMATCH", "HIGH",
                       lambda r: f"Ledger {r.amount_l:,.2f} vs bank {r.amount_b:,.2f} "
                                 f"(diff {r.amount_b - r.amount_l:+,.2f})."))

    # --- 4. Unmatched ledger -> booked, never cleared the bank
    ul = unmatched_l[~unmatched_l["entry_id"].isin(dup_ids)]
    parts.append(_rows(ul, "UNMATCHED_LEDGER", "HIGH",
                       lambda r: f"Posted {pd.Timestamp(r.posting_date):%d-%b} to {r.counterparty_name}; "
                                 f"no bank movement found."))

    # --- 5. Unmatched bank -> charges (expected) vs unrecorded movements
    ub = unmatched_b.copy()
    pat = "|".join(cfg.charge_keywords)
    is_chg = (ub["amount"] < 0) & (ub["amount"].abs() <= cfg.bank_charge_max_abs) & \
             ub["narrative"].str.upper().str.contains(pat, regex=True)
    ub = ub.rename(columns={"ref": "reference"})
    parts.append(_rows(ub[is_chg], "BANK_CHARGE", "INFO",
                       lambda r: f"'{r.narrative}' - post to Bank Charges expense.", entry="-"))
    parts.append(_rows(ub[~is_chg], "UNMATCHED_BANK", "HIGH",
                       lambda r: f"'{r.narrative}' on {pd.Timestamp(r.value_date):%d-%b} not in ledger"
                                 + (f"; likely relates to {r.reference}." if isinstance(r.reference, str) else "."),
                       entry="-"))

    exc = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    exc["_o"] = exc["severity"].map(SEVERITY_ORDER)
    exc = exc.sort_values(["_o", "kind", "account"]).drop(columns="_o").reset_index(drop=True)
    exc.insert(0, "exception_id", [f"E{i:05d}" for i in range(1, len(exc) + 1)])
    return exc


def evaluate(exceptions: pd.DataFrame, anomalies: pd.DataFrame,
             ledger: pd.DataFrame | None = None, bank: pd.DataFrame | None = None) -> dict:
    """Compare flagged exceptions with injected ground truth (kind + record must agree).

    Records that are indistinguishable (two ledger lines with the same reference
    and amount, e.g. equal instalments) are treated as equivalent: flagging
    either one is correct, since no reviewer could tell them apart either.
    """
    canon = {}
    if ledger is not None:
        k = ledger["reference"].astype(str) + "|" + ledger["amount"].round(2).astype(str)
        canon.update(dict(zip(ledger["entry_id"], "L|" + k)))
    if bank is not None:
        k = bank["account"] + "|" + bank["amount"].round(2).astype(str) + "|" + bank["narrative"].astype(str)
        canon.update(dict(zip(bank["bank_txn_id"], "B|" + k)))
    c = lambda x: canon.get(x, x)

    flagged = exceptions[exceptions["severity"] != "INFO"]
    f_keys = [{(kd, c(e)), (kd, c(b))} for kd, e, b in
              zip(flagged["kind"], flagged["entry_id"].fillna(""), flagged["bank_txn_id"].fillna(""))]
    flag_set = set().union(*f_keys) if f_keys else set()
    truth = [(kd, c(r)) for kd, r in zip(anomalies["kind"], anomalies["record_id"])]
    truth_set = set(truth)
    tp = sum(1 for t in truth if t in flag_set)
    fp = sum(1 for keys in f_keys if not keys & truth_set)
    per_kind = {}
    for kind in sorted(anomalies["kind"].unique()):
        ts = [t for t in truth if t[0] == kind]
        per_kind[kind] = {"injected": len(ts), "detected": sum(1 for t in ts if t in flag_set)}
    return {
        "injected": len(truth), "true_positives": tp, "false_positives": fp,
        "recall": round(tp / len(truth), 4) if truth else 1.0,
        "precision": round((len(flagged) - fp) / len(flagged), 4) if len(flagged) else 1.0,
        "per_kind": per_kind,
    }
