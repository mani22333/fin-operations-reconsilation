"""End-to-end run: load -> reconcile -> controls -> persist -> report."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DEFAULT, ReconConfig
from .controls import evaluate, run_controls
from .db import SQL_CATEGORY_SUMMARY, SQL_OPEN_ITEMS, Store
from .reconcile import reconcile
from .report import write_report

INPUTS = ("vendors", "customers", "invoices", "fx_rates", "ledger", "bank")
DATES = {"invoices": ["issue_date", "due_date"], "fx_rates": ["date"],
         "ledger": ["posting_date"], "bank": ["value_date"]}


def load_inputs(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    missing = [n for n in INPUTS if not (d / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files in {d}: {missing}")
    t = {n: pd.read_csv(d / f"{n}.csv", parse_dates=DATES.get(n, [])) for n in INPUTS}
    if (d / "anomalies.csv").exists():
        t["anomalies"] = pd.read_csv(d / "anomalies.csv")
    validate_inputs(t)
    return t


def validate_inputs(t: dict[str, pd.DataFrame]) -> None:
    """Schema & integrity checks - fail loudly before any numbers are produced."""
    req = {"ledger": {"entry_id", "account", "posting_date", "amount", "reference", "counterparty_name",
                      "currency_orig", "amount_orig", "payment_run"},
           "bank": {"bank_txn_id", "account", "value_date", "amount", "narrative"},
           "invoices": {"invoice_id", "type", "counterparty_id", "issue_date", "due_date", "currency", "amount"}}
    for name, cols in req.items():
        miss = cols - set(t[name].columns)
        if miss:
            raise ValueError(f"{name}: missing columns {sorted(miss)}")
    for name, key in (("ledger", "entry_id"), ("bank", "bank_txn_id"), ("invoices", "invoice_id")):
        if t[name][key].duplicated().any():
            raise ValueError(f"{name}: duplicate primary key values in {key}")
        if t[name][key].isna().any():
            raise ValueError(f"{name}: null {key}")
    for name, col in (("ledger", "amount"), ("bank", "amount")):
        if t[name][col].isna().any():
            raise ValueError(f"{name}: null amounts")


def _hash(t) -> str:
    h = hashlib.sha256()
    for n in INPUTS:
        h.update(pd.util.hash_pandas_object(t[n], index=False).values.tobytes())
    return h.hexdigest()


def run(data_dir: str | Path, out_dir: str | Path = "output", as_of: str | None = None,
        cfg: ReconConfig = DEFAULT, db_path: str | Path | None = None) -> dict:
    t0 = time.perf_counter()
    t = load_inputs(data_dir)
    as_of = as_of or pd.to_datetime(t["bank"]["value_date"]).max().strftime("%Y-%m-%d")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    store = Store(db_path or out / "finrecon.db")
    store.load(t)

    matches, ul, ub = reconcile(t["ledger"], t["bank"], cfg)
    exceptions = run_controls(store, matches, ul, ub, cfg)
    open_items = store.query(SQL_OPEN_ITEMS, as_of=as_of)
    cats = store.query(SQL_CATEGORY_SUMMARY)

    run_id = f"RUN-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    meta = {"run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"),
            "as_of": as_of, "input_hash": _hash(t)}
    summary = {
        "ledger_entries": len(t["ledger"]), "bank_lines": len(t["bank"]),
        "transactions_processed": len(t["ledger"]) + len(t["bank"]),
        "matched_pairs": len(matches),
        "match_types": matches["match_type"].value_counts().to_dict(),
        "ledger_match_rate": round(1 - len(ul) / len(t["ledger"]), 4),
        "bank_match_rate": round(1 - len(ub) / len(t["bank"]), 4),
        "exceptions": exceptions["kind"].value_counts().to_dict(),
        "open_ap_base": round(float(open_items.loc[open_items["type"] == "AP", "open_amount_base"].sum()), 2),
        "open_ar_base": round(float(open_items.loc[open_items["type"] == "AR", "open_amount_base"].sum()), 2),
    }
    # Independent check that the reconciliation closes to zero per account
    unexplained = {}
    for a in sorted(set(t["bank"]["account"]) | set(t["ledger"]["account"])):
        d = t["bank"].loc[t["bank"]["account"] == a, "amount"].sum() - \
            t["ledger"].loc[t["ledger"]["account"] == a, "amount"].sum()
        m = matches[matches["account"] == a]
        expl = ub.loc[ub["account"] == a, "amount"].sum() - ul.loc[ul["account"] == a, "amount"].sum() + \
            (m["amount_b"] - m["amount_l"]).sum()
        unexplained[a] = round(float(d - expl), 2)
    summary["unexplained_difference"] = unexplained
    if "anomalies" in t:
        summary["detection_vs_ground_truth"] = evaluate(exceptions, t["anomalies"], t["ledger"], t["bank"])

    meta["runtime_s"] = round(time.perf_counter() - t0, 2)
    summary["runtime_s"] = meta["runtime_s"]
    store.save_run(run_id, meta["started_at"], as_of, meta["input_hash"], json.dumps(summary, default=str),
                   matches, exceptions)
    xlsx = write_report(out / f"recon_report_{as_of}.xlsx", run_meta=meta, matches=matches, unmatched_l=ul,
                        unmatched_b=ub, exceptions=exceptions, open_items=open_items, category_summary=cats,
                        ledger=t["ledger"], bank=t["bank"], buckets=cfg.aging_buckets)
    exceptions.to_csv(out / "exceptions.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({**meta, **summary}, indent=2, default=str))
    return {"meta": meta, "summary": summary, "report": str(xlsx)}
