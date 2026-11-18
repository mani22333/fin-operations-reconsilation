import json

import pandas as pd
import pytest

from finrecon.config import DEFAULT
from finrecon.controls import run_controls
from finrecon.db import Store
from finrecon.generate import generate
from finrecon.pipeline import run, validate_inputs
from finrecon.reconcile import reconcile


def _ledger(rows):
    base = {"account": "HDFC-OPS", "counterparty_name": "Acme Traders 001", "currency_orig": "INR",
            "payment_run": None, "counterparty_id": "V0001", "counterparty_type": "vendor",
            "category": "Rent", "booking_rate": 1.0, "payment_id": "P"}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["amount_orig"] = df.get("amount_orig", df["amount"])
    return df


def _bank(rows):
    return pd.DataFrame([{"account": "HDFC-OPS", **r} for r in rows])


def test_exact_reference_and_variance_passes():
    L = _ledger([
        {"entry_id": "L1", "posting_date": "2026-09-01", "amount": -1000.0, "reference": "BILL-1"},
        {"entry_id": "L2", "posting_date": "2026-09-02", "amount": -2000.0, "reference": "BILL-2"},
    ])
    B = _bank([
        {"bank_txn_id": "B1", "value_date": "2026-09-02", "amount": -1000.0, "narrative": "NEFT/BILL-1/ACME"},
        {"bank_txn_id": "B2", "value_date": "2026-09-03", "amount": -2050.0, "narrative": "NEFT/BILL-2/ACME"},
    ])
    m, ul, ub = reconcile(L, B)
    assert dict(zip(m["entry_id"], m["match_type"])) == {"L1": "EXACT", "L2": "REFERENCE"}
    assert ul.empty and ub.empty


def test_batch_payment_run_matches_many_to_one():
    L = _ledger([
        {"entry_id": f"L{i}", "posting_date": "2026-09-05", "amount": -a, "reference": f"BILL-{i}",
         "payment_run": "RUN-0905-1"} for i, a in enumerate([100.0, 250.5, 49.5], 1)
    ])
    B = _bank([{"bank_txn_id": "B9", "value_date": "2026-09-06", "amount": -400.0,
                "narrative": "BULK PAYMENT/BATCH RUN-0905-1"}])
    m, ul, ub = reconcile(L, B)
    assert set(m["match_type"]) == {"BATCH"} and len(m) == 3
    assert m["amount_b"].sum() == pytest.approx(-400.0)


def test_heuristic_prefers_exact_amount_over_closer_date():
    # Two receipts from the same customer, bank lines carry no invoice reference.
    L = _ledger([
        {"entry_id": "L1", "posting_date": "2026-09-01", "amount": 52611.50, "reference": "INV-1",
         "counterparty_name": "Kaveri Pvt Ltd 121"},
        {"entry_id": "L2", "posting_date": "2026-09-03", "amount": 51806.13, "reference": "INV-2",
         "counterparty_name": "Kaveri Pvt Ltd 121"},
    ])
    B = _bank([
        {"bank_txn_id": "B1", "value_date": "2026-09-04", "amount": 52611.50, "narrative": "NEFT/KAVERI PVT LTD 121"},
        {"bank_txn_id": "B2", "value_date": "2026-09-06", "amount": 51806.13, "narrative": "RTGS/KAVERI PVT LTD 121"},
    ])
    m, _, _ = reconcile(L, B)
    assert dict(zip(m["entry_id"], m["bank_txn_id"])) == {"L1": "B1", "L2": "B2"}
    assert (m["amt_diff"] == 0).all()


def test_outside_date_window_is_not_matched():
    L = _ledger([{"entry_id": "L1", "posting_date": "2026-09-01", "amount": -10.0, "reference": "BILL-1"}])
    B = _bank([{"bank_txn_id": "B1", "value_date": "2026-09-20", "amount": -10.0, "narrative": "NEFT/BILL-1/X"}])
    m, ul, ub = reconcile(L, B)
    assert m.empty and len(ul) == 1 and len(ub) == 1


def test_validate_rejects_duplicate_keys(tmp_path):
    t = generate(tmp_path, n_invoices=300)
    t["ledger"] = pd.concat([t["ledger"], t["ledger"].head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_inputs(t)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("data"); o = tmp_path_factory.mktemp("out")
    generate(d, seed=11, n_invoices=3000)
    return run(d, o), o


def test_reconciliation_closes_to_zero(full_run):
    res, _ = full_run
    assert all(abs(v) < 0.05 for v in res["summary"]["unexplained_difference"].values())


def test_detection_quality_on_injected_anomalies(full_run):
    det = full_run[0]["summary"]["detection_vs_ground_truth"]
    assert det["recall"] >= 0.95 and det["precision"] >= 0.95
    for kind, v in det["per_kind"].items():
        assert v["detected"] >= 0.9 * v["injected"], kind


def test_outputs_written_and_run_persisted(full_run):
    res, o = full_run
    assert (o / "exceptions.csv").exists() and (o / "run_summary.json").exists()
    assert json.loads((o / "run_summary.json").read_text())["run_id"] == res["meta"]["run_id"]
    import sqlite3
    con = sqlite3.connect(o / "finrecon.db")
    assert con.execute("select count(*) from recon_runs").fetchone()[0] == 1
    assert con.execute("select count(*) from exceptions").fetchone()[0] > 0


def test_duplicate_control_ignores_legit_instalments(tmp_path):
    t = generate(tmp_path, n_invoices=200, anomaly_rate=0.0)
    s = Store(); s.load(t)
    m, ul, ub = reconcile(t["ledger"], t["bank"])
    exc = run_controls(s, m, ul, ub, DEFAULT)
    assert "DUPLICATE_PAYMENT" not in set(exc["kind"])
