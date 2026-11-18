"""Synthetic but realistic AP/AR + bank data generator.

Produces the five input tables the system consumes, plus a ground-truth file of
deliberately injected anomalies so detection accuracy can be measured rather
than asserted.

Tables
------
vendors / customers  : counterparty masters
invoices.csv         : AP bills and AR invoices (multi-currency)
fx_rates.csv         : daily rate of each currency to INR
ledger.csv           : general-ledger cash postings (what the books say)
bank.csv             : bank statement lines (what the bank says)
anomalies.csv        : ground truth of injected problems
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACCOUNTS = {"HDFC-OPS": "AP", "ICICI-COLL": "AR"}  # payables / collections accounts
CURRENCIES = {"INR": 1.0, "USD": 83.2, "EUR": 90.4, "GBP": 105.7}
CATEGORIES_AP = ["Raw Materials", "Logistics", "Utilities", "Software", "Rent",
                 "Professional Fees", "Marketing", "Maintenance"]
CATEGORIES_AR = ["Product Sales", "Services", "Subscriptions", "Export Sales"]
NAMES = ["Acme", "Bharat", "Coastal", "Delta", "Everest", "Futura", "Ganga", "Horizon",
         "Indus", "Jaguar", "Kaveri", "Lotus", "Meridian", "Nova", "Orion", "Pinnacle",
         "Quantum", "Royal", "Sapphire", "Titan", "Unity", "Vertex", "Western", "Zenith"]
SUFFIX = ["Traders", "Industries", "Logistics", "Tech", "Exports", "Pvt Ltd", "LLP", "Corp"]


def _counterparties(rng, n, prefix, foreign_share):
    rows = []
    for i in range(n):
        name = f"{rng.choice(NAMES)} {rng.choice(SUFFIX)} {i:03d}"
        cur = rng.choice(["USD", "EUR", "GBP"]) if rng.random() < foreign_share else "INR"
        rows.append({"counterparty_id": f"{prefix}{i:04d}", "name": name,
                     "currency": cur, "country": "IN" if cur == "INR" else "XX"})
    return pd.DataFrame(rows)


def _fx_table(rng, start, days):
    dates = pd.date_range(start, periods=days + 60, freq="D")
    rows = []
    for cur, base in CURRENCIES.items():
        walk = base * np.exp(np.cumsum(rng.normal(0, 0.002, len(dates)))) if cur != "INR" else np.ones(len(dates))
        rows += [{"date": d, "currency": cur, "rate_to_inr": round(r, 4)} for d, r in zip(dates, walk)]
    return pd.DataFrame(rows)


def generate(out_dir: str | Path = "data", seed: int = 7, n_invoices: int = 4200,
             period_start: str = "2026-09-01", period_days: int = 30,
             anomaly_rate: float = 0.05) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(period_start)
    period_end = start + pd.Timedelta(days=period_days - 1)

    vendors = _counterparties(rng, 120, "V", 0.15)
    customers = _counterparties(rng, 160, "C", 0.20)
    fx = _fx_table(rng, start - pd.Timedelta(days=60), period_days)
    fx_idx = fx.set_index(["date", "currency"])["rate_to_inr"]

    def rate(d, cur):
        return float(fx_idx[(pd.Timestamp(d).normalize(), cur)])

    # ---------------- invoices ----------------
    inv_rows = []
    for i in range(n_invoices):
        is_ap = rng.random() < 0.55
        cp = (vendors if is_ap else customers).sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        issue = start - pd.Timedelta(days=int(rng.integers(0, 75))) + pd.Timedelta(days=int(rng.integers(0, period_days)))
        terms = int(rng.choice([15, 30, 45, 60]))
        amt_inr = float(np.round(np.exp(rng.normal(10.5, 1.0)), 2))  # lognormal, median ~36k INR
        amt = round(amt_inr / CURRENCIES[cp.currency], 2)
        inv_rows.append({
            "invoice_id": f"{'BILL' if is_ap else 'INV'}-{i:05d}", "type": "AP" if is_ap else "AR",
            "counterparty_id": cp.counterparty_id, "counterparty_name": cp["name"],
            "issue_date": issue.normalize(), "due_date": (issue + pd.Timedelta(days=terms)).normalize(),
            "currency": cp.currency, "amount": amt,
            "category": rng.choice(CATEGORIES_AP if is_ap else CATEGORIES_AR),
        })
    invoices = pd.DataFrame(inv_rows)

    # ---------------- payments -> ledger + bank ----------------
    ledger, bank, anomalies = [], [], []
    seq = {"L": 0, "B": 0, "P": 0}

    def nid(k):
        seq[k] += 1
        return f"{k}{seq[k]:06d}"

    def narrative(inv, kind="NEFT", with_ref=True):
        ref = f"/{inv.invoice_id}" if with_ref else ""
        return f"{kind}{ref}/{inv.counterparty_name.upper()[:18]}"

    batches: dict[str, list] = {}
    for inv in invoices.itertuples():
        # ~70% of invoices see payment activity inside the statement period
        if rng.random() > 0.70:
            continue
        pay_date = max(inv.issue_date + pd.Timedelta(days=int(rng.integers(3, 50))), start)
        if pay_date > period_end:
            continue
        sign = -1.0 if inv.type == "AP" else 1.0
        account = "HDFC-OPS" if inv.type == "AP" else "ICICI-COLL"
        # partial payments in two instalments for ~8%
        parts = [0.5, 0.5] if rng.random() < 0.08 else ([0.7] if rng.random() < 0.05 else [1.0])
        for j, frac in enumerate(parts):
            pdate = min(pay_date + pd.Timedelta(days=7 * j), period_end)
            book_rate = rate(pdate, inv.currency)
            amt_orig = round(inv.amount * frac, 2)
            ledger_inr = round(amt_orig * book_rate, 2)
            pid = nid("P")
            payment_run = None
            if inv.type == "AP" and inv.currency == "INR" and rng.random() < 0.12:
                payment_run = f"RUN-{pdate:%m%d}-{int(rng.integers(1, 3))}"
            led = {"entry_id": nid("L"), "payment_id": pid, "account": account,
                   "posting_date": pdate, "amount": sign * ledger_inr, "reference": inv.invoice_id,
                   "counterparty_id": inv.counterparty_id, "counterparty_name": inv.counterparty_name,
                   "counterparty_type": "vendor" if inv.type == "AP" else "customer",
                   "category": inv.category, "currency_orig": inv.currency,
                   "amount_orig": sign * amt_orig, "booking_rate": book_rate,
                   "payment_run": payment_run}
            ledger.append(led)
            lag = int(rng.choice([0, 0, 1, 1, 2, 3]))
            # settlement rate drifts a little from booking rate for cross-border
            drift = rng.normal(0, 0.003) if inv.currency != "INR" else 0.0
            bank_inr = round(ledger_inr * (1 + drift), 2)
            if payment_run:
                batches.setdefault(payment_run, []).append((led, pdate + pd.Timedelta(days=lag), bank_inr))
                continue
            kind = "SWIFT" if inv.currency != "INR" else rng.choice(["NEFT", "RTGS", "IMPS", "UPI"])
            bank.append({"bank_txn_id": nid("B"), "account": account,
                         "value_date": pdate + pd.Timedelta(days=lag), "amount": sign * bank_inr,
                         "narrative": narrative(inv, kind, with_ref=rng.random() > 0.10),
                         "_entry_id": led["entry_id"]})

    for run, items in batches.items():
        total = round(sum(b for _, _, b in items), 2)
        bank.append({"bank_txn_id": nid("B"), "account": "HDFC-OPS", "value_date": max(d for _, d, _ in items),
                     "amount": -total, "narrative": f"BULK PAYMENT/BATCH {run}", "_entry_id": None})

    ledger = pd.DataFrame(ledger)
    bank = pd.DataFrame(bank)

    # ---------------- inject anomalies ----------------
    n_anom = int(len(ledger) * anomaly_rate)
    single = bank[bank["_entry_id"].notna()]
    eligible = ledger[ledger["entry_id"].isin(single["_entry_id"])]

    def pick(mask, k, exclude):
        pool = eligible[mask & ~eligible["entry_id"].isin(exclude)]
        return pool.sample(min(k, len(pool)), random_state=int(rng.integers(1e9)))

    used: set[str] = set()
    k = n_anom // 6
    dom = eligible["currency_orig"] == "INR"
    full = ~eligible["reference"].isin(ledger.groupby("reference").filter(lambda g: len(g) > 1)["reference"])

    # 1. Duplicate payments: same invoice paid again a few days later (ledger + bank)
    for r in pick(dom & full, k, used).itertuples():
        used.add(r.entry_id)
        d = r._asdict(); d.pop("Index")
        d.update(entry_id=nid("L"), payment_id=nid("P"),
                 posting_date=r.posting_date + pd.Timedelta(days=int(rng.integers(1, 5))))
        ledger = pd.concat([ledger, pd.DataFrame([d])], ignore_index=True)
        orig_b = bank[bank["_entry_id"] == r.entry_id].iloc[0]
        bank = pd.concat([bank, pd.DataFrame([{**orig_b.to_dict(), "bank_txn_id": nid("B"),
                                                "value_date": d["posting_date"], "_entry_id": d["entry_id"]}])],
                         ignore_index=True)
        anomalies.append({"kind": "DUPLICATE_PAYMENT", "record_id": d["entry_id"], "reference": r.reference})

    # 2. Missing in bank: ledger posted, bank never shows it
    for r in pick(eligible["entry_id"].notna(), k, used).itertuples():
        used.add(r.entry_id)
        bank = bank[bank["_entry_id"] != r.entry_id]
        anomalies.append({"kind": "UNMATCHED_LEDGER", "record_id": r.entry_id, "reference": r.reference})

    # 3. Missing in ledger: bank shows a receipt/payment the books never recorded
    for r in pick(eligible["entry_id"].notna(), k, used).itertuples():
        used.add(r.entry_id)
        btxn = bank.loc[bank["_entry_id"] == r.entry_id, "bank_txn_id"].iloc[0]
        ledger = ledger[ledger["entry_id"] != r.entry_id]
        anomalies.append({"kind": "UNMATCHED_BANK", "record_id": btxn, "reference": r.reference})

    # 4. Amount mismatch (domestic): bank settled a different amount than booked
    for r in pick(dom, k, used).itertuples():
        used.add(r.entry_id)
        m = bank["_entry_id"] == r.entry_id
        delta = abs(r.amount) * rng.uniform(0.005, 0.04) * rng.choice([-1, 1])
        bank.loc[m, "amount"] = round(bank.loc[m, "amount"].iloc[0] + np.sign(r.amount) * delta, 2)
        anomalies.append({"kind": "AMOUNT_MISMATCH", "record_id": r.entry_id, "reference": r.reference})

    # 5. FX variance: cross-border settled at a rate >1% away from booking rate
    for r in pick(~dom, k, used).itertuples():
        used.add(r.entry_id)
        m = bank["_entry_id"] == r.entry_id
        bank.loc[m, "amount"] = round(r.amount * (1 + rng.choice([-1, 1]) * rng.uniform(0.018, 0.045)), 2)
        anomalies.append({"kind": "FX_VARIANCE", "record_id": r.entry_id, "reference": r.reference})

    # 6. Orphan reference: payment booked against an invoice number that does not exist
    for r in pick(dom & full, k, used).itertuples():
        used.add(r.entry_id)
        bad = r.reference.replace("-", "-9")
        ledger.loc[ledger["entry_id"] == r.entry_id, "reference"] = bad
        m = bank["_entry_id"] == r.entry_id
        bank.loc[m, "narrative"] = bank.loc[m, "narrative"].str.replace(r.reference, bad, regex=False)
        anomalies.append({"kind": "ORPHAN_REFERENCE", "record_id": r.entry_id, "reference": bad})

    # Expected (non-anomalous) bank-only items: charges
    for acct in ACCOUNTS:
        for _ in range(int(rng.integers(6, 12))):
            bank = pd.concat([bank, pd.DataFrame([{
                "bank_txn_id": nid("B"), "account": acct,
                "value_date": start + pd.Timedelta(days=int(rng.integers(0, period_days))),
                "amount": -round(float(rng.uniform(15, 1800)), 2),
                "narrative": rng.choice(["SMS CHG", "NEFT CHARGES", "GST ON CHARGES", "ACCOUNT MAINT FEE"]),
                "_entry_id": None}])], ignore_index=True)

    bank = bank.drop(columns="_entry_id").sample(frac=1, random_state=seed).reset_index(drop=True)
    ledger = ledger.sort_values(["posting_date", "entry_id"]).reset_index(drop=True)
    anomalies = pd.DataFrame(anomalies)

    tables = {"vendors": vendors, "customers": customers, "invoices": invoices,
              "fx_rates": fx, "ledger": ledger, "bank": bank, "anomalies": anomalies}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic AP/AR + bank data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--invoices", type=int, default=4200)
    a = ap.parse_args()
    t = generate(a.out, a.seed, a.invoices)
    print({k: len(v) for k, v in t.items()})
