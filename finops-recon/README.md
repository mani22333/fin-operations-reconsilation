# Financial Operations & Reconciliation System

Automated bank reconciliation, AP/AR open-item processing, internal-control checks and an audit-ready Excel report, built with **Python, SQL (SQLite), pandas and openpyxl**.

```
 bank.csv ─┐                       ┌─► matches ─────────┐
 ledger.csv┼─► validate ─► SQLite ─┤                    ├─► exception register ─► Excel report
 invoices  │   (schema,    (normal-│   4-pass matcher   │   (controls: SQL +    │  + run_summary.json
 fx_rates ─┘    keys)       ised)  └─► unmatched L / B ─┘    pandas)            └─► recon_runs table (audit trail)
```

## What it does

| Area | Implementation |
|---|---|
| **Bank reconciliation** | 4-pass matcher (`reconcile.py`): **BATCH** (one bulk bank debit ↔ an ERP payment run), **EXACT** (account + invoice ref + amount), **REFERENCE** (same ref, amount variance kept for classification), **HEURISTIC** (no ref on the bank line: counterparty name + near amount + date window). Contested candidates are resolved with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), so one record never steals another's match. |
| **AP/AR processing** | SQL open-item query (`db.py → SQL_OPEN_ITEMS`) computes invoice balances net of partial payments, converts to INR at invoice-date FX, and buckets into Current / 1-30 / 31-60 / 61-90 / 90+ aging. |
| **Internal controls** | `controls.py` raises a severity-ranked exception for each of: **duplicate payment** (SQL: same invoice, same amount, total paid > invoiced; legitimate instalments are not flagged), **orphan reference** (payment against a non-existent invoice), **amount mismatch** (domestic), **FX variance** (cross-border settlement > 1% from booking rate), **unmatched ledger** (booked, never cleared), **unmatched bank** (unrecorded movement), **bank charges** (info only, with proposed journal entries). |
| **Audit-ready reporting** | `recon_report_<date>.xlsx`: a reconciliation statement built from **live SUMIFS/COUNTIFS formulas** over the detail sheets (it closes to 0 per account), an exception register with severity colouring, Matched / Unmatched sheets, proposed JEs, AP & AR aging, open items and a category roll-up. Each run is stored in `recon_runs` with an input SHA-256 hash. |

## Quick start

```bash
pip install -r requirements.txt
python -m finrecon.cli demo                 # generate sample data + reconcile + report
# or, with your own CSVs in ./data:
python -m finrecon.cli run --data data --out output [--as-of 2026-09-30] [--json]
python -m pytest -q
```

### Input format (`data/*.csv`)

| File | Required columns |
|---|---|
| `ledger.csv` | entry_id, payment_id, account, posting_date, amount (signed, INR), reference, counterparty_id, counterparty_name, counterparty_type, category, currency_orig, amount_orig, booking_rate, payment_run |
| `bank.csv` | bank_txn_id, account, value_date, amount (signed), narrative |
| `invoices.csv` | invoice_id, type (AP/AR), counterparty_id, issue_date, due_date, currency, amount, category |
| `fx_rates.csv` | date, currency, rate_to_inr |
| `vendors.csv`, `customers.csv` | counterparty_id, name, currency, country |

Input is validated before any numbers are produced: missing columns, duplicate or null keys, and null amounts all fail the run.

## Measured results (synthetic data, `python -m finrecon.cli demo`)

The generator (`generate.py`) produces realistic data: 8,000 invoices across 280 counterparties in INR/USD/EUR/GBP, partial payments, payment-run batches, bank-date lag, missing references in about 10% of bank narratives, and FX drift. It then **injects labelled anomalies**, so detection quality is *measured* against ground truth:

| Metric | Value |
|---|---|
| Transactions processed (ledger + bank) | 9,783 |
| Ledger / bank auto-match rate | 99.2% / 98.8% |
| Unexplained difference after reconciliation | 0.00 on every account |
| Anomaly recall / precision (seed 7) | 99.2% / 98.8% (244 of 246 found, 3 FP) |
| Recall range across seeds 1–3 | 98.8% – 100% |
| End-to-end runtime | ≈ 0.6 s (plus about 5 s to write Excel) |

The few misses come from genuinely ambiguous cases, for example an FX payment whose natural market drift is close to the 1% threshold.

A sample workbook from this run is in `sample_output/recon_report_sample.xlsx`.

## Configuration
All tolerances are in `finrecon/config.py`: date window, variance limits, FX threshold, duplicate window, charge keywords and aging buckets.

## Project layout
```
finrecon/
  config.py     tolerances & thresholds
  generate.py   synthetic data + ground-truth anomalies
  db.py         SQLite schema, indexes, SQL controls/aging queries, run persistence
  reconcile.py  multi-pass matcher with optimal assignment
  controls.py   exception register + detection evaluation
  report.py     formatted Excel workbook with live formulas
  pipeline.py   orchestration, validation, input hashing
  cli.py        `generate`, `run`, `demo`
tests/          unit + end-to-end tests
```
