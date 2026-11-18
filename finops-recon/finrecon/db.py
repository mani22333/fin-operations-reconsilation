"""SQLite persistence layer.

Normalised schema with keys + indexes; every reconciliation run is recorded so
results are reproducible and auditable. SQL is used for the set-based work
(aging, duplicates, category roll-ups); pandas for the matching logic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS counterparties (
    counterparty_id   TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('vendor','customer')),
    currency          TEXT NOT NULL,
    country           TEXT
);

CREATE TABLE IF NOT EXISTS fx_rates (
    date         TEXT NOT NULL,
    currency     TEXT NOT NULL,
    rate_to_inr  REAL NOT NULL,
    PRIMARY KEY (date, currency)
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id        TEXT PRIMARY KEY,
    type              TEXT NOT NULL CHECK (type IN ('AP','AR')),
    counterparty_id   TEXT NOT NULL REFERENCES counterparties(counterparty_id),
    issue_date        TEXT NOT NULL,
    due_date          TEXT NOT NULL,
    currency          TEXT NOT NULL,
    amount            REAL NOT NULL,
    category          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger (
    entry_id          TEXT PRIMARY KEY,
    payment_id        TEXT NOT NULL,
    account           TEXT NOT NULL,
    posting_date      TEXT NOT NULL,
    amount            REAL NOT NULL,          -- base currency, signed (+in / -out)
    reference         TEXT,
    counterparty_id   TEXT REFERENCES counterparties(counterparty_id),
    counterparty_type TEXT,
    category          TEXT,
    currency_orig     TEXT,
    amount_orig       REAL,
    booking_rate      REAL,
    payment_run       TEXT
);
CREATE INDEX IF NOT EXISTS ix_ledger_ref  ON ledger(reference);
CREATE INDEX IF NOT EXISTS ix_ledger_acct ON ledger(account, posting_date);

CREATE TABLE IF NOT EXISTS bank (
    bank_txn_id  TEXT PRIMARY KEY,
    account      TEXT NOT NULL,
    value_date   TEXT NOT NULL,
    amount       REAL NOT NULL,
    narrative    TEXT
);
CREATE INDEX IF NOT EXISTS ix_bank_acct ON bank(account, value_date);

CREATE TABLE IF NOT EXISTS recon_runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    as_of        TEXT NOT NULL,
    input_hash   TEXT NOT NULL,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    run_id       TEXT NOT NULL REFERENCES recon_runs(run_id),
    match_id     TEXT NOT NULL,
    match_type   TEXT NOT NULL,
    bank_txn_id  TEXT NOT NULL,
    entry_id     TEXT NOT NULL,
    PRIMARY KEY (run_id, bank_txn_id, entry_id)
);

CREATE TABLE IF NOT EXISTS exceptions (
    run_id        TEXT NOT NULL REFERENCES recon_runs(run_id),
    exception_id  TEXT NOT NULL,
    kind          TEXT NOT NULL,
    severity      TEXT NOT NULL,
    account       TEXT,
    entry_id      TEXT,
    bank_txn_id   TEXT,
    reference     TEXT,
    amount        REAL,
    details       TEXT,
    PRIMARY KEY (run_id, exception_id)
);
"""

# ---------------------------------------------------------------- queries
SQL_DUPLICATE_CANDIDATES = """
-- Payments against the same invoice with identical amounts, where the total
-- paid exceeds the invoice value (converted to base currency at booking rate).
WITH paid AS (
    SELECT reference, SUM(ABS(amount_orig)) AS paid_orig, COUNT(*) AS n
    FROM ledger GROUP BY reference
)
SELECT l.entry_id, l.reference, l.account, l.amount, l.posting_date,
       i.amount AS invoice_amount, p.paid_orig, p.n
FROM ledger l
JOIN paid p     ON p.reference = l.reference
JOIN invoices i ON i.invoice_id = l.reference
WHERE p.n > 1
  AND p.paid_orig > i.amount * 1.001
  AND EXISTS (
      SELECT 1 FROM ledger l2
      WHERE l2.reference = l.reference
        AND l2.entry_id < l.entry_id
        AND ABS(l2.amount_orig - l.amount_orig) < 0.01
        AND ABS(julianday(l.posting_date) - julianday(l2.posting_date)) <= :window
  )
ORDER BY l.reference, l.posting_date;
"""

SQL_ORPHAN_REFERENCES = """
SELECT l.entry_id, l.reference, l.account, l.amount
FROM ledger l
LEFT JOIN invoices i ON i.invoice_id = l.reference
WHERE i.invoice_id IS NULL;
"""

SQL_OPEN_ITEMS = """
-- Open balance per invoice in base currency (invoice-date rate) as of :as_of
WITH paid AS (
    SELECT reference, SUM(ABS(amount_orig)) AS paid_orig
    FROM ledger WHERE posting_date <= :as_of GROUP BY reference
)
SELECT i.invoice_id, i.type, i.counterparty_id, c.name AS counterparty_name,
       i.issue_date, i.due_date, i.currency, i.category,
       i.amount                                    AS invoice_amount,
       COALESCE(p.paid_orig, 0)                    AS paid_amount,
       ROUND(i.amount - COALESCE(p.paid_orig, 0), 2)                 AS open_amount,
       ROUND((i.amount - COALESCE(p.paid_orig, 0)) * f.rate_to_inr, 2) AS open_amount_base,
       CAST(julianday(:as_of) - julianday(i.due_date) AS INTEGER)   AS days_past_due
FROM invoices i
JOIN counterparties c ON c.counterparty_id = i.counterparty_id
LEFT JOIN paid p      ON p.reference = i.invoice_id
LEFT JOIN fx_rates f  ON f.currency = i.currency AND f.date = i.issue_date
WHERE i.issue_date <= :as_of
  AND i.amount - COALESCE(p.paid_orig, 0) > 0.01;
"""

SQL_CATEGORY_SUMMARY = """
SELECT category, counterparty_type,
       COUNT(*)              AS entries,
       ROUND(SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END), 2) AS outflow,
       ROUND(SUM(CASE WHEN amount > 0 THEN  amount ELSE 0 END), 2) AS inflow
FROM ledger
GROUP BY category, counterparty_type
ORDER BY counterparty_type, outflow DESC, inflow DESC;
"""


def _iso(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)

    def load(self, t: dict[str, pd.DataFrame]) -> None:
        c = self.conn
        for tbl in ("exceptions", "matches", "recon_runs", "bank", "ledger", "invoices",
                    "fx_rates", "counterparties"):
            c.execute(f"DELETE FROM {tbl}")
        cps = pd.concat([t["vendors"].assign(kind="vendor"), t["customers"].assign(kind="customer")])
        cps[["counterparty_id", "name", "kind", "currency", "country"]].to_sql(
            "counterparties", c, if_exists="append", index=False)
        _iso(t["fx_rates"], ["date"]).to_sql("fx_rates", c, if_exists="append", index=False)
        inv = _iso(t["invoices"], ["issue_date", "due_date"])
        inv.drop(columns=["counterparty_name"], errors="ignore").to_sql(
            "invoices", c, if_exists="append", index=False)
        led = _iso(t["ledger"], ["posting_date"]).drop(columns=["counterparty_name"], errors="ignore")
        led.to_sql("ledger", c, if_exists="append", index=False)
        _iso(t["bank"], ["value_date"]).to_sql("bank", c, if_exists="append", index=False)
        c.commit()

    def query(self, sql: str, **params) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def save_run(self, run_id, started_at, as_of, input_hash, summary_json, matches, exceptions):
        c = self.conn
        c.execute("INSERT INTO recon_runs VALUES (?,?,?,?,?)",
                  (run_id, started_at, as_of, input_hash, summary_json))
        if len(matches):
            matches.assign(run_id=run_id)[["run_id", "match_id", "match_type", "bank_txn_id", "entry_id"]] \
                .to_sql("matches", c, if_exists="append", index=False)
        if len(exceptions):
            cols = ["run_id", "exception_id", "kind", "severity", "account", "entry_id",
                    "bank_txn_id", "reference", "amount", "details"]
            exceptions.assign(run_id=run_id)[cols].to_sql("exceptions", c, if_exists="append", index=False)
        c.commit()
