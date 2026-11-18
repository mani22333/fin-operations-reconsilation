"""Bank <-> ledger matching engine.

Multi-pass, one-to-one (plus batch one-to-many) matching:

  1. BATCH       one bank debit == sum of a ledger payment run
  2. EXACT       same account, same reference, same amount, dates within window
  3. REFERENCE   same account + reference, amount within ref tolerance  -> variance classified later
  4. HEURISTIC   no reference on the bank line: counterparty name + near amount + date window

Each pass only sees records left unmatched by earlier passes. Within a pass,
candidate pairs are scored (amount difference dominates, then date gap) and
assigned optimally so every record is used at most once.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import DEFAULT, ReconConfig

REF_RE = re.compile(r"((?:INV|BILL)-\d+)")
BATCH_RE = re.compile(r"BATCH\s+(RUN-[\d-]+)")


def _prep(ledger: pd.DataFrame, bank: pd.DataFrame):
    ledger = ledger.copy()
    bank = bank.copy()
    ledger["posting_date"] = pd.to_datetime(ledger["posting_date"])
    bank["value_date"] = pd.to_datetime(bank["value_date"])
    bank["ref"] = bank["narrative"].fillna("").str.extract(REF_RE, expand=False)
    bank["batch"] = bank["narrative"].fillna("").str.extract(BATCH_RE, expand=False)
    bank["name_token"] = bank["narrative"].fillna("").str.split("/").str[-1].str.strip()
    ledger["name_token"] = ledger["counterparty_name"].fillna("").str.upper().str[:18].str.strip()
    return ledger, bank


def _assign(cands: pd.DataFrame, score: str, group: list[str]) -> pd.DataFrame:
    """One-to-one assignment of candidate (bank_txn_id, entry_id) pairs.

    Candidates are split into independent groups (e.g. same account + reference).
    Groups with a single candidate are taken directly; contested groups are
    solved with the Hungarian algorithm, which maximises the number of matches
    first and then minimises total score - unlike greedy best-first, it will not
    steal a record that a neighbour needs.
    """
    if cands.empty:
        return cands
    keep = []
    for _, g in cands.groupby(group, sort=False):
        if len(g) == 1:
            keep.append(g.index[0])
            continue
        bs, ls = g["bank_txn_id"].unique(), g["entry_id"].unique()
        if len(bs) == 1 or len(ls) == 1:
            keep.append(g[score].idxmin())
            continue
        bi = {b: i for i, b in enumerate(bs)}; li = {e: j for j, e in enumerate(ls)}
        big = 1e9
        cost = np.full((len(bs), len(ls)), big)
        idx = {}
        for ix, b, e, sc in zip(g.index, g["bank_txn_id"], g["entry_id"], g[score]):
            if sc < cost[bi[b], li[e]]:
                cost[bi[b], li[e]] = sc; idx[(bi[b], li[e])] = ix
        rows, cols = linear_sum_assignment(cost)
        keep += [idx[(r, c)] for r, c in zip(rows, cols) if cost[r, c] < big]
    return cands.loc[keep]


def _pairs(l_open, b_open, on, cfg):
    c = l_open.merge(b_open, on=on, suffixes=("_l", "_b"))
    c["day_gap"] = (c["value_date"] - c["posting_date"]).dt.days.abs()
    c["amt_diff"] = (c["amount_b"] - c["amount_l"]).abs()
    c["rel_diff"] = c["amt_diff"] / c["amount_l"].abs().clip(lower=1)
    same_sign = np.sign(c["amount_b"]) == np.sign(c["amount_l"])
    return c[(c["day_gap"] <= cfg.date_window_days) & same_sign]


def reconcile(ledger: pd.DataFrame, bank: pd.DataFrame, cfg: ReconConfig = DEFAULT):
    """Return (matches, unmatched_ledger, unmatched_bank).

    matches has one row per (bank_txn_id, entry_id) pair with match_type,
    amount_l, amount_b, amt_diff, rel_diff, day_gap.
    """
    ledger, bank = _prep(ledger, bank)
    L = ledger[["entry_id", "account", "posting_date", "amount", "reference", "currency_orig",
                "counterparty_name", "name_token", "payment_run"]]
    B = bank[["bank_txn_id", "account", "value_date", "amount", "narrative", "ref", "batch", "name_token"]]
    matched_l, matched_b, out = set(), set(), []

    def open_sets():
        return L[~L["entry_id"].isin(matched_l)], B[~B["bank_txn_id"].isin(matched_b)]

    def take(sel: pd.DataFrame, mtype: str):
        if sel.empty:
            return
        sel = sel.assign(match_type=mtype)
        matched_l.update(sel["entry_id"]); matched_b.update(sel["bank_txn_id"])
        out.append(sel)

    # Pass 1: payment-run batches — run first so members are not consumed one by one
    l_o, b_o = open_sets()
    runs = l_o.dropna(subset=["payment_run"]).groupby(["account", "payment_run"]).agg(
        total=("amount", "sum"), last_date=("posting_date", "max")).reset_index()
    bb = b_o.dropna(subset=["batch"]).merge(runs, left_on=["account", "batch"],
                                            right_on=["account", "payment_run"])
    bb = bb[(bb["amount"] - bb["total"]).abs() <= cfg.amount_tolerance_abs * 10]
    for r in bb.itertuples():
        members = l_o[(l_o["payment_run"] == r.payment_run) & (l_o["account"] == r.account)]
        take(pd.DataFrame({
            "bank_txn_id": r.bank_txn_id, "entry_id": members["entry_id"].values,
            "account": r.account, "amount_l": members["amount"].values,
            # bank total allocated to members so per-row variances sum correctly
            "amount_b": members["amount"].values + (r.amount - members["amount"].sum()) / len(members),
            "posting_date": members["posting_date"].values, "value_date": r.value_date,
            "currency_orig": members["currency_orig"].values, "ref": members["reference"].values,
            "amt_diff": 0.0, "rel_diff": 0.0, "day_gap": 0}), "BATCH")

    # Pass 2: exact
    l_o, b_o = open_sets()
    c = _pairs(l_o.rename(columns={"reference": "ref"}), b_o, ["account", "ref"], cfg)
    c = c[c["amt_diff"] <= cfg.amount_tolerance_abs]
    take(_assign(c.assign(s=c["day_gap"]), "s", ["account", "ref"]), "EXACT")

    # Pass 3: reference with amount variance
    l_o, b_o = open_sets()
    c = _pairs(l_o.rename(columns={"reference": "ref"}), b_o, ["account", "ref"], cfg)
    c = c[c["rel_diff"] <= cfg.ref_match_max_rel_diff]
    take(_assign(c.assign(s=c["rel_diff"] * 1e4 + c["day_gap"]), "s", ["account", "ref"]), "REFERENCE")

    # Pass 4: heuristic for bank lines without a usable reference
    l_o, b_o = open_sets()
    c = _pairs(l_o.rename(columns={"reference": "ref_l"}), b_o, ["account", "name_token"], cfg)
    c = c[c["rel_diff"] <= cfg.noref_match_max_rel_diff]
    c = c.drop(columns=["ref"]).rename(columns={"ref_l": "ref"})
    take(_assign(c.assign(s=c["rel_diff"] * 1e4 + c["day_gap"]), "s", ["account", "name_token"]), "HEURISTIC")

    cols = ["match_type", "bank_txn_id", "entry_id", "account", "ref", "posting_date", "value_date",
            "amount_l", "amount_b", "amt_diff", "rel_diff", "day_gap", "currency_orig"]
    matches = pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)
    matches = matches.rename(columns={"ref": "reference"})
    # stable match ids; batch lines share the bank txn's id
    ids = {b: f"M{i:06d}" for i, b in enumerate(matches["bank_txn_id"].unique(), 1)}
    matches.insert(0, "match_id", matches["bank_txn_id"].map(ids))

    unmatched_l = ledger[~ledger["entry_id"].isin(matched_l)].drop(columns=["name_token"])
    unmatched_b = bank[~bank["bank_txn_id"].isin(matched_b)].drop(columns=["name_token", "batch"])
    return matches, unmatched_l.reset_index(drop=True), unmatched_b.reset_index(drop=True)
