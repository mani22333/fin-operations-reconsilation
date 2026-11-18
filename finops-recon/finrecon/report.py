"""Audit-ready Excel workbook.

The bank reconciliation statement on the Summary sheet is built from live
SUMIFS/COUNTIFS formulas over the detail sheets, so a reviewer can trace every
number back to the underlying rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

NAVY = "1F3864"
HDR_FILL = PatternFill("solid", fgColor=NAVY)
HDR_FONT = Font(bold=True, color="FFFFFF")
INPUT_FONT = Font(color="0000FF")
BOLD = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
NUM = '#,##0.00;[Red](#,##0.00);"-"'
SEV_FILL = {"CRITICAL": "F8CBAD", "HIGH": "FCE4D6", "MEDIUM": "FFF2CC", "INFO": "E2EFDA"}


def _sheet(wb, name, df: pd.DataFrame, money_cols=(), widths=None):
    ws = wb.create_sheet(name)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
    for i, col in enumerate(df.columns, 1):
        L = get_column_letter(i)
        if col in money_cols:
            for cell in ws[L][1:]:
                cell.number_format = NUM
        w = (widths or {}).get(col) or min(max(len(str(col)), *(len(str(v)) for v in df[col].head(200)), 8) + 2, 60) \
            if len(df) else len(str(col)) + 4
        ws.column_dimensions[L].width = w
    if len(df):
        ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
        t = Table(displayName=name.replace(" ", ""), ref=ref)
        t.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        ws.add_table(t)
    else:
        for c in ws[1]:
            c.fill, c.font = HDR_FILL, HDR_FONT
    ws.freeze_panes = "A2"
    return ws


def _aging(open_items: pd.DataFrame, typ: str, buckets) -> pd.DataFrame:
    df = open_items[open_items["type"] == typ].copy()
    edges = list(buckets) + [10**6]
    labels = ["Current"] + [f"{edges[i] + 1}-{edges[i + 1]}" for i in range(len(edges) - 2)] + [f"{edges[-2]}+"]
    df["bucket"] = pd.cut(df["days_past_due"], [-10**6] + edges, labels=labels, right=True)
    pv = df.pivot_table(index="counterparty_name", columns="bucket", values="open_amount_base",
                        aggfunc="sum", fill_value=0, observed=False)
    pv["Total"] = pv.sum(axis=1)
    return pv.sort_values("Total", ascending=False).reset_index()


def write_report(path: str | Path, *, run_meta: dict, matches: pd.DataFrame, unmatched_l: pd.DataFrame,
                 unmatched_b: pd.DataFrame, exceptions: pd.DataFrame, open_items: pd.DataFrame,
                 category_summary: pd.DataFrame, ledger: pd.DataFrame, bank: pd.DataFrame,
                 buckets=(0, 30, 60, 90)) -> Path:
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"

    # ---------------- detail sheets ----------------
    _sheet(wb, "Exceptions", exceptions, money_cols=["amount"], widths={"details": 80})
    ws = wb["Exceptions"]
    sev_col = get_column_letter(list(exceptions.columns).index("severity") + 1)
    for sev, color in SEV_FILL.items():
        ws.conditional_formatting.add(f"{sev_col}2:{sev_col}{len(exceptions) + 1}",
                                      CellIsRule(operator="equal", formula=[f'"{sev}"'],
                                                 fill=PatternFill("solid", fgColor=color)))
    _sheet(wb, "Matched", matches, money_cols=["amount_l", "amount_b", "amt_diff"])
    ul = unmatched_l[["entry_id", "account", "posting_date", "amount", "reference", "counterparty_name",
                      "category", "currency_orig", "amount_orig"]]
    _sheet(wb, "Unmatched Ledger", ul, money_cols=["amount", "amount_orig"])
    ub = unmatched_b[["bank_txn_id", "account", "value_date", "amount", "narrative", "ref"]]
    _sheet(wb, "Unmatched Bank", ub, money_cols=["amount"])

    charges = exceptions[exceptions["kind"] == "BANK_CHARGE"]
    je = pd.DataFrame({
        "account": charges["account"], "bank_txn_id": charges["bank_txn_id"],
        "debit_gl": "6105 Bank Charges", "credit_gl": "1010 Cash - " + charges["account"],
        "amount": charges["amount"].abs(), "memo": charges["details"]})
    _sheet(wb, "Proposed JEs", je, money_cols=["amount"])

    ap = _aging(open_items, "AP", buckets)
    ar = _aging(open_items, "AR", buckets)
    _sheet(wb, "AP Aging", ap, money_cols=[c for c in ap.columns if c != "counterparty_name"])
    _sheet(wb, "AR Aging", ar, money_cols=[c for c in ar.columns if c != "counterparty_name"])
    _sheet(wb, "Open Items", open_items, money_cols=["invoice_amount", "paid_amount", "open_amount",
                                                     "open_amount_base"])
    _sheet(wb, "By Category", category_summary, money_cols=["outflow", "inflow"])

    # ---------------- summary / reconciliation statement ----------------
    s = summary
    s.column_dimensions["A"].width = 46
    for col in "BCDE":
        s.column_dimensions[col].width = 20
    s["A1"] = "Bank Reconciliation & AP/AR Control Report"; s["A1"].font = Font(bold=True, size=14, color=NAVY)
    meta = [("Run ID", run_meta["run_id"]), ("Period as of", run_meta["as_of"]),
            ("Generated", run_meta["started_at"]), ("Input SHA-256", run_meta["input_hash"][:16] + "…"),
            ("Runtime (s)", run_meta["runtime_s"])]
    for i, (k, v) in enumerate(meta, 3):
        s[f"A{i}"], s[f"B{i}"] = k, v
        s[f"A{i}"].font = BOLD

    r = 9
    s[f"A{r}"] = "Reconciliation statement (formulas over detail sheets)"; s[f"A{r}"].font = BOLD
    r += 1
    accounts = sorted(set(ledger["account"]) | set(bank["account"]))
    s.cell(r, 1, "Line item").font = HDR_FONT
    for j, a in enumerate(accounts, 2):
        s.cell(r, j, a).font = HDR_FONT
    for j in range(1, len(accounts) + 2):
        s.cell(r, j).fill = HDR_FILL
    hdr = r
    nm, nl, nb = len(matches) + 1, len(ul) + 1, len(ub) + 1
    bank_net = bank.groupby("account")["amount"].sum()
    led_net = ledger.groupby("account")["amount"].sum()
    lines = [
        ("Net movement per bank statement", lambda a, L: round(float(bank_net.get(a, 0)), 2), True),
        ("Net movement per ledger", lambda a, L: round(float(led_net.get(a, 0)), 2), True),
        ("Difference (bank − ledger)", lambda a, L: f"={L}{hdr + 1}-{L}{hdr + 2}", False),
        ("Less: unrecorded bank items", lambda a, L:
            f"=SUMIFS('Unmatched Bank'!D2:D{nb},'Unmatched Bank'!B2:B{nb},{L}${hdr})", False),
        ("Add: ledger items not yet in bank", lambda a, L:
            f"=SUMIFS('Unmatched Ledger'!D2:D{nl},'Unmatched Ledger'!B2:B{nl},{L}${hdr})", False),
        ("Less: settlement variances on matched pairs", lambda a, L:
            f"=SUMIFS(Matched!J2:J{nm},Matched!E2:E{nm},{L}${hdr})-SUMIFS(Matched!I2:I{nm},Matched!E2:E{nm},{L}${hdr})",
         False),
        ("Unexplained difference (should be 0)", lambda a, L:
            f"=ROUND({L}{hdr + 3}-{L}{hdr + 4}+{L}{hdr + 5}-{L}{hdr + 6},2)", False),
    ]
    # Matched sheet: I = amount_l, J = amount_b, E = account (match_id,match_type,bank,entry,account,...)
    assert list(matches.columns)[4] == "account" and list(matches.columns)[8:10] == ["amount_l", "amount_b"]
    for i, (label, fn, is_input) in enumerate(lines, hdr + 1):
        s.cell(i, 1, label)
        for j, a in enumerate(accounts, 2):
            L = get_column_letter(j)
            c = s.cell(i, j, fn(a, L))
            c.number_format = NUM
            if is_input:
                c.font = INPUT_FONT
            c.border = Border(top=THIN, bottom=THIN)
    last = hdr + len(lines)
    for j in range(1, len(accounts) + 2):
        s.cell(last, j).font = BOLD

    r = last + 3
    s[f"A{r}"] = "Exception register"; s[f"A{r}"].font = BOLD
    r += 1
    for j, h in enumerate(["Kind", "Count", "Amount", "Severity"], 1):
        c = s.cell(r, j, h); c.fill, c.font = HDR_FILL, HDR_FONT
    ne = len(exceptions) + 1
    kinds = exceptions.drop_duplicates("kind")[["kind", "severity"]]
    for i, k in enumerate(kinds.itertuples(), r + 1):
        s.cell(i, 1, k.kind)
        s.cell(i, 2, f'=COUNTIFS(Exceptions!B2:B{ne},A{i})')
        s.cell(i, 3, f'=SUMIFS(Exceptions!H2:H{ne},Exceptions!B2:B{ne},A{i})').number_format = NUM
        s.cell(i, 4, k.severity).fill = PatternFill("solid", fgColor=SEV_FILL[k.severity])
    assert list(exceptions.columns)[1] == "kind" and list(exceptions.columns)[7] == "amount"

    r = r + len(kinds) + 3
    s[f"A{r}"] = "Match statistics"; s[f"A{r}"].font = BOLD
    stats = [("Ledger entries", len(ledger)), ("Bank lines", len(bank)),
             ("Matched pairs", f"=COUNTA(Matched!A2:A{nm})"),
             ("Ledger match rate", f"=1-COUNTA('Unmatched Ledger'!A2:A{nl})/B{r + 1}"),
             ("Bank match rate", f"=1-COUNTA('Unmatched Bank'!A2:A{nb})/B{r + 2}"),
             ("Open AP (base ccy)", "=SUMIFS('Open Items'!L:L,'Open Items'!B:B,\"AP\")"),
             ("Open AR (base ccy)", "=SUMIFS('Open Items'!L:L,'Open Items'!B:B,\"AR\")")]
    assert list(open_items.columns)[11] == "open_amount_base"
    for i, (k, v) in enumerate(stats, r + 1):
        s.cell(i, 1, k); c = s.cell(i, 2, v)
        c.number_format = "0.0%" if "rate" in k else ("#,##0" if i < r + 4 else NUM)
    for row in s.iter_rows():
        for c in row:
            c.alignment = Alignment(vertical="center")

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
