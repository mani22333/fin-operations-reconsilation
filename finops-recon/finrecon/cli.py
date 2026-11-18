"""Command line entry point.

    python -m finrecon.cli generate --out data
    python -m finrecon.cli run --data data --out output
    python -m finrecon.cli demo           # generate + run in one go
"""
from __future__ import annotations

import argparse
import json
import sys

from .generate import generate
from .pipeline import run


def _print(res):
    s = res["summary"]
    print(f"\nRun {res['meta']['run_id']}  (as of {res['meta']['as_of']}, {s['runtime_s']}s)")
    print(f"  Transactions processed : {s['transactions_processed']:,}")
    print(f"  Matched pairs          : {s['matched_pairs']:,}  {s['match_types']}")
    print(f"  Match rate ledger/bank : {s['ledger_match_rate']:.2%} / {s['bank_match_rate']:.2%}")
    print(f"  Unexplained difference : {s['unexplained_difference']}")
    print("  Exceptions:")
    for k, v in s["exceptions"].items():
        print(f"    {k:<20} {v}")
    if "detection_vs_ground_truth" in s:
        d = s["detection_vs_ground_truth"]
        print(f"  Detection vs injected  : recall {d['recall']:.2%}, precision {d['precision']:.2%} "
              f"({d['true_positives']}/{d['injected']}, {d['false_positives']} FP)")
    print(f"  Report                 : {res['report']}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="finrecon", description="Bank reconciliation & AP/AR controls")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", default="data")
    g.add_argument("--seed", type=int, default=7); g.add_argument("--invoices", type=int, default=8000)
    r = sub.add_parser("run"); r.add_argument("--data", default="data"); r.add_argument("--out", default="output")
    r.add_argument("--as-of"); r.add_argument("--json", action="store_true")
    d = sub.add_parser("demo"); d.add_argument("--out", default="output")
    d.add_argument("--invoices", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "generate":
        t = generate(a.out, a.seed, a.invoices)
        print({k: len(v) for k, v in t.items()})
    elif a.cmd == "run":
        res = run(a.data, a.out, a.as_of)
        print(json.dumps(res, indent=2, default=str)) if a.json else _print(res)
    else:
        generate("data", n_invoices=a.invoices)
        _print(run("data", a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
