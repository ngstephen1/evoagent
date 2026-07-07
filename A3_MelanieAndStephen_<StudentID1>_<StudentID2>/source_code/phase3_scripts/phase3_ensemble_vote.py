"""
phase3_ensemble_vote.py — Majority-vote ensemble across several submission CSVs.

For each test id, collect the predicted_value from every input CSV, group values
that are numerically equal (within a tolerance), and keep the value with the most
votes. Ties are broken by the --priority CSV (your current best), so the ensemble
never loses to it on a coin-flip row. CPU-only; no model or GPU needed.

Usage:
    python3 phase3_ensemble_vote.py \
        --inputs a.csv b.csv c.csv \
        --priority b.csv \
        --test data/test.json \
        --output submission_ensemble.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_csv(path: str) -> dict[str, str]:
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows[str(r["id"])] = str(r.get("predicted_value", "")).strip()
    return rows


def to_float(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Majority-vote ensemble of submission CSVs.")
    p.add_argument("--inputs", nargs="+", required=True, help="Two or more submission CSVs.")
    p.add_argument("--priority", required=True, help="CSV whose value wins ties (your current best).")
    p.add_argument("--test", default="data/test.json", help="test.json for canonical id order.")
    p.add_argument("--output", required=True, help="Output ensemble CSV path.")
    p.add_argument("--tol", type=float, default=1e-4, help="Rel/abs tolerance for treating values as equal.")
    return p.parse_args()


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= max(tol, tol * max(abs(a), abs(b), 1.0))


def main() -> None:
    args = parse_args()
    subs = [load_csv(p) for p in args.inputs]
    prio = load_csv(args.priority)
    test = json.loads(Path(args.test).read_text())
    ids = [str(t["id"]) for t in test]

    out_rows = []
    changed_vs_priority = 0
    for qid in ids:
        # Gather this row's values from every input.
        vals = [s[qid] for s in subs if qid in s and s[qid] != ""]
        nums = [(v, to_float(v)) for v in vals]
        # Bucket numerically-equal values; vote by bucket size.
        buckets = []  # list of [repr_value_str, count, first_seen_order]
        for order, (vstr, vnum) in enumerate(nums):
            placed = False
            for b in buckets:
                if b["num"] is not None and vnum is not None and close(b["num"], vnum, args.tol):
                    b["count"] += 1
                    placed = True
                    break
                if b["num"] is None and vnum is None and b["str"] == vstr:
                    b["count"] += 1
                    placed = True
                    break
            if not placed:
                buckets.append({"str": vstr, "num": vnum, "count": 1, "order": order})

        prio_val = prio.get(qid, "")
        if not buckets:
            winner = prio_val
        else:
            max_count = max(b["count"] for b in buckets)
            top = [b for b in buckets if b["count"] == max_count]
            if len(top) == 1:
                winner = top[0]["str"]
            else:
                # Tie: prefer the priority CSV's value if it's among the tied buckets.
                pnum = to_float(prio_val)
                winner = prio_val
                if not any(
                    (b["num"] is not None and pnum is not None and close(b["num"], pnum, args.tol))
                    or (b["num"] is None and b["str"] == prio_val)
                    for b in top
                ):
                    # Priority not in the tie -> fall back to earliest-seen tied value.
                    winner = sorted(top, key=lambda b: b["order"])[0]["str"]

        if winner != prio_val:
            changed_vs_priority += 1
        out_rows.append((qid, winner))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Usage", "predicted_value"])
        for qid, val in out_rows:
            w.writerow([qid, "Public", val])

    print(f"inputs: {len(subs)} | rows: {len(out_rows)}")
    print(f"rows changed vs priority ({Path(args.priority).name}): {changed_vs_priority}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
