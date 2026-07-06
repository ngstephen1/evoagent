"""
phase3_scale_fix.py — targeted, category-gated post-processor for the ensemble CSV.

Fixes the single most common error class found in the ensemble error analysis
(2026-07-06): percent-vs-ratio scale mistakes. It is deliberately CONSERVATIVE —
broad numeric post-processing hurt the score in Run 005, so a row is only changed
when ALL of the following hold:

  1. The question is a ratio/percent question (cue words in Vietnamese).
  2. The current ensemble value |v| > 1.5   (too big to be a decimal ratio).
  3. Some MEMBER model actually produced the ÷100 form (v/100), i.e. the ratio
     reading is not invented by us — a model already voted for it.
  4. That ÷100 form lands in the plausible ratio band |v/100| <= 1.5.

Everything else (sign flips, unit/×1e9 outliers, zeros) is only REPORTED for
manual review, never auto-applied.

Dev check confirms the direction: for ratio/percent-cue questions the gold answer
is a decimal ratio (median |ans| = 0.375; 71% <= 1.0), NOT a percent.

Usage:
    python3 phase3_scale_fix.py           # dry-run: print the diff, write nothing
    python3 phase3_scale_fix.py --apply   # write submission_ensemble_v2.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

KAGGLE = Path(__file__).resolve().parent.parent / "A3_MelanieAndStephen_<StudentID1>_<StudentID2>" / "kaggle"
DATA = Path(__file__).resolve().parent / "data"
MEMBERS = {
    "run009": "final_submission.csv",
    "8b_sc5": "submission_8b_sc5.csv",
    "coder7b": "submission_coder7b.csv",
}
ENSEMBLE = "submission_ensemble.csv"
OUTPUT = "submission_ensemble_v2.csv"

RATIO_CUES = ["tỷ lệ", "phần trăm", "tăng trưởng", "thay đổi", "%", "tỷ trọng", "biên", "chiếm bao nhiêu"]
RATIO_BAND = 1.5     # a decimal ratio should sit within +-1.5
TOL = 1e-4


def load(path):
    out, order = {}, []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        for r in rd:
            qid = str(r["id"])
            out[qid] = r
            order.append(qid)
    return out, order, fields


def tof(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def close(a, b, tol=TOL):
    return abs(a - b) <= max(tol, tol * max(abs(a), abs(b), 1.0))


def is_ratio_q(q):
    ql = q.lower()
    return any(c in ql for c in RATIO_CUES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the fixed CSV; default is dry-run")
    args = ap.parse_args()

    ens_rows, order, fields = load(KAGGLE / ENSEMBLE)
    members = {k: load(KAGGLE / v)[0] for k, v in MEMBERS.items()}
    test = {str(t["id"]): t for t in json.loads((DATA / "test.json").read_text())}

    applied = []          # scale fixes we make
    sign_flags = []       # reported only
    unit_flags = []       # reported only

    for qid in order:
        row = ens_rows[qid]
        v = tof(row.get("predicted_value", ""))
        q = test.get(qid, {}).get("qa", {}).get("question", "")
        member_vals = [tof(members[m][qid]["predicted_value"]) for m in MEMBERS if qid in members[m]]
        member_vals = [x for x in member_vals if x is not None]

        # ---- Rule: percent -> ratio (the only auto-applied fix) ----
        if v is not None and is_ratio_q(q) and abs(v) > RATIO_BAND:
            target = v / 100.0
            member_has_ratio = any(close(mv, target) for mv in member_vals)
            if member_has_ratio and abs(target) <= RATIO_BAND:
                applied.append((qid, v, target, q))
                if args.apply:
                    row["predicted_value"] = repr(target)
                continue

        # ---- Report-only: sign disagreement (2 members share |val|, opposite sign to ensemble) ----
        if v is not None and member_vals:
            same_mag_opp = [mv for mv in member_vals if close(abs(mv), abs(v)) and mv * v < 0]
            if len(same_mag_opp) >= 2:
                sign_flags.append((qid, v, same_mag_opp[0], q))

        # ---- Report-only: extreme unit outlier ----
        if v is not None and abs(v) >= 1e6:
            unit_flags.append((qid, v, q))

    # -------- report --------
    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — targeted scale fixer over {ENSEMBLE}\n")
    print(f"=== AUTO-APPLIED: percent→ratio scale fixes ({len(applied)}) ===")
    for qid, old, new, q in applied:
        print(f"  {old:>14g}  ->  {new:<14g}   {qid[:58]}")
        print(f"                  Q: {q[:78]}")
    if not applied:
        print("  (none matched the strict gate)")

    print(f"\n=== REPORT-ONLY: sign-flip candidates ({len(sign_flags)}) — review manually, NOT changed ===")
    for qid, v, mv, q in sign_flags:
        print(f"  ensemble={v:g}  members show {mv:g}   {qid[:56]}")

    print(f"\n=== REPORT-ONLY: extreme unit/outlier candidates ({len(unit_flags)}) — review manually ===")
    for qid, v, q in unit_flags:
        print(f"  {v:g}   {qid[:60]}")

    if args.apply:
        outp = KAGGLE / OUTPUT
        with open(outp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for qid in order:
                w.writerow(ens_rows[qid])
        print(f"\nWrote {len(applied)} changes -> {outp}")
    else:
        print(f"\n(dry-run: nothing written. Re-run with --apply to produce {OUTPUT}.)")


if __name__ == "__main__":
    main()
