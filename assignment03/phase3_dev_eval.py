"""
phase3_dev_eval.py — LOCAL validation harness on the labeled dev set.

The Kaggle test set has no local gold labels, so every change so far had to be
probed blindly on Kaggle (which is how v3 regressed). This harness gives a local
accuracy number on dev.json (584 rows with gold), sliced by the same error
categories used in the ensemble analysis, so future changes can be measured
BEFORE spending a Kaggle submission.

It mirrors the project's own scoring (src/executor.py):
    is_correct = predicted AND (value_match OR exact_program_match)
    value_match = abs(pred_val - gold_val) <= TOL   (default 1e-4, absolute)

Two entry points:

  # 1. Self-test: score the GOLD programs against exe_ans. Needs NO model / GPU.
  #    Validates that the executor + metric faithfully reproduce the labels and
  #    establishes the local-fidelity ceiling. Run this now.
  python3 phase3_dev_eval.py --selftest

  # 2. Score a real predictions file produced by a model on dev:
  #    JSON  {id: program_string}  or  CSV with columns id,predicted_value
  python3 phase3_dev_eval.py --pred my_dev_preds.json --mode program
  python3 phase3_dev_eval.py --pred my_dev_preds.csv  --mode value
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.evaluator import evaluate_program          # noqa: E402
from src.executor import normalize_program          # noqa: E402

DATA = HERE / "data"
TOL = 1e-4
RATIO_CUES = ["tỷ lệ", "phần trăm", "tăng trưởng", "thay đổi", "%", "tỷ trọng", "biên", "chiếm bao nhiêu"]


def tof(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def gold_value(item):
    exe = item["qa"].get("exe_ans")
    v = tof(exe)
    if v is not None:
        return v
    try:
        return evaluate_program(item["qa"].get("program", ""), item.get("table", []))
    except Exception:
        return None


def categories(item):
    """Return the slice labels for one dev item."""
    qid = item["id"]
    q = item["qa"]["question"].lower()
    prog = item["qa"].get("program", "") or ""
    n_ops = prog.count("(") if prog else 0
    cats = {
        "source": "MASVN" if qid.split("/")[0] == "masvn" else "US-filing",
        "complexity": "1-op" if n_ops <= 1 else ("2-op" if n_ops == 2 else "3+op"),
        "kind": "table-lookup" if "table_" in prog.lower() else "arithmetic",
        "ratio_q": "ratio/percent-Q" if any(c in q for c in RATIO_CUES) else "plain-Q",
    }
    return cats


def score(pred_map, mode, tol=TOL):
    """pred_map: id -> program_str (mode=program) or id -> float-ish (mode=value)."""
    dev = json.loads((DATA / "dev.json").read_text())
    n = 0
    correct = 0
    by = defaultdict(lambda: [0, 0])           # slice-value -> [correct, total]
    misses = []                                # wrong rows for inspection
    for item in dev:
        qid = item["id"]
        gv = gold_value(item)
        if gv is None:
            continue
        n += 1
        raw = pred_map.get(qid)
        pv = None
        exact = False
        if raw is not None and str(raw).strip() != "":
            if mode == "program":
                try:
                    pv = evaluate_program(str(raw), item.get("table", []))
                except Exception:
                    pv = None
                exact = normalize_program(str(raw)) == normalize_program(item["qa"].get("program", ""))
            else:  # value
                pv = tof(raw)
        value_match = pv is not None and gv is not None and abs(pv - gv) <= tol
        ok = bool(raw is not None and (value_match or exact))
        if ok:
            correct += 1
        else:
            misses.append((qid, gv, pv, item["qa"]["question"]))
        for _, label in categories(item).items():
            by[label][1] += 1
            if ok:
                by[label][0] += 1
    return n, correct, by, misses


def report(n, correct, by, misses, show_misses=0):
    print(f"\nDev accuracy: {correct}/{n} = {100*correct/n:.2f}%\n")
    # group slice labels back under their dimension for tidy printing
    order = ["MASVN", "US-filing", "arithmetic", "table-lookup",
             "1-op", "2-op", "3+op", "ratio/percent-Q", "plain-Q"]
    print("By slice (accuracy — where the errors concentrate):")
    for label in order:
        if label in by:
            c, t = by[label]
            print(f"  {label:16s} {c:4d}/{t:<4d}  {100*c/t:5.1f}%")
    if show_misses:
        print(f"\nSample misses ({min(show_misses, len(misses))} of {len(misses)}):")
        for qid, gv, pv, q in misses[:show_misses]:
            print(f"  gold={gv:<12g} pred={pv}  {qid[:48]}")
            print(f"      Q: {q[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="Score the GOLD programs against exe_ans (no model needed).")
    ap.add_argument("--pred", help="Predictions file: .json {id:program} or .csv id,predicted_value")
    ap.add_argument("--mode", choices=["program", "value"], default="program")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--show-misses", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        dev = json.loads((DATA / "dev.json").read_text())
        pred_map = {it["id"]: it["qa"].get("program", "") for it in dev}
        print("=== SELF-TEST: gold programs vs exe_ans (validates executor + metric) ===")
        n, correct, by, misses = score(pred_map, "program", args.tol)
        report(n, correct, by, misses, args.show_misses)
        print(f"\nInterpretation: {100*correct/n:.1f}% is the local-fidelity ceiling — the fraction of dev")
        print("the executor can reproduce from gold. Rows below 100% are executor gaps or")
        print("non-numeric golds, NOT model errors. Any model eval is measured against this.")
        return

    if not args.pred:
        ap.error("provide --selftest or --pred FILE")
    p = Path(args.pred)
    field = "program" if args.mode == "program" else "predicted_value"
    if p.suffix == ".json":
        obj = json.loads(p.read_text())
        if isinstance(obj, list):
            # submit.py's *_details.json format: [{id, program, predicted_value, ...}]
            pred_map = {str(r["id"]): r.get(field, "") for r in obj}
        else:
            pred_map = {str(k): v for k, v in obj.items()}
    else:
        pred_map = {}
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            pred_map[str(r["id"])] = r.get("predicted_value", r.get("program", ""))
    print(f"=== EVAL: {p.name} (mode={args.mode}) ===")
    n, correct, by, misses = score(pred_map, args.mode, args.tol)
    report(n, correct, by, misses, args.show_misses)


if __name__ == "__main__":
    main()
