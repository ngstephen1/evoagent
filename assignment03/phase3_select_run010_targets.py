"""Select suspicious rows for Run010 agentic solver.

This is Phase 3-only tooling. It builds an auditable target_rows.csv from the
current best Run009-lite safe submission using automatic signals only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_OUTPUT = Path("runs/kaggle_run010_agentic_solver/target_rows.csv")
DEFAULT_RUN001 = Path("runs/kaggle_arc_best/submission_checked.csv")
DEFAULT_RUN002 = Path("runs/kaggle_iter003/submission_checked.csv")
DEFAULT_RUN004 = Path("runs/kaggle_iter004/submission_checked.csv")
DEFAULT_RUN006 = Path("runs/kaggle_run006_iterbest_ctx32768/submission_checked.csv")
DEFAULT_RUN008 = Path("runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv")
DEFAULT_RUN009 = Path("runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv")
DEFAULT_DETAIL_PATHS = [
    Path("runs/kaggle_arc_best/submission_details.json"),
    Path("runs/kaggle_iter003/submission_details.json"),
    Path("runs/kaggle_iter004/submission_details.json"),
    Path("runs/kaggle_run006_iterbest_ctx32768/submission_details.json"),
    Path("runs/kaggle_hybrid_001_002/submission_details.json"),
    Path("runs/kaggle_retry_run008/retry_details.json"),
    Path("runs/kaggle_retry_run009_lite/retry_details.json"),
]

EXTREME_THRESHOLD = 1e6
ZERO_NEAR_THRESHOLD = 1e-9
ZERO_VS_LARGE_THRESHOLD = 1000.0

ABSOLUTE_AMOUNT_TERMS = (
    "tăng bao nhiêu",
    "mức tăng",
    "mức giảm",
    "chênh lệch",
)
ADDITION_TERMS = (
    "tổng",
    "tổng số",
    "cộng",
    "kết quả kèm tỷ trọng",
    "bao nhiêu triệu đồng",
    "bao nhiêu tỷ đồng",
)
SUBTRACTION_TERMS = (
    "chênh lệch",
    "thay đổi",
    "cao hơn",
    "thấp hơn",
    "giảm bao nhiêu",
    "tăng bao nhiêu",
    "so sánh",
)
TABLE_OP_TERMS = (
    "giá trị lớn nhất",
    "giá trị nhỏ nhất",
    "trung bình",
    "thấp nhất",
    "cao nhất",
    "tỷ lệ",
    "tỷ suất",
    "biên lợi nhuận",
    "roe",
    "roa",
    "nim",
)
DIVISION_PERCENT_TERMS = (
    "phần trăm",
    "%",
    "tỷ lệ",
    "tỷ suất",
    "roe",
    "roa",
    "biên",
    "chiếm bao nhiêu",
    "so với",
)
AMOUNT_TERMS = (
    "giá trị",
    "doanh thu",
    "lợi nhuận",
    "lntt",
    "tiền",
    "chi phí",
    "tổng",
    "dòng tiền",
    "bao nhiêu triệu",
    "bao nhiêu tỷ",
    "cổ phiếu",
    "khoản",
)
RATIO_TERMS = (
    "tỷ lệ",
    "tỷ suất",
    "phần trăm",
    "%",
    "roe",
    "roa",
    "biên",
    "nim",
    "p/b",
    "p/e",
    "chiếm",
)

REASON_WEIGHTS = {
    "zero": 1000,
    "extreme": 900,
    "negative_absolute_wording": 850,
    "magnitude_mismatch": 650,
    "high_disagreement": 550,
    "suspicious_program": 450,
    "weak_operation_type": 180,
}
REASON_PRIORITY_GROUP = {
    "zero": 0,
    "extreme": 1,
    "negative_absolute_wording": 2,
    "magnitude_mismatch": 3,
    "high_disagreement": 4,
    "suspicious_program": 5,
    "weak_operation_type": 6,
}

OUTPUT_COLUMNS = [
    "id",
    "priority_score",
    "target_reason",
    "old_value",
    "question",
    "run001_value",
    "run002_value",
    "run004_value",
    "run006_value",
    "run008_value",
    "run009_value",
    "finite_factor_ratio",
    "max_abs_source_value",
    "min_nonzero_abs_source_value",
    "program_detail_status",
    "magnitude_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Run010 target_rows.csv from automatic suspicious-row signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-targets", type=int, default=150)
    parser.add_argument("--run001-submission", type=Path, default=DEFAULT_RUN001)
    parser.add_argument("--run002-submission", type=Path, default=DEFAULT_RUN002)
    parser.add_argument("--run004-submission", type=Path, default=DEFAULT_RUN004)
    parser.add_argument("--run006-submission", type=Path, default=DEFAULT_RUN006)
    parser.add_argument("--run008-submission", type=Path, default=DEFAULT_RUN008)
    parser.add_argument("--run009-submission", type=Path, default=DEFAULT_RUN009)
    parser.add_argument("--detail-path", type=Path, action="append", default=list(DEFAULT_DETAIL_PATHS))
    parser.add_argument("--dry-run", action="store_true", help="Print estimates without writing output.")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def format_value(value: float | None) -> str:
    return "" if value is None else format(float(value), ".15g")


def load_submission(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = str(row.get("id") or "")
            value = to_float(row.get("predicted_value"))
            if row_id and value is not None:
                rows[row_id] = value
    return rows


def load_test(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list")
    return rows


def load_details(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("predictions"), list):
            data = data["predictions"]
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict) and row.get("id") is not None:
                copied = dict(row)
                copied["_detail_path"] = str(path)
                merged[str(row["id"])].append(copied)
    return dict(merged)


def question_text(row: dict[str, Any]) -> str:
    qa = row.get("qa")
    if isinstance(qa, dict) and qa.get("question"):
        return str(qa["question"])
    return str(row.get("question") or "")


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def program_detail_status(details: list[dict[str, Any]]) -> str:
    if not details:
        return "missing_program_detail"
    programs = [str(row.get("program") or row.get("selected_program") or "") for row in details]
    raw_outputs = [str(row.get("raw_output") or "") for row in details]
    joined = "\n".join(programs + raw_outputs).lower()
    if any(not program.strip() for program in programs):
        return "missing_program_detail"
    if any(len(program) > 1000 or program.count(",") > 15 for program in programs):
        return "repeated_or_truncated_generation"
    if any(marker in joined for marker in ("traceback", "exception", "error", " nan", " none")):
        return "malformed_dsl"
    if "divide" in joined and "100" in joined:
        return "divide_by_100_pattern"
    if any(re.search(r"subtract\([^,]+,\s*-[^)]", program) for program in programs):
        return "reversed_subtraction_pattern"
    if any("#x" in program.lower() or "##" in program for program in programs):
        return "malformed_dsl"
    return "present"


def disagreement_stats(values: list[float]) -> tuple[float | None, float | None, float | None, bool, bool, bool]:
    finite = [value for value in values if math.isfinite(value)]
    abs_values = [abs(value) for value in finite]
    nonzero_abs = [value for value in abs_values if value > ZERO_NEAR_THRESHOLD]
    max_abs = max(abs_values) if abs_values else None
    min_nonzero_abs = min(nonzero_abs) if nonzero_abs else None
    ratio = (max_abs / min_nonzero_abs) if max_abs is not None and min_nonzero_abs else None
    factor_gt_10 = ratio is not None and ratio > 10
    zero_vs_large = any(value <= ZERO_NEAR_THRESHOLD for value in abs_values) and any(
        value > ZERO_VS_LARGE_THRESHOLD for value in abs_values
    )
    sign_conflict = any(value < 0 for value in finite) and any(value > 0 for value in finite)
    return ratio, max_abs, min_nonzero_abs, factor_gt_10, zero_vs_large, sign_conflict


def score_reasons(reasons: set[str]) -> int:
    return sum(REASON_WEIGHTS[reason] for reason in reasons) + 35 * max(0, len(reasons) - 1)


def build_targets(args: argparse.Namespace) -> list[dict[str, str]]:
    base = load_submission(args.base_submission)
    source_runs = {
        "run001": load_submission(args.run001_submission),
        "run002": load_submission(args.run002_submission),
        "run004": load_submission(args.run004_submission),
        "run006": load_submission(args.run006_submission),
        "run008": load_submission(args.run008_submission),
        "run009": load_submission(args.run009_submission),
    }
    if not source_runs["run009"]:
        source_runs["run009"] = base
    details = load_details(args.detail_path)
    test_rows = load_test(args.test)

    candidates: list[dict[str, Any]] = []
    for row in test_rows:
        row_id = str(row.get("id") or "")
        if row_id not in base:
            raise ValueError(f"base submission missing test id: {row_id}")
        question = question_text(row)
        q_lower = question.lower()
        current_value = base[row_id]
        values = [source[row_id] for source in source_runs.values() if row_id in source]
        ratio, max_abs, min_nonzero_abs, factor_gt_10, zero_vs_large, sign_conflict = disagreement_stats(values)

        reasons: set[str] = set()
        subreasons: set[str] = set()
        magnitude_flags: list[str] = []

        if current_value == 0.0:
            reasons.add("zero")
            subreasons.add("remaining_zero")
        if abs(current_value) > EXTREME_THRESHOLD:
            reasons.add("extreme")
            subreasons.add("abs_pred_gt_1e6")
        if current_value < 0 and has_any(q_lower, ABSOLUTE_AMOUNT_TERMS):
            reasons.add("negative_absolute_wording")
            subreasons.add("negative_abs_wording")
        if factor_gt_10 or zero_vs_large or (sign_conflict and has_any(q_lower, ABSOLUTE_AMOUNT_TERMS)):
            reasons.add("high_disagreement")
            if factor_gt_10:
                subreasons.add("cross_run_factor_gt_10")
            if zero_vs_large:
                subreasons.add("cross_run_zero_vs_large")
            if sign_conflict and has_any(q_lower, ABSOLUTE_AMOUNT_TERMS):
                subreasons.add("cross_run_sign_conflict")

        weak_flags = []
        if has_any(q_lower, ADDITION_TERMS):
            weak_flags.append("addition")
        if has_any(q_lower, SUBTRACTION_TERMS):
            weak_flags.append("subtraction")
        if has_any(q_lower, TABLE_OP_TERMS):
            weak_flags.append("table_op")
        if has_any(q_lower, DIVISION_PERCENT_TERMS):
            weak_flags.append("division_pct_ratio")
        if weak_flags:
            reasons.add("weak_operation_type")
            subreasons.update(f"weak_{flag}" for flag in weak_flags)

        status = program_detail_status(details.get(row_id, []))
        if status != "present":
            reasons.add("suspicious_program")
            subreasons.add(status)

        if has_any(q_lower, RATIO_TERMS) and abs(current_value) > 1000:
            reasons.add("magnitude_mismatch")
            subreasons.add("ratio_question_huge_answer")
            magnitude_flags.append("ratio_question_huge_answer")
        if has_any(q_lower, AMOUNT_TERMS) and 0 < abs(current_value) < 1:
            reasons.add("magnitude_mismatch")
            subreasons.add("amount_question_tiny_answer")
            magnitude_flags.append("amount_question_tiny_answer")

        if not reasons:
            continue

        priority_score = score_reasons(reasons)
        candidates.append(
            {
                "id": row_id,
                "priority_score": priority_score,
                "target_reason": "|".join(sorted(subreasons)),
                "category_reasons": "|".join(sorted(reasons)),
                "old_value": current_value,
                "question": question,
                "run001_value": source_runs["run001"].get(row_id),
                "run002_value": source_runs["run002"].get(row_id),
                "run004_value": source_runs["run004"].get(row_id),
                "run006_value": source_runs["run006"].get(row_id),
                "run008_value": source_runs["run008"].get(row_id),
                "run009_value": source_runs["run009"].get(row_id),
                "finite_factor_ratio": ratio,
                "max_abs_source_value": max_abs,
                "min_nonzero_abs_source_value": min_nonzero_abs,
                "program_detail_status": status,
                "magnitude_flags": "|".join(magnitude_flags),
            }
        )

    candidates.sort(
        key=lambda item: (
            min(REASON_PRIORITY_GROUP[reason] for reason in item["category_reasons"].split("|")),
            -int(item["priority_score"]),
            item["id"],
        )
    )

    selected = candidates[: max(0, args.max_targets)]
    return [
        {
            "id": item["id"],
            "priority_score": str(item["priority_score"]),
            "target_reason": item["target_reason"],
            "old_value": format_value(item["old_value"]),
            "question": item["question"],
            "run001_value": format_value(item["run001_value"]),
            "run002_value": format_value(item["run002_value"]),
            "run004_value": format_value(item["run004_value"]),
            "run006_value": format_value(item["run006_value"]),
            "run008_value": format_value(item["run008_value"]),
            "run009_value": format_value(item["run009_value"]),
            "finite_factor_ratio": format_value(item["finite_factor_ratio"]),
            "max_abs_source_value": format_value(item["max_abs_source_value"]),
            "min_nonzero_abs_source_value": format_value(item["min_nonzero_abs_source_value"]),
            "program_detail_status": item["program_detail_status"],
            "magnitude_flags": item["magnitude_flags"],
        }
        for item in selected
    ]


def print_summary(rows: list[dict[str, str]]) -> None:
    category_counts = Counter()
    subreason_counts = Counter()
    for row in rows:
        reasons = set(row["target_reason"].split("|")) if row["target_reason"] else set()
        subreason_counts.update(reason for reason in reasons if reason)
        if "remaining_zero" in reasons:
            category_counts["zero"] += 1
        if "abs_pred_gt_1e6" in reasons:
            category_counts["extreme"] += 1
        if "negative_abs_wording" in reasons:
            category_counts["negative_absolute_wording"] += 1
        if reasons & {"cross_run_factor_gt_10", "cross_run_zero_vs_large", "cross_run_sign_conflict"}:
            category_counts["high_disagreement"] += 1
        if any(reason.startswith("weak_") for reason in reasons):
            category_counts["weak_operation_type"] += 1
        if reasons & {
            "missing_program_detail",
            "malformed_dsl",
            "divide_by_100_pattern",
            "reversed_subtraction_pattern",
            "repeated_or_truncated_generation",
        }:
            category_counts["suspicious_program"] += 1
        if reasons & {"ratio_question_huge_answer", "amount_question_tiny_answer"}:
            category_counts["magnitude_mismatch"] += 1

    print(f"target_count={len(rows)}")
    print("category_breakdown=" + json.dumps(category_counts, ensure_ascii=False, sort_keys=True))
    print("reason_breakdown=" + json.dumps(subreason_counts, ensure_ascii=False, sort_keys=True))
    for row in rows[:25]:
        print(json.dumps(row, ensure_ascii=False))


def write_targets(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_targets(args)
    print_summary(rows)
    if args.dry_run:
        print("dry_run=true; output not written")
        return
    write_targets(args.output, rows)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
