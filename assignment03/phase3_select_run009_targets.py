"""Select suspicious rows for Run009-lite targeted retry.

This is Phase 3-only tooling. It creates an auditable target_rows.csv for a
narrow retry over the current best Run008 filtered submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_OUTPUT = Path("runs/kaggle_retry_run009_lite/target_rows.csv")
DEFAULT_RUN001 = Path("runs/kaggle_arc_best/submission_checked.csv")
DEFAULT_RUN002 = Path("runs/kaggle_iter003/submission_checked.csv")
DEFAULT_RUN004 = Path("runs/kaggle_iter004/submission_checked.csv")
DEFAULT_RUN006 = Path("runs/kaggle_run006_iterbest_ctx32768/submission_checked.csv")
DEFAULT_RUN001_DETAILS = Path("runs/kaggle_arc_best/submission_details.json")
DEFAULT_RUN002_DETAILS = Path("runs/kaggle_iter003/submission_details.json")
DEFAULT_RUN004_DETAILS = Path("runs/kaggle_iter004/submission_details.json")
DEFAULT_RUN006_DETAILS = Path("runs/kaggle_run006_iterbest_ctx32768/submission_details.json")
DEFAULT_RUN008_RETRY_DETAILS = Path("runs/kaggle_retry_run008/retry_details.json")

MAX_ABS_VALUE = 1e8
EXTREME_THRESHOLD = 1e6
ZERO_NEAR_THRESHOLD = 1e-9
ZERO_VS_LARGE_THRESHOLD = 1000.0

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
    "finite_factor_ratio",
    "max_abs_source_value",
    "min_nonzero_abs_source_value",
    "program_detail_status",
]

REASON_WEIGHTS = {
    "remaining_zero": 1000,
    "extreme_abs_gt_1e6": 800,
    "negative_absolute_wording": 700,
    "cross_run_factor_gt_10": 450,
    "cross_run_zero_vs_large": 400,
    "malformed_program_detail": 250,
    "weak_table_op_cue": 35,
    "weak_addition_cue": 25,
    "weak_subtraction_cue": 25,
    "missing_program_detail": 5,
}
REASON_PRIORITY_GROUP = {
    "remaining_zero": 0,
    "extreme_abs_gt_1e6": 1,
    "negative_absolute_wording": 2,
    "cross_run_factor_gt_10": 3,
    "cross_run_zero_vs_large": 3,
    "weak_addition_cue": 4,
    "weak_subtraction_cue": 4,
    "weak_table_op_cue": 4,
    "malformed_program_detail": 5,
    "missing_program_detail": 6,
}

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
)
SUBTRACTION_TERMS = (
    "chênh lệch",
    "thay đổi",
    "cao hơn",
    "thấp hơn",
    "giảm bao nhiêu",
    "tăng bao nhiêu",
)
TABLE_OP_TERMS = (
    "giá trị lớn nhất",
    "giá trị nhỏ nhất",
    "trung bình",
    "thấp nhất",
    "cao nhất",
    "tỷ lệ",
    "tỷ suất",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Run009-lite target_rows.csv from suspicious Run008 rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-targets", type=int, default=60)
    parser.add_argument("--run001-submission", type=Path, default=DEFAULT_RUN001)
    parser.add_argument("--run002-submission", type=Path, default=DEFAULT_RUN002)
    parser.add_argument("--run004-submission", type=Path, default=DEFAULT_RUN004)
    parser.add_argument("--run006-submission", type=Path, default=DEFAULT_RUN006)
    parser.add_argument("--run001-details", type=Path, default=DEFAULT_RUN001_DETAILS)
    parser.add_argument("--run002-details", type=Path, default=DEFAULT_RUN002_DETAILS)
    parser.add_argument("--run004-details", type=Path, default=DEFAULT_RUN004_DETAILS)
    parser.add_argument("--run006-details", type=Path, default=DEFAULT_RUN006_DETAILS)
    parser.add_argument("--run008-retry-details", type=Path, default=DEFAULT_RUN008_RETRY_DETAILS)
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
    merged: dict[str, list[dict[str, Any]]] = {}
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
                merged.setdefault(str(row["id"]), []).append(row)
    return merged


def question_text(row: dict[str, Any]) -> str:
    qa = row.get("qa")
    if isinstance(qa, dict) and qa.get("question"):
        return str(qa["question"])
    return str(row.get("question") or "")


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def program_detail_status(details: list[dict[str, Any]]) -> str:
    if not details:
        return "missing"
    programs = [
        str(row.get("program") or row.get("selected_program") or "")
        for row in details
    ]
    if any(not program.strip() for program in programs):
        return "missing_program"
    if any(len(program) > 3000 for program in programs):
        return "too_long"
    bad_markers = ("traceback", "error", "exception", "nan", "none")
    if any(any(marker in program.lower() for marker in bad_markers) for program in programs):
        return "malformed"
    return "present"


def disagreement_stats(values: list[float]) -> tuple[float | None, float | None, float | None, bool, bool]:
    finite = [value for value in values if math.isfinite(value)]
    abs_values = [abs(value) for value in finite]
    nonzero_abs = [value for value in abs_values if value > ZERO_NEAR_THRESHOLD]
    max_abs = max(abs_values) if abs_values else None
    min_nonzero_abs = min(nonzero_abs) if nonzero_abs else None
    ratio = None
    if min_nonzero_abs and max_abs is not None:
        ratio = max_abs / min_nonzero_abs
    factor_gt_10 = ratio is not None and ratio > 10.0
    zero_vs_large = any(value <= ZERO_NEAR_THRESHOLD for value in abs_values) and any(
        value >= ZERO_VS_LARGE_THRESHOLD for value in abs_values
    )
    return ratio, max_abs, min_nonzero_abs, factor_gt_10, zero_vs_large


def reasons_for_row(
    *,
    row_id: str,
    question: str,
    run008_value: float,
    source_values: list[float],
    detail_status: str,
) -> list[str]:
    reasons: list[str] = []
    lower_question = question.lower()
    ratio, _, _, factor_gt_10, zero_vs_large = disagreement_stats(source_values)

    if run008_value == 0.0:
        reasons.append("remaining_zero")
    if abs(run008_value) > EXTREME_THRESHOLD:
        reasons.append("extreme_abs_gt_1e6")
    if (
        run008_value < 0
        and has_any(lower_question, ABSOLUTE_AMOUNT_TERMS)
        and "tỷ lệ phần trăm thay đổi" not in lower_question
    ):
        reasons.append("negative_absolute_wording")
    if factor_gt_10:
        reasons.append("cross_run_factor_gt_10")
    if zero_vs_large:
        reasons.append("cross_run_zero_vs_large")
    if detail_status in {"too_long", "malformed"}:
        reasons.append("malformed_program_detail")
    elif detail_status in {"missing", "missing_program"}:
        reasons.append("missing_program_detail")
    if has_any(lower_question, TABLE_OP_TERMS):
        reasons.append("weak_table_op_cue")
    if has_any(lower_question, ADDITION_TERMS):
        reasons.append("weak_addition_cue")
    if has_any(lower_question, SUBTRACTION_TERMS):
        reasons.append("weak_subtraction_cue")

    # Weak cues alone are not enough for Run009-lite; they only break ties.
    strong_reasons = [
        reason
        for reason in reasons
        if not reason.startswith("weak_") and reason != "missing_program_detail"
    ]
    if not strong_reasons:
        return []
    return reasons


def main() -> None:
    args = parse_args()
    test_rows = load_test(args.test)
    base = load_submission(args.base_submission)
    sources = {
        "run001": load_submission(args.run001_submission),
        "run002": load_submission(args.run002_submission),
        "run004": load_submission(args.run004_submission),
        "run006": load_submission(args.run006_submission),
        "run008": base,
    }
    details = load_details(
        [
            args.run001_details,
            args.run002_details,
            args.run004_details,
            args.run006_details,
            args.run008_retry_details,
        ]
    )

    candidates: list[dict[str, str]] = []
    for order, row in enumerate(test_rows):
        row_id = str(row.get("id", ""))
        if row_id not in base:
            raise ValueError(f"base submission missing test id: {row_id}")
        question = question_text(row)
        source_values = [
            source[row_id]
            for source in sources.values()
            if row_id in source and source[row_id] is not None
        ]
        detail_status = program_detail_status(details.get(row_id, []))
        run008_value = base[row_id]
        row_reasons = reasons_for_row(
            row_id=row_id,
            question=question,
            run008_value=run008_value,
            source_values=source_values,
            detail_status=detail_status,
        )
        if not row_reasons:
            continue

        ratio, max_abs, min_nonzero_abs, _, _ = disagreement_stats(source_values)
        priority_score = sum(REASON_WEIGHTS.get(reason, 0) for reason in set(row_reasons))
        priority_group = min(REASON_PRIORITY_GROUP.get(reason, 99) for reason in row_reasons)
        candidates.append(
            {
                "id": row_id,
                "priority_score": str(priority_score),
                "target_reason": "|".join(row_reasons),
                "old_value": format_value(run008_value),
                "question": question,
                "run001_value": format_value(sources["run001"].get(row_id)),
                "run002_value": format_value(sources["run002"].get(row_id)),
                "run004_value": format_value(sources["run004"].get(row_id)),
                "run006_value": format_value(sources["run006"].get(row_id)),
                "run008_value": format_value(run008_value),
                "finite_factor_ratio": format_value(ratio),
                "max_abs_source_value": format_value(max_abs),
                "min_nonzero_abs_source_value": format_value(min_nonzero_abs),
                "program_detail_status": detail_status,
                "_priority_group": str(priority_group),
                "_order": str(order),
            }
        )

    candidates.sort(key=lambda item: (int(item["_priority_group"]), -int(item["priority_score"]), int(item["_order"])))
    capped = candidates[: max(0, args.max_targets)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in capped:
            writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})

    reason_counts = Counter()
    for row in capped:
        for reason in row["target_reason"].split("|"):
            if reason:
                reason_counts[reason] += 1

    print(f"wrote {len(capped)} targets to {args.output}")
    print("reason_counts=" + json.dumps(reason_counts, ensure_ascii=False, sort_keys=True))
    if capped:
        print("top_targets:")
        for row in capped[:10]:
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "priority_score": row["priority_score"],
                        "target_reason": row["target_reason"],
                        "old_value": row["old_value"],
                        "question": row["question"],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
