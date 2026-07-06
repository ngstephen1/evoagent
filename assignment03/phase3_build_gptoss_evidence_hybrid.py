"""Build a Run012 GPT-OSS evidence-to-DSL smoke hybrid.

This Phase 3-only builder preserves Run009-lite safe by default and only
applies accepted evidence-to-DSL retry details. It is smoke-oriented:
safe_to_submit is false unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv")
DEFAULT_RETRY_DETAILS = Path("runs/kaggle_run012_gptoss_evidence/retry_details.json")
DEFAULT_TARGET_ROWS = Path("runs/kaggle_run012_gptoss_evidence/target_rows.csv")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_hybrid_run012_gptoss_evidence")
OUTPUT_COLUMNS = ["id", "Usage", "predicted_value"]
CHANGE_COLUMNS = [
    "id",
    "old_value",
    "new_value",
    "program",
    "confidence_reason",
    "stage1_confidence",
    "stage2_confidence",
    "relevant_number_count",
    "question_operation",
    "target_reason",
    "question",
]
MAX_ABS_VALUE = 1e8
EXTREME_THRESHOLD = 1e6
STRONG_SIGNAL_REASONS = {
    "remaining_zero",
    "abs_pred_gt_1e6",
    "negative_abs_wording",
    "cross_run_factor_gt_10",
    "cross_run_zero_vs_large",
    "ratio_question_huge_answer",
    "amount_question_tiny_answer",
    "missing_program_detail",
    "malformed_dsl",
    "divide_by_100_pattern",
    "repeated_or_truncated_generation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conservative Run012 evidence-to-DSL hybrid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--retry-details", type=Path, default=DEFAULT_RETRY_DETAILS)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-changes-for-safe-submit", type=int, default=999)
    parser.add_argument("--allow-safe-submit", action="store_true")
    return parser.parse_args()


def to_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing value")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("blank value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def format_float(value: float) -> str:
    return format(float(value), ".15g")


def confidence_rank(value: Any) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value or "").lower(), -1)


def load_submission(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for column in OUTPUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"{path} contains empty id")
            if row_id in rows:
                raise ValueError(f"{path} contains duplicate id: {row_id}")
            rows[row_id] = row
    return rows


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return [row for row in data if isinstance(row, dict)]


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list")
    return rows


def load_target_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"id", "target_reason"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"{path} contains empty id")
            if row_id in rows:
                raise ValueError(f"{path} contains duplicate target id: {row_id}")
            rows[row_id] = row
    return rows


def has_strong_signal(target_reason: str) -> bool:
    reasons = set(reason for reason in target_reason.split("|") if reason)
    return bool(reasons & STRONG_SIGNAL_REASONS)


def valid_retry_for_base(
    detail: dict[str, Any],
    target: dict[str, str],
    old_value: float,
) -> tuple[bool, float | None, str | None]:
    if not detail.get("accepted"):
        return False, None, "retry not accepted"
    if not detail.get("selected_program"):
        return False, None, "missing selected program"
    try:
        selected_value = to_float(detail.get("selected_value"))
    except Exception as exc:
        return False, None, f"invalid selected value: {exc}"
    if selected_value == old_value:
        return False, selected_value, "selected value unchanged"
    if selected_value == 0.0:
        return False, selected_value, "selected value is zero"
    if abs(selected_value) > MAX_ABS_VALUE:
        return False, selected_value, f"selected value abs>{MAX_ABS_VALUE:g}"

    stage1_conf = confidence_rank(detail.get("stage1_confidence"))
    stage2_conf = confidence_rank(detail.get("stage2_confidence"))
    numbers = int(detail.get("relevant_number_count") or 0)
    operation = str(detail.get("question_operation") or "unknown").lower()
    target_reason = str(target.get("target_reason") or "")
    if old_value == 0.0:
        if stage1_conf < 1 or stage2_conf < 1:
            return False, selected_value, "zero-row confidence below medium"
        if operation != "table_lookup" and numbers < 2:
            return False, selected_value, "zero-row evidence has fewer than two numbers"
        return True, selected_value, None
    if stage1_conf < 2 or stage2_conf < 2:
        return False, selected_value, "nonzero-row confidence below high"
    if operation != "table_lookup" and numbers < 2:
        return False, selected_value, "nonzero-row evidence has fewer than two numbers"
    if not has_strong_signal(target_reason):
        return False, selected_value, "nonzero-row lacks strong suspicious signal"
    if abs(selected_value) > EXTREME_THRESHOLD and abs(old_value) <= EXTREME_THRESHOLD:
        return False, selected_value, "nonzero-row would introduce new extreme value"
    return True, selected_value, None


def validate_output(rows: list[dict[str, str]], expected_ids: list[str]) -> dict[str, Any]:
    actual_ids = [row["id"] for row in rows]
    values: list[float] = []
    missing = 0
    all_numeric = True
    for row in rows:
        try:
            values.append(to_float(row["predicted_value"]))
        except Exception:
            all_numeric = False
            missing += 1
    return {
        "total_rows": len(rows),
        "expected_rows": len(expected_ids),
        "id_order_exact": actual_ids == expected_ids,
        "duplicate_ids": len(actual_ids) - len(set(actual_ids)),
        "missing_predictions": missing,
        "all_numeric": all_numeric,
        "zero_count": sum(value == 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "extreme_abs_gt_1e6_count": sum(abs(value) > EXTREME_THRESHOLD for value in values),
        "max_abs_value": max((abs(value) for value in values), default=0.0),
    }


def main() -> None:
    args = parse_args()
    base = load_submission(args.base_submission)
    retry_details = {str(row["id"]): row for row in load_json_list(args.retry_details) if row.get("id") is not None}
    target_rows = load_target_rows(args.target_rows)
    expected_ids = [str(row.get("id") or "") for row in load_test_rows(args.test)]

    output_rows: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    before_values: list[float] = []
    after_values: list[float] = []
    operation_distribution = Counter()

    for row_id in expected_ids:
        if row_id not in base:
            raise ValueError(f"base submission missing test id: {row_id}")
        original = dict(base[row_id])
        old_value = to_float(original["predicted_value"])
        new_value = old_value
        before_values.append(old_value)
        detail = retry_details.get(row_id)
        target = target_rows.get(row_id)
        if detail and target:
            operation_distribution[str(detail.get("question_operation") or "unknown")] += 1
            ok, candidate_value, reason = valid_retry_for_base(detail, target, old_value)
            if ok and candidate_value is not None:
                new_value = candidate_value
                changes.append(
                    {
                        "id": row_id,
                        "old_value": format_float(old_value),
                        "new_value": format_float(new_value),
                        "program": str(detail.get("selected_program") or ""),
                        "confidence_reason": str(detail.get("confidence_reason") or ""),
                        "stage1_confidence": str(detail.get("stage1_confidence") or ""),
                        "stage2_confidence": str(detail.get("stage2_confidence") or ""),
                        "relevant_number_count": str(int(detail.get("relevant_number_count") or 0)),
                        "question_operation": str(detail.get("question_operation") or ""),
                        "target_reason": str(target.get("target_reason") or ""),
                        "question": str(target.get("question") or detail.get("question") or ""),
                    }
                )
            else:
                rejected.append(
                    {
                        "id": row_id,
                        "old_value": old_value,
                        "retry_value": candidate_value,
                        "target_reason": str(target.get("target_reason") or ""),
                        "reason": reason,
                    }
                )
        original["predicted_value"] = format_float(new_value)
        output_rows.append(original)
        after_values.append(new_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission_checked.csv"
    changes_path = args.output_dir / "changes.csv"
    summary_path = args.output_dir / "summary.json"
    with submission_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    with changes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(changes)

    validation = validate_output(output_rows, expected_ids)
    validation.update(
        {
            "zero_count_before": sum(value == 0.0 for value in before_values),
            "zero_count_after": sum(value == 0.0 for value in after_values),
            "negative_count_before": sum(value < 0.0 for value in before_values),
            "negative_count_after": sum(value < 0.0 for value in after_values),
            "extreme_abs_gt_1e6_count_before": sum(abs(value) > EXTREME_THRESHOLD for value in before_values),
            "extreme_abs_gt_1e6_count_after": sum(abs(value) > EXTREME_THRESHOLD for value in after_values),
            "changed_rows": len(changes),
            "target_rows": len(target_rows),
            "retry_detail_rows": len(retry_details),
            "rejected_retry_rows": len(rejected),
            "operation_distribution": dict(sorted(operation_distribution.items())),
            "safe_to_submit": (
                args.allow_safe_submit
                and validation["total_rows"] == validation["expected_rows"]
                and validation["id_order_exact"]
                and validation["duplicate_ids"] == 0
                and validation["missing_predictions"] == 0
                and validation["all_numeric"]
                and len(changes) >= args.min_changes_for_safe_submit
            ),
            "submission_path": str(submission_path),
            "changes_path": str(changes_path),
            "summary_path": str(summary_path),
            "rejected_retries": rejected[:50],
        }
    )
    summary_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
