"""Build conservative Run013 GPT-OSS variants on top of the team ensemble.

This Phase 3-only script consumes GPT-OSS evidence-to-DSL retry details and
builds three auditable hybrid candidates:

- variant_a_zero_only: replace only current-base zero rows.
- variant_b_zero_sign: variant A plus sign fixes for absolute-change wording.
- variant_c_high_conf: variant B plus stricter high-confidence nonzero fixes.

It does not submit to Kaggle and does not touch core EvoAgent logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_BASE_SUBMISSION = Path("runs/team_melanie_ensemble_v2/submission_checked.csv")
DEFAULT_RETRY_DETAILS = Path("runs/kaggle_run013_gptoss_on_team_v2/retry_details.json")
DEFAULT_TARGET_ROWS = Path("runs/kaggle_run013_gptoss_on_team_v2/target_rows.csv")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_OUTPUT_ROOT = Path("runs")
DEFAULT_OUTPUT_PREFIX = "kaggle_hybrid_run013_gptoss_on_team_v2"
OUTPUT_COLUMNS = ["id", "Usage", "predicted_value"]
CHANGE_COLUMNS = [
    "id",
    "old_value",
    "new_value",
    "program",
    "variant",
    "acceptance_reason",
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
ABSOLUTE_REASONS = {"negative_abs_wording"}
STRONG_NONZERO_REASONS = {
    "abs_pred_gt_1e6",
    "cross_run_factor_gt_10",
    "cross_run_zero_vs_large",
    "amount_question_tiny_answer",
    "ratio_question_huge_answer",
    "missing_program_detail",
    "malformed_dsl",
    "divide_by_100_pattern",
    "repeated_or_truncated_generation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Run013 GPT-OSS A/B/C variants from evidence retry details.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--retry-details", type=Path, default=DEFAULT_RETRY_DETAILS)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--min-safe-changes", type=int, default=1)
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


def numeric_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-6, 1e-4 * max(abs(left), abs(right), 1.0))


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


def load_test_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list")
    return [str(row.get("id") or "") for row in rows]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return [row for row in data if isinstance(row, dict)]


def load_target_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"id", "target_reason"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id") or "")
            if row_id:
                rows[row_id] = row
    return rows


def target_reasons(target: dict[str, str] | None) -> set[str]:
    if not target:
        return set()
    return {reason for reason in str(target.get("target_reason") or "").split("|") if reason}


def accepted_detail_value(detail: dict[str, Any]) -> tuple[bool, float | None, str]:
    if not detail.get("accepted"):
        return False, None, "retry not accepted"
    if not detail.get("selected_program"):
        return False, None, "missing selected program"
    try:
        value = to_float(detail.get("selected_value"))
    except Exception as exc:
        return False, None, f"invalid selected value: {exc}"
    if value == 0.0:
        return False, value, "selected value is zero"
    if abs(value) > MAX_ABS_VALUE:
        return False, value, f"selected value abs>{MAX_ABS_VALUE:g}"
    return True, value, ""


def detail_confidence_ok(detail: dict[str, Any], minimum: int) -> bool:
    return confidence_rank(detail.get("stage1_confidence")) >= minimum and confidence_rank(detail.get("stage2_confidence")) >= minimum


def relevant_number_count(detail: dict[str, Any]) -> int:
    try:
        return int(detail.get("relevant_number_count") or 0)
    except Exception:
        return 0


def accept_for_variant(
    variant: str,
    old_value: float,
    detail: dict[str, Any] | None,
    target: dict[str, str] | None,
) -> tuple[bool, float | None, str]:
    if not detail or not target:
        return False, None, "missing retry detail or target row"
    ok, candidate, reason = accepted_detail_value(detail)
    if not ok or candidate is None:
        return False, candidate, reason
    if numeric_close(old_value, candidate):
        return False, candidate, "selected value unchanged"

    reasons = target_reasons(target)
    operation = str(detail.get("question_operation") or "").lower()
    numbers = relevant_number_count(detail)

    if old_value == 0.0:
        if detail_confidence_ok(detail, 1) and (operation == "table_lookup" or numbers >= 2):
            return True, candidate, "zero_row_medium_plus_evidence"
        return False, candidate, "zero row lacks medium confidence or evidence"

    if variant == "variant_a_zero_only":
        return False, candidate, "variant A only accepts zero rows"

    if variant in {"variant_b_zero_sign", "variant_c_high_conf"}:
        sign_fix = (
            old_value < 0.0
            and candidate > 0.0
            and bool(reasons & ABSOLUTE_REASONS)
            and numeric_close(candidate, abs(old_value))
            and detail_confidence_ok(detail, 2)
            and operation in {"subtraction", "table_lookup"}
        )
        if sign_fix:
            return True, candidate, "negative_absolute_sign_fix"

    if variant == "variant_b_zero_sign":
        return False, candidate, "variant B only accepts zero rows and sign fixes"

    if variant == "variant_c_high_conf":
        strong_signal = bool(reasons & STRONG_NONZERO_REASONS)
        introduces_extreme = abs(candidate) > EXTREME_THRESHOLD and abs(old_value) <= EXTREME_THRESHOLD
        if detail_confidence_ok(detail, 2) and strong_signal and numbers >= 2 and not introduces_extreme:
            return True, candidate, "high_confidence_nonzero_strong_signal"
        return False, candidate, "nonzero row lacks high-confidence strong-signal evidence"

    raise ValueError(f"unknown variant: {variant}")


def validate_rows(rows: list[dict[str, str]], expected_ids: list[str]) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    values: list[float] = []
    bad = 0
    for row in rows:
        try:
            values.append(to_float(row["predicted_value"]))
        except Exception:
            bad += 1
    return {
        "total_rows": len(rows),
        "expected_rows": len(expected_ids),
        "id_order_exact": ids == expected_ids,
        "duplicate_ids": len(ids) - len(set(ids)),
        "missing_predictions": bad,
        "all_numeric": bad == 0,
        "zero_count": sum(value == 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "extreme_abs_gt_1e6_count": sum(abs(value) > EXTREME_THRESHOLD for value in values),
        "max_abs_value": max((abs(value) for value in values), default=0.0),
    }


def build_variant(
    *,
    variant: str,
    output_dir: Path,
    base: dict[str, dict[str, str]],
    retry_details: dict[str, dict[str, Any]],
    target_rows: dict[str, dict[str, str]],
    expected_ids: list[str],
    min_safe_changes: int,
) -> dict[str, Any]:
    output_rows: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    before_values: list[float] = []
    after_values: list[float] = []

    for row_id in expected_ids:
        if row_id not in base:
            raise ValueError(f"base submission missing test id: {row_id}")
        row = dict(base[row_id])
        old_value = to_float(row["predicted_value"])
        before_values.append(old_value)
        detail = retry_details.get(row_id)
        target = target_rows.get(row_id)
        accept, candidate, reason = accept_for_variant(variant, old_value, detail, target)
        new_value = candidate if accept and candidate is not None else old_value
        if accept and candidate is not None:
            changes.append(
                {
                    "id": row_id,
                    "old_value": format_float(old_value),
                    "new_value": format_float(candidate),
                    "program": str(detail.get("selected_program") or "") if detail else "",
                    "variant": variant,
                    "acceptance_reason": reason,
                    "confidence_reason": str(detail.get("confidence_reason") or "") if detail else "",
                    "stage1_confidence": str(detail.get("stage1_confidence") or "") if detail else "",
                    "stage2_confidence": str(detail.get("stage2_confidence") or "") if detail else "",
                    "relevant_number_count": str(relevant_number_count(detail or {})),
                    "question_operation": str(detail.get("question_operation") or "") if detail else "",
                    "target_reason": str((target or {}).get("target_reason") or ""),
                    "question": str((target or {}).get("question") or (detail or {}).get("question") or ""),
                }
            )
        elif detail and target:
            rejected.append(
                {
                    "id": row_id,
                    "old_value": old_value,
                    "retry_value": candidate,
                    "target_reason": str(target.get("target_reason") or ""),
                    "reason": reason,
                }
            )
        row["predicted_value"] = format_float(new_value)
        output_rows.append(row)
        after_values.append(new_value)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission_checked.csv"
    changes_path = output_dir / "changes.csv"
    summary_path = output_dir / "summary.json"
    with submission_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    with changes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(changes)

    validation = validate_rows(output_rows, expected_ids)
    validation.update(
        {
            "variant": variant,
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
            "safe_to_submit": (
                validation["total_rows"] == validation["expected_rows"]
                and validation["id_order_exact"]
                and validation["duplicate_ids"] == 0
                and validation["missing_predictions"] == 0
                and validation["all_numeric"]
                and len(changes) >= min_safe_changes
            ),
            "submission_path": str(submission_path),
            "changes_path": str(changes_path),
            "summary_path": str(summary_path),
            "rejected_retries": rejected[:50],
        }
    )
    summary_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    args = parse_args()
    base = load_submission(args.base_submission)
    retry_details = {str(row["id"]): row for row in load_json_list(args.retry_details) if row.get("id") is not None}
    target_rows = load_target_rows(args.target_rows)
    expected_ids = load_test_ids(args.test)

    variants = {
        "variant_a_zero_only": args.output_root / f"{args.output_prefix}_zero_only",
        "variant_b_zero_sign": args.output_root / f"{args.output_prefix}_zero_sign",
        "variant_c_high_conf": args.output_root / f"{args.output_prefix}_high_conf",
    }
    summaries = []
    for variant, output_dir in variants.items():
        summary = build_variant(
            variant=variant,
            output_dir=output_dir,
            base=base,
            retry_details=retry_details,
            target_rows=target_rows,
            expected_ids=expected_ids,
            min_safe_changes=args.min_safe_changes,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    all_summary_path = args.output_root / f"{args.output_prefix}_variants_summary.json"
    all_summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"variants_summary={all_summary_path}")


if __name__ == "__main__":
    main()
