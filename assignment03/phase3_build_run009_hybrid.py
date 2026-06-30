"""Build Run009-lite hybrid submission from high-confidence retry details.

This is Phase 3-only tooling. It preserves Run008 filtered by default and
replaces only rows listed in target_rows.csv with accepted, high-confidence
retry outputs.
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
DEFAULT_RETRY_DETAILS = Path("runs/kaggle_retry_run009_lite/retry_details.json")
DEFAULT_TARGET_ROWS = Path("runs/kaggle_retry_run009_lite/target_rows.csv")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_hybrid_retry_run009_lite")
OUTPUT_COLUMNS = ["id", "Usage", "predicted_value"]
CHANGE_COLUMNS = [
    "id",
    "old_value",
    "new_value",
    "program",
    "confidence_reason",
    "agreement_count",
    "target_reason",
    "question",
]
MAX_ABS_VALUE = 1e8
EXTREME_THRESHOLD = 1e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validated Run009-lite hybrid submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--retry-details", type=Path, default=DEFAULT_RETRY_DETAILS)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-changes-for-safe-submit", type=int, default=3)
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


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list")
    return rows


def load_retry_details(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    rows: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        row_id = str(row["id"])
        if row_id in rows:
            raise ValueError(f"{path} contains duplicate retry id: {row_id}")
        rows[row_id] = row
    return rows


def load_target_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "target_reason"}
        missing = required - set(reader.fieldnames or [])
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


def valid_retry_for_base(
    detail: dict[str, Any],
    old_value: float,
) -> tuple[bool, float | None, str | None]:
    if not detail.get("accepted"):
        return False, None, "retry not accepted"
    try:
        selected_value = to_float(detail.get("selected_value"))
    except Exception as exc:
        return False, None, f"invalid selected value: {exc}"
    if selected_value == 0.0:
        return False, selected_value, "selected value is zero"
    if abs(selected_value) > MAX_ABS_VALUE:
        return False, selected_value, f"selected value abs>{MAX_ABS_VALUE:g}"
    if not detail.get("selected_program"):
        return False, selected_value, "missing selected program"

    agreement_count = int(detail.get("agreement_count") or 0)
    confidence_reason = str(detail.get("confidence_reason") or "")
    if old_value == 0.0:
        if agreement_count < 2:
            return False, selected_value, "zero-row retry agreement_count < 2"
    else:
        if agreement_count < 3:
            return False, selected_value, "nonzero-row retry agreement_count < 3"
        if not confidence_reason.startswith("cluster_agreement_"):
            return False, selected_value, "nonzero-row retry is not cluster agreement"
    return True, selected_value, None


def validate_output(rows: list[dict[str, str]], expected_ids: list[str]) -> dict[str, Any]:
    actual_ids = [row["id"] for row in rows]
    duplicate_count = len(actual_ids) - len(set(actual_ids))
    missing_predictions = 0
    nonnumeric_count = 0
    zero_count = 0
    negative_count = 0
    extreme_count = 0
    max_abs_value = 0.0

    for row in rows:
        try:
            value = to_float(row.get("predicted_value"))
        except Exception:
            nonnumeric_count += 1
            missing_predictions += 1
            continue
        abs_value = abs(value)
        max_abs_value = max(max_abs_value, abs_value)
        if value == 0.0:
            zero_count += 1
        if value < 0:
            negative_count += 1
        if abs_value > EXTREME_THRESHOLD:
            extreme_count += 1

    return {
        "total_rows": len(rows),
        "expected_rows": len(expected_ids),
        "id_order_exact": actual_ids == expected_ids,
        "duplicate_ids": duplicate_count,
        "missing_predictions": missing_predictions,
        "all_numeric": nonnumeric_count == 0,
        "zero_count": zero_count,
        "negative_count": negative_count,
        "extreme_abs_gt_1e6_count": extreme_count,
        "max_abs_value": max_abs_value,
    }


def main() -> None:
    args = parse_args()
    base_rows = load_submission(args.base_submission)
    retry_details = load_retry_details(args.retry_details)
    target_rows = load_target_rows(args.target_rows)
    test_rows = load_test_rows(args.test)
    expected_ids = [str(row.get("id", idx)) for idx, row in enumerate(test_rows)]

    missing_base = [row_id for row_id in expected_ids if row_id not in base_rows]
    if missing_base:
        raise ValueError(f"base submission missing ids: {missing_base[:3]}")

    output_rows: list[dict[str, str]] = []
    change_rows: list[dict[str, str]] = []
    rejected_retry_rows: list[dict[str, Any]] = []
    agreement_distribution: Counter[str] = Counter()

    base_values = [to_float(base_rows[row_id]["predicted_value"]) for row_id in expected_ids]
    zero_before = sum(value == 0.0 for value in base_values)
    negative_before = sum(value < 0 for value in base_values)
    extreme_before = sum(abs(value) > EXTREME_THRESHOLD for value in base_values)

    for row_id in expected_ids:
        base_row = base_rows[row_id]
        old_value = to_float(base_row["predicted_value"])
        new_value = old_value
        target = target_rows.get(row_id)
        detail = retry_details.get(row_id) if target is not None else None

        if detail is not None:
            agreement_distribution[str(int(detail.get("agreement_count") or 0))] += 1
            is_valid, retry_value, reject_reason = valid_retry_for_base(detail, old_value)
            if is_valid and retry_value is not None:
                new_value = retry_value
                change_rows.append(
                    {
                        "id": row_id,
                        "old_value": format_float(old_value),
                        "new_value": format_float(new_value),
                        "program": str(detail.get("selected_program") or ""),
                        "confidence_reason": str(detail.get("confidence_reason") or ""),
                        "agreement_count": str(int(detail.get("agreement_count") or 0)),
                        "target_reason": str(target.get("target_reason") or ""),
                        "question": str(detail.get("question") or target.get("question") or ""),
                    }
                )
            else:
                rejected_retry_rows.append(
                    {
                        "id": row_id,
                        "old_value": old_value,
                        "retry_value": retry_value,
                        "target_reason": str(target.get("target_reason") or ""),
                        "reason": reject_reason or "retry rejected",
                    }
                )

        output_rows.append(
            {
                "id": row_id,
                "Usage": base_row.get("Usage", "Public") or "Public",
                "predicted_value": format_float(new_value),
            }
        )

    validation = validate_output(output_rows, expected_ids)
    validation["zero_count_before"] = zero_before
    validation["zero_count_after"] = validation["zero_count"]
    validation["negative_count_before"] = negative_before
    validation["negative_count_after"] = validation["negative_count"]
    validation["extreme_abs_gt_1e6_count_before"] = extreme_before
    validation["extreme_abs_gt_1e6_count_after"] = validation["extreme_abs_gt_1e6_count"]
    validation["changed_rows"] = len(change_rows)
    validation["target_rows"] = len(target_rows)
    validation["retry_detail_rows"] = len(retry_details)
    validation["rejected_retry_rows"] = len(rejected_retry_rows)
    validation["agreement_count_distribution"] = dict(sorted(agreement_distribution.items(), key=lambda item: int(item[0])))
    validation["safe_to_submit"] = (
        validation["total_rows"] == validation["expected_rows"]
        and validation["id_order_exact"]
        and validation["duplicate_ids"] == 0
        and validation["missing_predictions"] == 0
        and validation["all_numeric"]
        and validation["changed_rows"] >= args.min_changes_for_safe_submit
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission_checked.csv"
    changes_path = args.output_dir / "changes.csv"
    summary_path = args.output_dir / "summary.json"

    with submission_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    with changes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGE_COLUMNS)
        writer.writeheader()
        writer.writerows(change_rows)

    validation["submission_path"] = str(submission_path)
    validation["changes_path"] = str(changes_path)
    validation["summary_path"] = str(summary_path)
    validation["rejected_retries"] = rejected_retry_rows
    summary_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in validation.items() if k != "rejected_retries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
