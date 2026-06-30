"""Build Run008 hybrid submission from accepted retry details.

This script is Phase 3-only tooling. It preserves Run003 by default and
replaces only rows with accepted retry results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_001_002/submission_checked.csv")
DEFAULT_RETRY_DETAILS = Path("runs/kaggle_retry_run008/retry_details.json")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_hybrid_retry_run008")
OUTPUT_COLUMNS = ["id", "Usage", "predicted_value"]
CHANGE_COLUMNS = [
    "id",
    "old_value",
    "new_value",
    "program",
    "confidence_reason",
    "agreement_count",
    "question",
]
MAX_ABS_VALUE = 1e8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validated Run008 hybrid submission from retry details.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--retry-details", type=Path, default=DEFAULT_RETRY_DETAILS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-nonzero-replacements",
        action="store_true",
        help="Allow accepted retry values to replace nonzero base rows. Disabled for Run008.",
    )
    parser.add_argument(
        "--min-changes-for-safe-submit",
        type=int,
        default=3,
        help="Minimum accepted changes required for safe_to_submit=true.",
    )
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
        missing = [col for col in OUTPUT_COLUMNS if col not in (reader.fieldnames or [])]
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


def valid_retry_value(detail: dict[str, Any]) -> tuple[bool, float | None, str | None]:
    if not detail.get("accepted"):
        return False, None, "retry not accepted"
    try:
        value = to_float(detail.get("selected_value"))
    except Exception as exc:
        return False, None, f"invalid selected value: {exc}"
    if value == 0.0:
        return False, value, "selected value is zero"
    if abs(value) > MAX_ABS_VALUE:
        return False, value, f"selected value abs>{MAX_ABS_VALUE:g}"
    if not detail.get("selected_program"):
        return False, value, "missing selected program"
    return True, value, None


def validate_output(rows: list[dict[str, str]], expected_ids: list[str]) -> dict[str, Any]:
    actual_ids = [row["id"] for row in rows]
    duplicate_count = len(actual_ids) - len(set(actual_ids))
    missing_predictions = 0
    nonnumeric_count = 0
    max_abs_value = 0.0
    zero_count = 0

    for row in rows:
        try:
            value = to_float(row.get("predicted_value"))
        except Exception:
            nonnumeric_count += 1
            missing_predictions += 1
            continue
        max_abs_value = max(max_abs_value, abs(value))
        if value == 0.0:
            zero_count += 1

    return {
        "total_rows": len(rows),
        "expected_rows": len(expected_ids),
        "id_order_exact": actual_ids == expected_ids,
        "duplicate_ids": duplicate_count,
        "missing_predictions": missing_predictions,
        "all_numeric": nonnumeric_count == 0,
        "zero_count": zero_count,
        "max_abs_value": max_abs_value,
    }


def main() -> None:
    args = parse_args()
    base_rows = load_submission(args.base_submission)
    retry_details = load_retry_details(args.retry_details)
    test_rows = load_test_rows(args.test)
    expected_ids = [str(row.get("id", idx)) for idx, row in enumerate(test_rows)]

    missing_base = [row_id for row_id in expected_ids if row_id not in base_rows]
    if missing_base:
        raise ValueError(f"base submission missing ids: {missing_base[:3]}")

    output_rows: list[dict[str, str]] = []
    change_rows: list[dict[str, str]] = []
    rejected_retry_rows: list[dict[str, Any]] = []
    zero_before = 0

    for row_id in expected_ids:
        base_row = base_rows[row_id]
        old_value = to_float(base_row["predicted_value"])
        if old_value == 0.0:
            zero_before += 1

        new_value = old_value
        detail = retry_details.get(row_id)
        if detail is not None:
            is_valid, retry_value, reject_reason = valid_retry_value(detail)
            can_replace = args.allow_nonzero_replacements or old_value == 0.0
            if is_valid and retry_value is not None and can_replace:
                new_value = retry_value
                change_rows.append(
                    {
                        "id": row_id,
                        "old_value": format_float(old_value),
                        "new_value": format_float(new_value),
                        "program": str(detail.get("selected_program") or ""),
                        "confidence_reason": str(detail.get("confidence_reason") or ""),
                        "agreement_count": str(detail.get("agreement_count") or 0),
                        "question": str(detail.get("question") or ""),
                    }
                )
            elif is_valid and not can_replace:
                rejected_retry_rows.append(
                    {
                        "id": row_id,
                        "reason": "nonzero base row protected",
                        "old_value": old_value,
                        "retry_value": retry_value,
                    }
                )
            elif reject_reason:
                rejected_retry_rows.append({"id": row_id, "reason": reject_reason})

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
    validation["changed_rows"] = len(change_rows)
    validation["rejected_retry_rows"] = len(rejected_retry_rows)
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
