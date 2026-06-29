"""Conservative Phase 3 Kaggle post-processing.

This script is intentionally separate from the assignment graders and EvoAgent
core logic. It applies small, auditable numeric corrections to an existing
submission using only test-time artifacts: submission values, generated program
details, and the public test questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = ["id", "Usage", "predicted_value"]
CHANGE_COLUMNS = [
    "id",
    "old_value",
    "new_value",
    "rule_name",
    "question",
    "program",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply conservative Phase 3 numeric post-processing to a Kaggle submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--submission", required=True, type=Path, help="Input Kaggle submission CSV.")
    parser.add_argument("--details", required=True, type=Path, help="submit.py debug details JSON.")
    parser.add_argument("--test", required=True, type=Path, help="Official data/test.json file.")
    parser.add_argument("--output", required=True, type=Path, help="Corrected submission CSV path.")
    parser.add_argument("--changes", required=True, type=Path, help="Audit CSV of changed predictions.")
    parser.add_argument(
        "--enable-risky-ratio",
        action="store_true",
        help="Enable reciprocal division-ratio correction for margin/rate questions.",
    )
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    text = unicodedata.normalize("NFC", value or "")
    return text.casefold()


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def to_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing value")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("blank value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def clean_float(value: Any, default: float = 0.0) -> float:
    try:
        return to_float(value)
    except (TypeError, ValueError):
        return default


def format_float(value: float) -> str:
    if not math.isfinite(value):
        value = 0.0
    return format(value, ".15g")


def load_submission(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in OUTPUT_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"{path} contains a row with empty id")
            if row_id in rows:
                raise ValueError(f"{path} contains duplicate id: {row_id}")
            rows[row_id] = row
    return rows


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("test file must contain a list of rows")
    return rows


def test_question(row: dict[str, Any]) -> str:
    if isinstance(row.get("qa"), dict):
        return str(row["qa"].get("question", ""))
    return str(row.get("question", ""))


def load_details(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        print(f"Warning: details file not found: {path}. Program-aware rules will be skipped.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError("details JSON must be a list or contain a 'predictions' list")

    details: dict[str, dict[str, Any]] = {}
    for row in data:
        if isinstance(row, dict) and row.get("id") is not None:
            details[str(row["id"])] = row
    return details


def detail_question(detail: dict[str, Any], fallback: str) -> str:
    for key in ("question", "prompt_question"):
        if detail.get(key):
            return str(detail[key])
    return fallback


def detail_program(detail: dict[str, Any]) -> str:
    for key in ("program", "predicted_answer", "answer"):
        if detail.get(key):
            return str(detail[key])
    return ""


def apply_rules(
    value: float,
    question: str,
    program: str,
    enable_risky_ratio: bool,
) -> tuple[float, str | None, str | None]:
    q = normalize_text(question)
    p = normalize_text(program)

    if (
        value < 0
        and "subtract" in p
        and contains_any(q, ("tăng bao nhiêu", "mức tăng", "mức giảm", "chênh lệch"))
        and "tỷ lệ phần trăm thay đổi" not in q
    ):
        return abs(value), "abs_negative_difference", "Directional difference question with reversed subtract sign."

    if (
        abs(value) < 1
        and "divide" in p
        and "100" in p
        and contains_any(q, ("roe", "roa"))
        and contains_any(q, ("cao hơn", "chênh lệch", "so sánh", "khác biệt"))
    ):
        return value * 100, "percent_point_divided_by_100", "Percent-point ROE/ROA comparison was divided by 100."

    if (
        1 < value < 2
        and "divide" in p
        and contains_any(q, ("tăng trưởng", "tốc độ tăng trưởng"))
    ):
        return value - 1, "growth_ratio_to_growth_rate", "Growth ratio appears to include the original base value."

    if (
        enable_risky_ratio
        and value > 1
        and "divide" in p
        and contains_any(q, ("biên lợi nhuận", "tỷ suất", "chiếm bao nhiêu phần trăm"))
        and "lần so với" not in q
    ):
        return 1 / value, "division_reversal_ratio", "Margin/rate division appears reversed."

    return value, None, None


def validate_output(rows: list[dict[str, str]], expected_ids: list[str]) -> None:
    actual_ids = [row["id"] for row in rows]
    if len(actual_ids) != len(expected_ids):
        raise ValueError(f"row count mismatch: expected {len(expected_ids)}, got {len(actual_ids)}")
    if actual_ids != expected_ids:
        raise ValueError("output id order does not match test.json")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("output contains duplicate ids")
    for row in rows:
        to_float(row["predicted_value"])


def main() -> None:
    args = parse_args()

    submission_rows = load_submission(args.submission)
    test_rows = load_test_rows(args.test)
    details = load_details(args.details)

    expected_ids = [str(row.get("id", idx)) for idx, row in enumerate(test_rows)]
    missing_ids = [row_id for row_id in expected_ids if row_id not in submission_rows]
    extra_ids = sorted(set(submission_rows) - set(expected_ids))
    if missing_ids:
        raise ValueError(f"submission missing {len(missing_ids)} test ids; first missing: {missing_ids[:3]}")
    if extra_ids:
        raise ValueError(f"submission contains {len(extra_ids)} ids not in test.json; first extra: {extra_ids[:3]}")

    output_rows: list[dict[str, str]] = []
    change_rows: list[dict[str, str]] = []
    changes_by_rule: Counter[str] = Counter()
    zero_before = 0
    zero_after = 0

    for test_row in test_rows:
        row_id = str(test_row.get("id", ""))
        input_row = submission_rows[row_id]
        old_value = clean_float(input_row.get("predicted_value"))
        if old_value == 0.0:
            zero_before += 1

        detail = details.get(row_id, {})
        question = detail_question(detail, test_question(test_row))
        program = detail_program(detail)
        new_value, rule_name, reason = apply_rules(
            old_value,
            question,
            program,
            enable_risky_ratio=args.enable_risky_ratio,
        )
        if not math.isfinite(new_value):
            new_value = old_value
        if new_value == 0.0:
            zero_after += 1

        output_rows.append(
            {
                "id": row_id,
                "Usage": input_row.get("Usage", "Public") or "Public",
                "predicted_value": format_float(new_value),
            }
        )

        if rule_name is not None and new_value != old_value:
            changes_by_rule[rule_name] += 1
            change_rows.append(
                {
                    "id": row_id,
                    "old_value": format_float(old_value),
                    "new_value": format_float(new_value),
                    "rule_name": rule_name,
                    "question": question,
                    "program": program,
                    "reason": reason or "",
                }
            )

    validate_output(output_rows, expected_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.changes.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    with args.changes.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGE_COLUMNS)
        writer.writeheader()
        writer.writerows(change_rows)

    print(f"total rows: {len(output_rows)}")
    print(f"changed rows: {len(change_rows)}")
    print(f"changes by rule: {dict(changes_by_rule)}")
    print(f"zero count before: {zero_before}")
    print(f"zero count after: {zero_after}")
    print(f"output path: {args.output}")
    print(f"changes path: {args.changes}")
    print("valid submission: yes")


if __name__ == "__main__":
    main()
