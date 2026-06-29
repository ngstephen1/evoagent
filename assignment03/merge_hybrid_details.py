"""Merge submit.py detail JSON files for a hybrid Kaggle submission.

Hybrid submissions can combine numeric predictions from multiple runs. This
helper reconstructs an aligned `submission_details.json` so downstream
post-processing can still inspect the question and generated program that
produced each selected value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN001_SUBMISSION = Path("runs/kaggle_arc_best/submission_checked.csv")
DEFAULT_RUN001_DETAILS = Path("runs/kaggle_arc_best/submission_details.json")
DEFAULT_RUN002_SUBMISSION = Path("runs/kaggle_iter003/submission_checked.csv")
DEFAULT_RUN002_DETAILS = Path("runs/kaggle_iter003/submission_details.json")
DEFAULT_HYBRID_SUBMISSION = Path("runs/kaggle_hybrid_001_002/submission_checked.csv")
DEFAULT_OUTPUT = Path("runs/kaggle_hybrid_001_002/submission_details.json")
DEFAULT_TEST = Path("data/test.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create submission_details.json for hybrid Run003 from Run001 and Run002 details.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run001-submission", type=Path, default=DEFAULT_RUN001_SUBMISSION)
    parser.add_argument("--run001-details", type=Path, default=DEFAULT_RUN001_DETAILS)
    parser.add_argument("--run002-submission", type=Path, default=DEFAULT_RUN002_SUBMISSION)
    parser.add_argument("--run002-details", type=Path, default=DEFAULT_RUN002_DETAILS)
    parser.add_argument("--hybrid-submission", type=Path, default=DEFAULT_HYBRID_SUBMISSION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def to_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def same_value(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def load_submission(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames or "predicted_value" not in reader.fieldnames:
            raise ValueError(f"{path} must contain id and predicted_value columns")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"{path} contains an empty id")
            if row_id in rows:
                raise ValueError(f"{path} contains duplicate id: {row_id}")
            rows[row_id] = row
    return rows


def load_details(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list or a predictions list")

    rows: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        row_id = str(row["id"])
        if row_id in rows:
            raise ValueError(f"{path} contains duplicate detail id: {row_id}")
        rows[row_id] = row
    return rows


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return data


def test_question(row: dict[str, Any]) -> str:
    qa = row.get("qa")
    if isinstance(qa, dict):
        return str(qa.get("question", ""))
    return str(row.get("question", ""))


def require_all_ids(name: str, ids: list[str], mapping: dict[str, Any]) -> None:
    missing = [row_id for row_id in ids if row_id not in mapping]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} ids; first missing: {missing[:3]}")


def normalized_detail(
    source: dict[str, Any],
    row_id: str,
    question: str,
    hybrid_value: float,
    source_name: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "question": str(source.get("question") or question),
        "raw_output": str(source.get("raw_output") or ""),
        "program": str(source.get("program") or source.get("predicted_answer") or ""),
        "predicted_value": hybrid_value,
        "detail_source": source_name,
    }


def main() -> None:
    args = parse_args()

    run001_submission = load_submission(args.run001_submission)
    run002_submission = load_submission(args.run002_submission)
    hybrid_submission = load_submission(args.hybrid_submission)
    run001_details = load_details(args.run001_details)
    run002_details = load_details(args.run002_details)
    test_rows = load_test_rows(args.test)

    expected_ids = [str(row.get("id", idx)) for idx, row in enumerate(test_rows)]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("test.json contains duplicate ids")

    require_all_ids("Run001 submission", expected_ids, run001_submission)
    require_all_ids("Run002 submission", expected_ids, run002_submission)
    require_all_ids("Hybrid submission", expected_ids, hybrid_submission)
    require_all_ids("Run001 details", expected_ids, run001_details)
    require_all_ids("Run002 details", expected_ids, run002_details)

    merged: list[dict[str, Any]] = []
    used_run001 = 0
    used_run002 = 0

    for test_row in test_rows:
        row_id = str(test_row.get("id", ""))
        question = test_question(test_row)
        run001_value = to_float(run001_submission[row_id]["predicted_value"])
        run002_value = to_float(run002_submission[row_id]["predicted_value"])
        hybrid_value = to_float(hybrid_submission[row_id]["predicted_value"])

        use_run002 = (
            same_value(hybrid_value, run002_value)
            and same_value(run001_value, 0.0)
            and not same_value(run002_value, 0.0)
        )

        if use_run002:
            merged.append(normalized_detail(run002_details[row_id], row_id, question, hybrid_value, "run002"))
            used_run002 += 1
        else:
            merged.append(normalized_detail(run001_details[row_id], row_id, question, hybrid_value, "run001"))
            used_run001 += 1

    output_ids = [row["id"] for row in merged]
    if output_ids != expected_ids:
        raise ValueError("merged details id order does not match test.json")
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("merged details contain duplicate ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"details count: {len(merged)}")
    print("ids match test.json: yes")
    print("duplicate ids: 0")
    print(f"used Run001 details: {used_run001}")
    print(f"used Run002 details: {used_run002}")
    print(f"output path: {args.output}")


if __name__ == "__main__":
    main()
