"""Run008 targeted retry for failed Kaggle rows.

This is Phase 3-only tooling. It does not modify EvoAgent core logic or
graders. The default target set is the current best Run003 rows whose
prediction is exactly 0.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.evaluator import evaluate_program
from src.model import QwenInference, clean_and_fix_program, extract_answer


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_001_002/submission_checked.csv")
DEFAULT_BASE_DETAILS = Path("runs/kaggle_hybrid_001_002/submission_details.json")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_STRATEGY_PATH = Path("runs/exp_self_arc/iter_best_strategy.json")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_retry_run008")
DEFAULT_MODEL = "QuantTrio/Qwen3.5-4B-AWQ"
SUPPORTED_OPS = (
    "table_average",
    "table_max",
    "table_min",
    "table_sum",
    "multiply",
    "subtract",
    "divide",
    "greater",
    "add",
    "exp",
    "abs",
)
MAX_ABS_VALUE = 1e8
CLUSTER_REL_TOL = 1e-4
CLUSTER_ABS_TOL = 1e-6


@dataclass
class Candidate:
    row_id: str
    sample_index: int
    raw_output: str
    extracted_program: str | None = None
    repaired_program: str | None = None
    repair_reason: str | None = None
    value: float | None = None
    valid: bool = False
    reject_reason: str | None = None


@dataclass
class RetryResult:
    id: str
    question: str
    old_value: float
    accepted: bool
    selected_value: float | None = None
    selected_program: str | None = None
    confidence_reason: str | None = None
    agreement_count: int = 0
    target_reason: str = ""
    detail_source: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted multi-sample retry for Run003 zero/failure rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--base-details", type=Path, default=DEFAULT_BASE_DETAILS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--strategy-path", type=Path, default=DEFAULT_STRATEGY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--target-zero-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Target rows where the base prediction is exactly 0.0.",
    )
    parser.add_argument(
        "--include-suspicious",
        action="store_true",
        help="Also target suspicious nonzero rows. Disabled by default for Run008.",
    )
    parser.add_argument("--limit-targets", type=int, default=None)
    parser.add_argument(
        "--list-targets-only",
        action="store_true",
        help="Print target rows and exit before loading the model or datasets.",
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
        required = {"id", "Usage", "predicted_value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"{path} contains empty id")
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
        raise ValueError(f"{path} must contain a list or predictions list")

    rows: dict[str, dict[str, Any]] = {}
    for row in data:
        if isinstance(row, dict) and row.get("id") is not None:
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"{path} contains duplicate detail id: {row_id}")
            rows[row_id] = row
    return rows


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list")
    return rows


def test_question(row: dict[str, Any]) -> str:
    qa = row.get("qa")
    if isinstance(qa, dict):
        return str(qa.get("question", ""))
    return str(row.get("question", ""))


def row_text_context(row: dict[str, Any]) -> str:
    pre_text = " ".join(row.get("pre_text") or []).strip()
    table = row.get("table") or []
    post_text = " ".join(row.get("post_text") or []).strip()
    table_text = "\n".join(" | ".join(str(cell) for cell in line) for line in table)
    parts = []
    if pre_text:
        parts.append(pre_text)
    if table_text:
        parts.append("Bảng:\n" + table_text)
    if post_text:
        parts.append(post_text)
    return "\n\n".join(parts).strip()


def load_dataset_rows_by_id() -> dict[str, dict[str, Any]]:
    from src.data import load_dataset

    ds = load_dataset()
    return {str(row["id"]): dict(row) for row in ds["test"]}


def detail_question(detail: dict[str, Any], fallback: str) -> str:
    return str(detail.get("question") or detail.get("prompt_question") or fallback)


def is_suspicious_value(value: float, detail: dict[str, Any]) -> bool:
    program = str(detail.get("program") or "")
    if abs(value) > MAX_ABS_VALUE:
        return True
    if not program or len(program) > 3000:
        return True
    return False


def select_targets(
    submission: dict[str, dict[str, str]],
    details: dict[str, dict[str, Any]],
    test_rows: list[dict[str, Any]],
    target_zero_only: bool,
    include_suspicious: bool,
    limit_targets: int | None,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in test_rows:
        row_id = str(row.get("id", ""))
        if row_id not in submission:
            raise ValueError(f"submission missing test id: {row_id}")
        value = to_float(submission[row_id]["predicted_value"])
        detail = details.get(row_id, {})
        target_reason = ""
        if target_zero_only and value == 0.0:
            target_reason = "base_prediction_zero"
        elif include_suspicious and is_suspicious_value(value, detail):
            target_reason = "suspicious_nonzero"
        if target_reason:
            targets.append(
                {
                    "id": row_id,
                    "old_value": value,
                    "question": detail_question(detail, test_question(row)),
                    "target_reason": target_reason,
                    "detail_source": str(detail.get("detail_source") or ""),
                }
            )
    if limit_targets is not None:
        targets = targets[: max(0, limit_targets)]
    return targets


def strip_wrappers(text: str) -> str:
    text = re.sub(r"```(?:[a-zA-Z0-9_]*)\s*(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:program|chương\s*trình|answer|đáp\s*án)\s*[:\-]\s*", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def convert_identity(program: str) -> tuple[str, bool]:
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return f"add(0, {match.group(1).strip()})"

    converted = re.sub(r"\bidentity\s*\(\s*([^(),]+)\s*\)", repl, program, flags=re.IGNORECASE)
    return converted, changed


def remove_table_quotes(program: str) -> tuple[str, bool]:
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        op, first, second = match.group(1), match.group(2), match.group(3)
        return f"{op}({first}, {second})"

    pattern = r"\b(table_(?:average|max|min|sum))\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*([^)]+)\)"
    return re.sub(pattern, repl, program, flags=re.IGNORECASE), changed


def extract_balanced_calls(text: str) -> list[str]:
    calls: list[str] = []
    lower = text.lower()
    i = 0
    while i < len(text):
        match_op = None
        match_start = -1
        for op in SUPPORTED_OPS:
            pattern = op + "("
            idx = lower.find(pattern, i)
            if idx != -1 and (match_start == -1 or idx < match_start):
                match_start = idx
                match_op = op
        if match_op is None:
            break
        start = match_start
        pos = start + len(match_op)
        if pos >= len(text) or text[pos] != "(":
            i = start + 1
            continue
        depth = 0
        end = None
        for j in range(pos, len(text)):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            i = start + len(match_op)
            continue
        calls.append(text[start:end].strip())
        i = end
    return calls


def has_nested_call(call: str) -> bool:
    first_open = call.find("(")
    if first_open == -1:
        return False
    inner = call[first_open + 1 : -1]
    return any(re.search(rf"\b{re.escape(op)}\s*\(", inner, flags=re.IGNORECASE) for op in SUPPORTED_OPS)


def validate_program(program: str) -> tuple[bool, str | None]:
    calls = extract_balanced_calls(program)
    if not calls:
        return False, "no supported calls"
    if ", ".join(calls) != program:
        normalized = ", ".join(calls)
        if normalized.replace(" ", "") != program.replace(" ", ""):
            return False, "program contains text outside supported calls"
    for idx, call in enumerate(calls):
        op = call.split("(", 1)[0].strip().lower()
        if op not in SUPPORTED_OPS:
            return False, f"unsupported op {op}"
        if has_nested_call(call):
            return False, "nested function call"
        for ref in re.findall(r"#(\d+)", call):
            if int(ref) >= idx:
                return False, f"invalid forward reference #{ref}"
    return True, None


def repair_program(raw_program: str | None, raw_output: str) -> tuple[str | None, str | None]:
    text = raw_program or raw_output
    if not text:
        return None, "empty output"

    text = strip_wrappers(text)
    text, identity_changed = convert_identity(text)
    text, table_quote_changed = remove_table_quotes(text)
    calls = extract_balanced_calls(text)
    if not calls:
        return None, "no supported calls extracted"

    program = ", ".join(calls)
    program = clean_and_fix_program(program)
    valid, reason = validate_program(program)
    if not valid:
        return None, reason

    reasons = []
    if raw_program != program:
        reasons.append("extracted_supported_calls")
    if identity_changed:
        reasons.append("identity_to_add")
    if table_quote_changed:
        reasons.append("removed_table_quotes")
    if not reasons:
        reasons.append("as_extracted")
    return program, "+".join(reasons)


def is_strict_repair(reason: str | None) -> bool:
    if not reason:
        return False
    allowed = {"as_extracted", "extracted_supported_calls", "identity_to_add", "removed_table_quotes"}
    return set(reason.split("+")).issubset(allowed)


def candidate_from_output(row_id: str, sample_index: int, raw_output: str, table: list[list[str]]) -> Candidate:
    extracted = extract_answer(raw_output)
    repaired, repair_reason = repair_program(extracted, raw_output)
    cand = Candidate(
        row_id=row_id,
        sample_index=sample_index,
        raw_output=raw_output,
        extracted_program=extracted,
        repaired_program=repaired,
        repair_reason=repair_reason,
    )
    if not repaired:
        cand.reject_reason = repair_reason or "repair failed"
        return cand

    try:
        value = evaluate_program(repaired, table)
    except Exception as exc:
        cand.reject_reason = f"execution failed: {exc}"
        return cand

    if not math.isfinite(value):
        cand.reject_reason = "non-finite value"
        return cand
    if value == 0.0:
        cand.reject_reason = "zero value"
        cand.value = value
        return cand
    if abs(value) > MAX_ABS_VALUE:
        cand.reject_reason = f"extreme value abs>{MAX_ABS_VALUE:g}"
        cand.value = value
        return cand

    cand.value = float(value)
    cand.valid = True
    return cand


def same_cluster_value(left: float, right: float) -> bool:
    return abs(left - right) <= max(CLUSTER_ABS_TOL, CLUSTER_REL_TOL * max(abs(left), abs(right), 1.0))


def select_candidate(candidates: list[Candidate]) -> tuple[Candidate | None, str | None, int]:
    valid = [cand for cand in candidates if cand.valid and cand.value is not None]
    if not valid:
        return None, "no valid finite nonzero candidates", 0

    clusters: list[list[Candidate]] = []
    for cand in valid:
        assert cand.value is not None
        for cluster in clusters:
            assert cluster[0].value is not None
            if same_cluster_value(cand.value, cluster[0].value):
                cluster.append(cand)
                break
        else:
            clusters.append([cand])

    clusters.sort(key=lambda c: (-len(c), min(item.sample_index for item in c)))
    best = clusters[0]
    if len(best) >= 2:
        representative = best[0]
        return representative, f"cluster_agreement_{len(best)}", len(best)

    if len(valid) == 1 and is_strict_repair(valid[0].repair_reason):
        return valid[0], "single_strict_repair_no_conflict", 1

    values = [format_float(cand.value) for cand in valid if cand.value is not None]
    return None, f"insufficient agreement among valid values: {values}", 0


def build_retry_prompt(context: str, question: str, previous_program: str) -> str:
    previous = previous_program.strip() if previous_program else "(none)"
    return (
        "Bạn là bộ sửa lỗi chương trình toán tài chính. "
        "Chỉ trả lời đúng một dòng bắt đầu bằng PROGRAM: và không giải thích.\n\n"
        "DSL hợp lệ gồm: add, subtract, multiply, divide, exp, greater, abs, "
        "table_average, table_max, table_min, table_sum.\n"
        "Quy tắc:\n"
        "- Không lồng hàm.\n"
        "- Mỗi bước phân cách bằng dấu phẩy.\n"
        "- Dùng #0, #1 để tham chiếu kết quả bước trước.\n"
        "- Không dùng identity; nếu cần trả số x thì dùng add(0, x).\n"
        "- Không dùng markdown, không dùng tiếng Việt trong chương trình.\n"
        "- Kết quả phần trăm dùng dạng thập phân, không nhân 100 ở cuối.\n\n"
        f"Chương trình trước đó bị lỗi hoặc ra 0.0:\n{previous}\n\n"
        f"Bối cảnh:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "PROGRAM:"
    )


def print_targets(targets: list[dict[str, Any]]) -> None:
    print(f"target_count={len(targets)}")
    reasons = Counter(str(t["target_reason"]) for t in targets)
    print("target_reasons=" + json.dumps(reasons, ensure_ascii=False, sort_keys=True))
    for item in targets:
        print(
            json.dumps(
                {
                    "id": item["id"],
                    "old_value": item["old_value"],
                    "target_reason": item["target_reason"],
                    "detail_source": item.get("detail_source", ""),
                    "question": item.get("question", ""),
                },
                ensure_ascii=False,
            )
        )


def run_generation(args: argparse.Namespace, targets: list[dict[str, Any]]) -> list[RetryResult]:
    dataset_rows = load_dataset_rows_by_id()

    model = QwenInference(
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    model.load()

    results: list[RetryResult] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_programs.jsonl"

    with candidate_path.open("w", encoding="utf-8") as candidate_file:
        for target_idx, target in enumerate(targets, start=1):
            row_id = str(target["id"])
            if row_id not in dataset_rows:
                raise ValueError(f"dataset missing target id: {row_id}")
            row = dataset_rows[row_id]
            context = str(row.get("context") or row_text_context(row))
            question = str(row.get("question") or target["question"])
            detail_program = str(target.get("program") or "")
            prompt = build_retry_prompt(
                context=context,
                question=question,
                previous_program=detail_program,
            )
            formatted_prompt = model.format_prompt(
                system_message=(
                    "Bạn là trợ lý sửa chương trình DSL cho bài toán tài chính. "
                    "Luôn xuất đúng một dòng PROGRAM: <dsl>."
                ),
                user_message=prompt,
                enable_thinking=False,
            )

            candidates: list[Candidate] = []
            print(f"[{target_idx}/{len(targets)}] retry {row_id}")
            for sample_index in range(args.num_samples):
                try:
                    raw_output = model.generate_text(
                        formatted_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        presence_penalty=0.0,
                    )
                except Exception as exc:
                    cand = Candidate(
                        row_id=row_id,
                        sample_index=sample_index,
                        raw_output="",
                        reject_reason=f"generation failed: {exc}",
                    )
                else:
                    cand = candidate_from_output(row_id, sample_index, raw_output, row["table"])
                candidates.append(cand)
                candidate_file.write(json.dumps(asdict(cand), ensure_ascii=False) + "\n")
                candidate_file.flush()

            selected, reason, agreement = select_candidate(candidates)
            retry_result = RetryResult(
                id=row_id,
                question=question,
                old_value=float(target["old_value"]),
                accepted=selected is not None,
                selected_value=selected.value if selected else None,
                selected_program=selected.repaired_program if selected else None,
                confidence_reason=reason,
                agreement_count=agreement,
                target_reason=str(target["target_reason"]),
                detail_source=str(target.get("detail_source") or ""),
                candidates=[asdict(cand) for cand in candidates],
            )
            results.append(retry_result)
    return results


def write_outputs(output_dir: Path, results: list[RetryResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = [asdict(result) for result in results]
    predictions = [
        {
            "id": result.id,
            "predicted_value": result.selected_value,
            "program": result.selected_program,
            "accepted": result.accepted,
            "confidence_reason": result.confidence_reason,
        }
        for result in results
        if result.accepted
    ]
    (output_dir / "retry_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "retry_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    accepted = sum(result.accepted for result in results)
    print(f"target rows: {len(results)}")
    print(f"accepted retries: {accepted}")
    print(f"retry_details: {output_dir / 'retry_details.json'}")
    print(f"retry_predictions: {output_dir / 'retry_predictions.json'}")
    print(f"candidate_programs: {output_dir / 'candidate_programs.jsonl'}")


def main() -> None:
    args = parse_args()
    submission = load_submission(args.base_submission)
    details = load_details(args.base_details)
    test_rows = load_test_rows(args.test)
    targets = select_targets(
        submission=submission,
        details=details,
        test_rows=test_rows,
        target_zero_only=args.target_zero_only,
        include_suspicious=args.include_suspicious,
        limit_targets=args.limit_targets,
    )
    for target in targets:
        detail = details.get(str(target["id"]), {})
        target["program"] = str(detail.get("program") or "")

    if args.list_targets_only:
        print_targets(targets)
        return

    if not targets:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_outputs(args.output_dir, [])
        return

    results = run_generation(args, targets)
    write_outputs(args.output_dir, results)


if __name__ == "__main__":
    main()
