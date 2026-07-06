"""Run012 GPT-OSS evidence-to-DSL smoke solver.

This is Phase 3-only tooling. It talks to an already-running SGLang
OpenAI-compatible GPT-OSS server, extracts structured evidence, asks for a DSL
program, then deterministically executes and validates that program locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from phase3_retry_failures import (  # noqa: E402
    MAX_ABS_VALUE,
    format_duration,
    format_float,
    load_dataset_rows_by_id,
    load_details,
    load_submission,
    load_test_rows,
    repair_program,
    row_text_context,
    select_targets_from_csv,
    to_float,
)
from src.evaluator import evaluate_program  # noqa: E402


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv")
DEFAULT_BASE_DETAILS = Path("runs/kaggle_hybrid_001_002/submission_details.json")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_TARGET_ROWS = Path("runs/kaggle_run012_gptoss_evidence/target_rows.csv")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_run012_gptoss_evidence")
DEFAULT_SERVER_URL = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
STAGE1_FILE = "stage1_evidence.jsonl"
STAGE2_FILE = "stage2_dsl.jsonl"
EXECUTION_FILE = "execution_results.jsonl"
RETRY_DETAILS_FILE = "retry_details.json"
SUMMARY_FILE = "summary.json"
NUMERIC_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-OSS evidence extraction then DSL synthesis over suspicious rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--base-details", type=Path, default=DEFAULT_BASE_DETAILS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-targets", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--skip-server-check", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def extract_final_message(content: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker in content:
        content = content.rsplit(marker, 1)[-1]
    content = re.sub(r"<\\|end\\|>.*$", "", content, flags=re.DOTALL)
    return content.strip()


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    in_string = False
    escape = False
    depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def extract_json_object(text: str, required_keys: set[str] | None = None) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    parsed: list[dict[str, Any]] = []
    for candidate in _balanced_json_candidates(text):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            parsed.append(payload)
    if not parsed:
        return None
    if required_keys:
        for payload in reversed(parsed):
            if required_keys <= set(payload):
                return payload
    return parsed[-1]


def call_chat_completion(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        args.server_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {message[:1000]}") from exc


def models_url_from_chat_url(server_url: str) -> str:
    parsed = urllib.parse.urlparse(server_url)
    path = parsed.path
    if path.endswith("/v1/chat/completions"):
        path = path[: -len("/v1/chat/completions")] + "/v1/models"
    elif path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")] + "/models"
    else:
        path = "/v1/models"
    return urllib.parse.urlunparse(parsed._replace(path=path, query="", fragment=""))


def check_server_ready(args: argparse.Namespace) -> None:
    url = models_url_from_chat_url(args.server_url)
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=min(args.timeout_seconds, 15.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"GPT-OSS server is not reachable at {url}. Start SGLang first and confirm "
            f"`curl -s {url}` works. Original error: {exc}"
        ) from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(models, list) and models:
        model_ids = [str(model.get("id")) for model in models if isinstance(model, dict)]
        print(f"server_check=ok models={model_ids[:3]}", flush=True)
    else:
        print(f"server_check=ok url={url}", flush=True)


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def parse_stage_json(response: dict[str, Any], required_keys: set[str]) -> tuple[str, dict[str, Any] | None]:
    content = response_content(response)
    final_text = extract_final_message(content)
    payload = extract_json_object(final_text, required_keys)
    if payload is None:
        payload = extract_json_object(content, required_keys)
    return final_text, payload


def confidence_rank(confidence: Any) -> int:
    value = str(confidence or "").lower()
    return {"low": 0, "medium": 1, "high": 2}.get(value, -1)


def numeric_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-6, NUMERIC_TOL * max(abs(left), abs(right), 1.0))


def build_stage1_prompt(context: str, question: str, old_value: float, target_reason: str) -> str:
    return (
        "You are extracting evidence for financial numeric QA. Do not solve by guessing.\n"
        "Use only the provided context. No external lookup. No source PDF lookup.\n"
        "Your final answer must be exactly one JSON object. The first final character must be { and the last final character must be }.\n"
        "Do not include markdown, explanations, code fences, or channel tags in the final answer.\n"
        "Schema: {\"relevant_numbers\":[{\"value\":number,\"unit\":\"...\","
        "\"year_or_period\":\"...\",\"source_phrase\":\"...\",\"why_relevant\":\"...\"}],"
        "\"question_operation\":\"addition|subtraction|division|multiplication|table_lookup|percentage|growth|unknown\","
        "\"operation_explanation\":\"...\",\"expected_unit\":\"...\",\"confidence\":\"high|medium|low\"}.\n"
        "If evidence is missing, use an empty relevant_numbers list and confidence low.\n\n"
        f"Current prediction: {format_float(old_value)}\n"
        f"Suspicion reason: {target_reason}\n\n"
        f"Context:\n{context[:9000]}\n\n"
        f"Question: {question}\n\n"
        "Final JSON:"
    )


def build_stage2_prompt(context: str, question: str, evidence: dict[str, Any]) -> str:
    evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
    return (
        "You convert extracted evidence into executable FinQA DSL.\n"
        "Use only the evidence and context. Do not invent numbers.\n"
        "Your final answer must be exactly one JSON object. The first final character must be { and the last final character must be }.\n"
        "Do not include markdown, explanations, code fences, or channel tags in the final answer.\n"
        "Schema: {\"dsl_program\":\"<program>\",\"numeric_answer\":number,"
        "\"answer_unit\":\"...\",\"confidence\":\"high|medium|low\",\"self_check\":\"...\"}.\n"
        "DSL syntax examples: subtract(125, 100), divide(#0, 100), add(0, 650), table_max(revenue, none).\n"
        "Allowed ops: add, subtract, multiply, divide, exp, greater, abs, table_average, table_max, table_min, table_sum.\n"
        "Rules: comma-separated calls, #0/#1 references, no nested calls, no JSON operator lists.\n\n"
        f"Context:\n{context[:9000]}\n\n"
        f"Question: {question}\n\n"
        f"Evidence JSON:\n{evidence_text}\n\n"
        "Final JSON:"
    )


def relevant_number_count(evidence: dict[str, Any] | None) -> int:
    if not evidence:
        return 0
    values = evidence.get("relevant_numbers")
    return len(values) if isinstance(values, list) else 0


def stage_operation(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "unknown"
    return str(evidence.get("question_operation") or "unknown").lower()


def execute_stage2(payload: dict[str, Any] | None, table: list[list[str]]) -> dict[str, Any]:
    if not payload:
        return {"valid": False, "reason": "missing stage2 json"}
    program_raw = payload.get("dsl_program") or payload.get("program")
    program, repair_reason = repair_program(str(program_raw) if program_raw is not None else None, "")
    if not program:
        return {"valid": False, "reason": repair_reason or "program repair failed", "raw_program": program_raw}
    try:
        executed_value = float(evaluate_program(program, table))
    except Exception as exc:
        return {"valid": False, "reason": f"execution failed: {exc}", "program": program, "raw_program": program_raw}
    if not math.isfinite(executed_value):
        return {"valid": False, "reason": "non-finite executed value", "program": program, "executed_value": executed_value}
    if executed_value == 0.0:
        return {"valid": False, "reason": "executed value is zero", "program": program, "executed_value": executed_value}
    if abs(executed_value) > MAX_ABS_VALUE:
        return {"valid": False, "reason": f"executed value abs>{MAX_ABS_VALUE:g}", "program": program, "executed_value": executed_value}
    try:
        model_value = to_float(payload.get("numeric_answer"))
    except Exception as exc:
        return {"valid": False, "reason": f"invalid model numeric_answer: {exc}", "program": program, "executed_value": executed_value}
    if not numeric_close(executed_value, model_value):
        return {
            "valid": False,
            "reason": "executed value does not match model numeric_answer",
            "program": program,
            "executed_value": executed_value,
            "model_value": model_value,
        }
    return {
        "valid": True,
        "program": program,
        "repair_reason": repair_reason,
        "executed_value": executed_value,
        "model_value": model_value,
    }


def target_has_strong_signal(target_reason: str) -> bool:
    reasons = set(reason for reason in target_reason.split("|") if reason)
    strong = {
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
    return bool(reasons & strong)


def acceptance_reason(
    *,
    old_value: float,
    target_reason: str,
    evidence: dict[str, Any] | None,
    stage2: dict[str, Any] | None,
    execution: dict[str, Any],
) -> tuple[bool, str]:
    if not execution.get("valid"):
        return False, str(execution.get("reason") or "invalid execution")
    if not evidence:
        return False, "missing stage1 evidence"
    if not stage2:
        return False, "missing stage2 dsl"
    stage1_conf = confidence_rank(evidence.get("confidence"))
    stage2_conf = confidence_rank(stage2.get("confidence"))
    if relevant_number_count(evidence) == 0:
        return False, "no relevant numbers"
    if stage_operation(evidence) != "table_lookup" and relevant_number_count(evidence) < 2:
        return False, "fewer than two relevant numbers"
    if old_value == 0.0:
        if stage1_conf < 1 or stage2_conf < 1:
            return False, "zero-row confidence below medium"
        return True, "zero_row_evidence_dsl_valid"
    if stage1_conf < 2 or stage2_conf < 2:
        return False, "nonzero-row confidence below high"
    if not target_has_strong_signal(target_reason):
        return False, "nonzero-row lacks strong suspicious signal"
    return True, "nonzero_row_high_confidence_evidence_dsl_valid"


def main() -> None:
    args = parse_args()
    submission = load_submission(args.base_submission)
    details = load_details(args.base_details)
    test_rows = load_test_rows(args.test)
    targets = select_targets_from_csv(args.target_rows, submission, details, test_rows, args.limit_targets)
    dataset_rows = load_dataset_rows_by_id()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the target file beside outputs for artifact portability.
    target_copy = args.output_dir / "target_rows.csv"
    if args.target_rows.resolve() != target_copy.resolve():
        target_copy.write_text(args.target_rows.read_text(encoding="utf-8-sig"), encoding="utf-8")

    print(
        "Run012 GPT-OSS evidence solver configuration:\n"
        f"  server_url={args.server_url}\n"
        f"  model={args.model}\n"
        f"  output_dir={args.output_dir}\n"
        f"  targets={len(targets)} temperature={args.temperature} max_tokens={args.max_tokens}",
        flush=True,
    )
    if not args.skip_server_check:
        check_server_ready(args)

    results: list[dict[str, Any]] = []
    started_at = time.time()
    for index, target in enumerate(targets, start=1):
        row_id = str(target["id"])
        row = dataset_rows.get(row_id)
        if row is None:
            raise ValueError(f"dataset missing target id: {row_id}")
        context = str(row.get("context") or row_text_context(row))
        question = str(target.get("question") or "")
        old_value = float(target["old_value"])
        target_reason = str(target.get("target_reason") or "")
        print(f"[{index}/{len(targets)}] id={row_id} elapsed={format_duration(time.time() - started_at)}", flush=True)

        stage1_prompt = build_stage1_prompt(context, question, old_value, target_reason)
        try:
            stage1_response = call_chat_completion(args, stage1_prompt)
            stage1_final, evidence = parse_stage_json(stage1_response, {"relevant_numbers", "question_operation"})
            stage1_error = ""
        except Exception as exc:
            stage1_response = {"error": str(exc)}
            stage1_final, evidence, stage1_error = "", None, str(exc)
        append_jsonl(
            args.output_dir / STAGE1_FILE,
            {
                "id": row_id,
                "prompt": stage1_prompt,
                "raw_response": stage1_response,
                "final_text": stage1_final,
                "evidence": evidence,
                "error": stage1_error,
            },
        )

        if evidence:
            stage2_prompt = build_stage2_prompt(context, question, evidence)
            try:
                stage2_response = call_chat_completion(args, stage2_prompt)
                stage2_final, stage2 = parse_stage_json(stage2_response, {"dsl_program", "numeric_answer"})
                stage2_error = ""
            except Exception as exc:
                stage2_response = {"error": str(exc)}
                stage2_final, stage2, stage2_error = "", None, str(exc)
        else:
            stage2_prompt = ""
            stage2_response = {"skipped": "missing stage1 evidence"}
            stage2_final, stage2, stage2_error = "", None, "missing stage1 evidence"
        execution = execute_stage2(stage2, row.get("table") or [])
        accepted, reason = acceptance_reason(
            old_value=old_value,
            target_reason=target_reason,
            evidence=evidence,
            stage2=stage2,
            execution=execution,
        )
        result = {
            "id": row_id,
            "question": question,
            "old_value": old_value,
            "accepted": accepted,
            "selected_value": execution.get("executed_value") if accepted else None,
            "selected_program": execution.get("program") if accepted else None,
            "confidence_reason": reason,
            "agreement_count": 1 if accepted else 0,
            "target_reason": target_reason,
            "stage1_confidence": str((evidence or {}).get("confidence") or ""),
            "stage2_confidence": str((stage2 or {}).get("confidence") or ""),
            "relevant_number_count": relevant_number_count(evidence),
            "question_operation": stage_operation(evidence),
            "stage1_evidence": evidence,
            "stage2_dsl": stage2,
            "execution": execution,
        }
        results.append(result)
        append_jsonl(
            args.output_dir / STAGE2_FILE,
            {
                "id": row_id,
                "prompt": stage2_prompt,
                "raw_response": stage2_response,
                "final_text": stage2_final,
                "stage2": stage2,
                "error": stage2_error,
            },
        )
        append_jsonl(args.output_dir / EXECUTION_FILE, {"id": row_id, "execution": execution, "accepted": accepted, "reason": reason})
        value_text = format_float(result["selected_value"]) if result["selected_value"] is not None else "None"
        print(
            f"  accepted={accepted} value={value_text} reason={reason} "
            f"stage1={result['stage1_confidence']} stage2={result['stage2_confidence']} "
            f"numbers={result['relevant_number_count']}",
            flush=True,
        )

    (args.output_dir / RETRY_DETAILS_FILE).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "target_rows": len(results),
        "stage1_valid_json": sum(row["stage1_evidence"] is not None for row in results),
        "stage2_valid_json": sum(row["stage2_dsl"] is not None for row in results),
        "executable_dsl": sum(bool(row["execution"].get("valid")) for row in results),
        "accepted_replacements": sum(bool(row["accepted"]) for row in results),
        "stage1_confidence_distribution": dict(Counter(row["stage1_confidence"] or "missing" for row in results)),
        "stage2_confidence_distribution": dict(Counter(row["stage2_confidence"] or "missing" for row in results)),
        "operation_distribution": dict(Counter(row["question_operation"] for row in results)),
        "retry_details": str(args.output_dir / RETRY_DETAILS_FILE),
        "stage1_evidence": str(args.output_dir / STAGE1_FILE),
        "stage2_dsl": str(args.output_dir / STAGE2_FILE),
        "execution_results": str(args.output_dir / EXECUTION_FILE),
    }
    (args.output_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
