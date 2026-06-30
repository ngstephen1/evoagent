"""Run011 GPT-OSS-120B OpenAI-API smoke solver.

This is Phase 3-only tooling. It talks to an already-running SGLang
OpenAI-compatible server and checks whether gpt-oss-120b can produce
parseable, executable DSL programs on a tiny suspicious-row sample.
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
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from phase3_retry_failures import (  # noqa: E402
    Candidate,
    candidate_from_output,
    format_duration,
    format_float,
    load_dataset_rows_by_id,
    load_details,
    load_submission,
    load_test_rows,
    row_text_context,
    select_candidate,
    select_targets_from_csv,
)


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv")
DEFAULT_BASE_DETAILS = Path("runs/kaggle_hybrid_001_002/submission_details.json")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_TARGET_ROWS = Path("runs/kaggle_run011_model_smoke/target_rows.csv")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_run011_model_smoke")
DEFAULT_SERVER_URL = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_ABS_VALUE = 1e8
PROMPTS_FILE = "prompts.jsonl"
RESPONSES_FILE = "responses.jsonl"
PARSED_FILE = "parsed_candidates.jsonl"
RETRY_DETAILS_FILE = "retry_details.json"
SUMMARY_FILE = "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small GPT-OSS-120B DSL smoke over suspicious rows.",
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
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
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


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_ref(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("ref") or value.get("$ref") or value.get("value")
    text = str(value).strip()
    if text.startswith("$"):
        return "#" + text[1:]
    return text


def program_from_json_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        calls: list[str] = []
        for step in value:
            if not isinstance(step, dict):
                return None
            op = str(step.get("op") or step.get("operation") or "").strip()
            args = step.get("args") or step.get("arguments") or []
            if not op or not isinstance(args, list):
                return None
            calls.append(f"{op}({', '.join(normalize_ref(arg) for arg in args)})")
        return ", ".join(calls) if calls else None
    return None


def parse_model_content(content: str) -> dict[str, Any]:
    final_text = extract_final_message(content)
    payload = extract_json_object(final_text)
    program = None
    answer = None
    confidence = ""
    if payload:
        program = program_from_json_value(payload.get("program") or payload.get("dsl"))
        answer = payload.get("answer", payload.get("result"))
        confidence = str(payload.get("confidence") or payload.get("confidence_reason") or "")
    if program is None:
        match = re.search(
            r"\b(?:PROGRAM|program|dsl)\s*[:=]\s*(.+)",
            final_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            program = match.group(1).strip().splitlines()[0].strip()
    if answer is None:
        answer_match = re.search(r"\b(?:answer|result)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", final_text, flags=re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1)
    return {
        "final_text": final_text,
        "json": payload,
        "program": program,
        "answer": answer,
        "confidence": confidence.lower(),
        "unit_reason": str(payload.get("unit_reason") or "") if payload else "",
        "change_decision": str(payload.get("change_decision") or "") if payload else "",
    }


def build_prompt(context: str, question: str, old_value: float, target_reason: str, previous_program: str) -> str:
    return (
        "You solve financial numeric QA by producing executable FinQA DSL only.\n"
        "Return only the final JSON object. No markdown. No prose outside JSON.\n"
        "Required JSON schema: "
        "{\"program\":\"<dsl>\",\"answer\":number,\"confidence\":\"high|medium|low\","
        "\"change_decision\":\"change|keep|unsure\",\"unit_reason\":\"short reason\"}.\n"
        "The program value must match the answer exactly when executed.\n\n"
        "Allowed DSL syntax examples:\n"
        "- subtract(125, 100), divide(#0, 100)\n"
        "- add(600, 55)\n"
        "- table_max(revenue, none)\n"
        "Allowed DSL ops: add, subtract, multiply, divide, exp, greater, abs, "
        "table_average, table_max, table_min, table_sum.\n"
        "Forbidden formats: JSON operator lists, nested calls, equations, Python, markdown, explanations.\n"
        "DSL rules: calls are comma-separated; use #0, #1 for previous step results; "
        "do not nest function calls; use add(0, x) to return a single extracted number.\n"
        "For rates and percentages, follow the question wording: use decimal form only if the question asks for decimal form; "
        "otherwise use the numeric unit implied by the table/question.\n\n"
        "Conservative replacement policy:\n"
        "- The current prediction is the baseline. Keep it unless the context directly supports a different answer.\n"
        "- If unsure, set confidence to low and change_decision to unsure.\n"
        "- If the current prediction is 0.0, prefer a supported nonzero answer only when the relevant number/calculation is explicit.\n\n"
        f"Current suspicious prediction: {format_float(old_value)}\n"
        f"Suspicion reason: {target_reason}\n"
        f"Previous program if available: {previous_program or '(none)'}\n\n"
        f"Context:\n{context[:8000]}\n\n"
        f"Question: {question}\n\n"
        "Final JSON:"
    )


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


def content_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def candidate_from_parsed(row_id: str, sample_index: int, parsed: dict[str, Any], raw_content: str, table: list[list[str]]) -> Candidate:
    program = parsed.get("program")
    if not program:
        return Candidate(
            row_id=row_id,
            sample_index=sample_index,
            raw_output=raw_content,
            extracted_program=None,
            reject_reason="no executable program parsed",
        )
    synthetic_output = f"PROGRAM: {program}"
    candidate = candidate_from_output(row_id, sample_index, synthetic_output, table)
    candidate.raw_output = raw_content
    return candidate


def confidence_high(parsed_items: list[dict[str, Any]]) -> bool:
    return any(str(item.get("confidence") or "").lower() == "high" for item in parsed_items)


def change_supported(parsed_items: list[dict[str, Any]]) -> bool:
    return any(str(item.get("change_decision") or "").lower() == "change" for item in parsed_items)


def main() -> None:
    args = parse_args()
    submission = load_submission(args.base_submission)
    details = load_details(args.base_details)
    test_rows = load_test_rows(args.test)
    targets = select_targets_from_csv(args.target_rows, submission, details, test_rows, args.limit_targets)
    dataset_rows = load_dataset_rows_by_id()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Run011 GPT-OSS smoke configuration:\n"
        f"  server_url={args.server_url}\n"
        f"  model={args.model}\n"
        f"  output_dir={args.output_dir}\n"
        f"  targets={len(targets)} num_samples={args.num_samples}\n"
        f"  temperature={args.temperature} max_tokens={args.max_tokens}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    started_at = time.time()
    for index, target in enumerate(targets, start=1):
        row_id = str(target["id"])
        row = dataset_rows.get(row_id)
        if row is None:
            raise ValueError(f"dataset missing target id: {row_id}")
        detail = details.get(row_id, {})
        context = str(row.get("context") or row_text_context(row))
        question = str(target.get("question") or "")
        old_value = float(target["old_value"])
        previous_program = str(detail.get("program") or "")
        prompt = build_prompt(
            context=context,
            question=question,
            old_value=old_value,
            target_reason=str(target.get("target_reason") or ""),
            previous_program=previous_program,
        )
        append_jsonl(args.output_dir / PROMPTS_FILE, {"id": row_id, "prompt": prompt})
        print(
            f"[{index}/{len(targets)}] id={row_id} elapsed={format_duration(time.time() - started_at)}",
            flush=True,
        )

        candidates: list[Candidate] = []
        parsed_items: list[dict[str, Any]] = []
        for sample_index in range(args.num_samples):
            try:
                response = call_chat_completion(args, prompt)
                content = content_from_response(response)
                parsed = parse_model_content(content)
                candidate = candidate_from_parsed(row_id, sample_index, parsed, content, row.get("table") or [])
                finish_reason = ((response.get("choices") or [{}])[0] or {}).get("finish_reason")
            except Exception as exc:
                response = {"error": str(exc)}
                content = ""
                parsed = {"final_text": "", "json": None, "program": None, "answer": None, "confidence": ""}
                candidate = Candidate(row_id=row_id, sample_index=sample_index, raw_output="", reject_reason=f"request failed: {exc}")
                finish_reason = "error"
            candidates.append(candidate)
            parsed_items.append(parsed)
            append_jsonl(args.output_dir / RESPONSES_FILE, {"id": row_id, "sample_index": sample_index, "response": response})
            append_jsonl(
                args.output_dir / PARSED_FILE,
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "parsed": parsed,
                    "candidate": asdict(candidate),
                    "finish_reason": finish_reason,
                },
            )
            value_text = format_float(candidate.value) if candidate.value is not None else "None"
            print(
                f"  sample {sample_index + 1}/{args.num_samples} "
                f"valid={candidate.valid} value={value_text} "
                f"reason={candidate.reject_reason or candidate.repair_reason or 'ok'}",
                flush=True,
            )

        selected, confidence_reason, agreement_count = select_candidate(candidates)
        if selected and confidence_high(parsed_items):
            confidence_reason = f"{confidence_reason}+model_high_confidence" if confidence_reason else "model_high_confidence"
        if selected and change_supported(parsed_items):
            confidence_reason = f"{confidence_reason}+model_change_decision" if confidence_reason else "model_change_decision"
        result = {
            "id": row_id,
            "question": question,
            "old_value": old_value,
            "accepted": selected is not None,
            "selected_value": selected.value if selected else None,
            "selected_program": selected.repaired_program if selected else None,
            "confidence_reason": confidence_reason,
            "agreement_count": agreement_count,
            "target_reason": str(target.get("target_reason") or ""),
            "detail_source": str(target.get("detail_source") or ""),
            "model_confidence_high": confidence_high(parsed_items),
            "model_change_supported": change_supported(parsed_items),
            "candidates": [asdict(candidate) for candidate in candidates],
        }
        results.append(result)
        value_text = format_float(result["selected_value"]) if result["selected_value"] is not None else "None"
        print(
            f"  completed id={row_id} accepted={result['accepted']} "
            f"value={value_text} agreement={agreement_count} reason={confidence_reason}",
            flush=True,
        )

    (args.output_dir / RETRY_DETAILS_FILE).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "target_rows": len(results),
        "accepted_retries": sum(bool(row["accepted"]) for row in results),
        "agreement_count_distribution": dict(Counter(str(row["agreement_count"]) for row in results)),
        "model_high_confidence_rows": sum(bool(row.get("model_confidence_high")) for row in results),
        "model_change_supported_rows": sum(bool(row.get("model_change_supported")) for row in results),
        "prompts": str(args.output_dir / PROMPTS_FILE),
        "responses": str(args.output_dir / RESPONSES_FILE),
        "parsed_candidates": str(args.output_dir / PARSED_FILE),
        "retry_details": str(args.output_dir / RETRY_DETAILS_FILE),
    }
    (args.output_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
