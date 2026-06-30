"""Run010 agentic execution-verified solver.

This Phase 3-only script extends targeted retry with verifier decisions. It
does not modify core EvoAgent logic, graders, or existing submissions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from phase3_retry_failures import (  # noqa: E402
    CANDIDATE_PROGRAMS_FILE,
    MAX_ABS_VALUE,
    PROGRESS_STATE_FILE,
    Candidate,
    build_retry_prompt,
    candidate_from_output,
    format_duration,
    format_float,
    load_dataset_rows_by_id,
    load_details,
    load_submission,
    load_test_rows,
    repair_program,
    row_text_context,
    same_cluster_value,
    select_candidate,
    select_targets_from_csv,
    to_float,
)
from src.model import QwenInference  # noqa: E402


DEFAULT_BASE_SUBMISSION = Path("runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv")
DEFAULT_BASE_DETAILS = Path("runs/kaggle_hybrid_001_002/submission_details.json")
DEFAULT_TEST = Path("data/test.json")
DEFAULT_TARGET_ROWS = Path("runs/kaggle_run010_agentic_solver/target_rows.csv")
DEFAULT_OUTPUT_DIR = Path("runs/kaggle_run010_agentic_solver")
DEFAULT_MODEL = "QuantTrio/Qwen3.5-4B-AWQ"
PARTIAL_DETAILS_FILE = "retry_details.partial.jsonl"
VERIFIER_DECISIONS_FILE = "verifier_decisions.jsonl"
RETRY_DETAILS_FILE = "retry_details.json"
RETRY_PREDICTIONS_FILE = "retry_predictions.json"
SUMMARY_FILE = "summary.json"


@dataclass
class VerifierDecision:
    id: str
    raw_output: str
    selected_candidate_index: int | None
    decision: str
    confidence: str
    operation_match: bool
    unit_plausible: bool
    direction_plausible: bool
    magnitude_plausible: bool
    reason: str


@dataclass
class AgenticResult:
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
    verifier_decision: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run010 agentic solver over suspicious Kaggle rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE_SUBMISSION)
    parser.add_argument("--base-details", type=Path, default=DEFAULT_BASE_DETAILS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit-targets", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list-targets-only", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def result_from_dict(data: dict[str, Any]) -> AgenticResult:
    return AgenticResult(
        id=str(data.get("id", "")),
        question=str(data.get("question", "")),
        old_value=to_float(data.get("old_value", 0.0)),
        accepted=bool(data.get("accepted")),
        selected_value=to_float(data["selected_value"]) if data.get("selected_value") is not None else None,
        selected_program=str(data["selected_program"]) if data.get("selected_program") is not None else None,
        confidence_reason=str(data["confidence_reason"]) if data.get("confidence_reason") is not None else None,
        agreement_count=int(data.get("agreement_count") or 0),
        target_reason=str(data.get("target_reason") or ""),
        detail_source=str(data.get("detail_source") or ""),
        verifier_decision=dict(data.get("verifier_decision") or {}),
        candidates=list(data.get("candidates") or []),
    )


def load_partial_results(output_dir: Path, valid_ids: set[str]) -> dict[str, AgenticResult]:
    path = output_dir / PARTIAL_DETAILS_FILE
    if not path.exists():
        return {}
    rows: dict[str, AgenticResult] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                result = result_from_dict(json.loads(line))
            except Exception as exc:
                print(f"warning: skipped invalid checkpoint line {line_no}: {exc}", flush=True)
                continue
            if result.id in valid_ids:
                rows[result.id] = result
    return rows


def append_partial_result(output_dir: Path, result: AgenticResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / PARTIAL_DETAILS_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def reset_outputs(output_dir: Path) -> None:
    for filename in (
        PARTIAL_DETAILS_FILE,
        PROGRESS_STATE_FILE,
        CANDIDATE_PROGRAMS_FILE,
        VERIFIER_DECISIONS_FILE,
        RETRY_DETAILS_FILE,
        RETRY_PREDICTIONS_FILE,
        SUMMARY_FILE,
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def ordered_results(targets: list[dict[str, Any]], completed: dict[str, AgenticResult]) -> list[AgenticResult]:
    return [completed[str(target["id"])] for target in targets if str(target["id"]) in completed]


def cluster_candidates(candidates: list[Candidate]) -> list[dict[str, Any]]:
    clusters: list[list[Candidate]] = []
    valid = [cand for cand in candidates if cand.valid and cand.value is not None]
    for cand in valid:
        assert cand.value is not None
        for cluster in clusters:
            assert cluster[0].value is not None
            if same_cluster_value(cand.value, cluster[0].value):
                cluster.append(cand)
                break
        else:
            clusters.append([cand])
    clusters.sort(key=lambda cluster: (-len(cluster), min(cand.sample_index for cand in cluster)))
    summaries = []
    for idx, cluster in enumerate(clusters):
        representative = cluster[0]
        summaries.append(
            {
                "cluster_index": idx,
                "agreement_count": len(cluster),
                "representative_sample_index": representative.sample_index,
                "value": representative.value,
                "program": representative.repaired_program,
                "sample_indices": [cand.sample_index for cand in cluster],
            }
        )
    return summaries


def build_verifier_prompt(
    *,
    context: str,
    question: str,
    old_value: float,
    target_reason: str,
    clusters: list[dict[str, Any]],
) -> str:
    cluster_text = json.dumps(clusters[:12], ensure_ascii=False, indent=2)
    return (
        "Bạn là bộ kiểm chứng lời giải toán tài chính. "
        "Hãy đánh giá các chương trình DSL đã được thực thi và chọn ứng viên đáng tin nhất nếu có.\n\n"
        "Chỉ trả lời JSON hợp lệ với các khóa: selected_candidate_index, decision, confidence, "
        "operation_match, unit_plausible, direction_plausible, magnitude_plausible, reason.\n"
        "decision phải là accept hoặc reject. confidence là high, medium, hoặc low.\n"
        "Chỉ accept khi chương trình khớp phép toán câu hỏi, đơn vị hợp lý, hướng tử/mẫu hoặc phép trừ hợp lý, "
        "và độ lớn câu trả lời hợp lý. Nếu không chắc, reject.\n\n"
        f"Giá trị hiện tại: {format_float(old_value)}\n"
        f"Lý do nghi ngờ: {target_reason}\n\n"
        f"Bối cảnh:\n{context[:8000]}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Các cụm ứng viên đã thực thi:\n{cluster_text}\n\n"
        "JSON:"
    )


def parse_verifier(raw_output: str, row_id: str) -> VerifierDecision:
    text = raw_output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    payload: dict[str, Any] = {}
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            payload = {}
    selected_raw = payload.get("selected_candidate_index")
    try:
        selected_idx = int(selected_raw) if selected_raw is not None else None
    except Exception:
        selected_idx = None
    decision = str(payload.get("decision") or "reject").lower()
    confidence = str(payload.get("confidence") or "low").lower()
    return VerifierDecision(
        id=row_id,
        raw_output=raw_output,
        selected_candidate_index=selected_idx,
        decision=decision if decision in {"accept", "reject"} else "reject",
        confidence=confidence if confidence in {"high", "medium", "low"} else "low",
        operation_match=bool(payload.get("operation_match")),
        unit_plausible=bool(payload.get("unit_plausible")),
        direction_plausible=bool(payload.get("direction_plausible")),
        magnitude_plausible=bool(payload.get("magnitude_plausible")),
        reason=str(payload.get("reason") or "verifier_parse_fallback"),
    )


def verifier_accepts(decision: VerifierDecision) -> bool:
    return (
        decision.decision == "accept"
        and decision.confidence == "high"
        and decision.operation_match
        and decision.unit_plausible
        and decision.direction_plausible
        and decision.magnitude_plausible
    )


def choose_with_verifier(
    candidates: list[Candidate],
    verifier: VerifierDecision,
) -> tuple[Candidate | None, str | None, int]:
    selected, reason, agreement = select_candidate(candidates)
    valid = [cand for cand in candidates if cand.valid and cand.value is not None]
    if verifier_accepts(verifier) and verifier.selected_candidate_index is not None:
        for cand in valid:
            if cand.sample_index == verifier.selected_candidate_index:
                cluster_count = sum(
                    1
                    for other in valid
                    if other.value is not None and cand.value is not None and same_cluster_value(other.value, cand.value)
                )
                return cand, f"verifier_high_cluster_{cluster_count}", cluster_count
    if selected is not None:
        return selected, reason, agreement
    return None, reason, agreement


def progress_payload(
    args: argparse.Namespace,
    *,
    started_at: float,
    target_count: int,
    completed_count: int,
    accepted_count: int,
    last_row_id: str | None,
    status: str,
) -> dict[str, Any]:
    elapsed = time.time() - started_at
    avg_per_row = elapsed / completed_count if completed_count else None
    remaining = max(0, target_count - completed_count)
    eta_seconds = avg_per_row * remaining if avg_per_row is not None else None
    return {
        "status": status,
        "updated_at": utc_now_iso(),
        "model": args.model,
        "output_dir": str(args.output_dir),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "verifier_max_new_tokens": args.verifier_max_new_tokens,
        "max_model_len": args.max_model_len,
        "target_count": target_count,
        "completed_count": completed_count,
        "accepted_count": accepted_count,
        "remaining_count": remaining,
        "last_row_id": last_row_id,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed": format_duration(elapsed),
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "eta": format_duration(eta_seconds),
    }


def write_progress_state(
    args: argparse.Namespace,
    *,
    started_at: float,
    target_count: int,
    completed_count: int,
    accepted_count: int,
    last_row_id: str | None,
    status: str,
) -> None:
    atomic_write_json(
        args.output_dir / PROGRESS_STATE_FILE,
        progress_payload(
            args,
            started_at=started_at,
            target_count=target_count,
            completed_count=completed_count,
            accepted_count=accepted_count,
            last_row_id=last_row_id,
            status=status,
        ),
    )


def run_solver(
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
    completed: dict[str, AgenticResult],
    started_at: float,
) -> dict[str, AgenticResult]:
    dataset_rows = load_dataset_rows_by_id()
    model = QwenInference(
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    model.load()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_count = len(completed)
    accepted_count = sum(result.accepted for result in completed.values())
    pending = [target for target in targets if str(target["id"]) not in completed]
    candidate_path = args.output_dir / CANDIDATE_PROGRAMS_FILE
    verifier_path = args.output_dir / VERIFIER_DECISIONS_FILE

    with candidate_path.open("a", encoding="utf-8") as candidate_file, verifier_path.open("a", encoding="utf-8") as verifier_file:
        for pending_index, target in enumerate(pending, start=1):
            row_id = str(target["id"])
            row = dataset_rows.get(row_id)
            if row is None:
                raise ValueError(f"dataset missing target id: {row_id}")
            context = str(row.get("context") or row_text_context(row))
            question = str(target.get("question") or "")
            old_value = float(target["old_value"])
            previous_program = str(target.get("program") or "")

            elapsed = time.time() - started_at
            avg = elapsed / completed_count if completed_count else None
            remaining = len(targets) - completed_count
            eta = avg * remaining if avg is not None else None
            print(
                f"[{completed_count + 1}/{len(targets)}] id={row_id} "
                f"pending_index={pending_index}/{len(pending)} remaining={remaining} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                flush=True,
            )

            retry_prompt = build_retry_prompt(context=context, question=question, previous_program=previous_program)
            formatted_retry_prompt = model.format_prompt(
                system_message="Bạn là trợ lý tạo chương trình DSL tài chính. Luôn xuất đúng PROGRAM: <dsl>.",
                user_message=retry_prompt,
                enable_thinking=False,
            )

            candidates: list[Candidate] = []
            for sample_index in range(args.num_samples):
                try:
                    raw_output = model.generate_text(
                        formatted_retry_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        presence_penalty=0.0,
                    )
                    candidate = candidate_from_output(row_id, sample_index, raw_output, row["table"])
                except Exception as exc:
                    candidate = Candidate(row_id=row_id, sample_index=sample_index, raw_output="", reject_reason=f"generation failed: {exc}")
                candidates.append(candidate)
                candidate_file.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
                candidate_file.flush()
                value_text = format_float(candidate.value) if candidate.value is not None else "None"
                print(
                    f"  sample {sample_index + 1}/{args.num_samples} "
                    f"valid={candidate.valid} value={value_text} "
                    f"reason={candidate.reject_reason or candidate.repair_reason or 'ok'}",
                    flush=True,
                )

            clusters = cluster_candidates(candidates)
            if clusters:
                verifier_prompt = build_verifier_prompt(
                    context=context,
                    question=question,
                    old_value=old_value,
                    target_reason=str(target.get("target_reason") or ""),
                    clusters=clusters,
                )
                formatted_verifier_prompt = model.format_prompt(
                    system_message="Bạn là bộ kiểm chứng JSON cho chương trình DSL tài chính.",
                    user_message=verifier_prompt,
                    enable_thinking=False,
                )
                try:
                    raw_verifier = model.generate_text(
                        formatted_verifier_prompt,
                        max_new_tokens=args.verifier_max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        presence_penalty=0.0,
                    )
                except Exception as exc:
                    raw_verifier = json.dumps({"decision": "reject", "reason": f"verifier generation failed: {exc}"})
            else:
                raw_verifier = json.dumps({"decision": "reject", "reason": "no valid clusters"})

            verifier = parse_verifier(raw_verifier, row_id)
            verifier_file.write(json.dumps(asdict(verifier), ensure_ascii=False) + "\n")
            verifier_file.flush()

            selected, confidence_reason, agreement_count = choose_with_verifier(candidates, verifier)
            result = AgenticResult(
                id=row_id,
                question=question,
                old_value=old_value,
                accepted=selected is not None,
                selected_value=selected.value if selected else None,
                selected_program=selected.repaired_program if selected else None,
                confidence_reason=confidence_reason,
                agreement_count=agreement_count,
                target_reason=str(target.get("target_reason") or ""),
                detail_source=str(target.get("detail_source") or ""),
                verifier_decision=asdict(verifier),
                candidates=[asdict(candidate) for candidate in candidates],
            )
            completed[row_id] = result
            append_partial_result(args.output_dir, result)
            completed_count += 1
            if result.accepted:
                accepted_count += 1
            write_progress_state(
                args,
                started_at=started_at,
                target_count=len(targets),
                completed_count=completed_count,
                accepted_count=accepted_count,
                last_row_id=row_id,
                status="running",
            )
            value_text = format_float(result.selected_value) if result.selected_value is not None else "None"
            print(
                f"  completed id={row_id} accepted={result.accepted} value={value_text} "
                f"agreement={agreement_count} reason={confidence_reason} verifier={verifier.decision}/{verifier.confidence}",
                flush=True,
            )
    return completed


def write_outputs(output_dir: Path, results: list[AgenticResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = [asdict(result) for result in results]
    predictions = [
        {
            "id": result.id,
            "predicted_value": result.selected_value,
            "program": result.selected_program,
            "accepted": result.accepted,
            "confidence_reason": result.confidence_reason,
            "agreement_count": result.agreement_count,
        }
        for result in results
        if result.accepted
    ]
    (output_dir / RETRY_DETAILS_FILE).write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / RETRY_PREDICTIONS_FILE).write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "target_rows": len(results),
        "accepted_retries": sum(result.accepted for result in results),
        "agreement_count_distribution": Counter(str(result.agreement_count) for result in results),
        "verifier_decision_distribution": Counter(str((result.verifier_decision or {}).get("decision", "")) for result in results),
        "retry_details": str(output_dir / RETRY_DETAILS_FILE),
        "retry_predictions": str(output_dir / RETRY_PREDICTIONS_FILE),
        "candidate_programs": str(output_dir / CANDIDATE_PROGRAMS_FILE),
        "verifier_decisions": str(output_dir / VERIFIER_DECISIONS_FILE),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def print_targets(targets: list[dict[str, Any]]) -> None:
    print(f"target_count={len(targets)}")
    reasons = Counter()
    for target in targets:
        for reason in str(target.get("target_reason") or "").split("|"):
            if reason:
                reasons[reason] += 1
    print("target_reasons=" + json.dumps(reasons, ensure_ascii=False, sort_keys=True))
    for target in targets[:25]:
        print(json.dumps(target, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    submission = load_submission(args.base_submission)
    details = load_details(args.base_details)
    test_rows = load_test_rows(args.test)
    targets = select_targets_from_csv(args.target_rows, submission, details, test_rows, args.limit_targets)
    for target in targets:
        detail = details.get(str(target["id"]), {})
        target["program"] = str(detail.get("program") or "")

    if args.list_targets_only:
        print_targets(targets)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        reset_outputs(args.output_dir)
    started_at = time.time()
    valid_ids = {str(target["id"]) for target in targets}
    completed = load_partial_results(args.output_dir, valid_ids) if args.resume else {}
    print(
        "Run010 agentic solver configuration:\n"
        f"  model={args.model}\n"
        f"  output_dir={args.output_dir}\n"
        f"  targets={len(targets)} completed={len(completed)} pending={len(targets) - len(completed)}\n"
        f"  num_samples={args.num_samples} temperature={args.temperature} top_p={args.top_p}\n"
        f"  max_new_tokens={args.max_new_tokens} verifier_max_new_tokens={args.verifier_max_new_tokens}\n"
        f"  resume={args.resume}",
        flush=True,
    )
    completed = run_solver(args, targets, completed, started_at)
    results = ordered_results(targets, completed)
    write_outputs(args.output_dir, results)
    write_progress_state(
        args,
        started_at=started_at,
        target_count=len(targets),
        completed_count=len(results),
        accepted_count=sum(result.accepted for result in results),
        last_row_id=results[-1].id if results else None,
        status="completed",
    )


if __name__ == "__main__":
    main()
