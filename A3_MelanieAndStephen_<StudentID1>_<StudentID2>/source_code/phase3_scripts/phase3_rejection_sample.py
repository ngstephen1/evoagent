"""
phase3_rejection_sample.py — STaR / rejection sampling on the fine-tuned model.

For each TRAIN question, sample k programs from the fine-tuned model at a non-zero
temperature, execute each with the DSL evaluator, and keep only the DISTINCT
programs that execute to the gold answer. Those verified (question, program) pairs
become an augmented SFT dataset that reinforces correct reasoning — especially on
the hard (multi-step) questions where a greedy pass fails but a sampled one hits.

Runs on a GPU (SGLang inference). Output: <out> jsonl of {system,user,assistant}.

Usage (via Modal):
    python3 phase3_rejection_sample.py --model /runs/lora_merged_qwen3_8b \
        --strategy-path strategies/ft_strategy.json \
        --k 8 --temp 0.8 --max-per-q 3 --out /runs/rs/train_sft_rs.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.data import _load_json_split          # noqa: E402
from src.executor import build_prompt          # noqa: E402
from src.evaluator import evaluate_program     # noqa: E402
from src.model import QwenInference            # noqa: E402
from src.strategy import Strategy, CoTFormat   # noqa: E402

SYSTEM = (
    "Bạn là một trợ lý AI chuyên phân tích tài chính tiếng Việt. "
    "Nhiệm vụ của bạn là viết chương trình dạng các hàm toán học để trả lời câu hỏi "
    "dựa trên văn bản và bảng số liệu được cung cấp."
)


def tof(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def close(a, b, tol=1e-4):
    return a is not None and b is not None and abs(a - b) <= tol


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--strategy-path", default="strategies/ft_strategy.json")
    p.add_argument("--k", type=int, default=8, help="samples per question")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-per-q", type=int, default=3, help="cap distinct verified programs kept per question")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="/runs/rs/train_sft_rs.jsonl")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    strat = Strategy.from_json(Path(args.strategy_path).read_text(encoding="utf-8"))
    rows = _load_json_split(HERE / "data" / "train.json")
    if args.limit:
        rows = rows[:args.limit]

    model = QwenInference(
        model_name_or_path=args.model,
        max_new_tokens=256,
        temperature=0.0,
        use_4bit=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=16384,
        self_consistency_k=1,
    )
    model.load()

    prompts = []
    for r in rows:
        user = build_prompt(strat, r["context"], r["question"])
        prompts.append(model.format_prompt(
            system_message=SYSTEM, user_message=user,
            enable_thinking=(strat.cot_format != CoTFormat.NONE),
        ))

    k = args.k
    expanded = []
    for p in prompts:
        expanded.extend([p] * k)
    print(f"Sampling {k} per question x {len(prompts)} questions = {len(expanded)} generations at temp {args.temp}")
    gen = model.generate_batch(expanded, cot_format=False, temperature_override=args.temp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verified = []
    solved_q = 0
    for i, r in enumerate(rows):
        gold = tof(r["exe_ans"])
        if gold is None:
            continue
        user = build_prompt(strat, r["context"], r["question"])
        seen = set()
        kept = []
        for j in range(k):
            out = gen[i * k + j]
            prog = (out.predicted_answer or "").strip()
            if not prog or prog in seen:
                continue
            try:
                val = evaluate_program(prog, r["table"])
            except Exception:
                continue
            if close(val, gold):
                seen.add(prog)
                kept.append(prog)
            if len(kept) >= args.max_per_q:
                break
        if kept:
            solved_q += 1
        for prog in kept:
            verified.append({"system": SYSTEM, "user": user, "assistant": f"PROGRAM: {prog}"})

    with out_path.open("w", encoding="utf-8") as f:
        for e in verified:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Questions with >=1 verified program: {solved_q}/{len(rows)} ({100*solved_q/len(rows):.1f}%)")
    print(f"Total verified examples written: {len(verified)} -> {out_path}")


if __name__ == "__main__":
    main()
