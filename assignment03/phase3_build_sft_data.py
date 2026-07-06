"""
phase3_build_sft_data.py — build the LoRA SFT dataset from train.json.

Each example is a chat triple whose USER turn is exactly the inference prompt
(built with the same build_prompt + injected _DSL_BLOCK that submit.py uses) and
whose ASSISTANT turn is the gold program in the "PROGRAM: ..." form that
extract_answer() parses. Training on the exact inference distribution means the
fine-tuned adapter drops straight into the existing submit.py pipeline.

Output: data_sft/train_sft.jsonl  (+ a small held-out data_sft/eval_sft.jsonl)
Each line: {"system": ..., "user": ..., "assistant": "PROGRAM: <program>"}

Usage:  python3 phase3_build_sft_data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.data import _load_json_split          # noqa: E402
from src.executor import build_prompt          # noqa: E402
from src.strategy import Strategy              # noqa: E402

SYSTEM = (
    "Bạn là một trợ lý AI chuyên phân tích tài chính tiếng Việt. "
    "Nhiệm vụ của bạn là viết chương trình dạng các hàm toán học để trả lời câu hỏi "
    "dựa trên văn bản và bảng số liệu được cung cấp."
)
STRATEGY_PATH = HERE / "strategies" / "ft_strategy.json"
OUT_DIR = HERE / "data_sft"
EVAL_N = 200          # small held-out slice for train-time eval loss


def main() -> None:
    strat = Strategy.from_json(STRATEGY_PATH.read_text(encoding="utf-8"))
    rows = _load_json_split(HERE / "data" / "train.json")
    OUT_DIR.mkdir(exist_ok=True)

    examples = []
    skipped = 0
    for r in rows:
        program = (r.get("answer") or "").strip()      # gold program stored in 'answer'
        if not program:
            skipped += 1
            continue
        user = build_prompt(strat, r["context"], r["question"])
        examples.append({
            "system": SYSTEM,
            "user": user,
            "assistant": f"PROGRAM: {program}",
        })

    eval_rows = examples[:EVAL_N]
    train_rows = examples[EVAL_N:]
    with (OUT_DIR / "train_sft.jsonl").open("w", encoding="utf-8") as f:
        for e in train_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (OUT_DIR / "eval_sft.jsonl").open("w", encoding="utf-8") as f:
        for e in eval_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"train examples: {len(train_rows)}  eval: {len(eval_rows)}  skipped(no program): {skipped}")
    # quick length stats to size max_seq_len
    import statistics
    ulens = [len(e["user"]) for e in examples]
    print(f"user char length: median={statistics.median(ulens)} p95={sorted(ulens)[int(0.95*len(ulens))]} max={max(ulens)}")
    print("sample assistant target:", examples[0]["assistant"])


if __name__ == "__main__":
    main()
