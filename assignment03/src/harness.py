"""
harness.py — Full EvoAgent training loop.

Orchestrates T iterations of:
  1. Propose a new strategy (or use the seed for iteration 0).
  2. Evaluate on the train subset.
  3. Evaluate on the dev split.
  4. Reflect on the results.
  5. Save state to disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from tqdm import tqdm

from src.executor import EvalResult, evaluate, classify_question_type, TokenBudget
from src.model import QwenInference
from src.self_proposer import propose_self
from src.self_reflector import reflect_self
from src.strategy import Strategy, StrategyHistory, StrategyMetadata, make_seed_strategy

logger = logging.getLogger(__name__)


def select_parent_strategy(
    history: StrategyHistory,
    afo_mode: str,
    afo_prob_best: float = 0.4,
    afo_prob_original: float = 0.3,
    afo_prob_latest: float = 0.3,
) -> Optional[Strategy]:
    """
    Select the parent strategy to mutate based on the Always-From-Original (AFO) mode.

    TODO: Implement the Always-From-Original (AFO) parent selection policy.
    Modes:
    - 'none': Always mutate from the latest strategy in history.
    - 'best': Always mutate from the best strategy (highest dev accuracy) found so far.
    - 'original': Always mutate from the original seed strategy (iteration 0).
    - 'probabilistic': Select from Best, Original, and Latest based on the provided probabilities.
    """
    # --- YOUR CODE HERE ---
    # Delete the raise statement below and replace it with your implementation.
    raise NotImplementedError("select_parent_strategy() is not implemented yet.")


def select_curriculum_dataset(train_dataset: Dataset, iteration: int, train_size: int) -> Dataset:
    """
    Selects the train_subset dynamically based on curriculum learning.
    - Iteration 1: Easiest, shortest reading passages.
    - Iteration 2: Harder "table_op" or multi-step questions.
    - Other iterations (like 0, 3+): Easiest or standard subset.
    """
    rows = list(train_dataset)
    row_details = []
    for r in rows:
        passage = r.get("context") or r.get("article") or r.get("passage") or ""
        gold_program = r.get("answer") or ""
        
        q_type = classify_question_type(gold_program)
        is_hard = q_type in ["table_op", "division"] or ("," in gold_program)
        
        row_details.append({
            "row": r,
            "passage_len": len(passage),
            "is_hard": is_hard
        })

    if iteration == 2:
        # Prioritize rows with harder questions, then sort by passage length descending
        row_details.sort(key=lambda x: (0 if x["is_hard"] else 1, -x["passage_len"]))
    else:
        # Prioritize rows with easier questions, then sort by passage length ascending
        row_details.sort(key=lambda x: (1 if x["is_hard"] else 0, x["passage_len"]))

    selected_rows = [x["row"] for x in row_details[:min(train_size, len(row_details))]]
    return Dataset.from_list(selected_rows)


def run_smoke_test(
    strategy: Strategy,
    train_dataset: Dataset,
    model: QwenInference,
) -> bool:
    """
    Run the strategy on up to 5 examples to verify structural validity.
    Returns True if it passes, False if it fails.

    TODO: Implement the pre-flight smoke test.
    Steps:
      1. Select up to 5 examples from train_dataset.
      2. Temporarily set model.max_new_tokens based on strategy.cot_format (4096 if CoT, 256 if direct).
      3. Evaluate the strategy on this subset using evaluate().
      4. Check that at least one predicted answer is not None (i.e. program extraction succeeded).
      5. Check that average output tokens generated per question does not exceed 90% of the token limit 
         (to prevent infinite looping/truncation).
      6. Return True if valid, False otherwise. Make sure to restore model.max_new_tokens at the end.
    """
    # --- YOUR CODE HERE ---
    raise NotImplementedError("run_smoke_test() is not implemented yet.")


def run_evoagent(
    T: int,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    model: QwenInference,
    output_dir: Path,
    train_size: int = 100,
    resume_from: Optional[Path] = None,
    early_stop_accuracy: float = 1.0,
    afo_mode: str = "probabilistic",
    afo_prob_best: float = 0.4,
    afo_prob_original: float = 0.3,
    afo_prob_latest: float = 0.3,
    progressive_reflections: bool = True,
    use_curriculum: bool = False,
) -> StrategyHistory:
    """
    Run the EvoAgent loop for up to T iterations.

    TODO: Implement the EvoAgent optimization loop.
    For each iteration (from start_iteration up to T-1):
      1. Propose strategy:
         - Iteration 0: Use make_seed_strategy()
         - Iterations > 0: Mutate from a parent selected via select_parent_strategy().
           Try proposing up to 3 times, validating each using run_smoke_test().
      2. Set model.max_new_tokens dynamically (4096 if CoT, 256 if direct).
      3. Evaluate on train subset (curriculum or slice) and dev split.
      4. Accumulate token usage in TokenBudget.
      5. Reflect on errors (except on the last iteration T-1).
      6. Append/save strategies, evaluations, and reflections to the history file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.jsonl"
    history = StrategyHistory(history_path)

    if resume_from is not None:
        history.path = Path(resume_from)
        history.load()
        logger.info("Resumed from %s — %d strategies in history.", resume_from, len(history))
    elif history_path.exists():
        history.load()
        logger.info("Auto-resumed from %s — %d strategies in history.", history_path, len(history))

    budget = TokenBudget()
    start_iteration = len(history.strategies)

    if start_iteration >= T:
        logger.info("History already has %d strategies (T=%d). Nothing to do.", start_iteration, T)
        return history

    logger.info("Starting EvoAgent loop. Iterations %d–%d (T=%d).", start_iteration, T - 1, T)

    # --- YOUR CODE HERE ---
    # Delete the raise statement below and replace it with your implementation.
    raise NotImplementedError("run_evoagent() is not implemented yet.")


# ------------------------------------------------------------------
# Internal Helpers for saving results
# ------------------------------------------------------------------

def _save_strategy_json(strategy: Strategy, output_dir: Path, iteration: int) -> None:
    path = output_dir / f"iter_{iteration:03d}_strategy.json"
    path.write_text(strategy.to_json(), encoding="utf-8")


def _save_eval_result(
    result: EvalResult,
    output_dir: Path,
    iteration: int,
    tag: str,
) -> None:
    import json
    path = output_dir / f"iter_{iteration:03d}_eval_{tag}.json"
    data = result.to_dict()
    data["per_question"] = [
        {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "gold_val": r.gold_val,
            "predicted_answer": r.predicted_answer,
            "predicted_val": r.predicted_val,
            "is_correct": r.is_correct,
            "question_type": r.question_type,
            "raw_output": r.raw_output,
        }
        for r in result.per_question
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_reflection_json(reflection, output_dir: Path, iteration: int) -> None:
    import json
    path = output_dir / f"iter_{iteration:03d}_reflection.json"
    path.write_text(
        json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_leaderboard(history: StrategyHistory) -> None:
    rows = history.summary_table()
    if not rows:
        return
    header = f"{'Iter':>4}  {'ID':>8}  {'CoT':>10}  {'Dev Acc':>8}  {'Train Acc':>9}  {'Meta tok':>10}  {'Qwen tok':>8}"
    logger.info("Leaderboard:\n%s", header)
    for r in rows:
        dev = f"{r['dev_accuracy']:.3f}" if r["dev_accuracy"] is not None else "  —  "
        train = f"{r['train_accuracy']:.3f}" if r["train_accuracy"] is not None else "  —  "
        logger.info(
            "  %4d  %8s  %10s  %8s  %9s  %10d  %8d",
            r["iteration"],
            r["id"],
            r["cot_format"],
            dev,
            train,
            r["meta_tokens"],
            r["qwen_tokens"],
        )
