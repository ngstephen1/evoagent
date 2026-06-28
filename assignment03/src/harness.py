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
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from tqdm import tqdm

from src.executor import EvalResult, evaluate, classify_question_type, TokenBudget
from src.model import QwenInference
from src.self_proposer import propose_self
from src.self_reflector import reflect_self
from src.strategy import CoTFormat, Strategy, StrategyHistory, StrategyMetadata, make_seed_strategy

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
    if not history.strategies:
        return None

    latest = history.latest_strategy()
    original = history.strategies[0]
    best = history.best_strategy() or latest or original

    mode = (afo_mode or "probabilistic").lower()
    if mode == "none":
        return latest
    if mode == "best":
        return best
    if mode == "original":
        return original
    if mode != "probabilistic":
        logger.warning("Unknown AFO mode %r; falling back to latest strategy.", afo_mode)
        return latest

    weights = [
        ("best", max(0.0, float(afo_prob_best)), best),
        ("original", max(0.0, float(afo_prob_original)), original),
        ("latest", max(0.0, float(afo_prob_latest)), latest),
    ]
    total = sum(weight for _, weight, _ in weights)
    if total <= 0:
        return latest

    draw = random.random() * total
    cumulative = 0.0
    for _, weight, strategy in weights:
        cumulative += weight
        if draw <= cumulative:
            return strategy
    return latest


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
    original_max_new_tokens = getattr(model, "max_new_tokens", None)
    token_limit = 4096 if strategy.cot_format != CoTFormat.NONE else 256

    try:
        if hasattr(model, "max_new_tokens"):
            model.max_new_tokens = token_limit

        smoke_subset = _select_first_n(train_dataset, 5)
        if len(smoke_subset) == 0:
            logger.warning("Smoke test failed: empty training dataset.")
            return False

        result = evaluate(strategy, "smoke_test", smoke_subset, model)
        if not result.per_question:
            logger.warning("Smoke test failed: evaluation returned no per-question results.")
            return False

        extracted_count = sum(1 for row in result.per_question if row.predicted_answer)
        if extracted_count == 0:
            logger.warning("Smoke test failed: no program predictions were extracted.")
            return False

        avg_output_tokens = result.total_output_tokens / max(1, result.num_examples)
        if avg_output_tokens >= 0.9 * token_limit:
            logger.warning(
                "Smoke test failed: average output tokens %.1f exceeded 90%% of limit %d.",
                avg_output_tokens,
                token_limit,
            )
            return False

        return True
    except Exception as exc:
        logger.warning("Smoke test failed for strategy %s: %s", strategy.id, exc)
        return False
    finally:
        if original_max_new_tokens is not None and hasattr(model, "max_new_tokens"):
            model.max_new_tokens = original_max_new_tokens


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

    for iteration in tqdm(range(start_iteration, T), desc="EvoAgent"):
        train_subset = (
            select_curriculum_dataset(train_dataset, iteration, train_size)
            if use_curriculum
            else _select_first_n(train_dataset, train_size)
        )

        if iteration == 0 and not history.strategies:
            strategy = make_seed_strategy()
            strategy.metadata.iteration = iteration
        else:
            parent = select_parent_strategy(
                history,
                afo_mode=afo_mode,
                afo_prob_best=afo_prob_best,
                afo_prob_original=afo_prob_original,
                afo_prob_latest=afo_prob_latest,
            )
            strategy, proposal_tokens = _propose_with_smoke_test(
                history=history,
                model=model,
                train_dataset=train_subset,
                parent=parent,
            )
            budget.add_meta(proposal_tokens)
            strategy.metadata.iteration = iteration
            strategy.metadata.parent_id = parent.id if parent is not None else strategy.metadata.parent_id

        token_limit = 4096 if strategy.cot_format != CoTFormat.NONE else 256
        original_max_new_tokens = getattr(model, "max_new_tokens", None)
        if hasattr(model, "max_new_tokens"):
            model.max_new_tokens = token_limit

        try:
            train_result = evaluate(strategy, "train", train_subset, model)
            dev_result = evaluate(strategy, "dev", dev_dataset, model)
        finally:
            if original_max_new_tokens is not None and hasattr(model, "max_new_tokens"):
                model.max_new_tokens = original_max_new_tokens

        budget.add_eval(train_result)
        budget.add_eval(dev_result)

        strategy.metadata.train_accuracy = train_result.accuracy
        strategy.metadata.dev_accuracy = dev_result.accuracy
        strategy.metadata.token_cost_qwen = train_result.total_input_tokens + train_result.total_output_tokens
        strategy.metadata.token_cost_qwen += dev_result.total_input_tokens + dev_result.total_output_tokens
        strategy.metadata.token_cost_claude = budget.meta_total

        history.append_strategy(strategy)
        _save_strategy_json(strategy, output_dir, iteration)
        _save_eval_result(train_result, output_dir, iteration, "train")
        _save_eval_result(dev_result, output_dir, iteration, "dev")

        if dev_result.accuracy >= early_stop_accuracy:
            logger.info(
                "Early stop after iteration %d: dev accuracy %.3f >= %.3f.",
                iteration,
                dev_result.accuracy,
                early_stop_accuracy,
            )
            break

        if iteration < T - 1:
            try:
                reflection, reflection_tokens = reflect_self(
                    strategy,
                    dev_result,
                    model,
                    progressive=progressive_reflections,
                )
                budget.add_meta(reflection_tokens)
                strategy.metadata.token_cost_claude = budget.meta_total
                history.update_strategy_metadata(strategy.id, strategy.metadata)
                history.append_reflection(reflection)
                _save_reflection_json(reflection, output_dir, iteration)
            except Exception as exc:
                logger.warning("Reflection failed at iteration %d: %s", iteration, exc)

        _print_leaderboard(history)
        logger.info(budget.summary())

    return history


def _select_first_n(dataset: Dataset, n: int) -> Dataset:
    size = min(max(0, n), len(dataset))
    return dataset.select(range(size))


def _propose_with_smoke_test(
    history: StrategyHistory,
    model: QwenInference,
    train_dataset: Dataset,
    parent: Optional[Strategy],
) -> tuple[Strategy, int]:
    total_tokens = 0
    parent_id = parent.id if parent is not None else None

    for attempt in range(3):
        try:
            candidate, tokens = propose_self(
                history,
                model,
                parent_strategy_id=parent_id,
                train_dataset=train_dataset,
            )
            total_tokens += tokens
            if run_smoke_test(candidate, train_dataset, model):
                return candidate, total_tokens
            logger.warning("Rejected proposed strategy %s after smoke test.", candidate.id)
        except Exception as exc:
            logger.warning("Strategy proposal attempt %d/3 failed: %s", attempt + 1, exc)

    if parent is not None:
        logger.warning("All proposed strategies failed smoke tests; falling back to a parent clone.")
        return _clone_strategy_for_iteration(parent), total_tokens
    seed = make_seed_strategy()
    logger.warning("Proposal failed without a parent; using seed strategy.")
    return seed, total_tokens


def _clone_strategy_for_iteration(strategy: Strategy) -> Strategy:
    return Strategy(
        id=str(uuid.uuid4()),
        prompt_template=strategy.prompt_template,
        cot_format=strategy.cot_format,
        few_shot_examples=list(strategy.few_shot_examples),
        retrieval_config=strategy.retrieval_config,
        metadata=StrategyMetadata(parent_id=strategy.id),
    )


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
