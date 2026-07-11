# Phase 3 — Reproducibility & Next Steps (companion to `melanie_report.md`)

This is the operational companion to the main report (`melanie_report.md`). It holds
the exact commands to reproduce every result and the ranked plan for pushing the score
higher. The main report covers the method, results, and analysis; this file is the
"how to re-run it" and "what to try next" appendix.

## Reproducibility

The strategy was evolved with the 8B model and self-consistency on the GPU node:

```bash
python3 arc_proofs.py evolution --T 5 --train-size 200 --dev-size 240 \
  --output-dir runs/exp_qwen3_8b_sc5 --model Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.9 --dp-size 4 --self-consistency-k 5
```

The individual submissions were generated from that strategy:

```bash
python3 submit.py --strategy-path runs/exp_qwen3_8b_sc5/iter_best_strategy.json \
  --output-file runs/kaggle_8b_sc16/submission.csv --model Qwen/Qwen3-8B \
  --dp-size 4 --self-consistency-k 16

python3 submit.py --strategy-path runs/exp_qwen3_8b_sc5/iter_best_strategy.json \
  --output-file runs/kaggle_coder7b/submission.csv \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --dp-size 4 --self-consistency-k 8
```

The base ensemble was produced locally by majority vote:

```bash
python3 phase3_ensemble_vote.py \
  --inputs submission_8b_sc16.csv final_submission.csv submission_coder7b.csv \
  --priority final_submission.csv \
  --output submission_ensemble.csv
```

The final-day post-processing and validation steps (all CPU-only, deterministic):

```bash
# Local dev validation harness — sanity self-test then score any prediction file
python3 phase3_dev_eval.py --selftest                       # gold programs -> 100%
python3 phase3_dev_eval.py --pred <preds>.json --mode program

# Member-confirmed percent->ratio scale fix over the ensemble -> v2 (0.69838)
python3 phase3_scale_fix.py --apply                          # writes submission_ensemble_v2.csv

# Member-confirmed sign flips over v2 -> v4 (0.70242); v3 (blind, regressed) omitted
# (2 rows where >=2 of 3 members agree on the opposite sign of a difference question)
```

Candidate ensemble members for future expansion are generated on Modal with:

```bash
modal run --detach run_modal.py::predict \
  --model <hf-id> --sc-k <k> --tag <name> \
  --strategy-path strategies/<strategy>.json     # runs dev + test, downloads both
```

The LoRA fine-tune that produced the final 0.72267 model:

```bash
# 1. Build the SFT dataset (CPU, local)
python3 phase3_build_sft_data.py                 # -> data_sft/{train,eval}_sft.jsonl

# 2. QLoRA train + merge on a Modal A100-80GB (~1h50m)
modal run --detach run_modal.py::finetune \
  --base-model Qwen/Qwen3-8B --epochs 2 --tag qwen3_8b

# 3. Inference with the merged fine-tuned model (dev + test)
modal run --detach run_modal.py::predict \
  --model /runs/lora_merged_qwen3_8b --sc-k 1 --tag ft-qwen3-8b \
  --strategy-path strategies/ft_strategy.json

# 4. Score dev locally, then submit predict_out/ft-qwen3-8b_test.csv
python3 phase3_dev_eval.py --pred predict_out/ft-qwen3-8b_dev_details.json --mode program
```

Rejection sampling / STaR that produced the best 0.74696 model:

```bash
# 1. Sample the FT model on train, keep verified programs, combine with gold
modal run --detach run_modal.py::rejection_sample \
  --model /runs/lora_merged_qwen3_8b --k 8 --temp 0.8   # -> /runs/rs/train_sft_combined.jsonl

# 2. Re-fine-tune Qwen3-8B on the augmented dataset (5,611 examples)
modal run --detach run_modal.py::finetune \
  --base-model Qwen/Qwen3-8B --epochs 2 --tag qwen3_8b_rs \
  --train-file /runs/rs/train_sft_combined.jsonl

# 3. Inference + score; submit predict_out/ft-qwen3-8b-rs_test.csv
modal run --detach run_modal.py::predict \
  --model /runs/lora_merged_qwen3_8b_rs --sc-k 1 --tag ft-qwen3-8b-rs \
  --strategy-path strategies/ft_strategy.json
python3 phase3_dev_eval.py --pred predict_out/ft-qwen3-8b-rs_dev_details.json --mode program
```

The frontier-API three-way ensemble that produced the best 0.80161 (no GPU; keys in `.env`):

```bash
# 1. Solve dev + test with each frontier model (documents the exact model ids)
for split in dev test; do
  python3 phase3_api_solve.py --provider gemini   --model gemini-2.5-flash --split $split --max-tokens 4096
  python3 phase3_api_solve.py --provider deepseek --model deepseek-chat    --split $split
done

# 2. Three-way majority vote (STaR + Gemini + DeepSeek), ties -> STaR (priority)
python3 phase3_ensemble_vote.py \
  --inputs predict_out/ft-qwen3-8b-rs_test.csv \
           predict_out/api-gemini-gemini-2.5-flash_test.csv \
           predict_out/api-deepseek-deepseek-chat_test.csv \
  --priority predict_out/ft-qwen3-8b-rs_test.csv \
  --test data/test.json --output kaggle/submission_ensemble_api3.csv
```

## Next Steps

The final result is the **three-way frontier-API ensemble at 0.80161**, with the
**self-hosted STaR Qwen3-8B (0.74696)** kept as the primary non-API alternate.
Remaining levers, in order of expected value, if a higher score is wanted:

0. **Stronger / more API voters.** The dev oracle bound for the current three models
   is 89%, so headroom remains. Swapping Gemini 2.5-flash for a stronger model
   (gemini-3-pro / gemini-3.5-flash) or adding a fourth independent voter (Claude,
   GPT) in an odd (5-way) vote should lift the ensemble further, all CPU-only.

1. **A second round of rejection sampling / STaR.** Sample from the *0.74696* model
   (not the first FT model) to capture newly-solvable hard questions, and re-train on
   the further-enlarged verified set. STaR often gains across 2-3 rounds before
   plateauing.
2. **Target multi-step further.** The weakest slice is still 3+ operation programs
   (60.9% dev, up from 47.8%). Over-sampling multi-step verified programs, more epochs,
   or a larger LoRA rank should push it further.
3. **GRPO / RLVR.** The DSL evaluator is a binary verifiable reward, so GRPO can
   directly optimize execution accuracy on top of the SFT/STaR checkpoint. Highest
   ceiling, highest cost (multi-day).
4. **Final model on train+dev combined.** Once dev is no longer needed for validation,
   re-train on all 3,570 labeled examples for a small final gain.
5. **Within-team ensembling.** Voting the 0.74696 model against the teammate's
   independently-built pipeline is allowed (same team) and is a further source of
   diversity.

Because grading is rank-based and the public score is only a proxy for the private
leaderboard, **the frontier-API three-way ensemble (0.80161) is the primary final
candidate**, with the self-hosted STaR Qwen3-8B (0.74696) kept as the primary
non-API alternate and private-leaderboard hedge.

## Planning notes (scratch)

**Improvement plan:**
ensemble (now) → rejection sampling/STaR → target multi-step → (if time) GRPO → final train+dev

**To try tomorrow:**
Rejection sampling (STaR) with the second fine-tune runs.
