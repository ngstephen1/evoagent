# Phase 3 Report - Melanie's EvoAgent Improvements (Advanced NLP06 Assignment 03)

## Overview

This report documents the Phase 3 improvements made by Melanie to the EvoAgent
Kaggle pipeline, written for the team (Stephen) with enough technical detail to
reproduce and extend the work. Starting from the team's previous best public
score of 0.65789 (the Run009-lite hybrid built on a 4-billion-parameter model),
the work in this session raised the public leaderboard score to 0.69433, an
improvement of about 3.6 points. The gains came from three changes applied one at
a time: a larger base model, self-consistency inference, and a majority-vote
ensemble across diverse models.

All experiments ran on a compute node with four NVIDIA H100 80GB GPUs accessed
over SSH, using SGLang for inference. Every model used is an open-weight model of
at most nine billion parameters, in line with the Phase 3 competition rules.

Team members:

| Member | Student ID | Kaggle ID |
|---|---|---|
| Melanie | TBD | `yeyeezyzeus` |
| Nguyen Phan Nguyen - Stephen | TBD | `nguynphannguyn` |

## Methodology

The improvement process followed a strict single-variable discipline: exactly one
factor was changed per experiment, the effect was measured on the local dev split
and then on the Kaggle public leaderboard, and only validated changes were carried
forward. This made each gain attributable to a specific cause and avoided
confounding several changes at once. The prior 4B hybrid (0.65789) was retained
throughout as the control to beat.

## Model Upgrade (4B to 8B)

The first change replaced the quantized 4B model (`QuantTrio/Qwen3.5-4B-AWQ`) with
the full-precision 8B model `Qwen/Qwen3-8B`, holding the prompting strategy and
all loop settings fixed. The original model was AWQ-quantized only because the
Modal A10G had 24 GB of memory; with 80 GB H100s that constraint is gone, so the
8B model was loaded in bfloat16 (no quantization), which avoids the small quality
loss quantization introduces.

To use the four GPUs, data parallelism was chosen over tensor parallelism. An 8B
model fits comfortably on a single 80 GB card, so tensor parallelism (sharding one
model across GPUs) would only add cross-GPU communication overhead. Data
parallelism instead loads four independent replicas and splits the request batch
across them, giving roughly four times the throughput with identical outputs. This
is exposed through a new `--dp-size` flag that is passed to `sgl.Engine(...,
dp_size=N)` in `src/model.py`. The model change alone raised dev accuracy from
0.4833 to 0.650, the largest single quality gain to the base solver.

## Self-Consistency Inference

The second change introduced self-consistency at inference time. At temperature 0
the model produces a single greedy program per question; if that one attempt
misreads a table row or picks the wrong operation, the answer is wrong. Self-
consistency instead draws k samples per question at a non-zero temperature,
executes every candidate program with the local DSL evaluator, and keeps the
answer whose executed numeric value is the most common.

The mechanism is implemented in `src/executor.py`. When `self_consistency_k > 1`,
each prompt is replicated k times before the batched `generate_batch` call, a new
`temperature_override` argument forces non-zero sampling for that call only, and
the k outputs per question are grouped and passed to a new helper,
`_self_consistency_vote`. That helper executes each candidate with
`evaluate_program`, buckets the results by executed value, and returns the
candidate whose value has the most votes (falling back to the first sample if none
execute). Voting on the executed value, rather than on the raw program text, means
programs that differ syntactically but compute the same number are correctly
counted together. The same voting path is reused at submission time in
`submit.py`, so training-time and test-time behaviour match.

With k=5 the dev accuracy reached 0.704, and the count of zero-valued (failed)
predictions fell from 15 to 11 as k rose to 16. On the Kaggle public set the 8B
self-consistency submissions scored 0.64979 (k=5) and 0.65587 (k=16). The small
Kaggle gain from k=5 to k=16 shows this lever saturates quickly.

## Diverse-Model Ensemble

A single strong model did not by itself beat the engineered 4B hybrid on Kaggle.
The 8B self-consistency submission scored 0.65587 against the hybrid's 0.65789,
even though its dev accuracy (0.704) was far higher. This gap reflects two facts:
the dev and test distributions differ, and the hybrid benefits from explicit
failure patching that a single model does not perform.

The decisive gain came from ensembling. A second, differently trained model,
`Qwen/Qwen2.5-Coder-7B-Instruct`, was used with the same strategy and self-
consistency (k=8) to generate an additional submission. On its own this model
scored only 0.48178, but its errors are different from the Qwen3 models', so it
contributes useful diversity to a vote.

The final method is a per-row majority vote across three submissions: the 8B
self-consistency submission (k=16), the previous 4B hybrid, and the Coder-7B
submission. It is implemented in a new standalone script,
`phase3_ensemble_vote.py`, which runs locally on CPU with no model or GPU. For
each test id it collects the three predicted values, groups values that are
numerically equal within a relative/absolute tolerance, and keeps the value with
the most votes. Ties are broken in favour of a designated priority submission (the
previous hybrid), so the ensemble cannot lose to it on an evenly split row. A
comparison of the two strongest submissions showed they agree on about 65 percent
of rows and disagree on roughly 175 rows; the vote decides those disagreements
toward the more frequently supported answer. The resulting ensemble changed 70
rows relative to the previous best and scored 0.69433 on the public leaderboard.

## Code Changes

All changes are Phase 3 additions and do not modify EvoAgent core logic, the DSL
evaluator, or the graders. They are backward compatible: every new flag defaults
to the previous single-GPU, single-sample behaviour.

| File | Change |
|---|---|
| `src/model.py` | Added `tp_size`, `dp_size`, `self_consistency_k`, `self_consistency_temp` to `QwenInference`; pass `tp_size`/`dp_size` to `sgl.Engine`; add a `temperature_override` argument to `generate_batch` for sampling. |
| `src/executor.py` | Added `_self_consistency_vote` (execute-and-vote helper); `evaluate` now expands prompts k times and votes when `self_consistency_k > 1`. |
| `src/main.py` | Added `--tp-size`, `--dp-size`, `--self-consistency-k`, `--self-consistency-temp` CLI flags, forwarded to `QwenInference`. |
| `src/../arc_proofs.py` | Added the same four flags and forwards them to `main.py`. |
| `submit.py` | Added the same flags; applies self-consistency voting when generating test predictions. |
| `phase3_ensemble_vote.py` | New script: per-row majority-vote ensemble of N submission CSVs with a priority tiebreaker (CPU-only). |

## Kaggle Experiments

| Run | Method | Public Score | Decision |
|---|---|---:|---|
| 8B SC k=5 | Qwen3-8B + self-consistency (k=5) | 0.64979 | Not final |
| 8B SC k=16 | Qwen3-8B + self-consistency (k=16) | 0.65587 | Not final |
| Coder-7B | Qwen2.5-Coder-7B, standalone | 0.48178 | Diversity source only |
| Ensemble | 3-way majority vote (8B k16 + hybrid + Coder-7B) | 0.69433 | Primary final |

For reference, the previous team best was Run009-lite at 0.65789.

## Results Analysis

The largest quality gain to the base solver was the model upgrade, which added
16.7 points of dev accuracy. Self-consistency added a further gain on dev and
reduced failure rows. The largest Kaggle gain, however, came from the ensemble,
which exceeded every individual submission by about 3.5 points.

The most important lesson is that a model which is weak in isolation can still
strengthen an ensemble. The Coder-7B model scored only 0.48178 on its own, yet its
inclusion lifted the ensemble to 0.69433, because its errors differ from those of
the Qwen3 models and the majority vote exploits that diversity. Two corollaries
follow for future work: adding more independent models to an odd-sized vote is
likely the highest-value next step, and diversity of a candidate model matters
more than its standalone accuracy.

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

The final ensemble was produced locally by majority vote:

```bash
python3 phase3_ensemble_vote.py \
  --inputs submission_8b_sc16.csv final_submission.csv submission_coder7b.csv \
  --priority final_submission.csv \
  --output submission_ensemble.csv
```

## Next Steps

The remaining gap to a much higher score is large; the base solver caps in the
high-0.60s on this test set. The recommended next experiments, in order of
expected value:

1. Expand the ensemble to a 5-way odd vote by adding two more non-gated diverse
   models (for example `Qwen2.5-7B-Instruct` and `Mistral-7B-Instruct-v0.3`); an
   odd voter count removes tie-break ambiguity.
2. Confidence-weighted voting: weight each model on a row by its self-consistency
   agreement count instead of an equal vote.
3. Larger structural changes if time allows: LoRA fine-tuning on the train
   programs, or context/table-retrieval compression to fix wrong-row extraction.

Because grading is rank-based and the public score is only a proxy for the private
leaderboard, the ensemble (0.69433) should be kept as the primary final candidate
alongside the stable 4B hybrid.

## Integrity Declaration Summary

No hidden test labels, manual test labeling, or leaked answers were used. All
Kaggle outputs came from documented model inference runs and deterministic
ensemble rules that can be reproduced from the commands above. Hugging Face
tokens, Kaggle credentials, private keys, and model weights are excluded from the
repository.
