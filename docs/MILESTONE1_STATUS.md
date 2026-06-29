# Milestone 1 Status

## Summary

Milestone 1 for VietAI Advanced NLP06 Assignment 03 is locally verified.
Stages 0-4 are implemented, ARC-generated proof files are present, and all
local graders required for the EvoAgent implementation milestone pass.

## Current Branch

```bash
git branch --show-current
```

Expected branch:

```text
integration/evoagent-arc
```

Checkpoint commits:

```text
4459037 Add ARC-generated proof artifacts
675bb15 Sanitize ARC user placeholders
```

## Implemented Stages

| Stage | Area | Status |
|---|---|---|
| Stage 0 | Sandbox prediction | Implemented and proof verified |
| Stage 1 | Executor and token accounting | Implemented and grader verified |
| Stage 2 | Self reflection | Implemented and grader verified |
| Stage 3 | Self proposer | Implemented and grader verified |
| Stage 4 | Evolution harness | Implemented and proof verified |

## Proof Files

Proof generation was run on VT ARC GPU infrastructure instead of using Modal
as the primary execution path for this fork.

Generated proof files:

```text
assignment03/sandbox_proof.json
assignment03/smoke_proof.json
assignment03/evolution_proof.json
assignment03/runs/exp_self_arc/evolution_proof.json
```

Root proof files are currently tracked in git:

```text
assignment03/sandbox_proof.json
assignment03/smoke_proof.json
assignment03/evolution_proof.json
```

The full ARC run directory is intentionally treated as generated output and is
ignored by git:

```text
assignment03/runs/exp_self_arc/
```

Useful ARC run artifacts in that directory should be included in the final
ThinkFlic ZIP as evidence, but the large per-example run outputs should not be
committed unless explicitly required.

## ARC Environment

ARC target:

```text
Host: tinkercliffs1.arc.vt.edu
Remote path: /home/<VT_PID>/temp/evoagent
Allocation: llms-lab
Partition: a100_normal_q
QoS: tc_a100_normal_short
GPU request: 1 GPU
```

Proof adapter:

```text
assignment03/arc_proofs.py
scripts/arc_generate_proofs.sh
```

The proof run used the ARC local GPU path and produced the same grader-facing
proof JSON shapes expected by the assignment graders.

## Results

| Metric | Value |
|---|---:|
| Sandbox baseline accuracy | 0.000 |
| Evolution baseline accuracy | 0.420 |
| Best dev accuracy | 0.483 |
| Best iteration | 1 |

Exact values from `assignment03/evolution_proof.json`:

```text
baseline_accuracy = 0.42
best_dev_accuracy = 0.48333333333333334
best_iteration = 1
```

## Verification Commands

Run from the assignment workspace:

```bash
cd /Users/macbook/Hack/evoagent/assignment03

python3 graders/grade_stage0.py
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

Latest local verification status:

| Command | Status |
|---|---|
| `python3 graders/grade_stage0.py` | PASS |
| `python3 graders/grade_stage1_executor.py` | PASS |
| `python3 graders/grade_stage2_reflector.py` | PASS |
| `python3 graders/grade_stage3_proposer.py` | PASS |
| `PYTHONPATH=. python3 graders/grade_smoke_proof.py` | PASS |
| `python3 graders/grade_stage4_harness.py` | PASS |

## ThinkFlic Evidence Notes

The final ThinkFlic ZIP should include the generated evidence files required by
`assignment03/docs/THINKFLIC_SUBMISSION.md`.

Current Milestone 1 evidence available locally:

```text
assignment03/evolution_proof.json
assignment03/runs/exp_self_arc/failure_mode_report.pdf
assignment03/runs/exp_self_arc/learning_curve.pdf
assignment03/runs/exp_self_arc/iter_best_strategy.json
assignment03/runs/exp_self_arc/history.jsonl
```

`strategy_diversity.pdf` was not present in `assignment03/runs/exp_self_arc/`
during local verification. Generate it later if the final ThinkFlic package
requires that exact file.

Do not include secrets, `.env` files, local SFTP configuration, model weights,
cache directories, or large temporary artifacts in the final ZIP.

## Phase 3 Follow-Up

Kaggle experiments are tracked separately from the Milestone 1 proof workflow.
The current best public submission is Run 003, a hybrid fallback pipeline with
public score `0.64574`. Run 005 tested conservative numeric post-processing and
scored `0.64170`, so it remains an ablation rather than the selected best
submission. See `docs/PHASE3_EXPERIMENT_LOG.md` for commands, files, validation
checks, and public leaderboard results.
