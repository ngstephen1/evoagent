# Report Draft - EvoAgent Advanced NLP06 Assignment 03

## Overview

This project implements EvoAgent for Vietnamese financial programmatic question
answering. The system evolves reasoning strategies, evaluates them on local
financial QA splits, reflects on failures, proposes improved strategies, and
uses the best strategies for Kaggle prediction generation.

Milestone 1 is complete: Stages 0-4 pass local graders, and proof files were
generated on VT ARC GPU infrastructure. Phase 3 Kaggle experimentation is
currently frozen with Run009-lite safe selected as the primary final public
candidate.

Team members:

| Member | Student ID | Kaggle ID |
|---|---|---|
| Melanie | TBD | `yeyeezyzeus` |
| Nguyen Phan Nguyen - Stephen | TBD | `nguynphannguyn` |

## EvoAgent Implementation

The implementation covers the staged assignment components:

- Stage 0: sandbox prediction and proof generation.
- Stage 1: executor evaluation loop and token accounting.
- Stage 2: self-reflection over failure patterns.
- Stage 3: self-proposal with DSL validation and dynamic examples.
- Stage 4: evolution harness, parent selection, smoke testing, and history
  management.

The local grader suite passed:

```bash
python3 graders/grade_stage0.py
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

## Strategy Evolution

The ARC evolution run produced best iteration 1:

| Metric | Value |
|---|---:|
| Baseline dev accuracy | 0.42 |
| Best dev accuracy | 0.48333333333333334 |
| Best iteration | 1 |

The selected strategy for Run001 was:

```text
assignment03/runs/exp_self_arc/iter_best_strategy.json
```

Proof and analysis artifacts in this package:

- `evidence/evolution_proof.json`
- `evidence/failure_mode_report.pdf`
- `evidence/learning_curve.pdf`

## Kaggle Experiments

| Run | Method | Public Score | Decision |
|---|---|---:|---|
| Run001 | Best EvoAgent strategy baseline | 0.56477 | Baseline |
| Run002 | Iter003 table-op strategy | 0.47975 | Not final |
| Run003 | Hybrid Run001 fallback to Run002 nonzero | 0.64574 | Previous best |
| Run004 | Hybrid Run003 fallback to iter004 nonzero | 0.64574 | Alternate |
| Run005 | Conservative numeric post-processing | 0.64170 | Not final |
| Run006 | Context-expanded rerun | Not submitted | Not final |
| Run007 | Tiny Run003/Run006 hybrid | Not submitted | Not final |
| Run008 filtered | Targeted retry over Run003 zero rows, agreement >= 2 | 0.65587 | Previous best |
| Run009-lite safe | Suspicious-row targeted retry over Run008, filtered to safe meaningful changes | 0.65789 | Primary final |

## Final Hybrid Method

The primary final file is:

```text
kaggle/final_submission.csv
```

It is copied from Run009-lite safe:

```text
assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv
```

Run003 starts from Run001 and replaces only rows where Run001 predicted `0.0`
and Run002 predicted a nonzero value. This reduced zero-valued predictions
from 112 to 11 and improved public score from 0.56477 to 0.64574.

Run008 filtered starts from Run003, retries only the remaining zero-valued rows
with multi-sample DSL generation, executes valid candidate programs, and keeps
only retry recoveries with `agreement_count >= 2`. This changed 6 rows, reduced
the final zero-valued prediction count to 5, and improved public score from
0.64574 to 0.65587.

Run009-lite safe starts from Run008 filtered, retries a narrow set of 60
suspicious rows, and keeps only three meaningful high-confidence changes. It
explicitly excludes an accepted retry that would have introduced a new extreme
outlier, reducing the chance of public-leaderboard overfitting. This improved
public score from 0.65587 to 0.65789.

Run004 tied Run003 publicly but changed only four extra rows, so it remains an
alternate/private-leaderboard hedge rather than the primary final choice.

## Failure Analysis

The strongest single-strategy run still had frequent fallback-like `0.0`
predictions due to malformed or invalid generated programs. Iter003 reduced
zero predictions but was worse globally, indicating that confident nonzero
answers were often incorrect.

Hybrid fallback ensembling was the first effective method because it preserved
Run001's stronger global reasoning while selectively recovering useful answers
from a complementary strategy. Broad numeric post-processing changed 14 rows
but reduced public score, so it was rejected. Targeted retry then recovered
additional failures, but only after filtering out low-confidence candidates and
new extreme outliers.

## Reproducibility

Core environment setup:

```bash
cd assignment03
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Run001 generation command on ARC:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./runs/kaggle_arc_best/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

Run002 generation command on ARC:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_003_strategy.json \
  --output-file ./runs/kaggle_iter003/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

The final Run009-lite safe hybrid was created by deterministic CSV replacement
from accepted retry recoveries over Run008 filtered. Full experiment notes are in
`docs/PHASE3_EXPERIMENT_LOG.md` in the repository.

## Integrity Declaration Summary

We did not use hidden test labels, manual test labeling, leaked answers, or
test-specific external lookup. All Kaggle outputs came from documented model
inference runs and deterministic hybrid/post-processing experiments. Tokens,
private keys, and model weights are excluded from this package.
