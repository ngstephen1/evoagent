# Step-to-Improve Plan

## 1. Current State

Current best Kaggle candidate:

- Best score: `0.65789`
- Best submission file: `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv`
- Current final choice: Run009-lite safe as the primary final candidate
- Alternate candidates: Run008 filtered at `0.65587`, plus Run003 and Run004 at `0.64574`

| Run | Method | Public Score | Status | Key Signal |
|---|---:|---:|---|---|
| Run001 | ARC best strategy baseline | 0.56477 | Submitted | Strongest single strategy, but 112 zero predictions. |
| Run002 | iter003 table-op strategy | 0.47975 | Submitted | Only 23 zeros, but many confident wrong answers. |
| Run003 | Run001 fallback to Run002 nonzero | 0.64574 | Previous best | 101 replacements; first major public-score gain. |
| Run004 | Run003 fallback to iter004 nonzero | 0.64574 | Alternate | 4 extra replacements; tied public score. |
| Run005 | Conservative numeric postprocess | 0.64170 | Rejected | 14 changes hurt score. |
| Run006 | iter_best ctx32768 rerun | Not submitted | Rejected source | 122 zeros; worse fallback behavior. |
| Run007 | Run003 fallback to Run006 nonzero | Not submitted | Rejected | Only 2 changes, medium-risk. |
| Run008 filtered | Targeted retry over Run003 zero rows, agreement >= 2 | 0.65587 | Previous best | 6 high-confidence replacements; first targeted-retry gain. |
| Run009-lite safe | Suspicious-row retry over Run008, filtered to safe meaningful changes | 0.65789 | Primary final | 3 auditable changes; improved without increasing extreme outliers. |
| Run011 GPT-OSS smoke | `openai/gpt-oss-120b` adapter feasibility on ARC SGLang | Not submitted | Feasibility only | 120B loaded on 1x A100 and produced parseable DSL, but first smoke changed only one row. |

What worked:

- Run003 worked because it preserved Run001's stronger global reasoning and patched only the most obvious failure mode: `0.0` fallback predictions.
- Run002 was weak as a standalone submission, but it was useful as a specialist fallback source for Run001 failures.
- The best public gain came from selective replacement, not from replacing all predictions from one strategy with another.
- Run008 filtered worked because it retried only remaining Run003 zero rows and accepted only higher-confidence recovered answers.
- Run009-lite safe worked because it expanded retry to a narrow suspicious-row pool but submitted only three meaningful, auditable changes and skipped one new extreme outlier.

What failed:

- Run002 reduced zero predictions from 112 to 23 but scored worse, showing that fewer fallback values can still mean more confident wrong answers.
- Run005 changed 14 rows with conservative numeric rules and lowered score from `0.64574` to `0.64170`, so broad deterministic post-processing is not safe.
- Run006 increased fallback behavior: the context-expanded rerun had 122 zeros and recovered only a tiny number of useful rows for hybrid use.
- Run004 and Run007 show that same-family fallback ensembling plateaued after Run003; extra changes were too few or too uncertain to improve public score.
- The full Run008 retry included one suspicious single-candidate replacement, so the submitted variant filtered it out with `agreement_count >= 2`.

Why Run003 improved:

- Run001 had many malformed or looping Chain-of-Thought outputs that formatted to `0.0`.
- Run002's non-CoT prompt was less accurate overall, but often emitted executable programs for those same failed rows.
- Run003 changed only rows where Run001 was exactly `0.0` and Run002 was nonzero, avoiding broad replacement of Run001's stronger predictions.

## 2. Main Bottlenecks

Run009-lite safe still has 4 zero predictions. These are the cleanest remaining targets because the current best submission still has no useful answer for them.

Run001-style Chain-of-Thought remains high variance: iter001 dev accuracy was `0.4833`, with very high output-token usage and several malformed or repetitive programs. The visible failure mode is not just wrong arithmetic; many failures are extraction or syntax failures that become invalid programs.

Weak dev areas remain important:

- iter001 overall dev accuracy: `0.4833`
- iter001 addition: `0.3571`
- iter001 table_op: `0.25`
- iter001 subtraction: `0.5047`
- iter003 table_op: `0.575`, but overall dev accuracy dropped to `0.4542`

This means table-operation specialists can help in narrow cases, but routing them broadly is risky. Addition and subtraction also remain weak enough that a naive median or majority ensemble can easily swap a wrong answer for another wrong answer.

Broad replacement is risky because the public leaderboard already penalized apparently reasonable numeric corrections in Run005. Same-run ensembling also plateaued: Run004 changed only 4 rows from Run003 and tied the score, while Run007 changed only 2 rows and was not worth submitting. Run008 shows that a narrow retry can help, but low-agreement candidates should be filtered.

## 3. Ranked Improvement Ideas

### A. Targeted retry for Run003 zero/failure rows

- Expected impact: highest near-term upside.
- Effort: medium.
- Risk: low to medium if restricted to zero rows only.
- Method: retry only rows where Run003 has `predicted_value == 0.0`, ask the model for valid DSL only, execute candidate programs, and keep a retry answer only if it is finite, nonzero, non-extreme, and internally consistent.
- Upside: can improve the current best without touching nonzero Run003 answers.
- Integrity boundary: no hidden labels and no manual test-answer inference.

### B. Self-consistency / multi-sample inference

- Expected impact: medium to high on hard rows.
- Effort: medium to high.
- Risk: medium.
- Method: generate multiple programs per hard row, execute each one, and accept only when several samples agree numerically or the median is supported by close candidates.
- Scope: apply only to Run003 zero rows first; avoid changing confident nonzero predictions.
- Runtime estimate: small for 11 zero rows, moderate if expanded to suspicious nonzero rows.

### C. Program repair

- Expected impact: medium.
- Effort: medium.
- Risk: low if applied only to invalid or zero-output rows.
- Method: identify malformed DSL programs, repair syntax or obvious argument/reference errors, re-execute, and accept only if the repaired program is valid and produces a finite nonzero value.
- Good targets: repeated text, quoted table columns, unsupported literal assignments, malformed references, and outputs where a valid DSL program is embedded inside reasoning text.
- Scope: do not rewrite valid nonzero predictions by default.

### D. Type-aware specialist ensemble

- Expected impact: medium.
- Effort: medium.
- Risk: medium to high.
- Method: route by operation type only when dev evidence is strong, such as using iter003 for table_op-like failures.
- Why naive median failed conceptually: strategies make correlated extraction mistakes, and a value can be numerically plausible while still using the wrong denominator, year, or table row.
- Safer version: use specialist predictions only for zero rows or when a dev-backed question type and a conservative validity check both agree.

### E. Retrieval/context compression

- Expected impact: medium.
- Effort: high.
- Risk: medium.
- Method: extract only relevant numbers, years, and table rows before prompting the model, reducing long-context noise and repeated reasoning loops.
- Expected benefit: fewer malformed CoT outputs and fewer wrong-row table reads.
- Difficulty: requires careful context construction and validation against dev examples before Kaggle use.

### F. Dynamic few-shot selection

- Expected impact: medium.
- Effort: medium.
- Risk: medium.
- Method: retrieve similar train/dev examples by question type, wording, and operation pattern, then build few-shot examples per test row.
- Expected benefit: better operation selection for addition, subtraction, and table_op cases.
- Feasibility: practical as a Phase 3-only experiment, but must avoid using test labels or manual answer inference.

### G. Stronger model/API path if rules allow

- Expected impact: potentially high.
- Effort: medium.
- Risk: policy and reproducibility dependent.
- Method: check assignment and Kaggle rules first, then consider a stronger model only if it is allowed, reproducible, and documented.
- Current status: Run011 confirmed `openai/gpt-oss-120b` can load on `1x A100 80GB` through SGLang and can be adapted by stripping GPT-OSS channel markup before parsing final JSON/DSL.
- Next step: improve the Run011 prompt and repeat a 10-row smoke before scaling to 50-100 suspicious rows.
- Boundary: do not use hidden-label leakage, manual test labeling, non-reproducible private outputs, or any method that violates course/Kaggle rules.

## 4. Recommended Next Experiment

Run008: targeted retry for Run003 remaining zero rows. This experiment has now
been completed and submitted as Run008 filtered.

This was the best risk/reward experiment because it attacked the clearest
remaining failure mode while preserving every successful Run003 nonzero
prediction. The filtered submission recovered 6 higher-confidence rows and beat
`0.64574`, reaching `0.65587`.

Inputs:

- `runs/kaggle_hybrid_001_002/submission_checked.csv`
- `runs/kaggle_hybrid_001_002/submission_details.json`
- `runs/exp_self_arc/iter_best_strategy.json`
- `data/test.json`

Target rows:

- Default: only Run003 rows where `predicted_value == 0.0`.
- Optional suspicious nonzero rows may be considered later, but must be disabled by default.

Method:

- Generate multiple candidate programs per target row with a stricter "output only valid DSL" prompt.
- Execute candidates with the existing DSL evaluator.
- Accept only finite, nonzero, non-extreme numeric answers.
- Prefer candidates with repeated or median-agreeing numeric values.
- Keep Run003 unchanged for every non-target or unresolved row.

Outputs:

- `runs/kaggle_retry_run008/retry_predictions.json`
- `runs/kaggle_retry_run008/retry_details.json`
- `runs/kaggle_hybrid_retry_run008/submission_checked.csv`
- `runs/kaggle_hybrid_retry_run008/changes.csv`
- `runs/kaggle_hybrid_retry_run008/summary.json`

Submission decision outcome:

- Validation passed.
- 6 higher-confidence zero rows were recovered after filtering.
- No nonzero Run003 rows were changed.
- Public score improved to `0.65587`.

## 5. Implementation Plan for Recommended Experiment

Added Phase 3-only scripts:

- `assignment03/phase3_retry_failures.py`
- `assignment03/phase3_build_retry_hybrid.py`

Do not touch:

- graders
- `evaluator.py`
- core EvoAgent implementation
- existing submissions

ARC command outline:

```bash
cd /home/<VT_PID>/temp/evoagent/assignment03
export PYTHON_BIN=/home/<VT_PID>/.conda/envs/evoagent/bin/python
export HF_TOKEN=<set externally>

$PYTHON_BIN phase3_retry_failures.py \
  --base-submission runs/kaggle_hybrid_001_002/submission_checked.csv \
  --base-details runs/kaggle_hybrid_001_002/submission_details.json \
  --test data/test.json \
  --strategy-path runs/exp_self_arc/iter_best_strategy.json \
  --output-dir runs/kaggle_retry_run008 \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --num-samples 5 \
  --target-zero-only

$PYTHON_BIN phase3_build_retry_hybrid.py \
  --base-submission runs/kaggle_hybrid_001_002/submission_checked.csv \
  --retry-details runs/kaggle_retry_run008/retry_details.json \
  --test data/test.json \
  --output-dir runs/kaggle_hybrid_retry_run008
```

Local validation outline:

```bash
$PYTHON_BIN format_submission.py \
  --predictions runs/kaggle_hybrid_retry_run008/submission_checked.csv \
  --output-file runs/kaggle_hybrid_retry_run008/submission_checked.csv

python3 - <<'PY'
# Validate:
# - 494 rows
# - exact id order from data/test.json
# - no duplicate ids
# - no missing or non-numeric values
# - changed-row count
# - final zero count
# - summary checksum
PY
```

The submitted Run008 filtered hybrid kept only `agreement_count >= 2` retry
recoveries. Future variants should keep the same conservative default and avoid
broad replacement of nonzero Run008 filtered predictions.

## 6. Stop Criteria

- Stop if fewer than 3 useful new rows are recovered.
- Stop if dev proxy checks or candidate inspection suggest broad harm.
- Stop if 1-2 further narrow retry attempts do not beat `0.65789`.
- For Run011-style stronger-model experiments, stop before a long run if the 10-row smoke changes fewer than 3 rows or produces non-executable DSL.
- Stop if the only remaining changes are private-leaderboard gambles with weak evidence.
- Move to ThinkFlic packaging after that.

## 7. Reproducibility and Integrity

- Do not use hidden labels.
- Do not manually infer or hand-label test answers.
- Do not store HF tokens, Kaggle tokens, `.env` files, private keys, model weights, or cache artifacts in the repo.
- Document every Kaggle submission in the experiment log.
- Keep Run009-lite safe stable as the final submission unless a validated submission beats `0.65789`.
- Preserve enough details files, changes files, and summary files to reproduce every generated candidate.
