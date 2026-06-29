# Phase 3 Experiment Log

This log records Kaggle-facing experiments for Assignment 03 Phase 3. Each run
should include the strategy or pipeline used, validation checks, Kaggle result,
and enough reproduction detail to audit the submission later.

## Experiment Freeze

Kaggle experimentation is frozen after Run 007. Do not create or submit more
Kaggle runs unless the team explicitly reopens Phase 3 experimentation.

Final Kaggle candidates:

| Role | Run | File | Public Score | Decision |
|---|---|---|---:|---|
| Primary final | Run 003 | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` | 0.64574 | Use as primary final Kaggle file. |
| Alternate / private hedge | Run 004 | `assignment03/runs/kaggle_hybrid_003_004/submission_checked.csv` | 0.64574 | Keep as alternate only. |

Summary interpretation: hybrid fallback ensembling was the strongest method.
Broad numeric post-processing hurt public score, and further same-run ensembling
plateaued without producing a clearly better candidate.

## Run 001 - EvoAgent ARC Best Strategy Baseline

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | `675bb15` |
| Compute | VT ARC GPU |
| Model | `QuantTrio/Qwen3.5-4B-AWQ` |
| Strategy file | `assignment03/runs/exp_self_arc/iter_best_strategy.json` |
| Strategy id | `05ebde50-9303-45b6-a438-8aa840cc6574` |
| Strategy iteration | 1 |
| Strategy dev accuracy | 0.48333333333333334 |
| Strategy train accuracy | 0.535 |
| Output file | `assignment03/runs/kaggle_arc_best/submission.csv` |
| Submitted file | `assignment03/runs/kaggle_arc_best/submission_checked.csv` |
| Kaggle description | `EvoAgent ARC best strategy baseline` |
| Kaggle status | `COMPLETE` |
| Public score | 0.56477 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Generation Command

Run from `assignment03/` on an ARC GPU node:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./runs/kaggle_arc_best/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

### Validation Notes

- `assignment03/data/test.json` contains 494 unique test IDs.
- `submission_checked.csv` has columns `id,Usage,predicted_value`.
- Local validation found 494 rows and no blank or non-numeric predictions.
- 112 predictions were `0.0`; some may be legitimate, but this is also a
  useful signal for failed program extraction or execution fallback.

### Initial Interpretation

This run is a valid first Kaggle baseline and performs better on Kaggle public
score than the dev accuracy alone would suggest. The next experiments should
focus on reducing invalid/failed programs and improving weak operation types,
especially addition and table operations from the ARC dev analysis.

## Run 002 - Iteration 003 Non-CoT Table-Op Baseline

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | `177a7f2` |
| Compute | VT ARC GPU |
| Model | `QuantTrio/Qwen3.5-4B-AWQ` |
| Strategy file | `assignment03/runs/exp_self_arc/iter_003_strategy.json` |
| Strategy id | `f8d823c0-4e48-4147-a9fb-8ff135dbdadf` |
| Strategy iteration | 3 |
| Strategy dev accuracy | 0.45416666666666666 |
| Strategy train accuracy | 0.47 |
| Output file | `assignment03/runs/kaggle_iter003/submission.csv` |
| Submitted file | `assignment03/runs/kaggle_iter003/submission_checked.csv` |
| Kaggle description | `EvoAgent iter003 non-CoT table-op baseline` |
| Kaggle status | `COMPLETE` |
| Public score | 0.47975 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Generation Command

Run from `assignment03/` on an ARC GPU node:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_003_strategy.json \
  --output-file ./runs/kaggle_iter003/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

### Validation Notes

- `submission_checked.csv` has columns `id,Usage,predicted_value`.
- Local validation found 494 rows, no duplicate IDs, and no blank or
  non-numeric predictions.
- Zero-valued predictions dropped from 112 in Run 001 to 23 in Run 002.
- The checked submission hash was
  `ce9605bf8eb022bbd8d7ca986919641bfb40e10cb90fae17b28663bb7a6d4d1e`.

### Interpretation

Run 002 reduced fallback-like `0.0` predictions substantially, but Kaggle public
score dropped from 0.56477 to 0.47975. Avoiding `0.0` fallback alone is not
enough; the non-CoT iter003 strategy likely produced more confident but wrong
numeric answers. Current best remains Run 001.

## Run 003 - Hybrid Run001 Fallback to Iter003 Nonzero

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | `177a7f2` |
| Method | No-code CSV hybrid |
| Input 1 | `assignment03/runs/kaggle_arc_best/submission_checked.csv` |
| Input 2 | `assignment03/runs/kaggle_iter003/submission_checked.csv` |
| Output file | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Changes file | `assignment03/runs/kaggle_hybrid_001_002/changes.csv` |
| Kaggle description | `Hybrid run001 fallback to iter003 nonzero` |
| Kaggle status | `COMPLETE` |
| Public score | 0.64574 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Hybrid Rule

Start from Run 001 predictions. For each row:

1. If Run 001 `predicted_value == 0.0` and Run 002 `predicted_value != 0.0`,
   replace the value with Run 002's prediction.
2. Otherwise keep Run 001's prediction unchanged.

### Validation Notes

- Local validation found 494 rows, no duplicate IDs, and no blank or
  non-numeric predictions.
- 101 rows changed from Run 001.
- Final zero-valued predictions dropped from 112 in Run 001 to 11 in Run 003.
- The checked submission hash was
  `11dcc3ade60ef4570e0410559d2ef9a2e7b9343a20471f802281e0fb16c8b31b`.

### Interpretation

Run 002 was not globally better than Run 001, but it recovered useful answers
for many Run 001 fallback cases. This suggests fallback/hybrid ensembling is
the most promising current Phase 3 direction: preserve the strongest global
strategy, then selectively patch its failure modes with complementary runs.

## Run 004 - Hybrid Run003 Fallback to Iter004 Nonzero

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | `0123a1b` |
| Method | No-code CSV hybrid |
| Input 1 | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Input 2 | `assignment03/runs/kaggle_iter004/submission_checked.csv` |
| Output file | `assignment03/runs/kaggle_hybrid_003_004/submission_checked.csv` |
| Changes file | `assignment03/runs/kaggle_hybrid_003_004/changes.csv` |
| Kaggle description | `Hybrid run003 fallback to iter004 nonzero` |
| Kaggle status | `COMPLETE` |
| Public score | 0.64574 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Hybrid Rule

Start from Run 003 predictions. For each row:

1. If Run 003 `predicted_value == 0.0` and Run 004 `predicted_value != 0.0`,
   replace the value with Run 004's prediction.
2. Otherwise keep Run 003's prediction unchanged.

### Validation Notes

- Local validation found 494 rows, no duplicate IDs, and no blank or
  non-numeric predictions.
- 4 rows changed from Run 003.
- Final zero-valued predictions dropped from 11 in Run 003 to 7 in Run 004.
- The checked submission hash was
  `4582cb4e96e5b4d638a08585e3c0f8348c65da4c832e3a414247415e04f49b2a`.

### Interpretation

Run 004 is structurally valid but did not improve the public leaderboard score.
It may or may not help on the private leaderboard, but the public result shows
that patching the last few zero-valued rows is not automatically beneficial.
Current best remains Run 003 because it achieved the same public score with a
simpler and better-validated fallback rule. Keep Run 004 as an alternate only
if final submission selection allows multiple candidate choices.

## Run 005 - Hybrid Run003 Conservative Numeric Postprocess

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | Pending |
| Method | Phase 3 post-processing over Run 003 hybrid |
| Input submission | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Input details | `assignment03/runs/kaggle_hybrid_001_002/submission_details.json` |
| Output file | `assignment03/runs/kaggle_postprocess_run005/submission_checked.csv` |
| Changes file | `assignment03/runs/kaggle_postprocess_run005/changes.csv` |
| Kaggle description | `Hybrid run003 conservative numeric postprocess` |
| Kaggle status | `COMPLETE` |
| Public score | 0.64170 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Method

Start from Run 003, then apply conservative deterministic numeric corrections
using merged submission details. The risky reciprocal-ratio rule was disabled.

Enabled rules:

1. `abs_negative_difference`: 9 changes.
2. `percent_point_divided_by_100`: 4 changes.
3. `growth_ratio_to_growth_rate`: 1 change.

### Validation Notes

- Local validation found 494 rows in official `data/test.json` order.
- No duplicate IDs, no missing values, and all predictions were numeric.
- 14 rows changed from Run 003.
- Zero-valued predictions stayed at 11.

### Interpretation

The post-processing rules were structurally valid and mostly conservative, but
the 14 changed rows reduced public score from 0.64574 to 0.64170. This suggests
some apparent dev failure patterns, especially sign and percent-point
corrections, do not generalize cleanly to public test. Do not use Run 005 as
the final submission. Current best remains Run 003/Run 004 at 0.64574, with
Run 003 preferred as the primary final candidate because it is simpler and
clearly improves over the baseline. Future post-processing should be treated as
optional ablations and should prefer rules with stronger public/test-aligned
evidence.

## Run 006 - IterBest Context-Expanded Rerun

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | Pending |
| Compute | VT ARC GPU |
| Model | `QuantTrio/Qwen3.5-4B-AWQ` |
| Strategy file | `assignment03/runs/exp_self_arc/iter_best_strategy.json` |
| Method | Re-run best Run001 strategy with larger context window |
| Output file | `assignment03/runs/kaggle_run006_iterbest_ctx32768/submission.csv` |
| Checked file | `assignment03/runs/kaggle_run006_iterbest_ctx32768/submission_checked.csv` |
| Details file | `assignment03/runs/kaggle_run006_iterbest_ctx32768/submission_details.json` |
| Kaggle status | `NOT SUBMITTED` |
| Public score | Not submitted |
| Private score | Not submitted |
| Row count | 494 |

### Generation Command

Run from `assignment03/` on an ARC GPU node:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./runs/kaggle_run006_iterbest_ctx32768/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768
```

### Validation Notes

- Local validation found 494 rows in official `data/test.json` order.
- No duplicate IDs, no missing values, and all predictions were numeric.
- Standalone Run006 had 122 zero-valued predictions, worse than Run001's 112.
- It filled only 2 of Run003's 11 remaining zero-valued rows.
- The checked submission hash was
  `097bd225c1d16a18116c04dd1e85dd7fdd6f87d37911b1510b4dbeafe4ea77c4`.

### Interpretation

Run006 is a genuine new prediction source, but it is not a standalone final
candidate. Increasing context length did not reduce fallback behavior; zero
predictions increased relative to Run001. It was useful only as a tiny fallback
source for Run007, and Run007 was also held back.

## Run 007 - Hybrid Run003 Fallback to Run006 Nonzero

| Field | Value |
|---|---|
| Date | 2026-06-29 |
| Branch | `integration/evoagent-arc` |
| Commit | Pending |
| Method | Tiny CSV hybrid over Run 003 using Run 006 as fallback source |
| Input 1 | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Input 2 | `assignment03/runs/kaggle_run006_iterbest_ctx32768/submission_checked.csv` |
| Output file | `assignment03/runs/kaggle_hybrid_003_006/submission_checked.csv` |
| Changes file | `assignment03/runs/kaggle_hybrid_003_006/changes.csv` |
| Summary file | `assignment03/runs/kaggle_hybrid_003_006/summary.json` |
| Kaggle status | `NOT SUBMITTED` |
| Public score | Not submitted |
| Private score | Not submitted |
| Row count | 494 |

### Hybrid Rule

Start from Run 003. Replace a value only when Run 003 predicted `0.0` and
Run 006 produced a finite, nonzero value with absolute magnitude at most `1e8`.
No nonzero Run 003 rows were replaced, and no numeric post-processing was
applied.

### Validation Notes

- Local validation passed with 494 rows in official `data/test.json` order.
- No duplicate IDs, no missing values, and all predictions were numeric.
- 2 rows changed from Run 003.
- Final zero-valued predictions dropped from 11 to 9.
- One changed row overlaps with Run 004's four-row hybrid, but Run 006 gives a
  conflicting value on that row.

### Interpretation

Run 007 is valid but should not be submitted for now. It changes only 2 rows,
and one of those rows conflicts with the already-submitted Run 004 fallback
value. Since Run 004 already tied Run 003 publicly, Run 007 is only a
private-leaderboard gamble with medium risk and limited evidence. Primary final
candidate remains Run 003. Run 004 remains the alternate final candidate.

## Current Best

| Rank | Run | Public Score | Notes |
|---:|---|---:|---|
| 1 | Run 003 | 0.64574 | Primary final candidate; simpler hybrid and clearly improves over baseline. |
| 2 | Run 004 | 0.64574 | Alternate only; same public score as Run 003 with 4 extra fallback rows. |
| 3 | Run 005 | 0.64170 | Do not use as final; 14 numeric post-processing changes reduced public score. |
| - | Run 006 | Not submitted | Generated but not submitted; context-expanded rerun had more zeros than Run001. |
| - | Run 007 | Not submitted | Validated but held back; medium-risk private-LB gamble. |
| 4 | Run 001 | 0.56477 | Best single-strategy baseline so far. |
| 5 | Run 002 | 0.47975 | Fewer zero predictions, worse public score as standalone. |
