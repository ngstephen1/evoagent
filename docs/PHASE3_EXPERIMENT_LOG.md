# Phase 3 Experiment Log

This log records Kaggle-facing experiments for Assignment 03 Phase 3. Each run
should include the strategy or pipeline used, validation checks, Kaggle result,
and enough reproduction detail to audit the submission later.

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
simpler and better-validated fallback rule.

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
public score dropped from 0.64574 to 0.64170. This suggests some apparent dev
failure patterns, especially sign and percent-point corrections, do not
generalize cleanly to public test. Current best remains Run 003. Future
post-processing should be tested as optional ablations and should prefer
rules with stronger public/test-aligned evidence.

## Current Best

| Rank | Run | Public Score | Notes |
|---:|---|---:|---|
| 1 | Run 003 | 0.64574 | Current best; simpler and better-validated hybrid fallback. |
| 2 | Run 004 | 0.64574 | Same public score as Run 003; changed only 4 additional rows. |
| 3 | Run 005 | 0.64170 | Conservative numeric post-processing hurt public score slightly. |
| 4 | Run 001 | 0.56477 | Best single-strategy baseline so far. |
| 5 | Run 002 | 0.47975 | Fewer zero predictions, worse public score as standalone. |
