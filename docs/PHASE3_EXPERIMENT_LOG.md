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
