# Phase 3 Experiment Log

This log records Kaggle-facing experiments for Assignment 03 Phase 3. Each run
should include the strategy or pipeline used, validation checks, Kaggle result,
and enough reproduction detail to audit the submission later.

## Current Phase 3 Status

Run 009-lite safe is the current best public submission. Run 008 filtered
remains the strongest previous targeted-retry baseline, while Run 003 remains
the most important fallback-hybrid baseline.
Do not create more Kaggle runs unless the team explicitly reopens Phase 3
experimentation.

Final Kaggle candidates:

| Role | Run | File | Public Score | Decision |
|---|---|---|---:|---|
| Primary final | Run 009-lite safe | `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv` | 0.65789 | Use as primary final Kaggle file unless a later validated run beats it. |
| Alternate / previous best | Run 008 filtered | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv` | 0.65587 | Keep as the strongest previous targeted-retry candidate. |
| Alternate / private hedge | Run 003 | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` | 0.64574 | Keep as simpler fallback-only baseline. |
| Alternate / private hedge | Run 004 | `assignment03/runs/kaggle_hybrid_003_004/submission_checked.csv` | 0.64574 | Keep as alternate only. |

Summary interpretation: hybrid fallback ensembling created the first major
gain, and targeted retry over remaining zero rows created the next gain. Broad
numeric post-processing hurt public score, while weak-confidence retry outputs
needed filtering. Run 009-lite shows that suspicious-row retry can add a small
additional gain when changes are filtered to avoid new extreme outliers.

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
At the time, Run 003 remained preferred because it achieved the same public
score with a simpler and better-validated fallback rule. Run 004 is kept as an
alternate/private-leaderboard hedge.

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
the final submission. At the time, Run 003/Run 004 remained best at `0.64574`,
with Run 003 preferred because it was simpler and clearly improved over the
baseline. Future post-processing should be treated as optional ablations and
should prefer rules with stronger public/test-aligned evidence.

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
value. Since Run 004 already tied Run 003 publicly, Run 007 was only a
private-leaderboard gamble with medium risk and limited evidence. It was not
submitted, and the next successful public improvement came from Run 008
filtered.

## Run 008 - Targeted Retry for Run003 Zero Rows

| Field | Value |
|---|---|
| Date | 2026-06-30 |
| Branch | `integration/evoagent-arc` |
| Method | Multi-sample targeted retry plus DSL repair/execution over Run003 zero rows |
| Base submission | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Retry details | `assignment03/runs/kaggle_retry_run008/retry_details.json` |
| Full hybrid file | `assignment03/runs/kaggle_hybrid_retry_run008/submission_checked.csv` |
| Full hybrid changes | `assignment03/runs/kaggle_hybrid_retry_run008/changes.csv` |
| Filtered submitted file | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv` |
| Filtered changes | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/changes.csv` |
| Kaggle description | `Run008 filtered targeted retry agreement>=2` |
| Kaggle status | `COMPLETE` |
| Public score | 0.65587 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Retry Method

Run 008 starts from Run 003 and targets only rows where Run 003 predicted
`0.0`. For each target row, the retry script samples multiple DSL candidates,
repairs supported malformed outputs, executes valid programs with the existing
DSL evaluator, clusters numeric answers, and accepts only finite, nonzero,
non-extreme values.

The full retry recovered 7 of Run003's 11 zero rows:

- Zero count before retry: 11.
- Zero count after full retry: 4.
- Full retry changed rows: 7.
- Full retry rejected rows: 4.
- Full retry validation: 494 rows, exact ID order, no duplicates, no missing
  predictions, all numeric.

### Filtered Submission

Before submitting, the team inspected `changes.csv` and found one suspicious
single-candidate replacement:

- Question: comparing POW P/B in 2018 and 2019F.
- Table evidence: P/B was `1.5` in both years.
- Full Run008 replacement: `29.25`.
- Confidence: `single_strict_repair_no_conflict`, `agreement_count=1`.

To reduce risk, the submitted Run008 variant kept only replacements with
`agreement_count >= 2`.

Filtered validation:

- Row count: 494.
- Exact test ID order: true.
- Duplicate IDs: 0.
- Missing predictions: 0.
- All predictions numeric: true.
- Changed rows from Run003: 6.
- Final zero-valued predictions: 5.
- `safe_to_submit`: true.

### Interpretation

Run 008 confirms that targeted retry is more useful than broad numeric
post-processing. The gain over Run003 is modest but real: public score improved
from `0.64574` to `0.65587`. The agreement filter mattered because it removed a
likely bad single-candidate answer while preserving six higher-confidence
recoveries. Run 008 filtered became the first targeted-retry best and remains a
strong alternate now that Run 009-lite safe has slightly improved on it.

## Run 009-Lite - Safe Filtered Targeted Retry

| Field | Value |
|---|---|
| Date | 2026-06-30 |
| Branch | `integration/evoagent-arc` |
| Method | Suspicious-row targeted retry over Run008 filtered, then safe filtering |
| Base submission | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv` |
| Target rows | `assignment03/runs/kaggle_retry_run009_lite/target_rows.csv` |
| Full retry details | `assignment03/runs/kaggle_retry_run009_lite/retry_details.json` |
| Full hybrid file | `assignment03/runs/kaggle_hybrid_retry_run009_lite/submission_checked.csv` |
| Safe submitted file | `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv` |
| Safe changes | `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/changes.csv` |
| Kaggle description | `Run009-lite safe filtered targeted retry` |
| Kaggle status | `COMPLETE` |
| Public score | 0.65789 |
| Private score | Pending final leaderboard |
| Row count | 494 |

### Targeting Method

Run 009-lite starts from Run008 filtered and selects a narrow suspicious-row
pool instead of retrying all 494 test rows. The target selector prioritizes:

1. Remaining zero predictions after Run008.
2. Extreme predictions where `abs(predicted_value) > 1e6`.
3. Negative predictions with absolute-difference wording.
4. Strong disagreement among Run001, Run002, Run004, Run006, and Run008.
5. Weak operation-type cues for addition, subtraction, and table operations.

The full Run009-lite retry processed 60 target rows with 5 samples per row and
checkpoint/resume enabled. The full builder accepted 15 retry rows, but most
were numerically identical to the existing Run008 prediction. One accepted
replacement introduced a new extreme value:

```text
masvn/2020/2020077-V1_VN_Galvanized-steel-Outlook/page_13_QB4
0.0117678555 -> 11784259.5
```

To reduce risk, the submitted safe variant kept only meaningful replacements
that did not introduce a new `abs(value) > 1e6` extreme and skipped unchanged
accepted retries.

### Safe Validation

- Row count: 494.
- Exact test ID order: true.
- Duplicate IDs: 0.
- Missing predictions: 0.
- All predictions numeric: true.
- Changed rows from Run008 filtered: 3.
- Zero-valued predictions: 5 -> 4.
- Negative predictions: 50 -> 49.
- Extreme `abs(value) > 1e6` predictions: 9 -> 9.
- `safe_to_submit`: true.

Safe changed rows:

| ID | Old | New | Agreement | Reason |
|---|---:|---:|---:|---|
| `masvn/2020/2020010-190314_MBB_2018review_VN/page_1_QA4` | 36.5908489817509 | 22.97 | 5 | Table max ROE correction. |
| `C/2015/page_96.pdf-3` | -32.1 | 32.1 | 3 | Absolute difference wording; sign corrected. |
| `GPN/2013/page_87.pdf-4` | 0.0 | 170.0 | 2 | Remaining zero recovered by addition program. |

### Interpretation

Run 009-lite safe produced the best public score so far: `0.65789`, improving
over Run008 filtered by `0.00202`. The improvement is small but meaningful
because it came from only three auditable changes and avoided the full
Run009-lite candidate's new extreme outlier. This supports the same overall
lesson as Run008: narrow, executable, agreement-filtered changes are useful;
broad or high-magnitude replacements remain risky.

## Current Best

| Rank | Run | Public Score | Notes |
|---:|---|---:|---|
| 1 | Run 009-lite safe | 0.65789 | Primary final candidate; safe filtered retry over Run008 suspicious rows. |
| 2 | Run 008 filtered | 0.65587 | Strong previous best; targeted retry over Run003 zero rows with agreement filter. |
| 3 | Run 003 | 0.64574 | Previous fallback-hybrid best and useful private-LB hedge. |
| 4 | Run 004 | 0.64574 | Alternate only; same public score as Run 003 with 4 extra fallback rows. |
| 5 | Run 005 | 0.64170 | Do not use as final; 14 numeric post-processing changes reduced public score. |
| - | Run 006 | Not submitted | Generated but not submitted; context-expanded rerun had more zeros than Run001. |
| - | Run 007 | Not submitted | Validated but held back; medium-risk private-LB gamble. |
| 6 | Run 001 | 0.56477 | Best single-strategy baseline so far. |
| 7 | Run 002 | 0.47975 | Fewer zero predictions, worse public score as standalone. |

---

## Ensemble Error Analysis & Improvement Strategy (2026-07-06)

Analysis of `A3_MelanieAndStephen_.../kaggle/submission_ensemble.csv` and its three
member files (`final_submission.csv` = run009, `submission_8b_sc5.csv`,
`submission_coder7b.csv`). Goal: locate likely-wrong test predictions and lay out
the levers for improving the score.

### Key constraint: the test set has no local gold labels

`assignment03/data/test.json` (494 rows) contains only `qa.question` — no `exe_ans`.
So test rows cannot be scored directly. "Wrong answers" were surfaced via two proxies:

1. **Ensemble disagreement** across the 3 member models (low agreement => low confidence).
2. **Structural red flags** (scale, sign, zeros, extreme outliers).

The correct scaling convention was confirmed against the **labeled dev set**
(`data/dev.json`, 584 rows with gold).

Reproduction scripts (scratchpad): `analyze_ensemble.py`, `scale_errors.py`.
Full ranked suspicious-row table written to `kaggle/suspicious_rows.json`.

### Confidence distribution on the test submission

| Members agreeing with final value | Rows | % |
|---|---:|---:|
| 3/3 (unanimous) | 194 | 39% |
| 2/3 | 205 | 42% |
| 1/3 (one model carried it) | 95 | 19% |

**61% of test rows have at least one dissenting model** — this is where errors concentrate.

### Dominant error category: scale (power-of-10) and sign mistakes

**64 / 494 rows (13%)** are cases where the models agree on the *digits* but disagree
on *magnitude or sign*:

- **52 pure scale errors** — 43 are `×100` (percent-vs-ratio confusion), e.g.
  `0.198 vs 19.86`, `0.161 vs 16.12`, `0.653 vs 65.27`; plus a few `×1000` and unit-driven `×1e9`.
- **12 pure sign flips** — e.g. `-530 vs 530`, `2.5 vs -2.5`.

Direction of the fix, confirmed on dev: for ratio/percent-cue questions the gold answer
is a **decimal ratio** (median `|ans|` = 0.375; 71% ≤ 1.0), NOT a percent. So `×100`
disagreements should almost always resolve toward the **smaller/ratio** value. The
`RATIO_Q_BUT_BIG` flag caught **68 rows** where the ensemble picked a value > 1.5 on a
ratio-worded question — prime candidates for being 100× too large.

### Other error clusters

| Flag | Count | Interpretation |
|---|---:|---|
| `SPLIT_2` (2 vs 1) | 210 | One model dissents; medium confidence. |
| `FULL_DISAGREE` (all 3 differ) | 90 | Genuinely hard; often multi-step or table-lookup. |
| `RATIO_Q_BUT_BIG` | 68 | Ratio question, answer > 1.5 → likely 100× too large. |
| `EXTREME_HIGH` (≥1e6) | 10 | Unit confusion ("triệu"/"tỷ" = million/billion). |
| `ZERO` | 4 | Extraction failure. |

Disagreement (proxy error) rate by slice: US-filings 66.5% vs MASVN 56.5%;
plain-numeric Q 63.6% vs ratio/percent Q 58.7%.

### Interpretation

The models mostly *can* read the tables; they fail on **conventions** (scale, sign,
units), not comprehension. That is the cheapest class of error to fix — via
post-processing, prompting, and light finetuning.

---

## Improvement Strategy Menu (ranked by ROI for this dataset)

### Tier 1 — cheap, high impact, no training

1. **Post-processing (scale/sign fixer) — highest ROI now.** For ratio/percent-cue
   questions where the ensemble value > 1.5, divide by 100 (or prefer the ÷100 member
   value). Add "triệu"/"tỷ" unit normalization for the 10 extreme-high outliers, and
   canonicalize sign via `subtract(new, old)` for change questions. Targets ~64–68
   flagged rows. Validate with one Kaggle submission. NOTE: broad numeric post-processing
   previously hurt (Run 005) — keep changes *targeted and category-gated*, not global.
2. **Prompt engineering — fixes the root cause.** State the rules explicitly in the
   system prompt: "Answer as a decimal ratio (0.085), never a percent (8.5)"; include the
   DSL spec (reuse `data_description.md` §2 verbatim) and unit rules. Same fix as (1) but
   upstream, so it also reaches rows post-processing can't.
3. **Validation harness — prerequisite for everything below.** Currently there is no
   local score, only Kaggle's 0.658. Build dev.json eval: predicted programs → `src/evaluator.py`
   → exact-match (within tolerance) vs `exe_ans`, sliced by error category. Every strategy
   below needs this local signal to tune against.

### Tier 2 — strong gains, moderate effort, no training

4. **Few-shot examples.** Curate 4–8 exemplars from `finqa_qualified_reasonings.json` that
   demonstrate each failure mode (ratio→decimal, table_max, multi-step chain, sign case).
5. **Retrieval (RAG for few-shot selection).** Embed train.json questions; at inference,
   retrieve the k most similar gold (question→program) pairs as dynamic few-shot. High value
   because the data is stylistically clustered (MASVN vs US filings, recurring templates).
6. **Self-consistency (already have `8b_sc5`) — make it scale-aware.** Current SC votes on
   raw values, so `0.028` and `2.88` split the vote. Normalize magnitude before voting, or
   vote on the *program* rather than the number. Directly attacks the 52 scale-split rows.
7. **Ensembling (already have) — improve the tie-break.** Replace fixed run009-priority
   tie-break with per-category weighting (let the member that is best on table-lookups win
   those rows). Requires the validation harness to know per-category strengths.

### Tier 3 — biggest ceiling, highest cost, training

8. **Fine-tuning (SFT / LoRA).** Train on train.json's 2,986 gold programs; output the DSL
   *program* (auto-verifiable via `evaluator.py`), not the raw number. Over-sample
   scale/table/sign cases. LoRA/QLoRA on the existing 7-8B models via Modal (`run_modal.py`).
   Instruction JSONL: `{system: DSL spec + scaling rule, user: pre/post text + table +
   question, assistant: program}`. Full fine-tune is overkill.
9. **Post-training (rejection sampling → RL).** `evaluator.py` gives a *verifiable reward*,
   so this dataset is ideal for reward-based post-training:
   - *Rejection sampling / STaR (do first, cheaper):* sample N programs per train question,
     keep only those that execute to `exe_ans`, SFT on the survivors.
   - *RLVR / GRPO (fuller):* reward = 1 if executed answer matches gold; optimizes the metric directly.

### Suggested sequence

`Validation harness → Prompt + scale-aware SC (measure) → Post-processing fixer (submit)
→ Retrieval few-shot → LoRA SFT → rejection-sampling post-train`.

Single highest-leverage item: the **validation harness** — it is a dependency for judging
every other lever, and there is currently no local ground-truth signal at all.
