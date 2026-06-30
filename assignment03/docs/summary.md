# Project Summary

This document is a quick catch-up guide for Melanie joining the Advanced NLP06 Assignment 03 / EvoAgent project.

## 1. What This Project Is

Assignment 03 asks us to implement an EvoAgent-style strategy evolution system for financial and numeric question answering, then use it to compete in the Kaggle phase of the assignment.

The project has three practical tracks:

| Track | Goal | Current Status |
|---|---|---|
| Milestone 1 | Implement EvoAgent stages and generate proof artifacts. | Complete and locally verified. |
| Milestone 2 / Phase 3 | Improve Kaggle public score with generated predictions. | Current best public score is `0.65789`. |
| ThinkFlic final package | Prepare final source, evidence, report, integrity declaration, and final Kaggle file. | Scaffold exists, final metadata still pending. |

Team:

| Teammate | Kaggle ID |
|---|---|
| Melanie | `yeyeezyzeus` |
| Nguyen Phan Nguyen - Stephen | `nguynphannguyn` |

Current best final candidate:

- Best public score: `0.65789`
- Primary final run: Run009-lite safe
- Primary final file: `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv`
- Method: start from Run008 filtered, retry a narrow suspicious-row set, then keep only three auditable high-confidence changes that do not introduce a new extreme outlier
- Alternate final candidates: Run008 filtered at `0.65587`, plus Run003 and Run004 at `0.64574`

## 2. Implementation Summary

The implementation follows the staged EvoAgent assignment structure.

| File | Role |
|---|---|
| `assignment03/src/sandbox.py` | Stage 0 sandbox prediction and sandbox accuracy check. |
| `assignment03/src/executor.py` | Stage 1 prompt construction, DSL/program execution flow, evaluation result collection, and token accounting. |
| `assignment03/src/self_reflector.py` | Stage 2 failure reflection, weak-type/error selection, and reflection structure generation. |
| `assignment03/src/self_proposer.py` | Stage 3 strategy proposal, DSL validation, fallback proposal handling, and dynamic few-shot additions. |
| `assignment03/src/harness.py` | Stage 4 evolutionary loop: parent selection, smoke testing, evaluation, reflection, proposal, and history updates. |
| `assignment03/src/model.py` | Qwen/SGLang inference wrapper, prompt formatting, batch generation, and answer extraction. |
| `assignment03/src/evaluator.py` | DSL evaluator for arithmetic/table operations used to turn generated programs into numeric answers. |
| `assignment03/src/strategy.py` | Strategy, reflection, few-shot, metadata, and strategy-history representations. |
| `assignment03/submit.py` | Generates Kaggle test predictions from a selected strategy. |
| `assignment03/format_submission.py` | Cleans and formats prediction outputs into the Kaggle-required CSV schema. |

Phase 3 helper scripts currently present:

| File | Role |
|---|---|
| `assignment03/phase3_postprocess.py` | Applies conservative numeric post-processing rules for a Kaggle candidate. Run005 showed this was not strong enough for final use. |
| `assignment03/merge_hybrid_details.py` | Builds merged details JSON for hybrid submissions so later analysis/post-processing can trace the source prediction. |
| `assignment03/phase3_retry_failures.py` | Run008 targeted retry script with checkpoint/resume, candidate repair, execution, and agreement clustering. |
| `assignment03/phase3_build_retry_hybrid.py` | Builds validated Run008 hybrid submissions from accepted retry details. |
| `assignment03/phase3_select_run009_targets.py` | Selects suspicious Run009-lite rows from Run008 filtered and cross-run disagreement signals. |
| `assignment03/phase3_build_run009_hybrid.py` | Builds validated Run009-lite hybrids with stricter nonzero replacement rules. |

Additional future improvement ideas are described in `assignment03/docs/step_to_improve.md`.

## 3. How EvoAgent Works

At a high level, EvoAgent evolves prompting strategies for programmatic financial QA:

1. A `Strategy` defines prompt wording, CoT style, few-shot examples, metadata, and optional retrieval settings.
2. `executor.py` evaluates a strategy by building prompts, calling the Qwen/SGLang model, extracting a DSL program, executing it with `evaluator.py`, and comparing the numeric value against the gold answer.
3. `self_reflector.py` analyzes failures by type and produces notes about what went wrong.
4. `self_proposer.py` uses the current strategy, reflections, history, and selected examples to propose a new strategy.
5. `harness.py` coordinates the loop: evaluate, reflect, propose, smoke-test, and record accepted strategies.
6. The best saved strategy is then used by `submit.py` to generate Kaggle predictions for `data/test.json`.

The generated answer format is a small DSL such as:

```text
subtract(108.50, 100), divide(#0, 100)
table_max(Revenue, none)
add(-167.4, -53.3)
```

The model does not directly submit free-form text. It should emit a program, and the local evaluator executes that program to produce the final numeric value.

## 4. Milestone 1 Status

Milestone 1 is complete.

Implemented stages:

- Stage 0: sandbox prediction
- Stage 1: executor
- Stage 2: self reflection
- Stage 3: self proposal
- Stage 4: evolution harness

Local graders passed:

```bash
python3 graders/grade_stage0.py
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

Generated proof artifacts:

| Artifact | Path |
|---|---|
| Sandbox proof | `assignment03/sandbox_proof.json` |
| Smoke proof | `assignment03/smoke_proof.json` |
| Evolution proof | `assignment03/evolution_proof.json` |
| Run-local evolution proof | `assignment03/runs/exp_self_arc/evolution_proof.json` |

ARC evolution result:

| Metric | Value |
|---|---:|
| Baseline accuracy | `0.42` |
| Best dev accuracy | `0.48333333333333334` |
| Best iteration | `1` |

## 5. ARC / GPU Context

We used VT ARC GPU execution instead of Modal for the main proof and Kaggle inference workflow. Modal files remain in the assignment code, but this fork's practical GPU path uses ARC.

Observed ARC context:

| Item | Value |
|---|---|
| GPU node used | `tc-dgx006` |
| GPU type | NVIDIA A100-SXM4-80GB |
| Python env used on ARC | `/home/<VT_PID>/.conda/envs/evoagent/bin/python` |

Typical ARC helper files:

- `scripts/arc_login.sh`
- `scripts/arc_interact_a100.sh`
- `scripts/arc_setup_env.sh`
- `scripts/arc_run_local_graders.sh`
- `scripts/arc_generate_proofs.sh`
- `assignment03/arc_proofs.py`
- `docs/ARC_GPU_WORKFLOW.md`

Security warning:

- Do not expose or commit `HF_TOKEN`.
- Do not expose or commit Kaggle credentials or `kaggle.json`.
- Do not commit `.env`, SSH keys, private keys, model weights, caches, or checkpoints.
- Do not commit `.vscode/sftp.json`; use `.vscode/sftp.json.example` as the template.

## 6. Kaggle Run History

| Run | Description | Public Score | File / Strategy | Notes |
|---|---|---:|---|---|
| Run001 | EvoAgent ARC best strategy baseline | 0.56477 | Strategy: `assignment03/runs/exp_self_arc/iter_best_strategy.json`; submission: `assignment03/runs/kaggle_arc_best/submission_checked.csv` | Strongest single strategy, but had about 112 `0.0` fallback predictions. |
| Run002 | Full iter003/table-op strategy | 0.47975 | `assignment03/runs/kaggle_iter003/submission_checked.csv` | Fewer zero predictions, but worse overall; useful as fallback, bad as full replacement. |
| Run003 | Hybrid Run001 fallback to iter003 nonzero | 0.64574 | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` | Previous best. Biggest gain came from filling Run001 failure/zero rows with Run002 nonzero predictions. |
| Run004 | Hybrid Run003 fallback to iter004 nonzero | 0.64574 | `assignment03/runs/kaggle_hybrid_003_004/submission_checked.csv` | Tied Run003; changed only a few rows; useful as alternate/private hedge. |
| Run005 | Conservative numeric post-processing | 0.64170 | `assignment03/runs/kaggle_postprocess_run005/submission_checked.csv` | Changed 14 rows but hurt score; do not use as final. |
| Run006 | Context-expanded rerun of best strategy with max context 32768 | Not submitted | `assignment03/runs/kaggle_run006_iterbest_ctx32768/submission_checked.csv` | Worse zero count than Run001; not useful standalone. |
| Run007 | Tiny Run003 + Run006 fallback | Not submitted | `assignment03/runs/kaggle_hybrid_003_006/submission_checked.csv` | Valid, but only changed 2 rows and one conflicted with Run004; medium-risk private-leaderboard gamble; not recommended for final. |
| Run008 filtered | Targeted retry over Run003 zero rows, keeping only `agreement_count >= 2` recoveries | 0.65587 | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv` | Previous best. Recovered 6 higher-confidence zero/fallback rows from Run003. |
| Run009-lite safe | Suspicious-row targeted retry over Run008 filtered, excluding unchanged rows and new extreme outliers | 0.65789 | `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv` | Current primary final. Kept 3 auditable changes and improved public score without increasing extreme count. |

### Why Run009-Lite Safe Is Best So Far

Run001 had the strongest overall behavior, but it often failed by producing malformed, looping, or invalid programs that cleaned to `0.0`. Run002 was worse as a full strategy, but it often produced executable programs where Run001 failed. Run003 preserved all nonzero Run001 predictions and used Run002 only as a fallback source for Run001 zeros.

That selective rule gave the first major public-score improvement:

```text
Run001: 0.56477
Run003: 0.64574
```

Run008 then targeted only the 11 remaining Run003 zero rows. The full retry
recovered 7 rows, but one single-candidate recovery looked suspicious, so the
submitted filtered variant kept only replacements with `agreement_count >= 2`.
This changed 6 rows, reduced the final zero count to 5, and improved public
score again:

```text
Run003: 0.64574
Run008 filtered: 0.65587
```

Run009-lite then targeted only 60 suspicious Run008 rows and used stricter
filtering before submission. The full candidate accepted 15 retry rows, but
only 3 were meaningful and safe after removing unchanged values and one new
extreme outlier. That conservative variant improved the public score again:

```text
Run008 filtered: 0.65587
Run009-lite safe: 0.65789
```

### What Failed

- Replacing too broadly failed: Run002 had fewer zeros but scored much worse as a full submission.
- Numeric post-processing failed: Run005 changed 14 rows and reduced the public score.
- Bigger context failed: Run006 had more zero predictions than Run001.
- Extra fallback rows plateaued when they came from similar submissions: Run004 tied Run003, and Run007 was not worth submitting.
- Targeted retry worked only after confidence filtering; suspicious or extreme replacements were excluded from the submitted Run008 and Run009-safe files.

## 7. Important Lessons

The strongest lesson is that hybrid fallback ensembling worked better than trying to find one universally better strategy.

Useful principles:

- Preserve the best global strategy unless there is strong evidence a row failed.
- `0.0` can be a useful failure signal, but fewer zeros alone do not guarantee better Kaggle accuracy.
- Broad numeric corrections are risky because they can convert plausible-but-correct values into wrong values.
- Table-operation specialists can help narrowly, but they can hurt when used as full replacements.
- Same-family ensembling has plateaued; further improvement probably needs targeted retry, self-consistency, or program repair.

## 8. Current Files And Package Status

ThinkFlic scaffold exists:

```text
A3_MelanieAndStephen_<StudentID1>_<StudentID2>/
```

Important scaffold files:

| Path | Status |
|---|---|
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/README.md` | Present |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/report.md` | Draft present; final PDF still needed |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/integrity_declaration.md` | Draft present; signed/PDF form still needed |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/source_code/` | Present |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/kaggle/final_submission.csv` | Present; copied from Run009-lite safe |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/kaggle/submission_information.txt` | Present |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/evidence/evolution_proof.json` | Present |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/evidence/failure_mode_report.pdf` | Present |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/evidence/learning_curve.pdf` | Present |
| `A3_MelanieAndStephen_<StudentID1>_<StudentID2>/evidence/MISSING_STRATEGY_DIVERSITY_NOTE.md` | Present because `strategy_diversity.pdf` has not been generated |

Remaining blockers before final ThinkFlic ZIP:

- Real student IDs
- Video link
- Final `report.pdf`
- Signed or finalized `integrity_declaration.pdf`
- Final folder rename from placeholder IDs to real IDs
- Final ZIP creation

Suggested final folder naming once real IDs are known:

```text
A3_MelanieAndStephen_<REAL_ID_1>_<REAL_ID_2>
```

## 9. Next Improvement Roadmap

See `assignment03/docs/step_to_improve.md` for the detailed score-improvement roadmap.

Run008 and Run009-lite completed the previously recommended targeted-retry
experiments. If the team reopens Kaggle work, the next high-upside experiment
should build on Run009-lite safe rather than replacing it broadly:

- Target only the remaining Run009 zero/failure rows or clearly invalid retry candidates.
- Use multi-sample retry or self-consistency.
- Generate valid DSL programs only.
- Execute and validate programs locally.
- Hybrid recovered rows back into Run009-lite safe only when they are finite, nonzero, non-extreme, and consistent.
- Do not touch nonzero Run009-lite safe predictions by default.

The completed Run008 artifacts are:

| Item | Value |
|---|---|
| Base file | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Candidate output dir | `assignment03/runs/kaggle_retry_run008/` |
| Hybrid output dir | `assignment03/runs/kaggle_hybrid_retry_run008/` |
| Submitted filtered output | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv` |
| Public score | `0.65587` |

The completed Run009-lite safe artifacts are:

| Item | Value |
|---|---|
| Base file | `assignment03/runs/kaggle_hybrid_retry_run008_agree2/submission_checked.csv` |
| Target rows | `assignment03/runs/kaggle_retry_run009_lite/target_rows.csv` |
| Retry output dir | `assignment03/runs/kaggle_retry_run009_lite/` |
| Safe submitted output | `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv` |
| Public score | `0.65789` |

The practical recommendation is to freeze Kaggle with Run009-lite safe as the
primary final candidate unless the team explicitly decides to run another
narrow, auditable retry experiment.

## 10. Quick Commands

Run local graders from `assignment03/`:

```bash
python3 graders/grade_stage0.py
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

Generate a Kaggle submission from a strategy on ARC GPU:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./runs/kaggle_arc_best/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

Format a submission:

```bash
python3 format_submission.py \
  --predictions ./runs/kaggle_arc_best/submission.csv \
  --output-file ./runs/kaggle_arc_best/submission_checked.csv
```

Do not run Kaggle submissions or GPU jobs casually. Log every generated candidate and every submitted score in `docs/PHASE3_EXPERIMENT_LOG.md`.
