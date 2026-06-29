# Future Plan for Significant Score Improvement

This document captures ambitious future plans for improving the Advanced NLP06 Assignment 03 / EvoAgent Kaggle score. It goes beyond the immediate `assignment03/docs/step_to_improve.md` plan and includes both practical next experiments and higher-effort ideas that could matter if we had more time, compute, or engineering bandwidth.

Hard boundaries:

- Do not use hidden labels or infer test answers manually.
- Do not commit secrets, HF tokens, Kaggle tokens, `.env` files, private keys, model weights, or caches.
- Do not treat the public leaderboard as a tuning oracle through excessive submissions.
- Keep Run003 stable as the primary final candidate unless a validated method clearly beats it.

## 1. Current Baseline and Plateau

Current best public score: `0.64574`.

Current best submission:

- Run: Run003
- File: `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv`
- Method: start from Run001, then replace only rows where Run001 predicted `0.0` and Run002/iter003 produced a nonzero value

Run003 worked because Run001 was the strongest single strategy but had many failure-like `0.0` predictions. Run002 was weaker globally, but it often produced executable programs for rows where Run001 failed. The hybrid kept Run001's nonzero predictions and used Run002 only as a fallback source.

Later experiments showed a plateau:

- Run004 changed only a few additional rows and tied Run003.
- Run005 applied conservative numeric post-processing to 14 rows and lowered the score to `0.64170`.
- Run006 increased context length but produced too many zeros.
- Run007 changed only 2 rows and was not submitted because the evidence was weak.

The conclusion is that future gains need a stronger new signal. Simple fallback patching is mostly exhausted. The next improvement must recover failed rows with higher confidence, repair invalid programs, route examples to genuinely better specialists, or use better context/modeling under the assignment rules.

## 2. Improvement Options Summary Table

| Option | Main Idea | Expected Upside | Difficulty | Realistic Level | Compute Cost | Risk | Recommended Priority |
|---|---|---:|---|---|---|---|---:|
| Targeted retry on remaining zero/failure rows | Retry only Run003 zero rows and accept validated nonzero answers. | Low to medium | Medium | Highly realistic | Low | Low | 1 |
| Multi-sample self-consistency for hard rows | Generate several programs and accept only strong agreement. | Medium | Hard | Realistic | Medium | Medium | 2 |
| Program repair and re-execution | Fix malformed DSL outputs and re-run evaluator. | Low to medium | Medium | Highly realistic | Low | Low to medium | 3 |
| Type-aware specialist ensemble | Route by operation type to specialist strategies only when dev evidence supports it. | Medium | Medium | Realistic | Low to medium | Medium | 4 |
| Dynamic few-shot retrieval by question type | Select similar examples per test row to improve program generation. | Medium | Medium | Realistic | Medium | Medium | 5 |
| Better context/table compression | Extract relevant numbers and table rows before prompting. | Medium to high | Hard | Possible but uncertain | Medium | Medium | 6 |
| New EvoAgent evolution run focused on addition/subtraction | Optimize strategies for weak operation types. | Medium | Medium | Realistic | Medium to high | Medium | 7 |
| Train a lightweight selector/ranker over candidate predictions | Learn when to trust each candidate source from dev results. | Medium | Hard | Possible but uncertain | Low | High if overfit | 8 |
| Stronger model or hosted API path if rules allow | Use stronger allowed models for retry/verification. | Medium to high | Medium to Very Hard | Possible but uncertain | Medium to high | Rule-dependent | 9 |
| Fine-tuning or preference-tuning on train/dev generated programs | Train a better program generator or ranker. | High | Very Hard | Possible but uncertain | High | High | 10 |
| Tool-augmented financial table parser | Build deterministic extraction for tables, rows, years, and values. | High | Very Hard | Possible but uncertain | Medium | Medium | 11 |
| Full agentic solver with verification loop | Propose, execute, verify, and retry with feedback. | High | Very Hard | Possible but uncertain | High | Medium to high | 12 |
| Human-readable error taxonomy + automatic rule miner | Mine dev failures and propose safe targeted fixes. | Low to medium | Hard | Possible but uncertain | Low | Medium | 13 |
| Private-leaderboard robustness strategy | Choose final candidates with lower overfit risk. | Low to medium | Medium | Realistic | Low | Medium | 14 |

## 3. Detailed Option Plans

### 1. Targeted Retry on Remaining Zero/Failure Rows

- What it is: rerun only the rows where Run003 predicts `0.0`, with a stricter prompt and multiple attempts.
- Why it could improve score: these rows are the cleanest unresolved failure cases; any valid recovery can improve without changing known-good nonzero predictions.
- How to implement: create a Phase 3-only retry script that loads Run003, filters zero rows, builds stricter DSL-only prompts, executes candidate programs, and records accepted answers.
- Files/scripts likely involved: `assignment03/phase3_retry_failures.py`, `assignment03/phase3_build_retry_hybrid.py`, `assignment03/src/evaluator.py`, `assignment03/src/model.py`, `assignment03/runs/kaggle_hybrid_001_002/submission_details.json`.
- Expected artifacts: `runs/kaggle_retry_run008/retry_details.json`, `runs/kaggle_retry_run008/retry_predictions.json`, `runs/kaggle_hybrid_retry_run008/submission_checked.csv`, `runs/kaggle_hybrid_retry_run008/changes.csv`, `runs/kaggle_hybrid_retry_run008/summary.json`.
- Validation method: check 494 rows, exact test ID order, no duplicate IDs, no missing values, all numeric outputs, changed-row count, final zero count, and manual inspection of program validity without inferring labels.
- Difficulty: Medium.
- Realistic level: Highly realistic.
- Expected score upside estimate: small but meaningful, roughly `+0.005` to `+0.03` if several zero rows are true failures.
- Main risk: recovered nonzero answers may be confidently wrong.
- When to stop: stop if fewer than 3 useful rows are recovered or candidates disagree heavily.

### 2. Multi-Sample Self-Consistency for Hard Rows

- What it is: generate multiple candidate DSL programs per hard row and choose only when outputs agree strongly.
- Why it could improve score: one greedy output can fail from a bad extraction path; multiple samples may reveal a stable correct program.
- How to implement: sample 5-10 completions per target row with controlled temperature, execute all valid programs, cluster numeric answers, and accept only if a cluster has strong support and reasonable magnitude.
- Files/scripts likely involved: a new Phase 3 retry script, `assignment03/src/model.py`, `assignment03/src/evaluator.py`, `assignment03/format_submission.py`.
- Expected artifacts: per-row candidate logs, selected-program summaries, rejected-candidate summaries, hybrid CSV.
- Validation method: tune thresholds on dev-style failures, then apply unchanged to test candidates.
- Difficulty: Hard.
- Realistic level: Realistic.
- Expected score upside estimate: medium, roughly `+0.01` to `+0.05` if hard-row agreement is predictive.
- Main risk: sampled programs may agree on the same wrong number from a shared prompt bias.
- When to stop: stop if agreement clusters are rare or public score does not improve after one serious attempt.

### 3. Program Repair and Re-Execution

- What it is: recover valid DSL programs from malformed outputs before defaulting to `0.0`.
- Why it could improve score: many `0.0` cases are not natural zero answers; they are invalid extraction, looping text, unsupported operators, quoted table names, or broken references.
- How to implement: detect malformed DSL, extract candidate function calls, remove stray natural language, fix simple reference errors, optionally ask the model to repair only the failed program, then re-execute.
- Files/scripts likely involved: `assignment03/phase3_retry_failures.py`, `assignment03/src/evaluator.py`, `assignment03/src/model.py`, existing submission details JSON files.
- Expected artifacts: repaired-program JSON, repair reason codes, before/after execution values, hybrid CSV.
- Validation method: replay repair rules on dev eval outputs where gold answers are known, measure whether repairs improve accuracy before applying to test failures.
- Difficulty: Medium to Hard.
- Realistic level: Highly realistic for small gains, possible for larger gains.
- Expected score upside estimate: `+0.005` to `+0.04`.
- Main risk: repair rules can turn invalid text into a valid but wrong program.
- When to stop: stop if dev repair precision is low or repairs mostly produce one-off uncertain values.

### 4. Type-Aware Specialist Ensemble

- What it is: use different strategies for different operation types, such as table operations, addition, subtraction, or division.
- Why it could improve score: iter003 had better table_op dev accuracy than iter001, but worse overall. That suggests specialists can help if routed narrowly.
- How to implement: classify test questions by predicted operation type using question text and generated program hints, then switch from Run003 only when a specialist has strong dev evidence and a safe output.
- Files/scripts likely involved: candidate submission CSVs, submission details JSONs, `assignment03/src/executor.py` for `classify_question_type`, new Phase 3 ensemble builder.
- Expected artifacts: route decisions, per-type change report, hybrid submission, validation summary.
- Validation method: simulate routing on dev eval records and require type-level precision above a threshold before applying to test.
- Difficulty: Medium.
- Realistic level: Realistic but needs careful gating.
- Expected score upside estimate: `+0.01` to `+0.05`.
- Main risk: test question type classification may be wrong; broad specialist replacement already hurt in Run002.
- When to stop: stop if route decisions change many nonzero Run003 rows without strong dev evidence.

### 5. Dynamic Few-Shot Retrieval by Question Type

- What it is: choose few-shot examples dynamically for each test row based on question type, wording, and table structure.
- Why it could improve score: static few-shot examples do not cover all financial question forms, especially addition/subtraction/table_op variants.
- How to implement: index train/dev examples by operation type and lexical similarity, retrieve 2-4 close examples, and inject them into the prompt for targeted retry or new submission generation.
- Files/scripts likely involved: `assignment03/data/train.json`, `assignment03/data/dev.json`, `assignment03/src/data.py`, `assignment03/src/executor.py`, a new Phase 3 retrieval prompt script.
- Expected artifacts: retrieval logs, per-row prompt snapshots, candidate details JSON, submission CSV.
- Validation method: use dev questions as held-out queries and compare dynamic few-shot output against static strategy output.
- Difficulty: Medium.
- Realistic level: Realistic.
- Expected score upside estimate: `+0.01` to `+0.04`.
- Main risk: retrieved examples can bias the model toward the wrong operation or unit.
- When to stop: stop if prompt length grows too much or dev proxy accuracy does not improve.

### 6. Better Context/Table Compression

- What it is: preprocess the passage and table to show only relevant rows, columns, years, and nearby text.
- Why it could improve score: long financial contexts can distract the model and cause wrong-row extraction or repetitive Chain-of-Thought loops.
- How to implement: detect years/entities/keywords in the question, score table rows and text spans, build a compact context, and feed that to the model for retry or full prediction.
- Files/scripts likely involved: `assignment03/src/data.py`, `assignment03/src/executor.py`, new context-compression utility, `assignment03/submit.py` if promoted beyond Phase 3 scripts.
- Expected artifacts: compressed-context debug files, candidate predictions, row-selection explanations.
- Validation method: compare compressed vs full context on dev examples by type, especially addition, subtraction, and table_op.
- Difficulty: Hard.
- Realistic level: Possible but uncertain.
- Expected score upside estimate: `+0.02` to `+0.08` if wrong-context extraction is a major failure source.
- Main risk: compression can remove the needed number or table row.
- When to stop: stop if dev examples fail because evidence was removed.

### 7. New EvoAgent Evolution Run Focused on Addition/Subtraction

- What it is: run a new evolution loop optimized for weak operation categories rather than overall accuracy.
- Why it could improve score: iter001 remained weak on addition and table_op, and subtraction was only moderate. A focused run may create a useful specialist.
- How to implement: select training/dev subsets dominated by addition and subtraction, adjust reflection priority, propose strategies focused on direct extraction and operation choice, and generate a specialist Kaggle submission.
- Files/scripts likely involved: `assignment03/src/harness.py`, `assignment03/src/self_reflector.py`, `assignment03/src/self_proposer.py`, `assignment03/src/executor.py`, ARC run scripts.
- Expected artifacts: new `runs/exp_*` directory, strategy history, eval files, specialist submission CSV.
- Validation method: require dev improvement on target types and hybrid only into rows that are likely target-type failures.
- Difficulty: Medium.
- Realistic level: Realistic.
- Expected score upside estimate: `+0.01` to `+0.05`.
- Main risk: a specialist may improve target dev categories but generalize poorly to test.
- When to stop: stop if target-type dev gain does not exceed current strategy variance.

### 8. Train a Lightweight Selector/Ranker Over Candidate Predictions

- What it is: train a small model or rule-based ranker that chooses between candidate outputs from Run001, Run002, Run004, retries, and repairs.
- Why it could improve score: the hard part is not only generating candidates; it is deciding when to trust them.
- How to implement: build dev features from question text, predicted program, execution validity, output magnitude, source strategy, and operation type; train a simple logistic regression, decision tree, or calibrated rules.
- Files/scripts likely involved: dev eval JSONs, submission details JSONs, `assignment03/src/executor.py`, new Phase 3 selector script.
- Expected artifacts: selector training report, feature importance, chosen-candidate CSV, rejection log.
- Validation method: cross-validate on dev records and compare against simple hybrid baselines.
- Difficulty: Hard.
- Realistic level: Possible but uncertain.
- Expected score upside estimate: `+0.01` to `+0.06`.
- Main risk: dev set is small and selector overfitting is likely.
- When to stop: stop if cross-validation gain disappears or the selected changes look broad and unstable.

### 9. Stronger Model or Hosted API Path If Rules Allow

- What it is: use a stronger allowed model for retry, verification, or program repair.
- Why it could improve score: current failures often require careful financial extraction and program synthesis, where a stronger model may be more reliable.
- How to implement: first confirm assignment/Kaggle rules; if allowed, run stronger model only on hard rows or as a verifier to control cost and reduce broad behavior shifts.
- Files/scripts likely involved: `assignment03/src/model.py` if adding a new backend, or a Phase 3-only external inference adapter if rules allow.
- Expected artifacts: model/backend notes, reproducible commands, candidate details JSON, hybrid CSV.
- Validation method: compare on dev failures and require clear improvement before generating test candidate changes.
- Difficulty: Medium to Very Hard depending on model path.
- Realistic level: Possible but rule-dependent.
- Expected score upside estimate: `+0.02` to `+0.10`.
- Main risk: rule violation, unreproducibility, cost, or overreliance on public leaderboard feedback.
- When to stop: stop immediately if rules do not allow it or if reproducibility cannot be documented.

### 10. Fine-Tuning or Preference-Tuning on Train/Dev Generated Programs

- What it is: train a small model or adapter to generate better DSL programs or rank candidate programs.
- Why it could improve score: supervised train/dev programs provide direct examples of the desired output language.
- How to implement: prepare instruction-program pairs, train an adapter or preference ranker, evaluate on dev, then generate candidates for test under the same no-leakage rules.
- Files/scripts likely involved: `assignment03/data/train.json`, `assignment03/data/dev.json`, training scripts outside core graders, model config files.
- Expected artifacts: training data manifest, model checkpoint outside the repo, eval report, prediction details JSON.
- Validation method: held-out dev accuracy by type, invalid-program rate, and zero/fallback count.
- Difficulty: Very Hard.
- Realistic level: Possible but uncertain.
- Expected score upside estimate: medium to high, but highly uncertain.
- Main risk: time, compute, overfitting, and checkpoint-management complexity.
- When to stop: stop if setup consumes too much time before producing dev gains.

### 11. Tool-Augmented Financial Table Parser

- What it is: build deterministic tools for table lookup, year/entity matching, row/column selection, and unit normalization.
- Why it could improve score: many financial QA errors are table extraction errors rather than arithmetic errors.
- How to implement: parse tables into structured records, map question mentions to rows/columns/years, expose candidate numbers to the model, and execute arithmetic with less free-form extraction.
- Files/scripts likely involved: `assignment03/src/data.py`, new table parser utilities, new Phase 3 solver scripts.
- Expected artifacts: extracted table candidates, row/column match logs, candidate programs, validation reports.
- Validation method: dev table_op and subtraction/addition cases where the needed numbers are in tables.
- Difficulty: Very Hard.
- Realistic level: Possible but uncertain.
- Expected score upside estimate: high if table extraction is a dominant failure source.
- Main risk: table formats vary; deterministic matching can be brittle.
- When to stop: stop if parser rules become too manual or test-specific.

### 12. Full Agentic Solver with Verification Loop

- What it is: an agent loop where the model proposes a program, the executor runs it, a verifier checks wording/units/magnitude, and the solver retries with targeted feedback.
- Why it could improve score: it directly addresses invalid programs, wrong units, wrong denominators, and wrong table rows.
- How to implement: define a bounded retry loop, use evaluator output as tool feedback, ask the model to revise only failed candidates, and accept only verified answers.
- Files/scripts likely involved: new Phase 3 agentic solver, `assignment03/src/evaluator.py`, `assignment03/src/model.py`, `assignment03/src/executor.py`.
- Expected artifacts: per-row solver trace, executed candidate list, verifier decision, final hybrid CSV.
- Validation method: run on dev failures and measure whether verifier improves precision without rejecting too much.
- Difficulty: Very Hard.
- Realistic level: Possible but high engineering effort.
- Expected score upside estimate: medium to high.
- Main risk: verifier may be another model with the same blind spots; runtime can grow quickly.
- When to stop: stop if traces are hard to audit or the verifier accepts too many wrong revisions.

### 13. Human-Readable Error Taxonomy + Automatic Rule Miner

- What it is: classify dev failures into recurring patterns and mine narrow, testable transformation rules.
- Why it could improve score: Run005 showed broad rules hurt, but a taxonomy can still find safer micro-rules.
- How to implement: label dev failures by error type, generate candidate rules, test each rule on dev before applying it to test candidates.
- Files/scripts likely involved: dev eval JSONs, `assignment03/runs/exp_self_arc/failure_mode_report.txt`, post-processing scripts.
- Expected artifacts: error taxonomy, rule precision table, accepted/rejected rule report.
- Validation method: require high dev precision for any rule and keep changed-row count small.
- Difficulty: Hard.
- Realistic level: Possible but uncertain.
- Expected score upside estimate: low to medium.
- Main risk: public/test distribution may not match dev failure patterns.
- When to stop: stop if candidate rules resemble broad Run005 rules or change too many rows.

### 14. Private-Leaderboard Robustness Strategy

- What it is: choose final submissions based on simplicity, public score, change count, and overfit risk.
- Why it could improve final outcome: the best public score may not always be the best private score if a candidate is overfit or unstable.
- How to implement: keep Run003 primary because it is simple and strongly improved over baseline; keep Run004 as alternate only if multiple final choices are allowed; avoid submitting unstable micro-hybrids.
- Files/scripts likely involved: `docs/PHASE3_EXPERIMENT_LOG.md`, final ThinkFlic package files.
- Expected artifacts: final candidate rationale and submission information.
- Validation method: compare public score, method simplicity, changed-row counts, and dev rationale.
- Difficulty: Medium.
- Realistic level: Realistic.
- Expected score upside estimate: not a public-score increase, but can reduce final leaderboard risk.
- Main risk: choosing too conservatively may leave small gains unused.
- When to stop: stop when the team freezes Kaggle and packages the final submission.

## 4. Highest-Upside Advanced Methods

### A. Self-Consistency with Execution Verification

Generate multiple candidate DSL programs for each hard test row. Execute every candidate with the existing evaluator, discard invalid programs, and group the remaining numeric answers by closeness. Prefer answers that have agreement, valid execution, and reasonable magnitude.

This should be tuned on dev failures before touching test predictions. The safest usage is to replace Run003 only when confidence is high, such as when at least 3 of 5 candidates execute to the same value or a tight numeric cluster.

Difficulty: Hard.

Realistic level: Realistic to possible.

Why this could beat `0.64574`: Run003 has already exhausted simple fallback gains, but self-consistency can recover a new signal from the same model by reducing one-sample variance. It is especially attractive for the remaining zero rows and malformed-program failures.

### B. Program Repair Pipeline

Many `0.0` rows likely come from invalid or malformed programs rather than true zero answers. A repair pipeline should detect failed DSL outputs, extract valid function-call fragments, repair syntax or argument references, and re-execute the repaired program. Model-assisted repair can be used, but only for rows that already failed.

Difficulty: Medium to Hard.

Realistic level: Highly realistic for small gains, possible for larger gains.

Why this could beat `0.64574`: it directly targets the failure mode that made Run003 useful in the first place, while avoiding broad changes to nonzero rows.

### C. Type-Aware Specialist Ensemble

Train or select specialist strategies for addition, subtraction, table_op, and division. Use dev accuracy by type to decide where each specialist is trustworthy. Route test rows carefully using question text, generated programs, and output validity.

Naive median or broad ensembling should be avoided. Run002 showed that a strategy can reduce zeros while still lowering score, and Run005 showed that plausible numeric corrections can hurt. Any specialist routing must be narrow and evidence-backed.

Difficulty: Medium.

Realistic level: Realistic but needs careful gating.

Why this could beat `0.64574`: iter003's table_op strength shows that specialist behavior exists, but it must be applied only where it is genuinely complementary.

### D. Retrieval and Table Compression

Extract relevant numbers, years, entities, and table rows before prompting. The goal is to reduce context noise and make the required arithmetic path more obvious. This could help addition/subtraction questions and table operations, where wrong-number extraction is often the core error.

Difficulty: Hard.

Realistic level: Possible but uncertain.

Why this could beat `0.64574`: if current failures are caused by irrelevant context or table confusion, compression gives the model a cleaner problem without changing the scoring format.

### E. Stronger Model Path

First check the assignment and Kaggle rules. If larger hosted/API models are allowed and reproducible, use them only for retry, verification, or repair of hard rows. If hosted/API paths are not allowed, consider stronger open-weight models on ARC within the same reproducibility constraints.

Hidden labels, manual test-answer inference, and non-reproducible private outputs are not acceptable.

Difficulty: Medium to Very Hard depending on the model.

Realistic level: Possible but rule-dependent.

Why this could beat `0.64574`: stronger models may reduce extraction mistakes and malformed DSL, especially when used narrowly on hard rows.

### F. Full Verification Loop / Agentic Solver

Build a bounded solver loop:

1. The model proposes a DSL program.
2. The executor runs it.
3. A verifier checks whether the answer matches the question wording, units, magnitude, and table evidence.
4. If verification fails, the solver retries with targeted feedback.
5. The loop stops after a small fixed number of attempts.

Difficulty: Very Hard.

Realistic level: Possible but high engineering effort.

This is the most ambitious long-term path. It turns EvoAgent from a one-shot program generator into a tool-using financial QA solver with execution feedback.

## 5. Recommended Roadmap by Time Budget

### If We Have 2 Hours

Do the safest incremental work:

- Implement targeted retry for Run003 zero rows.
- Add simple program repair for failed rows.
- Build a fallback hybrid only if recovered answers are valid, finite, nonzero, and not extreme.
- Submit only if at least 3 useful rows are recovered and validation passes.

### If We Have 1 Day

Move from single retry to controlled self-consistency:

- Generate 5-10 candidate programs for Run003 zero and malformed rows.
- Add candidate clustering and agreement thresholds.
- Build a dev-calibrated selector.
- Produce Run008 or Run009 as a narrow hybrid candidate.

### If We Have 3-5 Days

Build complementary specialists:

- Create a type-aware specialist ensemble.
- Run addition/subtraction-focused EvoAgent evolution.
- Add dynamic few-shot retrieval by operation type and wording.
- Use dev eval records to gate any replacement of Run003 nonzero predictions.

### If We Have 1-2 Weeks

Invest in better problem representation:

- Add context/table compression.
- Combine program repair with a verifier loop.
- Test a stronger model verification path if assignment rules allow it.
- Produce a more robust private-leaderboard strategy instead of chasing public-score micro-gains.

### If We Have Unlimited Time

Build a full financial QA system:

- Tool-augmented financial table parser.
- Retrieval plus table compression.
- DSL solver with execution and verifier feedback.
- Fine-tuned program generator.
- Learned candidate selector/ranker.
- Full agentic solver with auditable traces.

## 6. Practical Recommendation

The next serious score attempt should still be narrow: targeted retry plus execution verification for Run003 zero rows. This is the best balance of upside, implementation speed, and risk control.

If that does not beat `0.64574` after one or two attempts, freeze Kaggle with Run003 as the primary final candidate and complete the ThinkFlic package. Longer-term ideas are worth documenting, but they should not destabilize a valid final submission.
