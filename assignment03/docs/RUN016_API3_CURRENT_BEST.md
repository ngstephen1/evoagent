# Run016 API3 Current Best

Run016 records the current shared-team best Kaggle submission.

Current best:

- File: `submission_ensemble_api3.csv`
- Public score: `0.80161`
- Local path: `assignment03/runs/team_melanie_ensemble_api3/submission_checked.csv`
- Source branch: `origin/melanie-evoagent-arc`
- Source commit: `714dc9d`
- Method: three-way ensemble of RS Qwen3-8B, Gemini 2.5 Flash, and DeepSeek-V3,
  with deterministic voting and priority/tie-breaking toward RS Qwen3-8B.
- Best self-hosted alternate: `submission_ft_qwen3_8b_rs.csv`, public score
  `0.74696`.

Validation status:

- 494 rows.
- Columns: `id, Usage, predicted_value`.
- Exact `data/test.json` ID order.
- No duplicate IDs.
- No missing, non-numeric, or non-finite predictions.

Important:

- `final_submission.csv` in Melanie's scaffold does not match
  `submission_ensemble_api3.csv` by hash. Copy API3 explicitly into the final
  package.
- No external web search, source PDF lookup, filenames, URLs, or hidden labels
  are used for answer recovery. API models are used only as reproducible solvers
  over assignment-provided context and the existing DSL/evaluator pipeline.

Recommended next action:

1. Freeze `submission_ensemble_api3.csv` as the primary final candidate.
2. Keep `submission_ft_qwen3_8b_rs.csv` as the self-hosted alternate.
3. Update ThinkFlic package `kaggle/final_submission.csv` from API3.
4. Document API model identifiers, prompt/evaluator reuse, vote rule, and saved
   prediction CSVs in the final report.
5. Do not spend more A100 hours unless attempting self-hosted follow-up training.
