# A3 Melanie and Stephen - ThinkFlic Final Submission

This folder is the draft final package for VietAI Advanced NLP06 Assignment 03.
It is not zipped yet because student IDs, final private leaderboard details,
final signatures, and video link still need final confirmation.

## Team

| Member | Student ID | Class | Kaggle ID | Contribution |
|---|---|---|---|---|
| Melanie | TBD | Advanced NLP06 | `yeyeezyzeus` | Phase 3 model upgrade, Qwen fine-tuning, rejection sampling/STaR, API ensemble, report figures. |
| Nguyen Phan Nguyen - Stephen | TBD | Advanced NLP06 | `nguynphannguyn` | EvoAgent implementation, ARC runs, early Kaggle hybrid/retry experiments, packaging and documentation. |

Designated ThinkFlic submitter: TBD.

Video presentation: TBD Google Drive link, set to "Anyone with the link can view".

## Final Kaggle Candidate

| Role | File | Public Score | Decision |
|---|---|---:|---|
| Primary final | `kaggle/final_submission.csv` | 0.80161 | Use for final Kaggle artifact. |
| Strong self-hosted alternate | `assignment03/runs/team_melanie_ft_qwen3_8b_rs/submission_checked.csv` | 0.74696 | Best non-API/self-hosted model result. |
| Previous team ensemble | `assignment03/runs/team_melanie_ensemble_v4/submission_checked.csv` | 0.70242 | Superseded by API3 ensemble. |
| Previous Stephen hybrid | `assignment03/runs/kaggle_hybrid_retry_run009_lite_safe/submission_checked.csv` | 0.65789 | Superseded by team fine-tuned/API ensemble results. |

`kaggle/final_submission.csv` is copied from:

```text
assignment03/runs/team_melanie_ensemble_api3/submission_checked.csv
```

The final Kaggle run was submitted as `submission_ensemble_api3.csv` and scored
`0.80161` on the public leaderboard. It is an API-assisted three-way ensemble
combining rejection-sampled Qwen3-8B, Gemini 2.5 Flash, and DeepSeek-V3
predictions using deterministic voting/tie-breaker logic. All prompts and
candidate answers used only assignment-provided row context; no hidden labels,
external source-document lookup, or manual test labeling were used.

## Package Contents

```text
README.md
report.pdf
report.md
integrity_declaration.pdf
integrity_declaration.md
source_code/
  src/
  phase3_scripts/
  requirements.txt
  run_instructions.md
kaggle/
  final_submission.csv
  submission_information.txt
  submission_ensemble_api3.csv
  submission_ft_qwen3_8b_rs.csv
  submission_ensemble_v4.csv
evidence/
  evolution_proof.json
  failure_mode_report.pdf
  learning_curve.pdf
  strategy_diversity.pdf
  results_progression.png
  results_star_slices.png
  kaggle_public_scores.md
  prediction_artifacts/
```

## Still Missing Before Final ZIP

- Replace `<StudentID1>` and `<StudentID2>` in the folder/ZIP name.
- Fill real student IDs in `README.md`, `integrity_declaration.md`, and
  `kaggle/submission_information.txt`.
- Add final signature confirmation if handwritten signatures are required.
- Add final Kaggle private score/rank after leaderboard close.
- Add Google Drive video link.
- Obtain and include the exact API-generation script if Melanie has a local
  `phase3_api_solve.py`; the saved API prediction artifacts are included, but
  that script was not present on `melanie-evoagent-arc` at packaging time.
