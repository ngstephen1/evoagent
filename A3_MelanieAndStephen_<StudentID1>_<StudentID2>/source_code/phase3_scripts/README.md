# Phase 3 Reproducibility Scripts

This folder contains the Phase 3 scripts available from the final team branch.
They are included in addition to the required `source_code/src/` directory so the
fine-tuning, rejection-sampling, self-consistency, and ensemble path can be
audited.

Included scripts:

- `submit.py`: Kaggle/dev prediction generation.
- `format_submission.py`: submission formatting and validation.
- `run_modal.py`: Modal/ARC workflow entrypoints used by the team.
- `arc_proofs.py`: ARC proof and prediction helper.
- `phase3_ensemble_vote.py`: deterministic majority-vote ensemble.
- `phase3_scale_fix.py`: targeted scale/sign post-processing helper.
- `phase3_dev_eval.py`: local dev-scoring harness.
- `phase3_build_sft_data.py`: supervised fine-tuning dataset builder.
- `train_lora.py`: QLoRA fine-tuning script.
- `merge_lora.py`: LoRA merge script.
- `phase3_rejection_sample.py`: rejection sampling / STaR data generation.

Missing at package time:

- `phase3_api_solve.py` was referenced in the final Phase 3 report but was not
  present on `origin/melanie-evoagent-arc` when this package was assembled
  (latest inspected commit: `714dc9d`, `report update`). The saved API
  prediction artifacts used by the final ensemble are included under
  `evidence/prediction_artifacts/`.

Do not place `.env`, API keys, Hugging Face tokens, Kaggle tokens, or model
weights in this folder.
