# Run Instructions

These commands reproduce the local grader checks and document the inference path
used for the final Kaggle artifact. Run from the repository root unless noted.

## Environment

```bash
cd assignment03
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

For ARC GPU runs, use the project ARC workflow documentation in
`docs/ARC_GPU_WORKFLOW.md`. Do not commit `.env`, Hugging Face tokens, Kaggle
tokens, model weights, or caches.

## Local Graders

```bash
cd assignment03
python3 graders/grade_stage0.py
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

## Final Kaggle Artifact

The selected final public candidate is Run003:

```text
assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv
```

This package copies it to:

```text
kaggle/final_submission.csv
```

Run003 is a no-code hybrid:

1. Start from Run001 best EvoAgent strategy predictions.
2. For rows where Run001 predicted `0.0`, use Run002 iter003 prediction if it
   is nonzero.
3. Keep all other Run001 predictions.

## ARC Strategy Submission Command

Run001 was generated on ARC with:

```bash
cd /home/<VT_PID>/temp/evoagent/assignment03
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./runs/kaggle_arc_best/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

Run002 was generated similarly with:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_003_strategy.json \
  --output-file ./runs/kaggle_iter003/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

The final hybrid CSV is documented in `kaggle/submission_information.txt`.
