# Run Instructions

These commands document the reproducibility path for the EvoAgent graders and
the final Kaggle artifact. Run from the local assignment workspace unless noted.
Use real API keys only through environment variables or `.env`; never commit
tokens, keys, model weights, or caches.

## Environment

```bash
cd assignment03
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

For ARC GPU runs, use the project ARC workflow documentation in
`docs/ARC_GPU_WORKFLOW.md`.

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

The selected final public candidate is:

```text
assignment03/runs/team_melanie_ensemble_api3/submission_checked.csv
```

This package copies it to:

```text
kaggle/final_submission.csv
```

It was submitted as `submission_ensemble_api3.csv` and scored `0.80161` on the
public leaderboard. The ensemble combines:

1. rejection-sampled fine-tuned Qwen3-8B predictions,
2. Gemini 2.5 Flash predictions,
3. DeepSeek-V3 predictions.

The best self-hosted alternate is `submission_ft_qwen3_8b_rs.csv`, public score
`0.74696`.

## EvoAgent Baseline Reproduction

Run001 baseline generation on ARC:

```bash
cd /home/<VT_PID>/temp/evoagent/assignment03
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./runs/kaggle_arc_best/submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7

python3 format_submission.py \
  --predictions ./runs/kaggle_arc_best/submission.csv \
  --output-file ./runs/kaggle_arc_best/submission_checked.csv
```

## Fine-Tuned Qwen / STaR Path

The final self-hosted path uses QLoRA fine-tuning on assignment training
programs, then rejection sampling / STaR with the DSL executor as verifier.
Representative commands:

```bash
cd assignment03

python3 phase3_build_sft_data.py \
  --train data/train.json \
  --output runs/ft_qwen3_8b/sft_train.jsonl

python3 train_lora.py \
  --model Qwen/Qwen3-8B \
  --train-file runs/ft_qwen3_8b/sft_train.jsonl \
  --output-dir runs/ft_qwen3_8b/lora

python3 merge_lora.py \
  --base-model Qwen/Qwen3-8B \
  --adapter runs/ft_qwen3_8b/lora \
  --output-dir runs/ft_qwen3_8b/merged

python3 phase3_rejection_sample.py \
  --model runs/ft_qwen3_8b/merged \
  --train data/train.json \
  --output runs/ft_qwen3_8b_rs/rejection_samples.jsonl
```

Exact GPU flags, batch sizes, and Modal/ARC launch options are in
`phase3_scripts/run_modal.py`, `phase3_scripts/train_lora.py`, and
`phase3_scripts/phase3_rejection_sample.py`.

## API Prediction Path

The final API3 ensemble used Gemini 2.5 Flash and DeepSeek-V3 predictions
generated over the assignment-provided row context only. API keys must be supplied
externally. Do not use source PDFs, filenames, URLs, hidden labels, or external
search for answer recovery.

The exact `phase3_api_solve.py` script was not present on the final Melanie
branch at package time. The saved API prediction artifacts used for the final
ensemble are included in:

```text
evidence/prediction_artifacts/
```

If the script is recovered, rerun with placeholder environment variables:

```bash
export GEMINI_API_KEY=<set externally>
export DEEPSEEK_API_KEY=<set externally>

python3 phase3_api_solve.py --provider gemini --model gemini-2.5-flash
python3 phase3_api_solve.py --provider deepseek --model deepseek-chat
```

## Ensemble and Validation

The final ensemble is a deterministic vote over saved prediction CSVs:

```bash
python3 phase3_ensemble_vote.py \
  --inputs \
    predict_out/ft-qwen3-8b-rs_test.csv \
    predict_out/api-gemini-gemini-2.5-flash_test.csv \
    predict_out/api-deepseek-deepseek-chat_test.csv \
  --priority predict_out/ft-qwen3-8b-rs_test.csv \
  --output runs/team_melanie_ensemble_api3/submission_checked.csv
```

Validate before any submission:

```bash
python3 - <<'PY'
import csv, json, math
sub = "runs/team_melanie_ensemble_api3/submission_checked.csv"
rows = list(csv.DictReader(open(sub, encoding="utf-8-sig")))
expected = [r["id"] for r in json.load(open("data/test.json", encoding="utf-8"))]
ids = [r["id"] for r in rows]
bad = []
for r in rows:
    try:
        v = float(r["predicted_value"])
        if not math.isfinite(v):
            bad.append(r["id"])
    except Exception:
        bad.append(r["id"])
print("rows", len(rows))
print("id_order_exact", ids == expected)
print("duplicate_ids", len(ids) - len(set(ids)))
print("bad_values", len(bad), bad[:5])
print("VALID", len(rows) == 494 and ids == expected and not bad)
PY
```

Earlier no-code and targeted-retry hybrid runs are documented in
`kaggle/submission_information.txt`.
