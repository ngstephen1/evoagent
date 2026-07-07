# Run015 GPT-OSS on Fine-Tuned Qwen3-8B RS

Status: superseded by `submission_ensemble_api3.csv` at `0.80161`.

Run015 starts from the best fully self-hosted shared-team Kaggle submission and
uses GPT-OSS 120B only as a narrow correction layer. Keep this as a fallback
plan if hosted-API submissions are disallowed or if the team wants to improve
the self-hosted alternate.

Current best:

- File: `submission_ft_qwen3_8b_rs.csv`
- Public score: `0.74696`
- Target local path:
  `assignment03/runs/team_melanie_ft_qwen3_8b_rs/submission_checked.csv`
- Artifact status: present on `origin/melanie-evoagent-arc` commit `f1f7679`,
  copied locally, and validated directly.

Do not use external search, source PDFs, filenames, URLs, page metadata, or
hidden labels for answer recovery.

## Sync to ARC

```bash
cd /Users/macbook/Hack/evoagent

export VT_PID=stephenallstar24
export LOGIN_HOSTNAME=tinkercliffs1
export REMOTE_DIR=/home/$VT_PID/temp/evoagent

rsync -avz \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".DS_Store" \
  --exclude ".vscode/sftp.json" \
  --exclude "assignment03/.env" \
  --exclude "assignment03/runs/" \
  --exclude "*.safetensors" \
  --exclude "*.bin" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  --exclude "*.ckpt" \
  --exclude "*.zip" \
  -e "ssh -i ~/.ssh/arc" \
  ./ "$VT_PID@$LOGIN_HOSTNAME.arc.vt.edu:$REMOTE_DIR/"

rsync -avz \
  -e "ssh -i ~/.ssh/arc" \
  assignment03/runs/team_melanie_ft_qwen3_8b_rs/ \
  "$VT_PID@$LOGIN_HOSTNAME.arc.vt.edu:$REMOTE_DIR/assignment03/runs/team_melanie_ft_qwen3_8b_rs/"
```

## Run on ARC

Confirm GPT-OSS 120B is served by SGLang:

```bash
cd /home/$VT_PID/temp/evoagent/assignment03
export PYTHON_BIN=/home/$VT_PID/.conda/envs/evoagent/bin/python

curl -s --max-time 10 http://127.0.0.1:30000/v1/models | head -40
```

Build targets:

```bash
$PYTHON_BIN phase3_select_run010_targets.py \
  --base-submission runs/team_melanie_ft_qwen3_8b_rs/submission_checked.csv \
  --test data/test.json \
  --output runs/kaggle_run015_gptoss_on_ft_qwen3_8b_rs/target_rows.csv \
  --max-targets 100
```

Run GPT-OSS evidence-to-DSL:

```bash
$PYTHON_BIN phase3_gptoss_evidence_solver.py \
  --base-submission runs/team_melanie_ft_qwen3_8b_rs/submission_checked.csv \
  --base-details runs/kaggle_hybrid_001_002/submission_details.json \
  --target-rows runs/kaggle_run015_gptoss_on_ft_qwen3_8b_rs/target_rows.csv \
  --output-dir runs/kaggle_run015_gptoss_on_ft_qwen3_8b_rs \
  --server-url http://127.0.0.1:30000/v1/chat/completions \
  --model openai/gpt-oss-120b \
  --limit-targets 100 \
  --temperature 0 \
  --max-tokens 2048
```

Build variants:

```bash
$PYTHON_BIN phase3_build_run013_variants.py \
  --base-submission runs/team_melanie_ft_qwen3_8b_rs/submission_checked.csv \
  --retry-details runs/kaggle_run015_gptoss_on_ft_qwen3_8b_rs/retry_details.json \
  --target-rows runs/kaggle_run015_gptoss_on_ft_qwen3_8b_rs/target_rows.csv \
  --test data/test.json \
  --output-root runs \
  --output-prefix kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs
```

Inspect before any submission:

```bash
cat runs/kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs_zero_only/summary.json
head -80 runs/kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs_zero_only/changes.csv

cat runs/kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs_zero_sign/summary.json
head -80 runs/kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs_zero_sign/changes.csv

cat runs/kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs_high_conf/summary.json
head -120 runs/kaggle_hybrid_run015_gptoss_on_ft_qwen3_8b_rs_high_conf/changes.csv
```

## Submission Rule

Submit only one conservative variant at a time:

- Prefer `zero_only` or `zero_sign` if it has clean auditable changes.
- Treat `high_conf` as risky unless every changed row is inspected.
- Stop if the first Run015 submission scores below `0.74696`.

Recommended compute request:

- `8` A100 hours for the GPT-OSS correction-layer run.
- `24` A100 hours only if reproducing or extending fine-tuning/rejection
  sampling.
