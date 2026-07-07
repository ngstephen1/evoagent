# Run014 GPT-OSS on Team Ensemble V4

Status: superseded. Melanie's fine-tuned Qwen3-8B rejection-sampled submission
now scores `0.74696`, so new GPT-OSS correction work should use
`assignment03/runs/team_melanie_ft_qwen3_8b_rs/submission_checked.csv` as the
base instead of ensemble v4.

Run014 starts from the previous team-best Kaggle submission and uses GPT-OSS
120B only as a narrow correction layer. Keep this file as a fallback reference
for command structure, not as the active plan.

Current best:

- File: `submission_ensemble_v4.csv`
- Public score: `0.70242`
- Target local path:
  `assignment03/runs/team_melanie_ensemble_v4/submission_checked.csv`
- Artifact status: present on `origin/melanie-evoagent-arc` commit `cdbc480`
  and copied locally to the target path. The scaffold's
  `submission_information.txt` is stale, so validate the CSV directly.

Do not use external search, source PDFs, filenames, URLs, page metadata, or
hidden labels for answer recovery.

## 1. Prepare V4 Locally

Copy the team-best file into the expected local path if it is missing:

```bash
cd /Users/macbook/Hack/evoagent

mkdir -p assignment03/runs/team_melanie_ensemble_v4

git fetch origin melanie-evoagent-arc

git show 'origin/melanie-evoagent-arc:A3_MelanieAndStephen_<StudentID1>_<StudentID2>/kaggle/submission_ensemble_v4.csv' \
  > assignment03/runs/team_melanie_ensemble_v4/submission_checked.csv
```

Do not substitute the older committed `submission_ensemble.csv`; that file is
associated with the `0.69433` run, not the `0.70242` v4 run.

Validate:

```bash
cd /Users/macbook/Hack/evoagent/assignment03

python3 - <<'PY'
import csv, json, math
sub = "runs/team_melanie_ensemble_v4/submission_checked.csv"
rows = list(csv.DictReader(open(sub, encoding="utf-8-sig")))
expected = [r["id"] for r in json.load(open("data/test.json", encoding="utf-8"))]
ids = [r["id"] for r in rows]
bad = []
vals = []
for r in rows:
    try:
        v = float(r["predicted_value"])
        if not math.isfinite(v):
            bad.append(r["id"])
        vals.append(v)
    except Exception:
        bad.append(r["id"])
print("rows", len(rows))
print("id_order_exact", ids == expected)
print("duplicate_ids", len(ids) - len(set(ids)))
print("bad_values", len(bad), bad[:5])
print("zero_count", sum(v == 0 for v in vals))
print("negative_count", sum(v < 0 for v in vals))
print("extreme_abs_gt_1e6", sum(abs(v) > 1e6 for v in vals))
print("VALID", len(rows) == 494 and ids == expected and len(set(ids)) == len(ids) and not bad)
PY
```

## 2. Sync to ARC

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
  assignment03/runs/team_melanie_ensemble_v4/ \
  "$VT_PID@$LOGIN_HOSTNAME.arc.vt.edu:$REMOTE_DIR/assignment03/runs/team_melanie_ensemble_v4/"
```

## 3. Run on ARC

Confirm GPT-OSS 120B is served by SGLang:

```bash
cd /home/$VT_PID/temp/evoagent/assignment03
export PYTHON_BIN=/home/$VT_PID/.conda/envs/evoagent/bin/python

curl -s --max-time 10 http://127.0.0.1:30000/v1/models | head -40
```

Build targets:

```bash
$PYTHON_BIN phase3_select_run010_targets.py \
  --base-submission runs/team_melanie_ensemble_v4/submission_checked.csv \
  --test data/test.json \
  --output runs/kaggle_run014_gptoss_on_team_v4/target_rows.csv \
  --max-targets 100
```

Run GPT-OSS evidence-to-DSL:

```bash
$PYTHON_BIN phase3_gptoss_evidence_solver.py \
  --base-submission runs/team_melanie_ensemble_v4/submission_checked.csv \
  --base-details runs/kaggle_hybrid_001_002/submission_details.json \
  --target-rows runs/kaggle_run014_gptoss_on_team_v4/target_rows.csv \
  --output-dir runs/kaggle_run014_gptoss_on_team_v4 \
  --server-url http://127.0.0.1:30000/v1/chat/completions \
  --model openai/gpt-oss-120b \
  --limit-targets 100 \
  --temperature 0 \
  --max-tokens 2048
```

Build variants with Run014 names:

```bash
$PYTHON_BIN phase3_build_run013_variants.py \
  --base-submission runs/team_melanie_ensemble_v4/submission_checked.csv \
  --retry-details runs/kaggle_run014_gptoss_on_team_v4/retry_details.json \
  --target-rows runs/kaggle_run014_gptoss_on_team_v4/target_rows.csv \
  --test data/test.json \
  --output-root runs \
  --output-prefix kaggle_hybrid_run014_gptoss_on_team_v4
```

Inspect before any submission:

```bash
cat runs/kaggle_hybrid_run014_gptoss_on_team_v4_zero_only/summary.json
head -80 runs/kaggle_hybrid_run014_gptoss_on_team_v4_zero_only/changes.csv

cat runs/kaggle_hybrid_run014_gptoss_on_team_v4_zero_sign/summary.json
head -80 runs/kaggle_hybrid_run014_gptoss_on_team_v4_zero_sign/changes.csv

cat runs/kaggle_hybrid_run014_gptoss_on_team_v4_high_conf/summary.json
head -120 runs/kaggle_hybrid_run014_gptoss_on_team_v4_high_conf/changes.csv
```

## 4. Submission Rule

Submit only one conservative variant at a time:

- Prefer `zero_only` or `zero_sign` if it has clean auditable changes.
- Treat `high_conf` as risky unless every changed row is inspected.
- Stop if the first Run014 submission scores below `0.70242`.
