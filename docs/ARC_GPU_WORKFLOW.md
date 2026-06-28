# VT ARC GPU Workflow

This fork keeps the original Modal workflow intact, but the preferred compute
path for this repository is VT ARC GPU execution on TinkerCliffs A100/H200
nodes. The goal is to run the same assignment code on ARC without committing
secrets, model weights, caches, or generated proof artifacts.

References:

- VT ARC documentation: <https://www.docs.arc.vt.edu/>
- VT ARC GPU resources: <https://www.docs.arc.vt.edu/resources/gpu.html>

## Why ARC Instead of Modal

- VT ARC gives direct access to institutional GPU nodes and allocations.
- Interactive TinkerCliffs jobs are easier to inspect with `nvidia-smi`,
  `squeue`, logs, and shell tools.
- Long-running experiments can be managed in `tmux` without depending on Modal
  volumes.
- Modal remains available in the original assignment files. Do not remove
  `assignment03/run_modal.py`; use it as a reference for equivalent commands.

## Login

Set your PID locally and SSH to the login node:

```bash
export VT_PID=<YOUR_VT_PID>
scripts/arc_login.sh
```

Equivalent raw command:

```bash
ssh -i ~/.ssh/arc "$VT_PID@tinkercliffs1.arc.vt.edu"
```

Do not put passwords, tokens, or private keys in this repository.

## SFTP Setup

Copy the example config into your personal VS Code SFTP config:

```bash
cp .vscode/sftp.json.example .vscode/sftp.json
```

Edit only your local `.vscode/sftp.json`:

- `username`: your VT PID.
- `remotePath`: for example `/home/<YOUR_VT_PID>/temp/evoagent`.
- `privateKeyPath`: keep as `~/.ssh/arc` if that is your ARC key.
- `uploadOnSave`: keep `false` unless you intentionally want uploads on every
  file save.

The example ignores `.git`, virtual environments, caches, runs, model weights,
checkpoints, archives, logs, outputs, and `.vscode` itself. Keep it conservative;
accidental uploads can be slow and can pollute the ARC workspace.

The committed file is only `.vscode/sftp.json.example`. Your real
`.vscode/sftp.json` is ignored by Git because it may contain personal usernames,
remote paths, and machine-specific settings.

## Interactive A100 Job

On TinkerCliffs:

```bash
cd /home/<YOUR_VT_PID>/temp/evoagent
scripts/arc_interact_a100.sh
```

Defaults:

```bash
ALLOCATION_ID=llms-lab
PARTITION=a100_normal_q
QOS=tc_a100_normal_short
NUM_GPUS=1
NUM_HOURS=1
NUM_CPUS_PER_TASK=16
```

Equivalent raw request:

```bash
interact -A llms-lab \
  --partition a100_normal_q \
  --qos tc_a100_normal_short \
  --cpus-per-task 16 \
  --time=1:00:00 \
  --gres=gpu:1 \
  --verbose
```

For longer runs, override environment variables before calling the script:

```bash
export NUM_HOURS=4
scripts/arc_interact_a100.sh
```

## tmux Workflow

Use `tmux` so long jobs survive terminal disconnects:

```bash
tmux new -s evoagent
```

Inside tmux, request a GPU job, set up the environment, and run experiments.

Detach:

```bash
Ctrl-b d
```

Reattach:

```bash
tmux attach -t evoagent
```

## Environment Setup

After the interactive allocation starts:

```bash
cd /home/<YOUR_VT_PID>/temp/evoagent
scripts/arc_setup_env.sh
```

The setup helper:

- attempts to load Miniforge3 if ARC exposes it as a module;
- creates or reuses a conda environment named `evoagent`, or falls back to a
  venv under `$HOME/.venvs/evoagent`;
- installs `assignment03/requirements.txt`;
- checks `python3 --version`, Torch CUDA visibility, SGLang import status, and
  `nvidia-smi`.

Important: `assignment03/requirements.txt` intentionally avoids installing the
full GPU runtime because Modal originally supplied SGLang, PyTorch, CUDA, and
Transformers through its container. On ARC, install the GPU stack deliberately
after confirming the module/CUDA runtime you are using.

## Hugging Face Token

Set `HF_TOKEN` only in your shell or job environment:

```bash
export HF_TOKEN=hf_your_token
```

Never commit:

- `.env`
- shell history with tokens
- model cache folders
- downloaded weights
- private keys

## Running Local Graders

Run non-GPU local graders:

```bash
scripts/arc_run_local_graders.sh
```

That script runs from `assignment03/`:

```bash
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
python3 graders/grade_stage4_harness.py
```

Proof-dependent checks require generated JSON files:

```bash
python3 graders/grade_stage0.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

Expected proof files:

- `sandbox_proof.json`
- `smoke_proof.json`
- `evolution_proof.json`

Do not hand-write or hand-edit these files.

## ARC Proof Generation

The adapter below mirrors the proof JSON shapes created by `run_modal.py`, but
runs the existing local code paths on ARC instead of Modal.

```bash
cd /home/<YOUR_VT_PID>/temp/evoagent/assignment03
export HF_TOKEN=hf_your_token

python3 arc_proofs.py sandbox
python3 arc_proofs.py smoke
python3 arc_proofs.py evolution
```

Run everything in sequence:

```bash
cd /home/<YOUR_VT_PID>/temp/evoagent/assignment03
export HF_TOKEN=hf_your_token
python3 arc_proofs.py all
```

Or from the repository root:

```bash
export HF_TOKEN=hf_your_token
scripts/arc_generate_proofs.sh
```

Useful overrides:

```bash
python3 arc_proofs.py evolution \
  --T 5 \
  --train-size 200 \
  --dev-size 240 \
  --output-dir runs/exp_self_arc \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

Outputs are written under `assignment03/` and `assignment03/runs/`:

- `sandbox_proof.json`
- `smoke_proof.json`
- `evolution_proof.json`
- `runs/exp_self_arc/evolution_proof.json`
- `runs/exp_self_arc/history.jsonl`
- `runs/exp_self_arc/iter_best_strategy.json`

Before relying on ARC proof files for final grading, confirm with course staff
that ARC-generated proof files are accepted in place of Modal-generated proof
files. The local graders validate filenames and JSON structure, but the
assignment prose still describes Modal as the official proof path.

## Kaggle Submission on ARC

After an evolution run produces a strategy JSON, generate predictions from
`assignment03/`:

```bash
python3 submit.py \
  --strategy-path ./runs/exp_self_arc/iter_best_strategy.json \
  --output-file ./submission.csv \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --gpu-memory-utilization 0.7
```

If your final pipeline emits predictions another way:

```bash
python3 format_submission.py \
  --predictions my_predictions.csv \
  --output-file submission.csv
```

## Monitoring

Useful commands inside the allocation:

```bash
echo "$SLURM_JOBID"
hostname
nvidia-smi
squeue -u "$USER"
squeue -j "$SLURM_JOBID" -o "%.18i %.2t %.10M %.10L %R"
```

Cancel a job:

```bash
scancel <jobid>
```

## Troubleshooting

### Missing CUDA or No GPU Visible

Symptoms:

- `torch.cuda.is_available()` prints `False`.
- `nvidia-smi` is missing or shows no GPU.

Checks:

```bash
echo "$SLURM_JOBID"
hostname
nvidia-smi
```

Fix: make sure you are inside an interactive GPU allocation, not just on the
login node.

### Missing Hugging Face Token

Symptoms:

- model download fails;
- authentication errors from Hugging Face.

Fix:

```bash
export HF_TOKEN=hf_your_token
```

Do not write the token into scripts or config files.

### Missing Python Environment

Symptoms:

- `ModuleNotFoundError`;
- `python3` points to the system Python.

Fix:

```bash
scripts/arc_setup_env.sh
which python3
python3 --version
```

### SGLang or PyTorch Import Fails

The assignment requirements do not install the heavy GPU stack. Confirm ARC
modules and install SGLang/PyTorch intentionally in the active environment.

### Out of Memory

Try:

- reduce `GPU_MEMORY_UTILIZATION`;
- reduce `--max-model-len`;
- reduce `--max-new-tokens`;
- keep the AWQ model default;
- request a larger GPU partition if available.

### Stuck Job

Check queue state:

```bash
squeue -j <jobid> -o "%.18i %.2t %.10M %.10L %R"
```

Cancel if needed:

```bash
scancel <jobid>
```

### Transport or SFTP Sync Issues

- Keep `uploadOnSave` disabled.
- Check `.vscode/sftp.json` has the correct `remotePath`.
- Avoid syncing `assignment03/runs/`, model caches, and checkpoints.
- Use `rsync` or manual SFTP for large one-time transfers instead of save-time
  sync.
