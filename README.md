<div align="center">

# EvoAgent — Advanced NLP06 Assignment 03

**Evolutionary self-improving reasoning agent for programmatic QA**

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Assignment](https://img.shields.io/badge/Assignment-03-6366F1)](assignment03/)
[![EvoAgent](https://img.shields.io/badge/EvoAgent-Self--Improving-10B981)](assignment03/README.md)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](assignment03/docs/PHASE3_KAGGLE.md)
[![Modal](https://img.shields.io/badge/Modal-GPU%20Runs-7C3AED)](assignment03/run_modal.py)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface&logoColor=black)](assignment03/requirements.txt)
[![Status](https://img.shields.io/badge/Status-In%20Progress-F59E0B)](#current-implementation-status)

<br />

[![Qwen](https://img.shields.io/badge/Qwen-Inference-111827)](assignment03/src/model.py)
[![SGLang](https://img.shields.io/badge/SGLang-Serving-0EA5E9)](assignment03/run_modal.py)
[![Pydantic](https://img.shields.io/badge/Pydantic-Typed%20Schemas-E92063?logo=pydantic&logoColor=white)](assignment03/requirements.txt)

</div>

---

## Overview

This repository is a team fork for **VietAI Advanced NLP06 Assignment 03**. It implements **EvoAgent**, an evolutionary agent loop that proposes, evaluates, reflects on, and improves reasoning strategies for Vietnamese financial programmatic question answering.

The project combines local graders for staged development with GPU execution for proof generation and Kaggle inference. The original assignment keeps a Modal path, while this fork uses **VT ARC TinkerCliffs** as the preferred compute environment.

## Assignment Milestones

| Milestone | Points | What it requires | Status |
| --- | ---: | --- | --- |
| EvoAgent Implementation | 6 | Complete staged EvoAgent components, run local graders, and generate proof artifacts. | Verified on local graders with ARC proof files |
| Kaggle Competition | 4 | Produce valid hidden-test predictions and compete on the private leaderboard. | Current best Run 003 public score `0.64574` |
| ThinkFlic Final Submission | - | Package source code, report, evidence, Kaggle artifacts, and integrity materials. | Pending |

## Deadlines

| Deliverable | Deadline |
| --- | --- |
| Kaggle competition | July 9, 2026, 23:59 UTC+7 |
| ThinkFlic final submission | July 11, 2026, 23:59 UTC+7 |

## Tech Stack

- **Python 3.10-3.12** for local development, graders, and orchestration.
- **Qwen / QwenInference** for program-generation inference.
- **SGLang** for high-throughput GPU model serving on ARC or Modal.
- **VT ARC GPU** for A100/H200 execution, proof generation, and Kaggle inference in this fork.
- **Modal** remains available through the original assignment workflow.
- **Hugging Face ecosystem** for datasets, model access, and token-authenticated downloads.
- **Pydantic** for typed structured outputs in reflection/proposal workflows.
- **Kaggle** for the final hidden-test competition workflow.

## Repository Structure

```text
.
└── assignment03/
    ├── src/
    │   ├── sandbox.py
    │   ├── executor.py
    │   ├── self_reflector.py
    │   ├── self_proposer.py
    │   ├── harness.py
    │   ├── model.py
    │   ├── evaluator.py
    │   └── strategy.py
    ├── graders/
    ├── docs/
    ├── data/
    ├── runs/
    ├── main.py
    ├── run_modal.py
    ├── submit.py
    ├── format_submission.py
    └── requirements.txt
```

## Quick Start

```bash
cd /Users/macbook/Hack/evoagent/assignment03
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python3 main.py --help
```

The local environment is intended for editing, lightweight checks, and graders. Full model inference should run on a suitable NVIDIA GPU environment such as VT ARC TinkerCliffs.

## VT ARC GPU Workflow

This fork's preferred GPU execution path is **Virginia Tech ARC TinkerCliffs** using A100/H200 GPU nodes. Modal remains available through the original assignment files, especially `assignment03/run_modal.py`, but ARC is the recommended path for interactive inference, proof-generation experiments, and Kaggle runs in this repository.

Start with [`docs/ARC_GPU_WORKFLOW.md`](docs/ARC_GPU_WORKFLOW.md). It covers SSH/SFTP setup, interactive A100 allocation, environment setup, safe Hugging Face token handling, monitoring, proof-generation command mapping, and troubleshooting.

## Local Graders

Run staged checks from `assignment03/`:

```bash
python3 graders/grade_stage0.py
python3 graders/grade_stage1_executor.py
python3 graders/grade_stage2_reflector.py
python3 graders/grade_stage3_proposer.py
PYTHONPATH=. python3 graders/grade_smoke_proof.py
python3 graders/grade_stage4_harness.py
```

`grade_smoke_proof.py` may require `PYTHONPATH=.` when run directly. Proof-dependent graders require their corresponding generated JSON artifacts. In this fork, those proof files were generated on VT ARC and the full Stage 0-4 local grader suite passes.

## Current Implementation Status

| Component | Status | Notes |
| --- | --- | --- |
| Stage 0 Sandbox | Verified | Local implementation and `sandbox_proof.json` pass. |
| Stage 1 Executor | Verified | Token accounting and evaluation loop pass. |
| Stage 2 Reflection | Verified | Self-reflection and fallback parsing pass. |
| Stage 3 Proposal | Verified | DSL validation and dynamic few-shot proposal pass. |
| Stage 4 Harness / Evolution | Verified | Smoke and evolution proof checks pass. |
| Kaggle Run 003 | Submitted | Hybrid Run001 fallback to iter003 nonzero, public score `0.64574`, private score pending. |

## Proof Generation

The original assignment describes Modal as the official GPU proof path. This fork keeps `assignment03/run_modal.py` intact, but uses VT ARC as the preferred path through `assignment03/arc_proofs.py`.

```bash
modal setup
export HF_TOKEN="hf_your-token"
modal secret create huggingface HF_TOKEN=hf_your-token
```

ARC proof generation:

```bash
cd /home/<VT_PID>/temp/evoagent/assignment03
export HF_TOKEN="hf_your_token"
python3 arc_proofs.py all
```

Generated proof files:

| Proof file | Generated by | Purpose |
| --- | --- | --- |
| `sandbox_proof.json` | `arc_proofs.py sandbox` or `run_modal.py::sandbox` | Stage 0 baseline evidence. |
| `smoke_proof.json` | `arc_proofs.py smoke` or `run_modal.py::smoke` | Stage 4 smoke-test evidence. |
| `evolution_proof.json` | `arc_proofs.py evolution` or Modal proof download | Stage 4 final evolution evidence. |

Do **not** hand-write or hand-edit proof JSON files. They should be generated by GPU runs and copied unchanged for grading/submission.

## Kaggle Workflow

Current best Kaggle submission is Run 003:

| Field | Value |
| --- | --- |
| File | `assignment03/runs/kaggle_hybrid_001_002/submission_checked.csv` |
| Description | `Hybrid run001 fallback to iter003 nonzero` |
| Public score | `0.64574` |
| Status | `COMPLETE` |
| Rows | `494` |

Tracked experiment notes live in [`docs/PHASE3_EXPERIMENT_LOG.md`](docs/PHASE3_EXPERIMENT_LOG.md). Earlier results include Run 001 at `0.56477` and Run 002 at `0.47975`.

A typical workflow is:

1. Join the competition from the official assignment link.
2. Generate predictions from the best validated EvoAgent strategy or another documented pipeline.
3. Format predictions into the required CSV schema.
4. Submit `submission.csv` to Kaggle.
5. Record the commit hash, strategy/artifact paths, model settings, and experiment notes.

Useful entrypoints:

```bash
python3 submit.py --strategy-path ./runs/exp_self_arc/iter_best_strategy.json --output-file ./runs/kaggle_arc_best/submission.csv
python3 format_submission.py --predictions my_predictions.csv --output-file submission.csv
```

Public leaderboard scores are interim signals only. Final grading depends on the private leaderboard after the Kaggle deadline.

## Team Workflow

- Work on feature branches and open pull requests for review.
- Keep `main` stable and avoid mixing unrelated stage changes.
- Do not commit secrets, `.env` files, Modal tokens, Hugging Face tokens, caches, checkpoints, model weights, or large generated artifacts.
- Keep experiment notes, generated run summaries, and final evidence organized under `assignment03/docs/` or ignored run-output locations.
- Prefer small, staged commits that map clearly to the assignment graders.

## Final ThinkFlic Checklist

The final ThinkFlic package should follow [`assignment03/docs/THINKFLIC_SUBMISSION.md`](assignment03/docs/THINKFLIC_SUBMISSION.md). At minimum, prepare:

- Implemented source code and dependency list.
- Generated proof files when available.
- Kaggle submission artifacts and submission metadata.
- Technical report with pipeline explanation, experiments, limitations, and reproducibility commands.
- Integrity declaration.
- Presentation video link.
- ZIP package following the required assignment structure.

## Academic Integrity

This is a course assignment repository. All work should follow the VietAI course rules, Kaggle competition rules, and the assignment's test-data boundary: do not retrieve, reconstruct, manually label, leak, or share hidden test answers.
