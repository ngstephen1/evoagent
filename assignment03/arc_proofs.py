"""
ARC-compatible proof generation for Assignment 03.

This script mirrors the proof JSON files produced by run_modal.py, but runs the
existing local code paths on a GPU node instead of through Modal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


ASSIGNMENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "QuantTrio/Qwen3.5-4B-AWQ"
DEFAULT_GPU_MEMORY = 0.7
DEFAULT_EVOLUTION_DIR = Path("runs/exp_self_arc")
BASELINE_ACCURACY = 0.42

sys.path.insert(0, str(ASSIGNMENT_DIR))


def _log(message: str) -> None:
    print(f"[arc_proofs] {message}", flush=True)


def _warn_if_missing_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        _log(
            "WARNING: HF_TOKEN is not set. Model downloads may fail. "
            "Export HF_TOKEN in your shell; do not commit it."
        )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Wrote {path.relative_to(ASSIGNMENT_DIR) if path.is_relative_to(ASSIGNMENT_DIR) else path}")


def _run_command(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    _log("Running: " + " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=ASSIGNMENT_DIR,
        check=False,
        text=True,
        capture_output=capture,
    )


def run_sandbox(args: argparse.Namespace) -> bool:
    _warn_if_missing_hf_token()
    _log(f"Starting sandbox proof generation with model={args.model}")

    from src.sandbox import get_model, run_sandbox_accuracy_check

    model = get_model(args.model)
    eval_res = run_sandbox_accuracy_check(model, dev_size=args.sandbox_dev_size)

    result = {
        "status": "success",
        "baseline_accuracy": eval_res.get("accuracy", 0.0),
        "num_correct": eval_res.get("num_correct", 0),
        "num_examples": eval_res.get("num_examples", 0),
        "message": (
            "Sandbox ran successfully on VT ARC. "
            f"Baseline Accuracy: {eval_res.get('accuracy', 0.0) * 100:.2f}% "
            f"({eval_res.get('num_correct')}/{eval_res.get('num_examples')})"
        ),
        "samples": eval_res.get("samples", []),
    }
    _write_json(ASSIGNMENT_DIR / "sandbox_proof.json", result)
    return True


def run_smoke(args: argparse.Namespace) -> bool:
    _warn_if_missing_hf_token()
    _log(f"Starting smoke proof generation with model={args.model}")

    proc = _run_command(
        [
            sys.executable,
            "main.py",
            "--smoke-test",
            "--dataset",
            "local_financial_qa",
            "--model",
            args.model,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ],
        capture=True,
    )

    passed = proc.returncode == 0
    result = {
        "status": "success" if passed else "failed",
        "smoke_test_passed": passed,
        "returncode": proc.returncode,
        "message": "ARC smoke test passed." if passed else "ARC smoke test failed.",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    _write_json(ASSIGNMENT_DIR / "smoke_proof.json", result)
    return passed


def run_evolution(args: argparse.Namespace) -> bool:
    _warn_if_missing_hf_token()
    output_dir = _resolve_output_dir(args.output_dir)
    _log(f"Starting evolution proof generation with model={args.model}")
    _log(f"Output directory: {output_dir}")

    if output_dir.exists() and not args.no_clean_output:
        _remove_output_dir_safely(output_dir)

    proc = _run_command(
        [
            sys.executable,
            "main.py",
            "--T",
            str(args.T),
            "--dataset",
            "local_financial_qa",
            "--output-dir",
            str(output_dir),
            "--train-size",
            str(args.train_size),
            "--dev-size",
            str(args.dev_size),
            "--model",
            args.model,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--progressive-reflections",
            "--use-curriculum",
            "--afo-mode",
            "best",
        ],
        capture=False,
    )
    if proc.returncode != 0:
        _log(f"Evolution run failed with return code {proc.returncode}")
        return False

    proof_data = _build_evolution_proof(output_dir)
    _write_json(output_dir / "evolution_proof.json", proof_data)
    _write_json(ASSIGNMENT_DIR / "evolution_proof.json", proof_data)
    return True


def run_all(args: argparse.Namespace) -> bool:
    steps: list[tuple[str, Callable[[argparse.Namespace], bool]]] = [
        ("sandbox", run_sandbox),
        ("smoke", run_smoke),
        ("evolution", run_evolution),
    ]
    all_ok = True
    for name, func in steps:
        _log(f"=== {name} ===")
        try:
            ok = func(args)
        except Exception as exc:
            ok = False
            _log(f"{name} failed with exception: {exc}")
        all_ok = all_ok and ok
        if not ok and not args.continue_on_error:
            _log(f"Stopping after failed step: {name}")
            return False
    return all_ok


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ASSIGNMENT_DIR / path
    return path.resolve()


def _remove_output_dir_safely(output_dir: Path) -> None:
    runs_dir = (ASSIGNMENT_DIR / "runs").resolve()
    if not output_dir.is_relative_to(runs_dir):
        raise ValueError(
            f"Refusing to delete output directory outside {runs_dir}: {output_dir}. "
            "Pass --no-clean-output or choose a path under runs/."
        )
    _log(f"Removing previous output directory: {output_dir}")
    shutil.rmtree(output_dir)


def _build_evolution_proof(output_dir: Path) -> dict:
    from src.strategy import StrategyHistory

    history_path = output_dir / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"history.jsonl not found: {history_path}")

    history = StrategyHistory(history_path)
    history.load()
    best_strategy = history.best_strategy()
    best_acc = best_strategy.metadata.dev_accuracy if best_strategy else 0.0
    if best_strategy is not None:
        best_path = output_dir / "iter_best_strategy.json"
        best_path.write_text(best_strategy.to_json(), encoding="utf-8")
        _log(f"Wrote {best_path.relative_to(ASSIGNMENT_DIR)}")

    return {
        "status": "success",
        "best_iteration": best_strategy.metadata.iteration if best_strategy else None,
        "best_dev_accuracy": best_acc,
        "baseline_accuracy": BASELINE_ACCURACY,
        "history": [
            {
                "iteration": strategy.metadata.iteration,
                "dev_accuracy": strategy.metadata.dev_accuracy,
            }
            for strategy in history.strategies
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Assignment 03 proof JSON files on VT ARC without Modal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["sandbox", "smoke", "evolution", "all"],
        help="Proof generation action to run.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local model path.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY,
        help="SGLang GPU memory utilization fraction.",
    )
    parser.add_argument("--sandbox-dev-size", type=int, default=50, help="Dev examples for sandbox accuracy.")
    parser.add_argument("--T", type=int, default=5, help="Evolution iterations.")
    parser.add_argument("--train-size", type=int, default=200, help="Evolution train subset size.")
    parser.add_argument("--dev-size", type=int, default=240, help="Evolution dev subset size.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVOLUTION_DIR),
        help="Evolution output directory, relative to assignment03 unless absolute.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not remove an existing evolution output directory before running.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="For 'all', continue to later proof steps after a failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ASSIGNMENT_DIR)

    try:
        if args.command == "sandbox":
            ok = run_sandbox(args)
        elif args.command == "smoke":
            ok = run_smoke(args)
        elif args.command == "evolution":
            ok = run_evolution(args)
        else:
            ok = run_all(args)
    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
