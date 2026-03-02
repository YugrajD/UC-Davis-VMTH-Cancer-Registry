"""Orchestrate a full training cycle in one command.

Steps:
  1. ml/model/build_training_pairs.py  — assemble training data from evaluation output
  2. ml/model/train_classifier.py      — train presence classifier on training pairs
  3. python -m petbert_scan            — re-run pipeline with the trained classifier
  4. ml/scripts/evaluate_predictions.py — score new predictions against keyword ground truth
  5. ml/utils/log_evaluation.py        — append result to evaluation_history.csv

Usage:
  python ml/scripts/run_training_cycle.py --label "classifier v2"
  python ml/scripts/run_training_cycle.py --label "v3" --epochs 30 --device cuda
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_CHECKPOINT = "ml/model/checkpoints/presence_classifier_best.pt"
_SCRIPTS = Path(__file__).parent
_MODEL   = _SCRIPTS.parent / "model"
_UTILS   = _SCRIPTS.parent / "utils"


def _step(title: str, cmd: list[str], **kwargs) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\nCycle aborted at: {title} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full PetBERT training cycle.")
    parser.add_argument(
        "--label", default="",
        help="Label written to evaluation_history.csv (default: auto-timestamp)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs passed to train_classifier.py (default: 20)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device for training and inference (default: auto)",
    )
    parser.add_argument(
        "--embedding-min-sim", type=float, default=0.5,
        help="Presence probability threshold for the pipeline step (default: 0.5)",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Use only locally cached PetBERT model files (no network calls)",
    )
    args = parser.parse_args()

    py    = sys.executable
    label = args.label or f"classifier {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Ensure `python -m petbert_scan` can find the package when run from project root
    env = os.environ.copy()
    ml_dir = str(_SCRIPTS.parent.resolve())
    env["PYTHONPATH"] = ml_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # 1 ── Build training pairs ────────────────────────────────────────────────
    _step(
        "Step 1/5 — Build training pairs",
        [py, str(_MODEL / "build_training_pairs.py")],
    )

    # 2 ── Train classifier ────────────────────────────────────────────────────
    _step(
        "Step 2/5 — Train classifier",
        [py, str(_MODEL / "train_classifier.py"),
         "--epochs", str(args.epochs),
         "--device", args.device],
    )

    # 3 ── Re-run pipeline with trained classifier ─────────────────────────────
    pipeline_cmd = [
        py, "-m", "petbert_scan",
        "--presence-classifier", _CHECKPOINT,
        "--embedding-min-sim",   str(args.embedding_min_sim),
        "--device",              args.device,
    ]
    if args.local_only:
        pipeline_cmd.append("--local-only")

    _step(
        "Step 3/5 — Re-run pipeline with trained classifier",
        pipeline_cmd,
        env=env,
    )

    # 4 ── Evaluate ────────────────────────────────────────────────────────────
    _step(
        "Step 4/5 — Evaluate predictions",
        [py, str(_SCRIPTS / "evaluate_predictions.py")],
    )

    # 5 ── Log ─────────────────────────────────────────────────────────────────
    _step(
        "Step 5/5 — Log evaluation results",
        [py, str(_UTILS / "log_evaluation.py"),
         "--label", label],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
