"""Orchestrate a full training cycle in one command.

Steps:
  0. (first run only) python -m petbert_scan — embed reports and labels, save embedding cache
  1. ml/scripts/utils/build_training_pairs.py — assemble training data from evaluation output
  2. ml/scripts/utils/train_classifier.py     — train presence classifier using cached embeddings
  3. python -m petbert_scan            — re-run pipeline with the trained classifier (uses cache)
  4. ml/scripts/utils/evaluate_predictions.py — score new predictions against keyword ground truth
  5. ml/scripts/utils/log_evaluation.py — append result to evaluation_history.csv

Step 0 runs only when the embedding cache doesn't exist yet.  After the first cycle,
Steps 2 and 3 load embeddings from cache — PetBERT is never called again.

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
_UTILS   = _SCRIPTS / "utils"


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
    parser.add_argument(
        "--pos-weight", type=float, default=1.0,
        help="BCEWithLogitsLoss pos_weight passed to train_classifier.py (default: 1.0)",
    )
    parser.add_argument(
        "--recall-weight", type=float, default=0.5,
        help="Checkpoint selection recall weight passed to train_classifier.py (default: 0.5)",
    )
    parser.add_argument(
        "--max-pos-per-group", type=int, default=0,
        help="Cap positive training examples per taxonomy group (0 = no cap). "
             "Passed to build_training_pairs.py. Recommended: 80.",
    )
    parser.add_argument(
        "--co-neg-per-case", type=int, default=3,
        help="Cap completely-off negatives per case (0 = no cap, default: 3). "
             "Passed to build_training_pairs.py.",
    )
    parser.add_argument(
        "--fp-neg-per-case", type=int, default=10,
        help="Extra random taxonomy labels to sample per unique false-positive case "
             "(default: 10). Passed to build_training_pairs.py.",
    )
    parser.add_argument(
        "--co-neg-extra-csv", default="",
        help="Optional second evaluation.csv to pull extra CO negatives from "
             "(e.g. a saved best-cycle evaluation). Passed to build_training_pairs.py.",
    )
    parser.add_argument(
        "--co-neg-bank-csv",
        default="ml/output/evaluation/evaluation_co_bank.csv",
        help="Path to the rolling CO-negative bank (default: ml/output/evaluation/evaluation_co_bank.csv). "
             "Automatically updated after each evaluate step and used as CO source in the next cycle. "
             "Pass empty string to disable.",
    )
    parser.add_argument(
        "--embedding-cache", default="ml/data/embedding_cache.npz",
        help="Path to embedding cache npz. Embeddings are computed on first run and "
             "reused every cycle after that, skipping PetBERT inference in Steps 2 and 3. "
             "Pass --embedding-cache '' to disable caching. (default: ml/data/embedding_cache.npz)",
    )
    args = parser.parse_args()

    py    = sys.executable
    label = args.label or f"classifier {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Ensure `python -m petbert_scan` can find the package when run from project root
    env = os.environ.copy()
    ml_dir = str(_SCRIPTS.parent.resolve())
    env["PYTHONPATH"] = ml_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # 0 ── Build embedding cache (first run only) ──────────────────────────────
    # petbert_scan saves the cache on any run where the cache is missing or stale.
    # Subsequent cycles load it instantly, skipping all PetBERT inference.
    if args.embedding_cache and not os.path.exists(args.embedding_cache):
        pre_embed_cmd = [
            py, "-m", "petbert_scan",
            "--embedding-cache", args.embedding_cache,
            "--device",          args.device,
        ]
        if args.local_only:
            pre_embed_cmd.append("--local-only")
        _step("Step 0/5 — Build embedding cache (first run only)", pre_embed_cmd, env=env)

    # 1 ── Build training pairs ────────────────────────────────────────────────
    build_cmd = [py, str(_UTILS / "build_training_pairs.py")]
    if args.max_pos_per_group > 0:
        build_cmd += ["--max-pos-per-group", str(args.max_pos_per_group)]
    build_cmd += ["--co-neg-per-case", str(args.co_neg_per_case)]
    build_cmd += ["--fp-neg-per-case", str(args.fp_neg_per_case)]
    if args.co_neg_extra_csv:
        build_cmd += ["--co-neg-extra-csv", args.co_neg_extra_csv]
    if args.co_neg_bank_csv:
        build_cmd += ["--co-neg-bank-csv", args.co_neg_bank_csv]
    _step("Step 1/5 — Build training pairs", build_cmd)

    # 2 ── Train classifier ────────────────────────────────────────────────────
    train_cmd = [
        py, str(_UTILS / "train_classifier.py"),
        "--epochs",        str(args.epochs),
        "--device",        args.device,
        "--pos-weight",    str(args.pos_weight),
        "--recall-weight", str(args.recall_weight),
    ]
    if args.embedding_cache:
        train_cmd += ["--embedding-cache", args.embedding_cache]
    _step("Step 2/5 — Train classifier", train_cmd)

    # 3 ── Re-run pipeline with trained classifier ─────────────────────────────
    pipeline_cmd = [
        py, "-m", "petbert_scan",
        "--presence-classifier", _CHECKPOINT,
        "--embedding-min-sim",   str(args.embedding_min_sim),
        "--device",              args.device,
    ]
    if args.local_only:
        pipeline_cmd.append("--local-only")
    if args.embedding_cache:
        pipeline_cmd += ["--embedding-cache", args.embedding_cache]

    _step(
        "Step 3/5 — Re-run pipeline with trained classifier",
        pipeline_cmd,
        env=env,
    )

    # 4 ── Evaluate ────────────────────────────────────────────────────────────
    _step(
        "Step 4/5 — Evaluate predictions",
        [py, str(_UTILS / "evaluate_predictions.py")],
    )

    # 4.5 ── Update rolling CO-negative bank ──────────────────────────────────
    if args.co_neg_bank_csv:
        _step(
            "Step 4.5/5 — Update rolling CO-negative bank",
            [py, str(_UTILS / "update_co_bank.py"),
             "--evaluation-csv", "ml/output/evaluation/evaluation.csv",
             "--bank-csv", args.co_neg_bank_csv],
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
