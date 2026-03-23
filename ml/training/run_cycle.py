"""Orchestrate a full binary-classifier training cycle in one command.

Steps:
  0. (first run only) Build embedding cache via petbert_pipeline
  1. build_training_pairs   — assemble training data from evaluation output
  2. train                  — train presence classifier using cached embeddings
  3. petbert_pipeline       — re-run with the trained classifier (uses cache)
  4. evaluate               — score new predictions against keyword ground truth
  5. log_evaluation         — append result to evaluation_history.csv

Step 0 runs only when the embedding cache doesn't exist yet.  After the first
cycle, Steps 2 and 3 load embeddings from cache — PetBERT is never called again.

Usage:
  env PYTHONPATH=ml python ml/training/run_cycle.py --label "classifier v2"
  env PYTHONPATH=ml python ml/training/run_cycle.py --label "v3" --epochs 30 --device mps
"""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

from model.constants import DEFAULT_TEXT_COLS
from production.petbert_pipeline.pipeline import run_scan
from production.petbert_pipeline.types import ScanConfig
from training.binary.build_training_pairs import build_pairs
from evaluation.evaluate import evaluate
from evaluation.log_evaluation import log_evaluation
from training.binary.train import train
from training.binary.update_co_bank import update_co_bank

def _subdir(model: str) -> str:
    """Return the output subdirectory name based on the model used."""
    return "contrastive" if "contrastive" in model else "binary"


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _scan(title: str, config: ScanConfig) -> None:
    """Run the petbert_pipeline directly — no subprocess."""
    _print_banner(title)
    run_scan(config)


def _make_scan_config(args, *, out_dir: str, presence_classifier_path: str | None = None) -> ScanConfig:
    """Build a ScanConfig from cycle CLI args."""
    return ScanConfig(
        csv_path="ml/data/report.csv",
        id_col="case_id",
        text_cols=DEFAULT_TEXT_COLS,
        col_weights={},
        model_name=args.model,
        local_only=args.local_only,
        out_dir=out_dir,
        max_rows=None,
        batch_size=16,
        max_length=512,
        neighbors_k=3,
        task="categorize",
        embedding_min_sim=args.embedding_min_sim,
        device=args.device,
        labels_csv_path="ml/ICD_labels/labels.csv",
        presence_classifier_path=presence_classifier_path,
        embedding_cache_path=args.embedding_cache or None,
        enrich_labels_csv_path=args.enrich_labels_csv or None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a full PetBERT training cycle.")
    parser.add_argument(
        "--label", default="",
        help="Label written to evaluation_history.csv (default: auto-timestamp)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs (default: 20)",
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
        "--model", default="SAVSNET/PetBERT",
        help="HuggingFace model name or local path for PetBERT embeddings "
             "(default: SAVSNET/PetBERT). Pass a contrastive checkpoint here "
             "after running --mode contrastive.",
    )
    parser.add_argument(
        "--pos-weight", type=float, default=1.0,
        help="BCEWithLogitsLoss pos_weight (default: 1.0)",
    )
    parser.add_argument(
        "--recall-weight", type=float, default=0.5,
        help="Checkpoint selection recall weight (default: 0.5)",
    )
    parser.add_argument(
        "--max-pos-per-group", type=int, default=0,
        help="Cap positive training examples per taxonomy group (0 = no cap).",
    )
    parser.add_argument(
        "--co-neg-per-case", type=int, default=3,
        help="Cap completely-off negatives per case (0 = no cap, default: 3).",
    )
    parser.add_argument(
        "--fp-neg-per-case", type=int, default=10,
        help="Extra random taxonomy labels to sample per unique false-positive case (default: 10).",
    )
    parser.add_argument(
        "--co-neg-extra-csv", default="",
        help="Optional second evaluation.csv to pull extra CO negatives from.",
    )
    parser.add_argument(
        "--co-neg-bank-csv",
        default=None,
        help="Path to the rolling CO-negative bank (default: auto-derived from --model). "
             "Pass empty string to disable.",
    )
    parser.add_argument(
        "--embedding-cache", default="ml/data/embedding_cache.npz",
        help="Path to embedding cache npz. Pass empty string to disable. "
             "(default: ml/data/embedding_cache.npz)",
    )
    parser.add_argument(
        "--enrich-labels-csv", default="",
        help="Path to keyword_annotation.csv for label embedding enrichment. "
             "(default: disabled)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="MLP hidden layer size (default: 256). Try 512 or 768 to reduce compression.",
    )
    args = parser.parse_args(argv)

    subdir          = _subdir(args.model)
    checkpoint      = f"ml/model/checkpoints/{subdir}/presence_classifier_current.pt"
    checkpoint_best = f"ml/model/checkpoints/{subdir}/presence_classifier_best.pt"
    production_out  = f"ml/output/production/{subdir}"
    evaluation_out  = f"ml/output/evaluation/{subdir}"
    history_csv     = f"ml/output/evaluation/{subdir}/evaluation_history.csv"

    if args.co_neg_bank_csv is None:
        args.co_neg_bank_csv = f"ml/output/training/{subdir}/evaluation_co_bank.csv"

    label = args.label or f"classifier {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # 0 ── Build embedding cache (first run only) ──────────────────────────────
    if args.embedding_cache and not Path(args.embedding_cache).exists():
        _scan(
            "Step 0/5 — Build embedding cache (first run only)",
            _make_scan_config(args, out_dir=production_out),
        )

    # 1 ── Build training pairs ────────────────────────────────────────────────
    _print_banner("Step 1/5 — Build training pairs")
    build_pairs(
        co_neg_per_case=args.co_neg_per_case,
        fp_neg_per_case=args.fp_neg_per_case,
        max_pos_per_group=args.max_pos_per_group,
        co_neg_extra_csv=args.co_neg_extra_csv,
        co_neg_bank_csv=args.co_neg_bank_csv,
    )

    # 2 ── Train classifier ────────────────────────────────────────────────────
    _print_banner("Step 2/5 — Train classifier")
    train(
        epochs=args.epochs,
        device=args.device,
        pos_weight=args.pos_weight,
        recall_weight=args.recall_weight,
        embedding_cache=args.embedding_cache or None,
        hidden_dim=args.hidden_dim,
        model_name=args.model,
    )

    # 3 ── Re-run pipeline with trained classifier ─────────────────────────────
    _scan(
        "Step 3/5 — Re-run pipeline with trained classifier",
        _make_scan_config(args, out_dir=production_out, presence_classifier_path=checkpoint),
    )

    # 4 ── Evaluate ────────────────────────────────────────────────────────────
    _print_banner("Step 4/5 — Evaluate predictions")
    evaluate(
        prediction_csv=Path(f"{production_out}/petbert_predictions.csv"),
        expectation_csv=Path("ml/output/annotation/keyword/keyword_annotation.csv"),
        out_dir=Path(evaluation_out),
    )

    # 4.5 ── Update rolling CO-negative bank ──────────────────────────────────
    if args.co_neg_bank_csv:
        _print_banner("Step 4.5/5 — Update rolling CO-negative bank")
        update_co_bank(
            evaluation_csv=f"{evaluation_out}/evaluation.csv",
            bank_csv=args.co_neg_bank_csv,
        )

    # 5 ── Log ─────────────────────────────────────────────────────────────────
    _print_banner("Step 5/5 — Log evaluation results")
    log_evaluation(
        summary=f"{evaluation_out}/evaluation_summary.csv",
        history=history_csv,
        label=label,
    )

    # 5.5 ── Save best checkpoint ──────────────────────────────────────────────
    history_path    = Path(history_csv)
    checkpoint_path = Path(checkpoint)
    production_path = Path(checkpoint_best)

    if history_path.exists() and checkpoint_path.exists():
        with open(history_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best_gs = max(
                float(r.get("good_pct", 0) or 0) + float(r.get("slightly_off_pct", 0) or 0)
                for r in rows
            )
            current_gs = (
                float(rows[-1].get("good_pct", 0) or 0)
                + float(rows[-1].get("slightly_off_pct", 0) or 0)
            )
            if current_gs >= best_gs:
                shutil.copy2(checkpoint, checkpoint_best)
                print(
                    f"\n* New best Good+Slight: {current_gs:.1f}% -- "
                    f"checkpoint saved to {_CHECKPOINT_PRODUCTION}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
