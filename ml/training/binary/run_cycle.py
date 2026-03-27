"""Run one full label-classifier training cycle.

A cycle is the core unit of iterative training:

  Step 0  Embed all reports into vector space (first run only — skipped once cache exists)
  Step 1  Assemble training data: positives from verified labels + negatives from
          previous cycle's wrong predictions
  Step 2  Train the label presence classifier on those pairs
  Step 3  Score all reports with the updated classifier
  Step 4  Evaluate predictions against verified labels → verdicts (good / slightly_off /
          completely_off / false_positive / false_negative)
  Step 4.5 Record wrong-group predictions in the rolling feedback bank
  Step 5  Record cycle results to evaluation history; promote checkpoint if new best

After the first cycle, Step 0 is skipped — PetBERT never runs again; all subsequent
training and inference reads embeddings from the cache.

Typical trajectory with the adapted backbone (Phase 17):
  c1 ~50%  →  c3 ~65%  →  c8 ~69% (plateau)

Usage:
  python ml/training/binary/run_cycle.py --label "c1" --device xpu --local-only
"""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import config
from model.constants import DEFAULT_TEXT_COLS
from production.petbert_pipeline import run_scan, ScanConfig
from training.binary.build_training_pairs import build_pairs
from evaluation.evaluate import evaluate
from evaluation.log_evaluation import log_evaluation
from training.binary.train import train
from training.binary.update_co_bank import update_co_bank


def _subdir(model: str) -> str:
    """Return the output subdirectory for a given model path.

    Uses 'contrastive' when the model path points to the adapted backbone
    checkpoint directory; 'binary' for the frozen-backbone baseline.
    """
    return "contrastive" if "contrastive" in model else "binary"


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _score_reports(title: str, scan_config: ScanConfig) -> None:
    """Run the scoring pipeline — wrapped here to keep cycle steps readable."""
    _banner(title)
    run_scan(scan_config)


def _make_scan_config(args, *, out_dir: str, classifier_path: str | None = None) -> ScanConfig:
    """Build a ScanConfig from cycle CLI args."""
    return ScanConfig(
        csv_path=config.REPORTS_CSV,
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
        labels_csv_path=config.LABELS_CSV,
        presence_classifier_path=classifier_path,
        embedding_cache_path=args.embedding_cache or None,
        enrich_labels_csv_path=args.enrich_labels_csv or None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one full label-classifier training cycle."
    )
    parser.add_argument(
        "--label", default="",
        help="Label written to evaluation_history.csv (default: auto-timestamp)",
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Training epochs (default: 25)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--embedding-min-sim", type=float, default=0.05,
        help="Minimum score threshold for predictions (default: 0.05; "
             "scores are mean-subtracted so 0.05 suits the label classifier)",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Use only locally cached model files (no network calls)",
    )
    parser.add_argument(
        "--model", default="SAVSNET/PetBERT",
        help="HuggingFace model name or local path for embeddings "
             "(default: SAVSNET/PetBERT). Pass the adapted backbone path "
             "after running --mode adapt-backbone.",
    )
    parser.add_argument(
        "--pos-weight", type=float, default=1.0,
        help="BCEWithLogitsLoss pos_weight (default: 1.0)",
    )
    parser.add_argument(
        "--recall-weight", type=float, default=0.25,
        help="Checkpoint selection recall weight (default: 0.25)",
    )
    parser.add_argument(
        "--max-pos-per-group", type=int, default=0,
        help="Cap positive training examples per taxonomy group (0 = no cap).",
    )
    parser.add_argument(
        "--co-neg-per-case", type=int, default=3,
        help="Cap wrong-group negatives per case (0 = no cap, default: 3).",
    )
    parser.add_argument(
        "--fp-neg-per-case", type=int, default=10,
        help="Extra random labels to sample per false-positive case (default: 10).",
    )
    parser.add_argument(
        "--co-neg-extra-csv", default="",
        help="Optional second evaluation.csv to pull extra wrong-group negatives from.",
    )
    parser.add_argument(
        "--co-neg-bank-csv",
        default=None,
        help="Path to the rolling wrong-label feedback bank "
             "(default: auto-derived from --model). Pass empty string to disable.",
    )
    parser.add_argument(
        "--embedding-cache", default=config.EMBEDDING_CACHE_NPZ,
        help=f"Path to embedding cache file. Pass empty string to disable. "
             f"(default: {config.EMBEDDING_CACHE_NPZ})",
    )
    parser.add_argument(
        "--enrich-labels-csv", default="",
        help="Annotation CSV for label embedding enrichment (default: disabled).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=512,
        help="MLP hidden layer size (default: 512).",
    )
    parser.add_argument(
        "--annotation-csv", default=config.KEYWORD_ANNOTATION_CSV,
        help="Verified label annotations for training positives and evaluation. "
             f"(default: {config.KEYWORD_ANNOTATION_CSV})",
    )
    args = parser.parse_args(argv)

    subdir = _subdir(args.model)
    ckpt_dir        = config.CHECKPOINT_CONTRASTIVE_DIR if subdir == "contrastive" else config.CHECKPOINT_BINARY_DIR
    checkpoint      = f"{ckpt_dir}/presence_classifier_current.pt"
    checkpoint_best = f"{ckpt_dir}/presence_classifier_best.pt"
    production_out  = f"{config.OUTPUT_PRODUCTION_DIR}/{subdir}"
    evaluation_out  = f"{config.OUTPUT_EVALUATION_DIR}/{subdir}"
    history_csv     = f"{config.OUTPUT_EVALUATION_DIR}/{subdir}/evaluation_history.csv"

    if args.co_neg_bank_csv is None:
        args.co_neg_bank_csv = f"{config.OUTPUT_TRAINING_DIR}/{subdir}/evaluation_co_bank.csv"

    label = args.label or f"cycle {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # 0 ── Embed reports (first run only) ─────────────────────────────────────
    if args.embedding_cache and not Path(args.embedding_cache).exists():
        _score_reports(
            "Step 0/5 — Embed reports into vector space (first run only)",
            _make_scan_config(args, out_dir=production_out),
        )

    # 1 ── Assemble training data ──────────────────────────────────────────────
    _banner("Step 1/5 — Assemble training data")
    build_pairs(
        expectation_csv=args.annotation_csv,
        evaluation_csv=f"{evaluation_out}/evaluation.csv",
        co_neg_per_case=args.co_neg_per_case,
        fp_neg_per_case=args.fp_neg_per_case,
        max_pos_per_group=args.max_pos_per_group,
        co_neg_extra_csv=args.co_neg_extra_csv,
        co_neg_bank_csv=args.co_neg_bank_csv,
    )

    # 2 ── Train label presence classifier ────────────────────────────────────
    _banner("Step 2/5 — Train label presence classifier")
    train(
        epochs=args.epochs,
        device=args.device,
        pos_weight=args.pos_weight,
        recall_weight=args.recall_weight,
        embedding_cache=args.embedding_cache or None,
        hidden_dim=args.hidden_dim,
        model_name=args.model,
        out_dir=ckpt_dir,
    )

    # 3 ── Score reports with updated classifier ───────────────────────────────
    _score_reports(
        "Step 3/5 — Score reports with updated classifier",
        _make_scan_config(args, out_dir=production_out, classifier_path=checkpoint),
    )

    # 4 ── Evaluate predictions ────────────────────────────────────────────────
    _banner("Step 4/5 — Score predictions against verified labels")
    evaluate(
        prediction_csv=Path(f"{production_out}/petbert_predictions.csv"),
        expectation_csv=Path(args.annotation_csv),
        out_dir=Path(evaluation_out),
    )

    # 4.5 ── Record wrong-group feedback ──────────────────────────────────────
    if args.co_neg_bank_csv:
        _banner("Step 4.5/5 — Record wrong-group predictions in feedback bank")
        update_co_bank(
            evaluation_csv=f"{evaluation_out}/evaluation.csv",
            bank_csv=args.co_neg_bank_csv,
        )

    # 5 ── Record cycle results ────────────────────────────────────────────────
    _banner("Step 5/5 — Record cycle results")
    log_evaluation(
        summary=f"{evaluation_out}/evaluation_summary.csv",
        history=history_csv,
        label=label,
    )

    # 5.5 ── Promote checkpoint if new best ───────────────────────────────────
    history_path    = Path(history_csv)
    checkpoint_path = Path(checkpoint)

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
                    f"\n* New best: {current_gs:.1f}% correct or close — "
                    f"checkpoint saved to {checkpoint_best}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
