"""Train and retrain classifiers against verified cancer labels.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Modes
-----
  train-classifier   Iterative cycle: embed reports → build training data → train
                     label presence classifier → score reports → evaluate → repeat.
                     Run 6–8 cycles; each takes ~10 minutes after the first.

  train-groups       One-shot: train a multi-label group classifier.
                     Not yet competitive at current data volume (~5,800 cases);
                     re-run when confirmed cases reach ~15,000.

  adapt-backbone     Fine-tune the embedding model (PetBERT) so that report text
                     and cancer label text land closer together in vector space.
                     Run once; then cold-start and retrain with train-classifier.

  build-knn          Build a K-nearest-neighbour lookup structure from the
                     embedding cache for group-based gating.

  calibrate          One-shot: compute per-label score offsets that correct for
                     systematic bias in the score distribution after mean-centering.
                     Saves ml/output/calibration/label_offsets.json; apply at
                     inference with --calibration-offsets in run_production.py.

Usage
-----
  python ml/scripts/run_training.py --mode train-classifier --label "c1"
  python ml/scripts/run_training.py --mode adapt-backbone --device xpu --local-only
  python ml/scripts/run_training.py --mode train-groups --device xpu
  python ml/scripts/run_training.py --mode build-knn
  python ml/scripts/run_training.py --mode calibrate --device xpu
"""

import argparse
import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model.constants import DEFAULT_HIDDEN_DIM
from annotation import annotate_keyword
from training.binary.run_cycle import main as run_classifier_cycle
from training.group.build_training_data import build_training_data
from training.group.train import train as train_group
from training.contrastive.build_contrastive_dataset import build_contrastive_pairs
from training.contrastive.train_contrastive import train as train_contrastive
from model.knn_group_selector import KnnGroupSelector


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train classifiers to predict cancer labels from veterinary pathology reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train-classifier", "train-groups", "adapt-backbone", "build-knn", "calibrate"],
        default="train-classifier",
        help=(
            "What to train: "
            "train-classifier (label presence model, iterative — default), "
            "train-groups (group classifier, one-shot), "
            "adapt-backbone (fine-tune embedding model), "
            "build-knn (K-nearest-neighbour lookup), "
            "calibrate (per-label score offsets, one-shot)."
        ),
    )
    parser.add_argument(
        "--force-reannotate",
        action="store_true",
        help="Re-run label annotation even if the annotation file already exists. "
             "Only applies when using the default keyword annotation file.",
    )

    # ------------------------------------------------------------------
    # Shared args
    # ------------------------------------------------------------------
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached model files (no HuggingFace download)")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs. "
             "Defaults: train-classifier=25, adapt-backbone=3, train-groups=50.",
    )
    parser.add_argument(
        "--annotation-csv", default=config.KEYWORD_ANNOTATION_CSV,
        help="Verified label annotations used as training supervision and evaluation "
             "ground truth. Accepts keyword_annotation.csv (default) or "
             "llm_annotation.csv — both share the same format. "
             f"(default: {config.KEYWORD_ANNOTATION_CSV})",
    )

    # ------------------------------------------------------------------
    # train-classifier args
    # ------------------------------------------------------------------
    parser.add_argument("--label", default="",
                        help="[train-classifier] Label for the evaluation history row")
    parser.add_argument("--recall-weight", type=float, default=0.25,
                        help="[train-classifier] Recall weight for checkpoint selection (default: 0.25)")
    parser.add_argument("--co-neg-per-case", type=int, default=5,
                        help="[train-classifier] Wrong-group negatives per case (default: 5)")
    parser.add_argument("--fp-neg-per-case", type=int, default=10,
                        help="[train-classifier] Extra negatives per false-positive case (default: 10)")
    parser.add_argument("--embedding-min-sim", type=float, default=0.05,
                        help="[train-classifier] Minimum score threshold for predictions (default: 0.05)")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM,
                        help=f"[train-classifier] MLP hidden layer size (default: {DEFAULT_HIDDEN_DIM})")
    parser.add_argument("--co-neg-bank-csv", default=None,
                        help="[train-classifier] Path to rolling wrong-label feedback bank "
                             "(default: auto-derived from --model). Pass empty string to disable.")

    # ------------------------------------------------------------------
    # adapt-backbone args
    # ------------------------------------------------------------------
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV,
                        help=f"[adapt-backbone] Report text CSV (default: {config.REPORTS_CSV})")
    parser.add_argument("--pairs-csv", default=config.CONTRASTIVE_PAIRS_CSV,
                        help="[adapt-backbone] Output/input path for (report, label) training pairs CSV")
    parser.add_argument("--backbone-out-dir", default=config.CHECKPOINT_CONTRASTIVE_DIR,
                        help="[adapt-backbone] Directory for the adapted model checkpoint")
    parser.add_argument(
        "--model", default="SAVSNET/PetBERT",
        help="HuggingFace model name or local checkpoint path. "
             "[train-classifier] embedding model; "
             "[adapt-backbone] starting weights. "
             "Default: SAVSNET/PetBERT",
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[adapt-backbone] Batch size / number of in-batch negatives (default: 32)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate. "
                             "[adapt-backbone] default: 2e-5; [train-groups] default: 5e-5")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="[adapt-backbone] Contrastive loss temperature (default: 0.07)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="[adapt-backbone] Max BERT token length (default: 256)")
    parser.add_argument("--skip-pair-build", action="store_true",
                        help="[adapt-backbone] Skip building training pairs (reuse existing pairs CSV)")
    parser.add_argument("--hard-neg-csv", default=None,
                        help="[adapt-backbone] Path to hard-negative triplets CSV. "
                             "If omitted, only InfoNCE loss is used. "
                             f"Build with: build_contrastive_dataset.py --mode build-hard-neg "
                             f"(default: {config.HARD_NEG_PAIRS_CSV})")
    parser.add_argument("--hard-neg-weight", type=float, default=0.5,
                        help="[adapt-backbone] Weight for the hard-negative margin loss (default: 0.5)")
    parser.add_argument("--hard-neg-margin", type=float, default=0.3,
                        help="[adapt-backbone] Margin for hard-negative loss (default: 0.3)")

    # ------------------------------------------------------------------
    # build-knn args
    # ------------------------------------------------------------------
    parser.add_argument("--knn-cache", default=config.EMBEDDING_CACHE_NPZ,
                        help=f"[build-knn] Embedding cache file (default: {config.EMBEDDING_CACHE_NPZ})")
    parser.add_argument("--knn-out", default=config.KNN_SELECTOR_NPZ,
                        help="[build-knn] Output path for the KNN lookup structure")
    parser.add_argument("--knn-k", type=int, default=10,
                        help="[build-knn] Nearest neighbours to vote with (default: 10)")
    parser.add_argument("--knn-min-cases", type=int, default=10,
                        help="[build-knn] Drop groups with fewer confirmed cases (default: 10)")
    parser.add_argument("--knn-mean-only", action="store_true",
                        help="[build-knn] Use mean embeddings (768-dim) instead of per-column (2304-dim)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Ensure verified labels are available
    # ------------------------------------------------------------------
    # Three cases:
    #   1. Annotation file already exists → skip (unless --force-reannotate).
    #   2. Default keyword annotation file is missing → auto-run keyword annotation.
    #   3. A custom annotation file is missing → error (user must annotate first).
    if args.mode in ("train-classifier", "train-groups", "adapt-backbone", "calibrate"):
        annotation_path = Path(args.annotation_csv)
        is_default_keyword_file = (
            annotation_path.resolve() == Path(config.KEYWORD_ANNOTATION_CSV).resolve()
        )

        if annotation_path.exists() and not args.force_reannotate:
            print(f"\n=== Step 1: Verified labels found — skipping annotation ({args.annotation_csv}) ===")
        elif is_default_keyword_file:
            print("\n=== Step 1: Annotating diagnoses with labels (keyword method) ===")
            annotate_keyword()
        else:
            print(f"\nError: annotation file not found: {args.annotation_csv}")
            print("Run annotation first, e.g.:")
            print("  python ml/scripts/run_annotation.py --method llm")
            return 1

    # ------------------------------------------------------------------
    # Step 2: Mode-specific training
    # ------------------------------------------------------------------

    if args.mode == "train-classifier":
        print("\n=== Step 2: Train label presence classifier ===")
        epochs = args.epochs if args.epochs is not None else 25
        cycle_argv = [
            "--label", args.label,
            "--epochs", str(epochs),
            "--device", args.device,
            "--recall-weight", str(args.recall_weight),
            "--co-neg-per-case", str(args.co_neg_per_case),
            "--fp-neg-per-case", str(args.fp_neg_per_case),
            "--embedding-min-sim", str(args.embedding_min_sim),
            "--hidden-dim", str(args.hidden_dim),
            "--model", args.model,
            "--annotation-csv", args.annotation_csv,
        ] + (["--local-only"] if args.local_only else []) \
          + (["--co-neg-bank-csv", args.co_neg_bank_csv] if args.co_neg_bank_csv is not None else [])
        run_classifier_cycle(argv=cycle_argv)

    elif args.mode == "train-groups":
        epochs = args.epochs if args.epochs is not None else 50
        group_lr = args.lr if args.lr is not None else 5e-5

        print("\n=== Step 2a: Build group classifier training data ===")
        build_training_data(
            cache_path=config.EMBEDDING_CACHE_NPZ,
            expectation_csv_path=args.annotation_csv,
            out_path=config.GROUP_TRAINING_DATA_NPZ,
        )
        print("\n=== Step 2b: Train group classifier ===")
        train_group(
            training_data_path=config.GROUP_TRAINING_DATA_NPZ,
            out_path=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_current.pt",
            epochs=epochs,
            lr=group_lr,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            val_frac=0.2,
            threshold=0.3,
            device_arg=args.device,
            weight_decay=0.0,
            max_class_weight=0.0,
            min_group_cases=10,
            max_group_cases=0,
            dropout=0.3,
        )

    elif args.mode == "adapt-backbone":
        epochs = args.epochs if args.epochs is not None else 3
        backbone_lr = args.lr if args.lr is not None else 2e-5

        if not args.skip_pair_build:
            print("\n=== Step 2a: Build (report, label) training pairs ===")
            build_contrastive_pairs(
                reports_csv=args.reports_csv,
                annotation_csv=args.annotation_csv,
                out_csv=args.pairs_csv,
            )
        else:
            print("\n=== Step 2a: Skipped — reusing existing pairs CSV ===")

        print("\n=== Step 2b: Adapt embedding backbone ===")
        train_contrastive(
            pairs_csv=args.pairs_csv,
            out_dir=args.backbone_out_dir,
            model_name=args.model,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=backbone_lr,
            temperature=args.temperature,
            max_length=args.max_length,
            device_arg=args.device,
            local_only=args.local_only,
            hard_neg_csv=args.hard_neg_csv,
            hard_neg_weight=args.hard_neg_weight,
            hard_neg_margin=args.hard_neg_margin,
        )

        print("\n=== Cold-start required ===")
        print("The embedding space has changed. Before retraining the label classifier:")
        print(f"  rm -f {config.EMBEDDING_CACHE_NPZ}")
        print(f"  rm -f {config.OUTPUT_TRAINING_DIR}/contrastive/evaluation_co_bank.csv")
        print(f"  rm -f {args.backbone_out_dir}/presence_classifier_current.pt")
        print(f"Then retrain with: --mode train-classifier --model {args.backbone_out_dir} --local-only")

    elif args.mode == "build-knn":
        print("\n=== Build KNN group lookup ===")
        selector = KnnGroupSelector.build(
            cache_path=args.knn_cache,
            labels_csv_path=args.annotation_csv,
            k=args.knn_k,
            per_column=not args.knn_mean_only,
            min_group_cases=args.knn_min_cases,
        )
        selector.save(args.knn_out)
        print(f"Saved KNN lookup to {args.knn_out}")

    elif args.mode == "calibrate":
        print("\n=== Calibrate per-label score offsets ===")
        from training.binary.calibrate import calibrate
        calibrate(
            annotation_csv=args.annotation_csv,
            model_path=None if args.model == "SAVSNET/PetBERT" else args.model,
            device_arg=args.device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
