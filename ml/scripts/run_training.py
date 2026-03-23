"""End-to-end training pipeline: keyword scan → train → evaluate.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Usage:
  python ml/scripts/run_training.py --mode binary --label "v12"
  python ml/scripts/run_training.py --mode group  --epochs 50
  python ml/scripts/run_training.py --skip-keyword-scan --mode binary

Modes:
  binary  Run the full binary PresenceClassifier training cycle (default)
          Steps: keyword_pipeline → build_pairs → train → petbert_pipeline → evaluate → log
  group   Build GroupClassifier training data and train
          Steps: keyword_pipeline → build_group_data → train_group
"""

import argparse
import sys
from pathlib import Path

# Add ml/ to the path so all packages are importable without env PYTHONPATH=ml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation.keyword_pipeline.pipeline import KeywordConfig, run_keyword_scan
from training.run_cycle import main as run_binary_cycle
from training.group.build_training_data import build_training_data
from training.group.train import train as train_group


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline (keyword scan → train → evaluate)."
    )
    parser.add_argument("--mode", choices=["binary", "group"], default="binary",
                        help="Training mode: binary PresenceClassifier (default) or group GroupClassifier")
    parser.add_argument("--skip-keyword-scan", action="store_true",
                        help="Skip keyword pipeline step (reuse existing ml/output/annotation/keyword/keyword_annotation.csv)")
    # Binary training args forwarded to run_binary_cycle:
    parser.add_argument("--label", default="",
                        help="Label for evaluation history (binary mode only)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs (default: 25)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--recall-weight", type=float, default=0.25,
                        help="Recall weight for checkpoint selection (binary mode, default: 0.25)")
    parser.add_argument("--co-neg-per-case", type=int, default=10,
                        help="CO negatives per case (binary mode, default: 10)")
    parser.add_argument("--fp-neg-per-case", type=int, default=10,
                        help="FP extra negatives per case (binary mode, default: 10)")
    parser.add_argument("--embedding-min-sim", type=float, default=0.05,
                        help="Min embedding similarity threshold (binary mode, default: 0.05)")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached PetBERT model files")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="MLP hidden layer size (binary mode, default: 256). Try 512 or 768.")
    args = parser.parse_args()

    # Step 1: Keyword scan (generate ground truth labels)
    if not args.skip_keyword_scan:
        print("\n=== Step 1: Keyword pipeline ===")
        run_keyword_scan(KeywordConfig(
            csv_path="ml/data/diagnoses.csv",
            id_col="case_id",
            diag_num_col="diagnosis_number",
            text_col="diagnosis",
            labels_csv_path="ml/labels/labels.csv",
            out_dir="ml/output/annotation/keyword",
            max_rows=None,
        ))

    # Step 2: Train
    if args.mode == "binary":
        print("\n=== Step 2: Binary training cycle ===")
        cycle_argv = [
            "--label", args.label,
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--recall-weight", str(args.recall_weight),
            "--co-neg-per-case", str(args.co_neg_per_case),
            "--fp-neg-per-case", str(args.fp_neg_per_case),
            "--embedding-min-sim", str(args.embedding_min_sim),
            "--hidden-dim", str(args.hidden_dim),
        ] + (["--local-only"] if args.local_only else [])
        run_binary_cycle(argv=cycle_argv)

    elif args.mode == "group":
        print("\n=== Step 2a: Build group training data ===")
        build_training_data(
            cache_path="ml/data/embedding_cache.npz",
            expectation_csv_path="ml/output/annotation/llm/llm_annotation.csv",
            out_path="ml/output/training/group/group_training_data.npz",
        )
        print("\n=== Step 2b: Train group classifier ===")
        train_group(
            training_data_path="ml/output/training/group/group_training_data.npz",
            out_path="ml/model/checkpoints/group_classifier_current.pt",
            epochs=args.epochs,
            lr=1e-3,
            hidden_dim=256,
            val_frac=0.2,
            threshold=0.3,
            device_arg=args.device,
            weight_decay=0.0,
            max_class_weight=0.0,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
