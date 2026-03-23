"""End-to-end training pipeline: keyword scan → train → evaluate.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Usage:
  python ml/scripts/run_training.py --mode binary --label "v12"
  python ml/scripts/run_training.py --mode group  --epochs 50
  python ml/scripts/run_training.py --mode contrastive-finetuning --device xpu --local-only
  python ml/scripts/run_training.py --mode knn
  python ml/scripts/run_training.py --skip-keyword-scan --mode binary

Modes:
  binary       Run the full binary PresenceClassifier training cycle (default)
               Steps: keyword_pipeline → build_pairs → train → petbert_pipeline → evaluate → log

  group        Build GroupClassifier training data and train
               Steps: keyword_pipeline → build_group_data → train_group

  contrastive-finetuning  Contrastive InfoNCE fine-tuning of PetBERT backbone
               Steps: [keyword_pipeline →] build_contrastive_pairs → train_contrastive
               After this completes, cold-start and re-run in binary mode with
               --model <contrastive-out-dir> --local-only.

  knn          Build a KnnGroupSelector from the embedding cache
               Steps: KnnGroupSelector.build → save
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
from training.finetune.build_contrastive_dataset import build_contrastive_pairs
from training.finetune.train_contrastive import train as train_contrastive
from model.knn_group_selector import KnnGroupSelector


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline (keyword scan → train → evaluate)."
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "group", "contrastive-finetuning", "knn"],
        default="binary",
        help=(
            "Training mode: binary PresenceClassifier (default), "
            "group GroupClassifier, contrastive-finetuning InfoNCE PetBERT fine-tuning, "
            "or knn KnnGroupSelector build."
        ),
    )
    parser.add_argument(
        "--skip-keyword-scan",
        action="store_true",
        help="Skip keyword pipeline step (reuse existing keyword_annotation.csv). "
             "Applies to binary, group, and contrastive modes.",
    )

    # ------------------------------------------------------------------
    # Shared args
    # ------------------------------------------------------------------
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "xpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--local-only", action="store_true",
                        help="Use only locally cached model files (no HuggingFace download)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs (binary default: 25; contrastive default: 3)")

    # ------------------------------------------------------------------
    # Binary mode args
    # ------------------------------------------------------------------
    parser.add_argument("--label", default="",
                        help="[binary] Label for evaluation history row")
    parser.add_argument("--recall-weight", type=float, default=0.25,
                        help="[binary] Recall weight for checkpoint selection (default: 0.25)")
    parser.add_argument("--co-neg-per-case", type=int, default=10,
                        help="[binary] CO negatives per case (default: 10; use 5 for per-column arch)")
    parser.add_argument("--fp-neg-per-case", type=int, default=10,
                        help="[binary] FP extra negatives per case (default: 10)")
    parser.add_argument("--embedding-min-sim", type=float, default=0.05,
                        help="[binary] Min embedding similarity threshold (default: 0.05)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="[binary] MLP hidden layer size (default: 256; Phase 16 uses 512)")

    # ------------------------------------------------------------------
    # Contrastive mode args
    # ------------------------------------------------------------------
    parser.add_argument("--reports-csv", default="ml/data/report.csv",
                        help="[contrastive] Report text CSV (default: ml/data/report.csv)")
    parser.add_argument("--annotation-csv",
                        default="ml/output/annotation/keyword/keyword_annotation.csv",
                        help="[contrastive] Keyword annotation CSV")
    parser.add_argument("--pairs-csv", default="ml/data/contrastive_pairs.csv",
                        help="[contrastive] Output/input path for (report, label) pairs CSV")
    parser.add_argument("--contrastive-out-dir",
                        default="ml/model/checkpoints/contrastive",
                        help="[contrastive] Directory for the fine-tuned HuggingFace checkpoint")
    parser.add_argument("--model", default="SAVSNET/PetBERT",
                        help="HuggingFace model name or local path. "
                             "[binary] model used for embeddings; "
                             "[contrastive-finetuning] starting weights. "
                             "Default: SAVSNET/PetBERT")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[contrastive] Batch size / number of in-batch negatives (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="[contrastive] Peak learning rate (default: 2e-5)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="[contrastive] InfoNCE temperature (default: 0.07)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="[contrastive] Max BERT token length (default: 256)")
    parser.add_argument("--skip-dataset-build", action="store_true",
                        help="[contrastive] Skip pair-building step (reuse existing pairs CSV)")

    # ------------------------------------------------------------------
    # KNN mode args
    # ------------------------------------------------------------------
    parser.add_argument("--knn-cache", default="ml/data/embedding_cache.npz",
                        help="[knn] Embedding cache npz (default: ml/data/embedding_cache.npz)")
    parser.add_argument("--knn-labels-csv",
                        default="ml/output/annotation/llm/llm_annotation.csv",
                        help="[knn] Predictions CSV with case_id and matched_group columns")
    parser.add_argument("--knn-out", default="ml/model/checkpoints/knn/knn_group_selector.npz",
                        help="[knn] Output path for the KnnGroupSelector")
    parser.add_argument("--knn-k", type=int, default=10,
                        help="[knn] Number of nearest neighbours to vote with (default: 10)")
    parser.add_argument("--knn-min-cases", type=int, default=10,
                        help="[knn] Drop groups with fewer confirmed cases (default: 10)")
    parser.add_argument("--knn-mean-only", action="store_true",
                        help="[knn] Use mean embeddings (768-dim) instead of per-column concat (2304-dim)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Keyword scan (binary, group, contrastive modes)
    # ------------------------------------------------------------------
    if args.mode in ("binary", "group", "contrastive-finetuning") and not args.skip_keyword_scan:
        print("\n=== Step 1: Keyword pipeline ===")
        run_keyword_scan(KeywordConfig(
            csv_path="ml/data/diagnoses.csv",
            id_col="case_id",
            diag_num_col="diagnosis_number",
            text_col="diagnosis",
            labels_csv_path="ml/ICD_labels/labels.csv",
            out_dir="ml/output/annotation/keyword",
            max_rows=None,
        ))

    # ------------------------------------------------------------------
    # Step 2: Mode-specific training
    # ------------------------------------------------------------------

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
            "--model", args.model,
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
            out_path="ml/model/checkpoints/group/group_classifier_current.pt",
            epochs=args.epochs,
            lr=1e-3,
            hidden_dim=256,
            val_frac=0.2,
            threshold=0.3,
            device_arg=args.device,
            weight_decay=0.0,
            max_class_weight=0.0,
        )

    elif args.mode == "contrastive-finetuning":
        if not args.skip_dataset_build:
            print("\n=== Step 2a: Build contrastive pairs ===")
            build_contrastive_pairs(
                reports_csv=args.reports_csv,
                annotation_csv=args.annotation_csv,
                out_csv=args.pairs_csv,
            )
        else:
            print("\n=== Step 2a: Skipped (--skip-dataset-build) ===")

        # Default epochs to 3 for contrastive if the user didn't override
        epochs = args.epochs if args.epochs != 25 else 3

        print("\n=== Step 2b: Contrastive InfoNCE fine-tuning ===")
        train_contrastive(
            pairs_csv=args.pairs_csv,
            out_dir=args.contrastive_out_dir,
            model_name=args.model,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            max_length=args.max_length,
            device_arg=args.device,
            local_only=args.local_only,
        )

        print("\n=== Cold-start reminder ===")
        print("The embedding space has changed. Before retraining the PresenceClassifier:")
        print("  rm -f ml/data/embedding_cache.npz")
        print("  rm -f ml/output/training/contrastive/evaluation_co_bank.csv")
        print("  rm -f ml/model/checkpoints/contrastive/presence_classifier_current.pt")
        print(f"Then re-run in binary mode with --model {args.contrastive_out_dir} --local-only")

    elif args.mode == "knn":
        print("\n=== Build KnnGroupSelector ===")
        selector = KnnGroupSelector.build(
            cache_path=args.knn_cache,
            labels_csv_path=args.knn_labels_csv,
            k=args.knn_k,
            per_column=not args.knn_mean_only,
            min_group_cases=args.knn_min_cases,
        )
        selector.save(args.knn_out)
        print(f"Saved KnnGroupSelector to {args.knn_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
