"""Command-line interface for running the PetBERT scan pipeline."""

import argparse

from .pipeline import run_scan
from .types import ScanConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a CSV of clinical diagnosis strings with PetBERT and produce categorizations (and optional nearest neighbors)."
    )
    parser.add_argument("--csv", default="ml/data/data.csv", help="Path to input CSV")
    parser.add_argument("--id-col", default="anon_id", help="ID column name")
    parser.add_argument("--text-col", default="Clinical Diagnoses", help="Text column name")
    parser.add_argument("--model", default="SAVSNET/PetBERT", help="HF model name or local path")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local cached model files (no network calls).",
    )
    parser.add_argument("--out-dir", default="ml/output/data", help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max_length")
    parser.add_argument("--neighbors-k", type=int, default=3, help="Neighbors per row")
    parser.add_argument(
        "--task",
        default="categorize",
        choices=["categorize", "neighbors", "both"],
        help="Run categorization only, neighbors only, or both.",
    )
    parser.add_argument(
        "--embedding-min-sim",
        type=float,
        default=0.6,
        help="Minimum embedding similarity to accept embedding fallback category.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device",
    )
    parser.add_argument(
        "--labels-csv",
        default="ml/labels/labels.csv",
        help="Path to labels taxonomy CSV.",
    )
    parser.add_argument(
        "--carcinoma-csv",
        default="ml/data/dataCarcinoma.csv",
        help="Auxiliary label CSV containing carcinoma-positive anon_ids.",
    )
    parser.add_argument(
        "--sarcoma-csv",
        default="ml/data/dataSarcoma.csv",
        help="Auxiliary label CSV containing sarcoma-positive anon_ids.",
    )
    parser.add_argument(
        "--use-auxiliary-labels",
        action="store_true",
        help="Use carcinoma/sarcoma CSVs as extra supervision by anon_id.",
    )
    return parser


def build_config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        text_col=args.text_col,
        model_name=args.model,
        local_only=args.local_only,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        neighbors_k=args.neighbors_k,
        task=args.task,
        embedding_min_sim=args.embedding_min_sim,
        device=args.device,
        labels_csv_path=args.labels_csv,
        carcinoma_csv_path=args.carcinoma_csv,
        sarcoma_csv_path=args.sarcoma_csv,
        use_auxiliary_labels=args.use_auxiliary_labels,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = run_scan(build_config(args))

    print("Wrote:")
    print(outputs.predictions_csv)
    print(outputs.provenance_csv)
    print(outputs.similarity_csv)
    print(outputs.visualization_csv)
    if outputs.neighbors_csv is not None:
        print(outputs.neighbors_csv)
    print(outputs.npz)
    print(outputs.summary_json)
    return 0
