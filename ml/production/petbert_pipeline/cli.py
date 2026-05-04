"""Command-line interface for running the PetBERT scan pipeline."""

import argparse

import config
from .pipeline import run_scan
from .types import ScanConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a reportText CSV with PetBERT and produce categorizations (and optional nearest neighbors)."
    )
    parser.add_argument("--csv", default=config.REPORTS_CSV, help="Path to input CSV")
    parser.add_argument("--id-col", default="case_id", help="ID column name")
    parser.add_argument(
        "--text-cols",
        default="",
        help=(
            "Comma-separated column names to embed independently. "
            "Empty string (default) activates TF-IDF multi-column text selection."
        ),
    )
    parser.add_argument("--model", default="SAVSNET/PetBERT", help="HF model name or local path")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local cached model files (no network calls).",
    )
    parser.add_argument("--out-dir", default=config.PETBERT_SCAN_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length")
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
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--labels-csv",
        default=config.LABELS_CSV,
        help="Path to labels taxonomy CSV.",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help=(
            "Path to an embedding cache npz file. "
            "If the cache exists and is valid, PetBERT embedding is skipped entirely. "
            "If it doesn't exist yet, embeddings are computed and saved here for future runs."
        ),
    )
    parser.add_argument(
        "--group-classifier",
        default=None,
        help=(
            "Path to a trained GroupClassifier checkpoint (.pt). "
            "Enables 3-stage categorization: "
            "(1) CasePresenceClassifier gates non-cancer cases (if --case-presence-classifier is set), "
            "(2) GroupClassifier predicts which cancer group(s) a case belongs to, "
            "(3) ICD-O behavior keyword matching selects the best term within each group."
        ),
    )
    parser.add_argument(
        "--group-classifier-threshold",
        type=float,
        default=0.3,
        help=(
            "Probability threshold for GroupClassifier group selection (default: 0.3). "
            "Lower = more predictions (higher recall, lower precision). "
            "Only used when --group-classifier is set."
        ),
    )
    parser.add_argument(
        "--case-presence-classifier",
        default=None,
        help=(
            "Path to a trained CasePresenceClassifier checkpoint (.pt). "
            "When set alongside --group-classifier, acts as a first-stage gate: "
            "cases whose cancer probability is below --case-presence-threshold are "
            "predicted Uncategorized without reaching the GroupClassifier."
        ),
    )
    parser.add_argument(
        "--case-presence-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for CasePresenceClassifier gate (default: 0.5). "
            "Lower = more cases pass (higher recall, higher FP). "
            "Only used when --case-presence-classifier is set."
        ),
    )
    parser.add_argument(
        "--no-group-classifier-fallback-to-argmax",
        dest="group_classifier_fallback_to_argmax",
        action="store_false",
        default=True,
        help=(
            "Disable argmax fallback: cases where no group clears the threshold are "
            "predicted 'Unidentified Cancer' instead of using the top-1 group. "
            "Default: fallback is enabled."
        ),
    )
    parser.add_argument(
        "--tfidf-vectorizer",
        default=config.TFIDF_VECTORIZER_PATH,
        help=(
            "Path to the fitted TF-IDF vectorizer joblib file. "
            "Used for multi-column text selection when --text-cols is empty. "
            f"Default: {config.TFIDF_VECTORIZER_PATH}"
        ),
    )
    return parser


def build_config(args: argparse.Namespace) -> ScanConfig:
    text_cols = tuple(c.strip() for c in args.text_cols.split(",") if c.strip())
    return ScanConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        text_cols=text_cols,
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
        presence_classifier_path=None,
        embedding_cache_path=args.embedding_cache,
        group_classifier_path=args.group_classifier,
        group_classifier_threshold=args.group_classifier_threshold,
        group_classifier_fallback_to_argmax=args.group_classifier_fallback_to_argmax,
        case_presence_classifier_path=args.case_presence_classifier,
        case_presence_threshold=args.case_presence_threshold,
        tfidf_vectorizer_path=args.tfidf_vectorizer,
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
