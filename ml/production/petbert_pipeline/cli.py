"""Command-line interface for running the PetBERT scan pipeline."""

import argparse

import config
from model.constants import DEFAULT_TEXT_COLS
from .pipeline import run_scan
from .types import ScanConfig

_DEFAULT_TEXT_COLS = ",".join(DEFAULT_TEXT_COLS)
_DEFAULT_COL_WEIGHTS = "FINAL COMMENT:2.0,HISTOPATHOLOGICAL SUMMARY:1.5,ANCILLARY TESTS:0.5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a reportText CSV with PetBERT and produce categorizations (and optional nearest neighbors)."
    )
    parser.add_argument("--csv", default=config.REPORTS_CSV, help="Path to input CSV")
    parser.add_argument("--id-col", default="case_id", help="ID column name")
    parser.add_argument(
        "--text-cols",
        default=_DEFAULT_TEXT_COLS,
        help=(
            "Comma-separated column names to embed independently and weighted-average. "
            f"Default: '{_DEFAULT_TEXT_COLS}'"
        ),
    )
    parser.add_argument(
        "--col-weights",
        default=_DEFAULT_COL_WEIGHTS,
        help=(
            "Per-column embedding weights as 'COL:weight,...' pairs. "
            "Columns absent from this list default to 1.0. "
            f"Default: '{_DEFAULT_COL_WEIGHTS}'"
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
        "--presence-classifier",
        default=None,
        help=(
            "Path to a trained PresenceClassifier checkpoint (.pt). "
            "When set, presence probabilities replace cosine similarity for scoring labels."
        ),
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
        "--enrich-labels-csv",
        default=None,
        help=(
            "Path to keyword_annotation.csv. When provided, each taxonomy label embedding "
            "is averaged 50/50 with the mean cached report embedding of its keyword-confirmed "
            "cases, pulling label representations toward the clinical language in real reports. "
            "Requires --embedding-cache. Enriched embeddings are stored in the cache so "
            "subsequent runs reuse them without recomputing."
        ),
    )
    parser.add_argument(
        "--group-classifier",
        default=None,
        help=(
            "Path to a trained GroupClassifier checkpoint (.pt). "
            "When set, uses two-stage categorization: GroupClassifier predicts cancer group(s), "
            "then cosine similarity selects the best term within each group. "
            "Replaces --presence-classifier and eliminates the completely-off floor. "
            "Train with: python ml/training/train_group_classifier.py"
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
        "--finetuned-model-path",
        default=None,
        help=(
            "Path to a fine-tuned PetBERT sequence classification model checkpoint. "
            "When set, the pipeline skips independent column embedding and instead "
            "passes the concatenated report text directly to the fine-tuned model "
            "to predict cancer groups. Within the predicted group, cosine similarity "
            "is still used to select the best specific ICD term."
        ),
    )
    parser.add_argument(
        "--categorization-mode",
        default="default",
        choices=["default", "group-keyword"],
        help=(
            "Categorization strategy within the binary-classifier path. "
            "'group-keyword' uses the top-scoring label's group as Stage 1, "
            "then ICD-O behavior keyword matching selects the specific term "
            "within that group (Stage 2). Requires --presence-classifier. "
            "Aims to convert 'slightly off' predictions into 'good' ones."
        ),
    )
    return parser


def _parse_col_weights(raw: str) -> dict[str, float]:
    """Parse 'COL:weight,...' into a dict. Silently skips malformed pairs."""
    result: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        col, _, val = token.rpartition(":")
        try:
            result[col.strip()] = float(val.strip())
        except ValueError:
            pass
    return result


def build_config(args: argparse.Namespace) -> ScanConfig:
    text_cols = tuple(c.strip() for c in args.text_cols.split(",") if c.strip())
    col_weights = _parse_col_weights(args.col_weights)
    return ScanConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        text_cols=text_cols,
        col_weights=col_weights,
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
        presence_classifier_path=args.presence_classifier,
        embedding_cache_path=args.embedding_cache,
        enrich_labels_csv_path=args.enrich_labels_csv,
        group_classifier_path=args.group_classifier,
        group_classifier_threshold=args.group_classifier_threshold,
        finetuned_model_path=args.finetuned_model_path,
        categorization_mode=args.categorization_mode,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = run_scan(build_config(args))

    print("Wrote:")
    print(outputs.predictions_csv)
    print(outputs.column_scores_csv)
    print(outputs.provenance_csv)
    print(outputs.similarity_csv)
    print(outputs.visualization_csv)
    if outputs.neighbors_csv is not None:
        print(outputs.neighbors_csv)
    print(outputs.npz)
    print(outputs.summary_json)
    return 0
