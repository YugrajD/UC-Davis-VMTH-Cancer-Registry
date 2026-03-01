"""Command-line interface for running the PetBERT scan pipeline."""

import argparse

from .pipeline import run_scan
from .types import ScanConfig

_DEFAULT_TEXT_COLS = "HISTOPATHOLOGICAL SUMMARY,FINAL COMMENT,ANCILLARY TESTS"
_DEFAULT_COL_WEIGHTS = "FINAL COMMENT:2.0,HISTOPATHOLOGICAL SUMMARY:1.5,ANCILLARY TESTS:0.5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a reportText CSV with PetBERT and produce categorizations (and optional nearest neighbors)."
    )
    parser.add_argument("--csv", default="ml/data/reportText.csv", help="Path to input CSV")
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
    parser.add_argument("--out-dir", default="ml/output/reportText", help="Output directory")
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
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--labels-csv",
        default="ml/labels/labels.csv",
        help="Path to labels taxonomy CSV.",
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
