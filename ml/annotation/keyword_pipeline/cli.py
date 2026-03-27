"""Command-line interface for rule-based label annotation (keyword method)."""

import argparse

import config
from .pipeline import KeywordConfig, KeywordOutputs, run_keyword_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate diagnosis text with Vet-ICD-O cancer labels using keyword matching. "
                    "Fast, no model required. Does not handle negation or abbreviations."
    )
    parser.add_argument(
        "--csv",
        default="database/data/output/diagnoses.csv",
        help="Path to input diagnoses CSV.",
    )
    parser.add_argument("--id-col", default="case_id", help="Case ID column name.")
    parser.add_argument(
        "--diag-num-col",
        default="diagnosis_number",
        help="Diagnosis number column name (optional, included in output if present).",
    )
    parser.add_argument("--text-col", default="diagnosis", help="Diagnosis text column name.")
    parser.add_argument(
        "--labels-csv",
        default=config.LABELS_CSV,
        help="Path to Vet-ICD-O taxonomy CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=config.KEYWORD_ANNOTATION_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on input rows (useful for testing).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = KeywordConfig(
        csv_path=args.csv,
        id_col=args.id_col,
        diag_num_col=args.diag_num_col,
        text_col=args.text_col,
        labels_csv_path=args.labels_csv,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
    )
    outputs: KeywordOutputs = run_keyword_scan(config)
    print("Wrote:")
    print(outputs.predictions_csv)
    print(outputs.summary_json)
    return 0
