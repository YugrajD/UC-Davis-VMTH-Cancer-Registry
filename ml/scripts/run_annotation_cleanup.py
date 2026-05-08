"""Ensemble verification cleanup pass over llm_annotation.csv.

For every confirmed positive (Exact / Fuzzy / LLM), runs the row past two
diverse local LM-Studio / Ollama models and applies unanimous-agreement
resolution rules. Writes a cleaned annotation CSV plus a diff and summary.

LLM use is training-time only. Production is unaffected.

Usage:
  python ml/scripts/run_annotation_cleanup.py
  python ml/scripts/run_annotation_cleanup.py --models llama3.3:70b,qwen2.5:72b
  python ml/scripts/run_annotation_cleanup.py --tiebreaker mistral-large
  python ml/scripts/run_annotation_cleanup.py --max-rows 200
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from annotation.llm_pipeline.cleanup import CleanupConfig, run_cleanup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ensemble verification over confirmed annotations.",
    )
    parser.add_argument(
        "--input",
        default=config.LLM_ANNOTATION_CSV,
        help="Input annotation CSV.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(config.LLM_ANNOTATION_DIR, "llm_annotation_cleaned.csv"),
        help="Output cleaned annotation CSV.",
    )
    parser.add_argument(
        "--diff",
        default=os.path.join(config.LLM_ANNOTATION_DIR, "cleanup_diff.csv"),
        help="Diff CSV (only changed rows).",
    )
    parser.add_argument(
        "--summary",
        default=os.path.join(config.LLM_ANNOTATION_DIR, "cleanup_summary.json"),
        help="Aggregate summary JSON.",
    )
    parser.add_argument(
        "--labels-csv",
        default=config.LABELS_CSV,
        help="Vet-ICD-O taxonomy CSV.",
    )
    parser.add_argument(
        "--models",
        default="llama3.3:70b,qwen2.5:72b",
        help="Comma-separated list of LLM model names to vote with.",
    )
    parser.add_argument(
        "--tiebreaker",
        default=None,
        help="Optional third model used only when the primary two disagree.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Seconds to wait per LLM call.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap on input rows (testing only).",
    )
    args = parser.parse_args()

    cfg = CleanupConfig(
        input_csv=args.input,
        output_csv=args.output,
        diff_csv=args.diff,
        summary_json=args.summary,
        labels_csv_path=args.labels_csv,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        tiebreaker_model=args.tiebreaker,
        timeout=args.timeout,
        max_rows=args.max_rows,
    )
    run_cleanup(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
