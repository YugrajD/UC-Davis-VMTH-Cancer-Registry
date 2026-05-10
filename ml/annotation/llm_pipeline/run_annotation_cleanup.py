"""Ensemble verification cleanup pass over llm_annotation.csv.

This is the same cleanup that `run_annotation.py` invokes automatically as the
final stage of the annotation pipeline. Use this script when re-running cleanup
on an existing llm_annotation.csv (e.g. with different verifier models) without
redoing the three-tier cascade.

For every confirmed positive (Exact / Fuzzy / LLM), runs the row past two
diverse local LM-Studio models and applies unanimous-agreement resolution
rules. Writes a cleaned annotation CSV plus a diff and summary.

LLM use is training-time only. Production is unaffected.

Usage:
  python ml/annotation/llm_pipeline/run_annotation_cleanup.py
  python ml/annotation/llm_pipeline/run_annotation_cleanup.py --models google/gemma-4-31b,qwen/qwen3.6-27b
  python ml/annotation/llm_pipeline/run_annotation_cleanup.py --tiebreaker nvidia/nemotron-3-nano-omni
  python ml/annotation/llm_pipeline/run_annotation_cleanup.py --max-rows 200

Default verifier pair was selected from a 6-model bake-off on 26 Tier-3 rows
(2026-05-09): gemma-4-31b and qwen3.6-27b had the highest adjudicated correctness
and were architecturally diverse, so unanimous votes carry signal. medgemma-27b
and llama-3.3-70b are deliberately excluded — both produced the most fabricated
subtype labels in that comparison.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
        default=config.LLM_ANNOTATION_CLEANED_CSV,
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
        default="google/gemma-4-31b,qwen/qwen3.6-27b",
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
