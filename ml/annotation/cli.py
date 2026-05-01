"""Annotate diagnosis text with Vet-ICD-O cancer labels.

Prefer run_annotation.py in ml/scripts/ for top-level usage (no PYTHONPATH needed).
This module entry point requires PYTHONPATH=ml.

Usage:
  python -m annotation --method keyword [options...]
  python -m annotation --method llm     [options...]

Run with --method keyword --help or --method llm --help for per-method options.
"""

import argparse
import sys 

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate diagnosis text with Vet-ICD-O cancer labels.",
        add_help=False,
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "llm"],
        required=True,
        help="keyword: fast rule-based matching (no model). "
             "llm: LLM-assisted matching (authoritative, handles negation).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.method == "keyword":
        from annotation.keyword_pipeline.cli import main as _main
    else:
        from annotation.llm_pipeline.cli import main as _main

    return _main()
