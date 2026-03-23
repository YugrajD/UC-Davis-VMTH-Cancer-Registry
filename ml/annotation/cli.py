"""Unified annotation pipeline — choose between keyword or LLM matching.

Usage:
  python -m annotation --method keyword [keyword options...]
  python -m annotation --method llm     [llm options...]

Run with --method keyword --help or --method llm --help for per-method options.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the annotation pipeline.",
        add_help=False,
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "llm"],
        required=True,
        help="Annotation method: keyword (fast, no model) or llm (keyword + LLM tiers).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.method == "keyword":
        from annotation.keyword_pipeline.cli import main as _main
    else:
        from annotation.llm_pipeline.cli import main as _main

    return _main()
