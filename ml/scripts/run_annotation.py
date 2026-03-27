"""Annotate diagnosis text with Vet-ICD-O cancer labels.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Reads diagnoses.csv and maps each diagnosis field to a Vet-ICD-O taxonomy label
(term, group, ICD code). The output annotation file is used as training supervision
and evaluation ground truth for all classifiers.

Two methods are available:

  keyword   Fast rule-based matching against a curated keyword dictionary.
            No model required. ~19% coverage. Does not handle negation or
            abbreviations. Use for quick testing or when the Ollama server
            is unavailable.

  llm       LLM-assisted matching using a four-tier cascade:
              1. Exact keyword match
              2. Fuzzy token overlap
              3. Ollama LLM (for rows containing a cancer signal term)
              4. Claude API fallback (opt-in with --use-claude)
            Handles negation, hedged language, and abbreviations (e.g. HSA,
            MCT). This is the authoritative annotation source.

Usage:
  python ml/scripts/run_annotation.py --method keyword
  python ml/scripts/run_annotation.py --method llm
  python ml/scripts/run_annotation.py --method llm --max-rows 100
  python ml/scripts/run_annotation.py --method llm --use-claude
  python ml/scripts/run_annotation.py --method llm --list-models
"""

import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse


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

    # Delegate to the appropriate pipeline CLI with remaining args
    sys.argv = [sys.argv[0]] + remaining
    if args.method == "keyword":
        from annotation.keyword_pipeline.cli import main as _main
    else:
        from annotation.llm_pipeline.cli import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
