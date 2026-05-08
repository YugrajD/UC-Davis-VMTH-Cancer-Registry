"""Annotate diagnosis text with Vet-ICD-O cancer labels (LLM pipeline).

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Reads diagnoses.csv and maps each diagnosis field to a Vet-ICD-O taxonomy label
(term, group, ICD code) using a three-tier cascade:
  1. Exact keyword match (with negation masking)
  2. Fuzzy token overlap (behavior-code aware)
  3. LM-Studio-hosted LLM (for rows containing a cancer signal term)

The output annotation file is used as training supervision and evaluation
ground truth for all classifiers.

Usage:
  python ml/scripts/run_annotation.py
  python ml/scripts/run_annotation.py --max-rows 100
  python ml/scripts/run_annotation.py --list-models
"""

import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation import llm_main


if __name__ == "__main__":
    raise SystemExit(llm_main())
