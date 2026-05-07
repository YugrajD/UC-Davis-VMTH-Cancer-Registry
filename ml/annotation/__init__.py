"""Programmatic API for the annotation package.

For CLI usage:  python ml/scripts/run_annotation.py --method keyword|llm
For scripted use, import from here — not from annotation.cli.
"""

import config
from annotation.keyword_pipeline import annotate_with_defaults


def annotate_keyword() -> None:
    """Annotate diagnoses with verified labels using keyword matching.

    Uses project-wide defaults from config.py. Equivalent to:
      python ml/scripts/run_annotation.py --method keyword
    """
    annotate_with_defaults(
        csv_path=config.DIAGNOSES_CSV,
        labels_csv_path=config.LABELS_CSV,
        out_dir=config.KEYWORD_ANNOTATION_DIR,
    )


def keyword_main() -> int:
    """Entry point for the keyword annotation CLI (deferred import)."""
    from annotation.keyword_pipeline.cli import main
    return main()


def llm_main() -> int:
    """Entry point for the LLM annotation CLI (deferred import)."""
    from annotation.llm_pipeline.cli import main
    return main()
