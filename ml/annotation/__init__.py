"""Annotation package public API.

For CLI usage:  python ml/scripts/run_annotation.py
"""

from annotation.llm_pipeline.cli import main as llm_main

__all__ = ["llm_main"]
