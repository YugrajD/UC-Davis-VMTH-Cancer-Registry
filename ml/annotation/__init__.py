"""Annotation package public API.

For CLI usage:  python ml/scripts/run_annotation.py

`llm_main` is resolved lazily: the LLM cascade pulls in `dotenv` and the API
client, and the gold sub-package (`annotation.gold`) is pure stdlib. Importing
the cascade eagerly here made the gold tooling unrunnable on any machine that
lacks the LLM dependencies, for no reason.
"""

__all__ = ["llm_main"]


def __getattr__(name):
    if name == "llm_main":
        from annotation.llm_pipeline.cli import main as llm_main
        return llm_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
