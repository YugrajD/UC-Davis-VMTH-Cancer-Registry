"""Multi-column TF-IDF text selection — used by both production and training."""

from .text_selector import SOURCE_COLS, TextSelector, get_selector

__all__ = ["SOURCE_COLS", "TextSelector", "get_selector"]
