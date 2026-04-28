"""Shared architecture constants for PetBERT-based classifiers."""

PETBERT_EMB_DIM: int = 768    # PetBERT mean-pooled embedding dimension
DEFAULT_HIDDEN_DIM: int = 512  # MLP hidden layer size (Phase 17+ production default)
DEFAULT_DROPOUT: float = 0.3   # MLP dropout probability

# Default clinical report columns to embed independently
DEFAULT_TEXT_COLS: tuple[str, ...] = (
    "HISTOPATHOLOGICAL SUMMARY",
    "FINAL COMMENT",
    "ANCILLARY TESTS",
)
