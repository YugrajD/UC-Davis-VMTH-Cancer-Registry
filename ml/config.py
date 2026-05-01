"""Project-wide path defaults and shared helpers.

All paths are relative to the project root (the directory containing ml/).
Import these constants instead of hardcoding paths in individual scripts.

To override any path, pass the corresponding CLI argument to the relevant
script — every path here is used as the argparse default, not as a hard lock.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
REPORTS_CSV = "ml/data/report.csv"
DIAGNOSES_CSV = "ml/data/diagnoses.csv"
LABELS_CSV = "ml/ICD_labels/labels.csv"

# ---------------------------------------------------------------------------
# Annotation outputs
# ---------------------------------------------------------------------------
ANNOTATION_DIR = "ml/output/annotation"
ANNOTATION_CSV = f"{ANNOTATION_DIR}/annotation.csv"
KEYWORD_ANNOTATION_DIR = "ml/output/annotation/keyword"
LLM_ANNOTATION_DIR = "ml/output/annotation/llm"
KEYWORD_ANNOTATION_CSV = f"{KEYWORD_ANNOTATION_DIR}/keyword_annotation.csv"
LLM_ANNOTATION_CSV = f"{LLM_ANNOTATION_DIR}/llm_annotation.csv"

# ---------------------------------------------------------------------------
# Training intermediates
# ---------------------------------------------------------------------------
EMBEDDING_CACHE_NPZ = "ml/output/training/embedding_cache.npz"
TFIDF_VECTORIZER_PATH = "ml/output/training/tfidf_selector.joblib"
TRAINING_PAIRS_CSV = "ml/output/training/binary/training_pairs.csv"
CONTRASTIVE_PAIRS_CSV = "ml/output/training/contrastive/contrastive_pairs.csv"
HARD_NEG_PAIRS_CSV = "ml/output/training/contrastive/hard_neg_pairs.csv"

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_CONTRASTIVE_DIR = "ml/output/checkpoints/contrastive"
CHECKPOINT_GROUP_DIR = "ml/output/checkpoints/group"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_TRAINING_DIR = "ml/output/training"
OUTPUT_EVALUATION_DIR = "ml/output/evaluation"
OUTPUT_PRODUCTION_DIR = "ml/output/production"
PETBERT_SCAN_OUTPUT_DIR = "ml/output/report"  # default for standalone petbert_pipeline CLI

# Train/test split files (generated once by ml/training/data/create_split.py)
SPLITS_DIR      = "ml/output/splits"
TRAIN_CASES_TXT = "ml/output/splits/train_cases.txt"
TEST_CASES_TXT  = "ml/output/splits/test_cases.txt"

# Derived structured outputs
GROUP_TRAINING_DATA_NPZ = "ml/output/training/group/group_training_data.npz"
CASE_PRESENCE_DATASET_NPZ = "ml/output/training/binary/case_presence_dataset.npz"
CASE_PRESENCE_CLASSIFIER_PT = "ml/output/checkpoints/contrastive/case_presence_classifier.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_checkpoint_info() -> tuple[str | None, str]:
    """Return (checkpoint_path, output_subdir) for the best saved classifier.

    Always returns the contrastive backbone classifier if it exists.
    Returns (None, 'contrastive') if no checkpoint exists yet.
    """
    p = Path(f"{CHECKPOINT_CONTRASTIVE_DIR}/presence_classifier_best.pt")
    if p.exists():
        return str(p), "contrastive"
    return None, "contrastive"


def best_predictions_subdir() -> str:
    """Return the output subdirectory that contains the most recent predictions."""
    p = Path(f"{OUTPUT_PRODUCTION_DIR}/contrastive/petbert_predictions.csv")
    if p.exists():
        return "contrastive"
    return "contrastive"
