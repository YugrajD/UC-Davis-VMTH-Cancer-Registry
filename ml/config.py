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
KEYWORD_ANNOTATION_DIR = "ml/output/annotation/keyword"
LLM_ANNOTATION_DIR = "ml/output/annotation/llm"
KEYWORD_ANNOTATION_CSV = f"{KEYWORD_ANNOTATION_DIR}/keyword_annotation.csv"
LLM_ANNOTATION_CSV = f"{LLM_ANNOTATION_DIR}/llm_annotation.csv"

# ---------------------------------------------------------------------------
# Training intermediates
# ---------------------------------------------------------------------------
EMBEDDING_CACHE_NPZ = "ml/data/embedding_cache.npz"
TRAINING_PAIRS_CSV = "ml/data/training_pairs.csv"
CONTRASTIVE_PAIRS_CSV = "ml/data/contrastive_pairs.csv"
HARD_NEG_PAIRS_CSV = "ml/data/hard_neg_pairs.csv"
FINETUNE_DATASET_DIR = "ml/data/finetune_dataset"

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_BINARY_DIR = "ml/output/checkpoints/binary"
CHECKPOINT_CONTRASTIVE_DIR = "ml/output/checkpoints/contrastive"
CHECKPOINT_GROUP_DIR = "ml/output/checkpoints/group"
CHECKPOINT_KNN_DIR = "ml/output/checkpoints/knn_selector"
CHECKPOINT_FINETUNE_DIR = "ml/output/checkpoints/finetune"
KNN_SELECTOR_NPZ = "ml/output/checkpoints/knn_selector/knn_group_selector.npz"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_TRAINING_DIR = "ml/output/training"
OUTPUT_EVALUATION_DIR = "ml/output/evaluation"
OUTPUT_PRODUCTION_DIR = "ml/output/production"
PETBERT_SCAN_OUTPUT_DIR = "ml/output/report"  # default for standalone petbert_pipeline CLI

# ---------------------------------------------------------------------------
# Calibration outputs
# ---------------------------------------------------------------------------
CALIBRATION_OFFSETS_JSON = "ml/output/calibration/label_offsets.json"

# Derived structured outputs (build paths programmatically from the dirs above
# rather than hardcoding every leaf; see run_cycle.py for the pattern)
GROUP_TRAINING_DATA_NPZ = "ml/output/training/group/group_training_data.npz"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Classifier subdirectories in preference order: best-first.
# Used by run_production.py and run_evaluation.py to auto-select the
# strongest available checkpoint without hardcoding the names in each script.
_CLASSIFIER_SUBDIRS = ("contrastive", "binary")


def best_checkpoint_info() -> tuple[str | None, str]:
    """Return (checkpoint_path, output_subdir) for the best saved classifier.

    Searches checkpoint directories in order: contrastive backbone first,
    then binary-only. Returns (None, 'binary') if no checkpoint exists yet.
    """
    for subdir in _CLASSIFIER_SUBDIRS:
        ckpt_dir = CHECKPOINT_CONTRASTIVE_DIR if subdir == "contrastive" else CHECKPOINT_BINARY_DIR
        p = Path(f"{ckpt_dir}/presence_classifier_best.pt")
        if p.exists():
            return str(p), subdir
    return None, "binary"


def best_predictions_subdir() -> str:
    """Return the output subdirectory that contains the most recent predictions.

    Used by run_evaluation.py to auto-locate the predictions file without
    knowing which classifier produced them.
    """
    for subdir in _CLASSIFIER_SUBDIRS:
        p = Path(f"{OUTPUT_PRODUCTION_DIR}/{subdir}/petbert_predictions.csv")
        if p.exists():
            return subdir
    return _CLASSIFIER_SUBDIRS[0]
