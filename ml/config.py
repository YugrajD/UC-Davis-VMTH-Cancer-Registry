"""Project-wide path defaults and shared helpers.

All paths are relative to the project root (the directory containing ml/).
Import these constants instead of hardcoding paths in individual scripts.

To override any path, pass the corresponding CLI argument to the relevant
script — every path here is used as the argparse default, not as a hard lock.
"""

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
LLM_ANNOTATION_DIR = "ml/output/annotation/llm"
LLM_ANNOTATION_CSV = f"{LLM_ANNOTATION_DIR}/llm_annotation.csv"
LLM_ANNOTATION_CLEANED_CSV = f"{LLM_ANNOTATION_DIR}/llm_annotation_cleaned.csv"

# ---------------------------------------------------------------------------
# Training intermediates
# ---------------------------------------------------------------------------
EMBEDDING_CACHE_NPZ = "ml/output/training/embedding_cache.npz"
TFIDF_VECTORIZER_PATH = "ml/output/training/tfidf_selector.joblib"
CONTRASTIVE_PAIRS_CSV = "ml/output/training/contrastive/contrastive_pairs.csv"
HARD_NEG_PAIRS_CSV = "ml/output/training/contrastive/hard_neg_pairs.csv"

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_CONTRASTIVE_DIR = "ml/output/checkpoints/contrastive"
CHECKPOINT_GROUP_DIR = "ml/output/checkpoints/group"
CHECKPOINT_CASE_PRESENCE_DIR = "ml/output/checkpoints/case_presence"
CHECKPOINT_LABEL_PRESENCE_DIR = "ml/output/checkpoints/label_presence"
LABEL_PRESENCE_THRESHOLDS_JSON = "ml/output/checkpoints/label_presence/lp_thresholds.json"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_TRAINING_DIR = "ml/output/training"
OUTPUT_EVALUATION_DIR = "ml/output/evaluation"
OUTPUT_PRODUCTION_DIR = "ml/output/production"
DATA_ANALYSIS_DIR = "ml/output/data_analysis"
PETBERT_SCAN_OUTPUT_DIR = "ml/output/report"  # default for standalone petbert_pipeline CLI
BEST_PREDICTIONS_SUBDIR = "contrastive"

# Train/test split files (generated once by ml/training/data/create_split.py)
SPLITS_DIR      = "ml/output/splits"
TRAIN_CASES_TXT = "ml/output/splits/train_cases.txt"
TEST_CASES_TXT  = "ml/output/splits/test_cases.txt"

# Derived structured outputs
GROUP_TRAINING_DATA_NPZ = "ml/output/training/group/group_training_data.npz"
UNCOMMON_GROUPS_TXT = "ml/output/training/group/uncommon_groups.txt"
CASE_PRESENCE_DATASET_NPZ = "ml/output/training/binary/case_presence_dataset.npz"
CASE_PRESENCE_CLASSIFIER_PT = "ml/output/checkpoints/case_presence/case_presence_classifier.pt"
