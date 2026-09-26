"""Every path used by the ml rewrite, as ``pathlib.Path`` objects.

In-tree assets (e.g. ``taxonomy/labels.csv``) resolve relative to this
package (``PACKAGE_ROOT``) so they are found the same way whether this file
lives at ``ml/next/config.py`` (pre-cutover) or ``ml/config.py``
(post-cutover). Data (``ml/data/...``) and outputs (``ml/output/...``)
resolve relative to ``ML_ROOT``, the project's ``ml/`` directory.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# Pre-cutover this file lives one level below ml/ (at ml/next/), so ML_ROOT
# must step up out of "next" to land on ml/ itself. Post-cutover this file
# lives directly at ml/, so PACKAGE_ROOT is already ML_ROOT.
ML_ROOT = PACKAGE_ROOT
if ML_ROOT.name == "next":
    ML_ROOT = ML_ROOT.parent  # removed at cutover

DATA_DIR = ML_ROOT / "data"
OUTPUT_DIR = ML_ROOT / "output"

# ---------------------------------------------------------------------------
# Raw inputs
# ---------------------------------------------------------------------------
REPORT_CSV = DATA_DIR / "report.csv"
DIAGNOSES_CSV = DATA_DIR / "diagnoses.csv"

# ---------------------------------------------------------------------------
# Taxonomy (in-tree asset — resolves from the package, not ML_ROOT)
# ---------------------------------------------------------------------------
LABELS_CSV = PACKAGE_ROOT / "taxonomy" / "labels.csv"

# ---------------------------------------------------------------------------
# The split every WP defaults to once the three-way split exists (see
# generations/splits.py's create_three_way / import_legacy for the others).
# ---------------------------------------------------------------------------
DEFAULT_SPLIT_ID = "three-way-v1"

# ---------------------------------------------------------------------------
# Silver generation (diagnosis_mapping) — output/silver/<silver_id>/...
# ---------------------------------------------------------------------------
SILVER_DIR = OUTPUT_DIR / "silver"
# Coverage-stats artifacts (diagnosis_mapping/stats.py) — one subdir per silver
# generation analysed. Separate from SILVER_DIR since silver directories are
# write-once and stats are reproducible, derived output.
DIAGNOSIS_MAPPING_STATS_DIR = OUTPUT_DIR / "diagnosis_mapping_stats"

# ---------------------------------------------------------------------------
# Manual audit: sheets, audit store, gold store, eval-batch ledger, cause store
# ---------------------------------------------------------------------------
MANUAL_AUDIT_DIR = OUTPUT_DIR / "manual_audit"
AUDIT_STORE_CSV = MANUAL_AUDIT_DIR / "audit_store.csv"
GOLD_STORE_CSV = MANUAL_AUDIT_DIR / "gold_store.csv"
EVAL_BATCH_LEDGER_CSV = MANUAL_AUDIT_DIR / "eval_batch_ledger.csv"
CAUSE_STORE_CSV = MANUAL_AUDIT_DIR / "cause_store.csv"

# Tier-3 audit sheets (manual_audit/tier3_audit.py) — row-level, writes to
# AUDIT_STORE_CSV above, never to the gold store.
TIER3_AUDIT_DIR = MANUAL_AUDIT_DIR / "tier3_audit"
TIER3_AUDIT_REVIEW_CSV = TIER3_AUDIT_DIR / "tier3_audit_review.csv"
TIER3_AUDIT_KEY_CSV = TIER3_AUDIT_DIR / "tier3_audit_key.csv"
TIER3_AUDIT_INSTRUCTIONS_MD = TIER3_AUDIT_DIR / "tier3_audit_instructions.md"
TIER3_AUDIT_TAXONOMY_CSV = TIER3_AUDIT_DIR / "tier3_audit_taxonomy.csv"
TIER3_AUDIT_PILOT_CSV = TIER3_AUDIT_DIR / "tier3_audit_pilot_review.csv"
TIER3_AUDIT_REMAINDER_CSV = TIER3_AUDIT_DIR / "tier3_audit_remainder_review.csv"
TIER3_AUDIT_PILOT_INSTRUCTIONS_MD = TIER3_AUDIT_DIR / "tier3_audit_pilot_instructions.md"

# Case-level eval batches (manual_audit/eval_batch.py) — sheets live here; the
# ledger of record is EVAL_BATCH_LEDGER_CSV above.
EVAL_BATCH_DIR = MANUAL_AUDIT_DIR / "eval_batch"

# The Tier-3 audit's batch-1 case-id exclusion list, as eval_batch.py needs it
# (a one-time copy off the legacy sheets dir — see eval_batch.py). Deliberately
# its own path, not TIER3_AUDIT_DIR/tier3_audit_batch1_cases.txt: that name is
# also what tier3_audit.sample(batch=1) writes its OWN case ledger to, and the
# two must never collide.
TIER3_AUDIT_BATCH1_EXCLUSION_TXT = MANUAL_AUDIT_DIR / "tier3_audit_batch1_exclusion.txt"

# ---------------------------------------------------------------------------
# Coding: corrected annotations, adopted codes, review queue
# ---------------------------------------------------------------------------
CODING_DIR = OUTPUT_DIR / "coding"
CORRECTED_ANNOTATIONS_CSV = CODING_DIR / "corrected_annotations.csv"
ADOPTED_CODES_CSV = CODING_DIR / "adopted_codes.csv"
REVIEW_QUEUE_CSV = CODING_DIR / "review_queue.csv"

# ---------------------------------------------------------------------------
# Split generations — output/splits/<split_id>/{train,calibration,test}_cases.txt
# ---------------------------------------------------------------------------
SPLITS_DIR = OUTPUT_DIR / "splits"

# ---------------------------------------------------------------------------
# Report-mapping generations (this layout is also the cloud bundle layout)
# ---------------------------------------------------------------------------
REPORT_MAPPING_DIR = OUTPUT_DIR / "report_mapping"
REPORT_MAPPING_CURRENT_DIR = REPORT_MAPPING_DIR / "current"
REPORT_MAPPING_CANDIDATE_DIR = REPORT_MAPPING_DIR / "candidate"
# Content-hash keyed embedding cache. Sits beside the generations; never bundled.
EMBEDDING_CACHE_DIR = REPORT_MAPPING_DIR / "embedding_cache"

# ---------------------------------------------------------------------------
# Predictions and evaluation outputs
# ---------------------------------------------------------------------------
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
EVAL_DIR = OUTPUT_DIR / "eval"
# One line per silver-eval run (evaluation/silver_eval.py), carried over from log_evaluation.py.
SILVER_EVAL_HISTORY_CSV = EVAL_DIR / "silver_eval_history.csv"

# ---------------------------------------------------------------------------
# Handoff (cloud file contracts)
# ---------------------------------------------------------------------------
HANDOFF_DIR = OUTPUT_DIR / "handoff"
HANDOFF_INBOX_DIR = HANDOFF_DIR / "inbox"
HANDOFF_OUTBOX_DIR = HANDOFF_DIR / "outbox"
# Cumulative landing table merged from every pending_diagnoses_<export>.csv
# import (handoff/imports.py) — case_id-keyed, a later export's case replaces
# its earlier rows. Raw per-export copies + sidecars stay in HANDOFF_INBOX_DIR
# under their own export-stamped filename.
HANDOFF_PENDING_DIAGNOSES_CSV = HANDOFF_INBOX_DIR / "pending_diagnoses.csv"
# Bundle tarballs (handoff/exports.py export_bundle) — kept out of
# HANDOFF_OUTBOX_DIR's flat file list since a bundle is large and binary.
HANDOFF_BUNDLES_DIR = HANDOFF_OUTBOX_DIR / "bundles"

# ---------------------------------------------------------------------------
# Archive — written only by generations/ (promote.py); nothing loads from it.
# ---------------------------------------------------------------------------
ARCHIVE_ROOT = OUTPUT_DIR / "archive"

# ---------------------------------------------------------------------------
# Legacy paths — the old ml/ tree, read by the parity harness and by the
# one-time importers (legacy split, legacy silver, gen-0 checkpoint copy).
# Every LEGACY_* constant is removed after cutover, along with the old tree.
# ---------------------------------------------------------------------------
LEGACY_ANNOTATION_CSV = OUTPUT_DIR / "annotation" / "annotation.csv"  # removed after cutover
LEGACY_TRAIN_CASES_TXT = OUTPUT_DIR / "splits" / "train_cases.txt"  # removed after cutover
LEGACY_TEST_CASES_TXT = OUTPUT_DIR / "splits" / "test_cases.txt"  # removed after cutover
LEGACY_TRAIN_CASES_TEMPORAL_TXT = OUTPUT_DIR / "splits" / "train_cases_temporal.txt"  # removed after cutover
LEGACY_TEST_CASES_TEMPORAL_TXT = OUTPUT_DIR / "splits" / "test_cases_temporal.txt"  # removed after cutover
LEGACY_CHECKPOINT_CONTRASTIVE_DIR = OUTPUT_DIR / "checkpoints" / "contrastive"  # removed after cutover
LEGACY_CHECKPOINT_GROUP_BEST_PT = OUTPUT_DIR / "checkpoints" / "group" / "group_classifier_best.pt"  # removed after cutover
LEGACY_CHECKPOINT_CASE_PRESENCE_PT = OUTPUT_DIR / "checkpoints" / "case_presence" / "case_presence_classifier.pt"  # removed after cutover
LEGACY_CHECKPOINT_LABEL_PRESENCE_DIR = OUTPUT_DIR / "checkpoints" / "label_presence"  # removed after cutover
LEGACY_LP_THRESHOLDS_JSON = OUTPUT_DIR / "checkpoints" / "label_presence" / "lp_thresholds.json"  # removed after cutover
LEGACY_UNCOMMON_GROUPS_TXT = OUTPUT_DIR / "training" / "group" / "uncommon_groups.txt"  # removed after cutover
LEGACY_EMBEDDING_CACHE_NPZ = OUTPUT_DIR / "training" / "embedding_cache.npz"  # removed after cutover
LEGACY_PREDICTIONS_CSV = OUTPUT_DIR / "production" / "petbert_predictions.csv"  # removed after cutover
LEGACY_TIER3_AUDIT_SHEETS_DIR = OUTPUT_DIR / "annotation"  # removed after cutover
# The old end-to-end scorer; parity runs it as a subprocess to freeze the reference.
LEGACY_EVALUATE_PY = ML_ROOT / "evaluation" / "evaluate.py"  # removed after cutover
# The old production entry point; parity drives it read-only to regenerate the reference predictions.
LEGACY_RUN_PRODUCTION_PY = ML_ROOT / "scripts" / "run_production.py"  # removed after cutover
# The old taxonomy copy the legacy pipeline reads; parity hashes it (its mtime check is bypassed).
LEGACY_LABELS_CSV = ML_ROOT / "ICD_labels" / "labels.csv"  # removed after cutover

# ---------------------------------------------------------------------------
# Parity reference pack — the frozen legacy baseline scripts/parity.py compares
# against (case halves, predictions, verdict tables, thresholds, hashes).
# ---------------------------------------------------------------------------
PARITY_REFERENCE_DIR = OUTPUT_DIR / "parity_reference"  # removed after cutover
