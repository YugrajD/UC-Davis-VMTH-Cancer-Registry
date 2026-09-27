"""Every path used by the ml pipeline, as ``pathlib.Path`` objects.

In-tree assets (e.g. ``taxonomy/labels.csv``) resolve relative to this
package (``PACKAGE_ROOT``), which is also ``ML_ROOT``, the project's ``ml/``
directory. Data (``ml/data/...``) and outputs (``ml/output/...``) resolve
relative to ``ML_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ML_ROOT = PACKAGE_ROOT

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
# generations/splits.py's create_three_way for how it and other splits are made).
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

# Diagnosis-Mapping audit (manual_audit/diagnosis_mapping_audit.py, formerly the
# Tier-3 audit): per batch, a key CSV and a case-ID list. Batch 1's case list is
# also what eval_batch.py excludes from its frame (it refuses to draw without it).
DIAGNOSIS_MAPPING_AUDIT_DIR = MANUAL_AUDIT_DIR / "diagnosis_mapping_audit"
DIAGNOSIS_MAPPING_AUDIT_BATCH1_TXT = DIAGNOSIS_MAPPING_AUDIT_DIR / "diagnosis_mapping_audit_batch1.txt"

# Report-Mapping audit (manual_audit/report_mapping_audit.py): per batch, a
# ledger CSV and a case-ID list.
REPORT_MAPPING_AUDIT_DIR = MANUAL_AUDIT_DIR / "report_mapping_audit"

# The universal audit list (manual_audit/audit_list.py): one ledger of every
# case put on a list and its gold origin; the lists themselves go to the outbox.
AUDIT_LIST_LEDGER_CSV = MANUAL_AUDIT_DIR / "audit_list_ledger.csv"

# Case-level eval batches (manual_audit/eval_batch.py) — sheets live here; the
# ledger of record is EVAL_BATCH_LEDGER_CSV above.
EVAL_BATCH_DIR = MANUAL_AUDIT_DIR / "eval_batch"

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
# k-fold out-of-fold gate scores on train cases (train.py --stage oof); read by the Report-Mapping audit.
OOF_DIR = REPORT_MAPPING_DIR / "oof"

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
