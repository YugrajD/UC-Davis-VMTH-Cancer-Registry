"""Synthetic fixture factories shared by every ml/next work package.

Everything here is offline and invented — no network access, no HuggingFace
download, and no real veterinary text. Real files are touched only to read
public column headers / the public taxonomy's first few lines, or to count
distinct categorical values (never to copy text fields).

Each ``build_*`` / ``make_*`` function is a plain factory that returns a path
or object; ``conftest.py`` wraps the ones every test needs as pytest fixtures.
Later work packages can call these factories directly to compose new fixture
combinations without going through pytest.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig, BertForMaskedLM, BertTokenizer

# ---------------------------------------------------------------------------
# Tiny BERT (stand-in for the real PetBERT checkpoint)
# ---------------------------------------------------------------------------

# Word-level vocab covering every word used in the synthetic report/diagnosis
# text below, so tokenization never falls back to [UNK].
VOCAB_WORDS = [
    "mast", "cell", "cells", "tumor", "tumors", "carcinoma", "sarcoma",
    "malignant", "benign", "skin", "margin", "margins", "clear", "excision",
    "complete", "incomplete", "no", "evidence", "of", "neoplasia",
    "neoplastic", "present", "absent", "biopsy", "specimen", "final",
    "comment", "ancillary", "tests", "immunohistochemistry", "positive",
    "negative", "inflammation", "mild", "moderate", "severe", "consistent",
    "with", "a", "the", "is", "and", "fibrosarcoma", "squamous", "epithelial",
    "grade", "low", "high", "section", "shows", "mitotic", "index",
]

TINY_EMB_DIM = 32  # BERT hidden_size below; also the emb_dim fed to the heads


def build_tiny_bert(model_dir: Path) -> Path:
    """Build and save a tiny random-init BERT + tokenizer in HF format.

    2 layers, hidden 32, small word-level vocab — small enough to run on CPU
    in well under a second, so tests never need the real PetBERT download.
    Loadable via AutoModel/AutoModelForMaskedLM + AutoTokenizer.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + VOCAB_WORDS
    vocab_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(str(model_dir))

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=TINY_EMB_DIM,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        type_vocab_size=2,
    )
    # Saved as a MaskedLM checkpoint to match production's AutoModelForMaskedLM
    # loading (ml/production/petbert_pipeline/embedding.py); AutoModel can
    # still load it since transformers strips the "bert." prefix.
    model = BertForMaskedLM(config)
    model.save_pretrained(str(model_dir))
    return model_dir


def embed_report_text(model_dir: Path, text: str) -> np.ndarray:
    """Tokenize + mean-pool one text through the tiny BERT at model_dir.

    Minimal stand-in for the real embedding pipeline (mean-pool over attended
    tokens, matching ml/production/petbert_pipeline/embedding.py's algorithm)
    until report_mapping/model/backbone.py lands in WP4.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    enc = tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return emb.squeeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic taxonomy (labels.csv shape)
# ---------------------------------------------------------------------------

# Group -> [(code, term), ...]. "Neoplasms, NOS" mirrors the real taxonomy's
# always-forced-in group. "Mast Cell Tumors" has enough terms to stand in for
# a common group; the "Rare *" groups stand in for ones that merge into
# Uncommon once case counts are assigned by a later work package.
TAXONOMY_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Neoplasms, NOS": [("9999/0", "Neoplasm, benign")],
    "Mast Cell Tumors": [
        ("1001/3", "Mast cell tumor, malignant"),
        ("1002/0", "Mast cell tumor, benign"),
        ("1003/3", "Mast cell sarcoma"),
    ],
    "Rare Sarcomas": [("2001/3", "Fibrosarcoma, NOS")],
    "Rare Carcinomas": [("3001/3", "Squamous cell carcinoma, NOS")],
}

LABELS_HEADER = ["Vet-ICD-O-canine-1 code", "Group", "Term", "level", "Topography", "obs", ""]


def make_labels_csv(path: Path) -> Path:
    """Write a synthetic labels.csv, mirroring the real file's shape and its
    title-row quirk (BOM, quoted title row, real header on row 2)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([" Vet-ICD-O-canine-1, First Edition (synthetic) ", "", "", "", "", "", ""])
        writer.writerow(LABELS_HEADER)
        for group, entries in TAXONOMY_GROUPS.items():
            for code, term in entries:
                writer.writerow([code, group, term, "Preferred", "", "", ""])
    return path


# ---------------------------------------------------------------------------
# Synthetic reports / diagnoses (report.csv / diagnoses.csv shape)
# ---------------------------------------------------------------------------

CASE_IDS = [f"CASE-{i:04d}" for i in range(1, 9)]

REPORT_COLUMNS = [
    "case_id", "CLINICAL ABSTRACT", "GROSS DESCRIPTION",
    "HISTOPATHOLOGICAL SUMMARY", "COMMENT", "FINAL COMMENT",
    "ANCILLARY TESTS", "ADDENDUM", "ADDITIONAL INFORMATION",
]

# One invented (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT) pair per case,
# covering both cancer-like and no-evidence text so section splitting has
# something to chew on.
_REPORT_TEXT = [
    ("sections show a malignant mast cell tumor with high mitotic index", "margins incomplete"),
    ("no evidence of neoplasia in the skin specimen", "benign consistent with mild inflammation"),
    ("a mast cell sarcoma is present with severe inflammation", "margins clear complete excision"),
    ("fibrosarcoma cells are present in the biopsy specimen", "malignant grade high"),
    ("squamous cell carcinoma of the skin is present", "margins clear"),
    ("no evidence of neoplastic cells", "benign"),
    ("a benign mast cell tumor is present with mild inflammation", "margins clear complete excision"),
    ("sections show moderate inflammation and no neoplasia", "consistent with a benign specimen"),
]


def make_reports_csv(path: Path, case_ids: list[str] = CASE_IDS) -> Path:
    """Write a synthetic report.csv with invented section text."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(REPORT_COLUMNS)
        for case_id, (summary, final_comment) in zip(case_ids, _REPORT_TEXT):
            writer.writerow([case_id, "", "", summary, "", final_comment, "", "", ""])
    return path


DIAGNOSES_COLUMNS = ["case_id", "diagnosis_number", "diagnosis"]

# Aligned with _REPORT_TEXT: matches a taxonomy term where a cascade would
# find one, "NO EVIDENCE OF NEOPLASIA" where it would not.
_DIAGNOSIS_TEXT = [
    "MAST CELL TUMOR",
    "NO EVIDENCE OF NEOPLASIA",
    "MAST CELL SARCOMA",
    "FIBROSARCOMA",
    "SQUAMOUS CELL CARCINOMA",
    "NO EVIDENCE OF NEOPLASIA",
    "MAST CELL TUMOR BENIGN",
    "NO EVIDENCE OF NEOPLASIA",
]


def make_diagnoses_csv(path: Path, case_ids: list[str] = CASE_IDS) -> Path:
    """Write a synthetic diagnoses.csv with invented diagnosis text."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(DIAGNOSES_COLUMNS)
        for case_id, diagnosis in zip(case_ids, _DIAGNOSIS_TEXT):
            writer.writerow([case_id, 1, diagnosis])
    return path


# ---------------------------------------------------------------------------
# Synthetic silver annotation.csv
# ---------------------------------------------------------------------------

ANNOTATION_COLUMNS = [
    "case_id", "diagnosis_number", "diagnosis", "matched_term", "matched_group",
    "matched_code", "matched_keyword", "method", "confidence", "decision_stage",
]

# One row per (decision_stage, method) combination actually observed in the
# real annotation.csv (counted with pandas — no text was read or copied).
_ANNOTATION_ROWS = [
    # case_id,        diag#, diagnosis,                matched_term,            matched_group,   matched_code, matched_keyword, method,      confidence, decision_stage
    ("CASE-0001", 1, "NO EVIDENCE OF NEOPLASIA",   "", "", "", "", "No Match", 0.0, "no_signal"),
    ("CASE-0002", 1, "MAST CELL TUMOR",            "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3", "mast cell tumor", "Exact", 1.0, "tier1_exact"),
    ("CASE-0003", 1, "MAST CELL SARCOMA, PROBABLE", "Mast cell sarcoma", "Mast Cell Tumors", "1003/3", "mast cell sarcoma", "Fuzzy", 0.8, "tier2_fuzzy"),
    ("CASE-0004", 1, "FIBROSARCOMA OF SKIN",       "Fibrosarcoma, NOS", "Rare Sarcomas", "2001/3", "", "LLM", 0.7, "tier3_llm"),
    ("CASE-0005", 1, "AMBIGUOUS GROWTH",           "", "", "", "", "Uncertain", 0.4, "tier3_llm"),
    ("CASE-0006", 1, "UNRELATED FINDING",          "", "", "", "", "No Match", 0.1, "tier3_llm"),
    ("CASE-0007", 1, "UNCLASSIFIABLE LESION",      "", "", "", "", "No Match", 0.0, "tier3_no_candidates"),
]


def make_annotation_csv(path: Path) -> Path:
    """Write a synthetic silver annotation.csv covering every decision_stage."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(ANNOTATION_COLUMNS)
        for row in _ANNOTATION_ROWS:
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Legacy-style split (train/test case-id lists)
# ---------------------------------------------------------------------------


def make_split(split_dir: Path, case_ids: list[str] = CASE_IDS) -> tuple[Path, Path]:
    """Write train_cases.txt / test_cases.txt (one case_id per line, no header)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = case_ids[:6], case_ids[6:]
    train_path = split_dir / "train_cases.txt"
    test_path = split_dir / "test_cases.txt"
    train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    return train_path, test_path


# ---------------------------------------------------------------------------
# Model heads (random init, legacy-compatible state_dict layout)
# ---------------------------------------------------------------------------

TINY_HIDDEN_DIM = 16
GROUP_NAMES = list(TAXONOMY_GROUPS.keys())


def _head_classes():
    """report_mapping/model/heads.py's three head classes.

    State_dict-compatible with the pre-rewrite ml/model classes (WP4); this
    used to reach into the old ml/ tree by path (see git history) and no
    longer does, per the comment that used to be here.
    """
    from report_mapping.model.heads import CasePresenceClassifier, GroupClassifier, LabelPresenceClassifier

    return CasePresenceClassifier, GroupClassifier, LabelPresenceClassifier


def build_case_presence_head():
    CasePresenceClassifier, _, _ = _head_classes()
    return CasePresenceClassifier(emb_dim=TINY_EMB_DIM, hidden_dim=TINY_HIDDEN_DIM)


def build_group_head(group_names: list[str] = GROUP_NAMES):
    _, GroupClassifier, _ = _head_classes()
    return GroupClassifier(num_groups=len(group_names), emb_dim=TINY_EMB_DIM, hidden_dim=TINY_HIDDEN_DIM)


def build_label_presence_head():
    """n_cols=3 for the concat-3 section spec (HISTOPATHOLOGICAL SUMMARY,
    FINAL COMMENT, COMMENT)."""
    _, _, LabelPresenceClassifier = _head_classes()
    return LabelPresenceClassifier(emb_dim=TINY_EMB_DIM, hidden_dim=TINY_HIDDEN_DIM, n_cols=3)


# ---------------------------------------------------------------------------
# Report-mapping generation bundle (WP4): a full synthetic current/ directory
# ---------------------------------------------------------------------------

# The GroupClassifier's own output space: two "common" groups plus one merged
# "Uncommon" bucket standing in for the taxonomy's two rare groups.
REPORT_MAPPING_MERGED_GROUP_NAMES = ["Neoplasms, NOS", "Mast Cell Tumors", "Uncommon"]
REPORT_MAPPING_UNCOMMON_GROUPS = ["Rare Sarcomas", "Rare Carcinomas"]

REPORT_MAPPING_THRESHOLDS = {
    "case_presence_gate": 0.5,
    "group": 0.5,
    "tail_max_predictions": 2,
    "tail_max_group_prob_gap": 0.08,
    "label_presence_fallback": 0.5,
}


def build_report_mapping_bundle(root: Path, tiny_bert_dir: Path) -> Path:
    """Build a full synthetic generation directory at ``root`` in the report-mapping
    bundle layout: ``petbert/`` (a copy of ``tiny_bert_dir``), ``labels/labels.csv``,
    ``checkpoints/{case_presence_classifier.pt, group_classifier_best.pt,
    label_presence/*.pt, lp_thresholds.json, thresholds.json, uncommon_groups.txt}``,
    and a manifest with a real embedding fingerprint. Case-presence and group
    heads are sized for the 3x concat-3 input (``3 * TINY_EMB_DIM``); the LP
    head is already per-section (``build_label_presence_head``, n_cols=3).
    """
    import shutil

    from generations.manifest import write_manifest
    from report_mapping.model.generation import compute_embedding_fingerprint, generation_paths, safe_filename
    from report_mapping.model.heads import CasePresenceClassifier, GroupClassifier

    paths = generation_paths(root)
    shutil.copytree(tiny_bert_dir, paths.petbert_dir)
    paths.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    make_labels_csv(paths.labels_csv)

    concat_dim = 3 * TINY_EMB_DIM
    paths.case_presence_pt.parent.mkdir(parents=True, exist_ok=True)
    CasePresenceClassifier(emb_dim=concat_dim, hidden_dim=TINY_HIDDEN_DIM).save(paths.case_presence_pt)
    GroupClassifier(
        num_groups=len(REPORT_MAPPING_MERGED_GROUP_NAMES), emb_dim=concat_dim, hidden_dim=TINY_HIDDEN_DIM,
    ).save(paths.group_pt, REPORT_MAPPING_MERGED_GROUP_NAMES)

    paths.label_presence_dir.mkdir(parents=True, exist_ok=True)
    for group_name in REPORT_MAPPING_MERGED_GROUP_NAMES:
        build_label_presence_head().save(paths.label_presence_dir / f"{safe_filename(group_name)}.pt")
    (paths.lp_thresholds_json).write_text(
        json.dumps({g: 0.5 for g in REPORT_MAPPING_MERGED_GROUP_NAMES}, indent=2) + "\n", encoding="utf-8"
    )
    paths.thresholds_json.write_text(json.dumps(REPORT_MAPPING_THRESHOLDS, indent=2) + "\n", encoding="utf-8")
    paths.uncommon_groups_txt.write_text("\n".join(REPORT_MAPPING_UNCOMMON_GROUPS) + "\n", encoding="utf-8")

    fingerprint = compute_embedding_fingerprint(paths.petbert_dir)
    write_manifest(root, {
        "kind": "report_mapping_generation",
        "generation_id": "test-gen",
        "embedding_fingerprint": fingerprint,
        # A ready-to-use test generation is calibrated by default, like "current" in
        # production; tests exercising an uncalibrated candidate patch this themselves.
        "calibration": {"status": "calibrated"},
    })
    return root


# ---------------------------------------------------------------------------
# Split generations (generations/): config redirection + legacy flat files
# ---------------------------------------------------------------------------


def point_generations_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path generations/ reads or writes into ``root``.

    Legacy flat split files sit in ``root/output/splits/`` beside the new
    ``<split_id>/`` dirs, as they do in the real tree.
    """
    import config

    splits_dir = root / "output" / "splits"
    monkeypatch.setattr(config, "ML_ROOT", root)
    monkeypatch.setattr(config, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(config, "LEGACY_TRAIN_CASES_TXT", splits_dir / "train_cases.txt")
    monkeypatch.setattr(config, "LEGACY_TEST_CASES_TXT", splits_dir / "test_cases.txt")
    monkeypatch.setattr(config, "LEGACY_TRAIN_CASES_TEMPORAL_TXT", splits_dir / "train_cases_temporal.txt")
    monkeypatch.setattr(config, "LEGACY_TEST_CASES_TEMPORAL_TXT", splits_dir / "test_cases_temporal.txt")
    monkeypatch.setattr(config, "GOLD_STORE_CSV", root / "output" / "manual_audit" / "gold_store.csv")
    monkeypatch.setattr(config, "CORRECTED_ANNOTATIONS_CSV", root / "output" / "coding" / "corrected_annotations.csv")


SPLIT_CASE_IDS = [f"CASE-{i:04d}" for i in range(1, 41)]


def make_legacy_split_files(case_ids: list[str] = SPLIT_CASE_IDS) -> None:
    """Write the four legacy flat split files at the (redirected) config paths.

    Sorted IDs with CRLF line endings, like the real files. Random split: first
    80% train; temporal split: last 5 cases test.
    """
    import config

    def write(path: Path, ids: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("".join(f"{c}\r\n" for c in sorted(ids)).encode("utf-8"))

    n_train = len(case_ids) * 4 // 5
    write(config.LEGACY_TRAIN_CASES_TXT, case_ids[:n_train])
    write(config.LEGACY_TEST_CASES_TXT, case_ids[n_train:])
    write(config.LEGACY_TRAIN_CASES_TEMPORAL_TXT, case_ids[:-5])
    write(config.LEGACY_TEST_CASES_TEMPORAL_TXT, case_ids[-5:])


# ---------------------------------------------------------------------------
# Silver generations (diagnosis_mapping/): config redirection
# ---------------------------------------------------------------------------


def point_diagnosis_mapping_config_at(monkeypatch, root: Path) -> None:
    """Redirect config.SILVER_DIR and config.LEGACY_ANNOTATION_CSV into ``root``.

    Mirrors ``point_generations_config_at`` for the silver-generation tests:
    SILVER_DIR and LEGACY_ANNOTATION_CSV are module-level constants computed
    once at import, so they need their own monkeypatch even when ML_ROOT is
    also redirected.
    """
    import config

    monkeypatch.setattr(config, "ML_ROOT", root)
    monkeypatch.setattr(config, "SILVER_DIR", root / "output" / "silver")
    monkeypatch.setattr(config, "LEGACY_ANNOTATION_CSV", root / "output" / "annotation" / "annotation.csv")


# ---------------------------------------------------------------------------
# Manual audit (manual_audit/): config redirection + a stratifiable silver csv
# ---------------------------------------------------------------------------


def point_manual_audit_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path manual_audit/ reads or writes into ``root``.

    Builds on ``point_generations_config_at`` (manual_audit tests also build
    split generations) and additionally redirects the audit/gold/ledger/cause
    stores, the Tier-3 sheet paths, the eval-batch sheet dir, the silver dir,
    and the legacy Tier-3 sheets dir (source of the batch-1 exclusion ledger).
    """
    import config

    point_generations_config_at(monkeypatch, root)
    manual_audit_dir = root / "output" / "manual_audit"
    tier3_dir = manual_audit_dir / "tier3_audit"
    monkeypatch.setattr(config, "MANUAL_AUDIT_DIR", manual_audit_dir)
    monkeypatch.setattr(config, "AUDIT_STORE_CSV", manual_audit_dir / "audit_store.csv")
    monkeypatch.setattr(config, "GOLD_STORE_CSV", manual_audit_dir / "gold_store.csv")
    monkeypatch.setattr(config, "EVAL_BATCH_LEDGER_CSV", manual_audit_dir / "eval_batch_ledger.csv")
    monkeypatch.setattr(config, "CAUSE_STORE_CSV", manual_audit_dir / "cause_store.csv")
    monkeypatch.setattr(config, "TIER3_AUDIT_DIR", tier3_dir)
    monkeypatch.setattr(config, "TIER3_AUDIT_REVIEW_CSV", tier3_dir / "tier3_audit_review.csv")
    monkeypatch.setattr(config, "TIER3_AUDIT_KEY_CSV", tier3_dir / "tier3_audit_key.csv")
    monkeypatch.setattr(config, "TIER3_AUDIT_INSTRUCTIONS_MD", tier3_dir / "tier3_audit_instructions.md")
    monkeypatch.setattr(config, "TIER3_AUDIT_TAXONOMY_CSV", tier3_dir / "tier3_audit_taxonomy.csv")
    monkeypatch.setattr(config, "TIER3_AUDIT_PILOT_CSV", tier3_dir / "tier3_audit_pilot_review.csv")
    monkeypatch.setattr(config, "TIER3_AUDIT_REMAINDER_CSV", tier3_dir / "tier3_audit_remainder_review.csv")
    monkeypatch.setattr(config, "TIER3_AUDIT_PILOT_INSTRUCTIONS_MD", tier3_dir / "tier3_audit_pilot_instructions.md")
    monkeypatch.setattr(config, "EVAL_BATCH_DIR", manual_audit_dir / "eval_batch")
    monkeypatch.setattr(config, "SILVER_DIR", root / "output" / "silver")
    monkeypatch.setattr(config, "LEGACY_TIER3_AUDIT_SHEETS_DIR", root / "output" / "annotation")
    # Without this, eval_batch's one-time copy writes a test id list into the real output tree.
    monkeypatch.setattr(config, "TIER3_AUDIT_BATCH1_EXCLUSION_TXT", manual_audit_dir / "tier3_audit_batch1_exclusion.txt")


# Case ids for a stratifiable silver csv: enough cases per group to exercise
# allocation with a floor, plus two multi-group cases exercising the "lowest
# diagnosis_number with a non-empty group" rule.
EVAL_BATCH_CASE_IDS = [f"CASE-{i:04d}" for i in range(1, 31)]


def make_eval_batch_silver_csv(path: Path, case_ids: list[str] = EVAL_BATCH_CASE_IDS) -> Path:
    """Write a silver-shaped annotation.csv for eval_batch stratification tests.

    Groups: CASE-0001..0010 -> "Mast Cell Tumors"; CASE-0011..0018 -> "Rare
    Sarcomas"; CASE-0019..0020 -> "Rare Carcinomas"; CASE-0021..0028 -> no
    non-empty group (no_cancer stratum). Two multi-group cases exercise the
    stratum rule: CASE-0029 has an empty-group row at diagnosis_number 1 and a
    "Mast Cell Tumors" row at 2 (expected stratum: Mast Cell Tumors, its only
    non-empty group); CASE-0030 has "Rare Sarcomas" at diagnosis_number 1 and
    "Rare Carcinomas" at 2 (expected stratum: Rare Sarcomas, the lowest
    diagnosis_number with a non-empty group).
    """
    assert len(case_ids) == 30, "make_eval_batch_silver_csv assumes the default 30-case layout"
    group_of = {}
    for cid in case_ids[0:10]:
        group_of[cid] = "Mast Cell Tumors"
    for cid in case_ids[10:18]:
        group_of[cid] = "Rare Sarcomas"
    for cid in case_ids[18:20]:
        group_of[cid] = "Rare Carcinomas"
    for cid in case_ids[20:28]:
        group_of[cid] = ""

    rows = []
    for cid in case_ids[:28]:
        group = group_of[cid]
        rows.append((cid, 1, "", "" if not group else "invented term", group,
                     "" if not group else "9001/3", "", "Exact" if group else "No Match",
                     1.0 if group else 0.0, "tier1_exact" if group else "no_signal"))
    rows.append((case_ids[28], 1, "", "", "", "", "", "No Match", 0.0, "no_signal"))
    rows.append((case_ids[28], 2, "", "invented term", "Mast Cell Tumors", "9001/3", "", "Exact", 1.0, "tier1_exact"))
    rows.append((case_ids[29], 1, "", "invented term", "Rare Sarcomas", "9002/3", "", "Exact", 1.0, "tier1_exact"))
    rows.append((case_ids[29], 2, "", "invented term", "Rare Carcinomas", "9003/3", "", "Exact", 1.0, "tier1_exact"))

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(ANNOTATION_COLUMNS)
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# report_mapping/training/: config redirection, a real (directly-written)
# split generation, and a synthetic labels table + report.csv sized to clear
# every trainer's minimum-count floor (gate class balance, group's
# min_group_cases=10, LP's >= 10 pairs).
# ---------------------------------------------------------------------------


def point_training_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path report_mapping.training reads or writes into
    ``root``: builds on ``point_generations_config_at`` (split dirs, gold
    store, corrected annotations) and adds ``report.csv``, ``labels.csv``, and
    the report-mapping generation/cache directories."""
    import config

    point_generations_config_at(monkeypatch, root)
    monkeypatch.setattr(config, "REPORT_CSV", root / "data" / "report.csv")
    monkeypatch.setattr(config, "LABELS_CSV", root / "labels.csv")
    report_mapping_dir = root / "output" / "report_mapping"
    monkeypatch.setattr(config, "REPORT_MAPPING_DIR", report_mapping_dir)
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", report_mapping_dir / "current")
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", report_mapping_dir / "candidate")
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", report_mapping_dir / "embedding_cache")


def make_two_way_split_generation(split_id: str, train_ids: list[str], test_ids: list[str] = ()) -> None:
    """Write a minimal, directly-loadable train/test split generation (no
    calibration partition) for training tests that need a real split without
    going through the legacy-file-shaped ``import_legacy()``."""
    from generations.manifest import write_manifest
    from generations.splits import TEST_FILE, TRAIN_FILE, split_dir

    directory = split_dir(split_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TRAIN_FILE).write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    if test_ids:
        (directory / TEST_FILE).write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    write_manifest(directory, {
        "split_id": split_id, "method": "synthetic test split", "seed": None, "stratification": None,
        "parent": None, "counts": {"train": len(train_ids), "calibration": 0, "test": len(test_ids)},
    })


TRAINING_CASE_IDS = [f"TCASE-{i:04d}" for i in range(1, 31)]
_TRAINING_GROUPS_CYCLE = ["Mast Cell Tumors", "Rare Sarcomas", "Rare Carcinomas"]


def make_training_annotation_csv(path: Path, case_ids: list[str] = TRAINING_CASE_IDS) -> Path:
    """15 cancer-positive cases (5 per group across 3 groups, each group with
    < 200 cases so ``report_mapping.training.group`` merges every group into
    the shared "Uncommon" head) + 15 non-cancer cases — sized so the gate,
    group (``min_group_cases=10``) and LP (>= 10 pairs) trainers all clear
    their floors on synthetic data."""
    assert len(case_ids) == 30, "make_training_annotation_csv assumes the default 30-case layout"
    rows = []
    cancer_ids, noncancer_ids = case_ids[:15], case_ids[15:]
    for i, cid in enumerate(cancer_ids):
        group = _TRAINING_GROUPS_CYCLE[i % len(_TRAINING_GROUPS_CYCLE)]
        entries = TAXONOMY_GROUPS[group]
        code, term = entries[i % len(entries)]
        rows.append((cid, 1, "invented diagnosis text", term, group, code, "", "Exact", 1.0, "tier1_exact"))
    for cid in noncancer_ids:
        rows.append((cid, 1, "invented diagnosis text", "", "", "", "", "No Match", 0.0, "no_signal"))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(ANNOTATION_COLUMNS)
        writer.writerows(rows)
    return path


def make_training_reports_csv(path: Path, case_ids: list[str] = TRAINING_CASE_IDS) -> Path:
    """report.csv-shaped file for the same 30 cases, with all three concat-3
    sections non-empty and long enough (>= 10 chars) on every row, so every
    case contributes full pairs to both the embedding cache and the
    contrastive-pairs builder."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(REPORT_COLUMNS)
        for i, case_id in enumerate(case_ids):
            summary, final_comment = _REPORT_TEXT[i % len(_REPORT_TEXT)]
            comment = "consistent with mild inflammation"
            ancillary = "immunohistochemistry positive"
            writer.writerow([case_id, "", "", summary, comment, final_comment, ancillary, "", ""])
    return path


# ---------------------------------------------------------------------------
# Silver generations for manual_audit tests (diagnosis_mapping.silver.load_silver)
# ---------------------------------------------------------------------------


def make_silver_generation(silver_id: str, rows: list[tuple], columns: list[str] = ANNOTATION_COLUMNS) -> str:
    """Write a minimal silver generation (annotation.csv + manifest.json) at the
    (redirected) config.SILVER_DIR, loadable via diagnosis_mapping.silver.load_silver.

    The manifest is written last, per generations.manifest's convention (a
    manifest may not list files added after it was written).
    """
    import config
    from generations.manifest import write_manifest

    directory = config.SILVER_DIR / silver_id
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "annotation.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    write_manifest(directory, {"silver_id": silver_id, "llm_enabled": True})
    return silver_id


def make_tier3_silver_generation(silver_id: str = "tier3-test-silver") -> str:
    """Same rows as ``make_annotation_csv``, as a loadable silver generation."""
    return make_silver_generation(silver_id, _ANNOTATION_ROWS)


def make_eval_batch_silver_generation(
    silver_id: str = "eval-batch-test-silver", case_ids: list[str] = EVAL_BATCH_CASE_IDS,
) -> str:
    """A silver generation for eval_batch's code-targeted allocation tests.

    Same 30-case group layout as ``make_eval_batch_silver_csv`` (see its
    docstring for the group assignments and the two multi-group cases), plus
    one extra row on CASE-0001 giving it a second distinct code in its own
    group ("Mast Cell Tumors"), so that stratum's codes_per_case_h > 1 and the
    code-targeted formula's division has something to divide.
    """
    assert len(case_ids) == 30, "make_eval_batch_silver_generation assumes the default 30-case layout"
    group_of = {}
    for cid in case_ids[0:10]:
        group_of[cid] = "Mast Cell Tumors"
    for cid in case_ids[10:18]:
        group_of[cid] = "Rare Sarcomas"
    for cid in case_ids[18:20]:
        group_of[cid] = "Rare Carcinomas"
    for cid in case_ids[20:28]:
        group_of[cid] = ""

    rows = []
    for cid in case_ids[:28]:
        group = group_of[cid]
        rows.append((cid, 1, "", "" if not group else "invented term", group,
                     "" if not group else "9001/3", "", "Exact" if group else "No Match",
                     1.0 if group else 0.0, "tier1_exact" if group else "no_signal"))
    rows.append((case_ids[0], 2, "", "invented term two", "Mast Cell Tumors", "9004/3", "",
                "Exact", 1.0, "tier1_exact"))
    rows.append((case_ids[28], 1, "", "", "", "", "", "No Match", 0.0, "no_signal"))
    rows.append((case_ids[28], 2, "", "invented term", "Mast Cell Tumors", "9001/3", "", "Exact", 1.0, "tier1_exact"))
    rows.append((case_ids[29], 1, "", "invented term", "Rare Sarcomas", "9002/3", "", "Exact", 1.0, "tier1_exact"))
    rows.append((case_ids[29], 2, "", "invented term", "Rare Carcinomas", "9003/3", "", "Exact", 1.0, "tier1_exact"))

    return make_silver_generation(silver_id, rows)


# ---------------------------------------------------------------------------
# Calibration (report_mapping/training/calibrate.py): a three-way split + labels
# table over the synthetic report cases
# ---------------------------------------------------------------------------

# Aligned with _REPORT_TEXT / CASE_IDS: the cancer cases carry a bundle taxonomy term.
CALIBRATION_LABEL_ROWS = [
    ("CASE-0001", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3"),
    ("CASE-0002", "", "", ""),
    ("CASE-0003", "Mast cell sarcoma", "Mast Cell Tumors", "1003/3"),
    ("CASE-0004", "Fibrosarcoma, NOS", "Rare Sarcomas", "2001/3"),
    ("CASE-0005", "Squamous cell carcinoma, NOS", "Rare Carcinomas", "3001/3"),
    ("CASE-0006", "", "", ""),
    ("CASE-0007", "Mast cell tumor, benign", "Mast Cell Tumors", "1002/0"),
    ("CASE-0008", "", "", ""),
]


def make_calibration_labels_csv(path: Path) -> Path:
    """A labels table (case_id, matched_term, matched_group, matched_code) for CASE_IDS."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "matched_term", "matched_group", "matched_code"])
        writer.writerows(CALIBRATION_LABEL_ROWS)
    return path


def make_three_way_split_generation(split_id: str, train: list[str], calibration: list[str], test: list[str]) -> str:
    """Write a three-way split generation at the (redirected) config.SPLITS_DIR, manifest last."""
    import config
    from generations.manifest import write_manifest

    directory = config.SPLITS_DIR / split_id
    directory.mkdir(parents=True)
    for name, ids in (("train_cases.txt", train), ("calibration_cases.txt", calibration), ("test_cases.txt", test)):
        (directory / name).write_text("\n".join(ids) + "\n", encoding="utf-8")
    write_manifest(directory, {"split_id": split_id, "parent": None})
    return split_id


# ---------------------------------------------------------------------------
# coding/ (rule, adopt, corrected, queue): config redirection, a bronze
# predictions csv factory, and a direct gold-store writer.
# ---------------------------------------------------------------------------


def point_coding_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path coding/ reads or writes into ``root``.

    Builds on ``point_manual_audit_config_at`` (silver dir, gold store, split
    dirs, eval-batch ledger) and adds the coding outputs.
    """
    import config

    point_manual_audit_config_at(monkeypatch, root)
    coding_dir = root / "output" / "coding"
    monkeypatch.setattr(config, "CODING_DIR", coding_dir)
    monkeypatch.setattr(config, "CORRECTED_ANNOTATIONS_CSV", coding_dir / "corrected_annotations.csv")
    monkeypatch.setattr(config, "ADOPTED_CODES_CSV", coding_dir / "adopted_codes.csv")
    monkeypatch.setattr(config, "REVIEW_QUEUE_CSV", coding_dir / "review_queue.csv")


BRONZE_PREDICTIONS_COLUMNS = [
    "case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
    "case_presence_prob", "confidence", "group_prob", "method", "generation_id",
]


def make_bronze_predictions_csv(path: Path, rows: list[tuple]) -> Path:
    """Write a synthetic bronze predictions.csv (``report_mapping.inference.
    predict``'s output shape) for coding/ tests. Each row is a full
    ``BRONZE_PREDICTIONS_COLUMNS`` tuple."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(BRONZE_PREDICTIONS_COLUMNS)
        writer.writerows(rows)
    return path


def make_gold_store_csv(rows: list[dict]) -> None:
    """Write a gold store CSV directly at the (redirected) config.GOLD_STORE_CSV,
    bypassing ``manual_audit.gold.ingest_gold``'s validation — for coding/
    tests that need a specific gold shape without a full ingest. ``rows`` are
    dicts; any ``GOLD_STORE_FIELDS`` column left out defaults to ``""``.
    """
    import config
    import io_utils
    from manual_audit.gold import GOLD_STORE_FIELDS

    full_rows = [{col: row.get(col, "") for col in GOLD_STORE_FIELDS} for row in rows]
    df = pd.DataFrame(full_rows, columns=GOLD_STORE_FIELDS)
    config.GOLD_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, config.GOLD_STORE_CSV)


# One shared coding/ scenario, covering gold/silver/bronze precedence, the
# vagueness split, and every bronze low-confidence trigger (method flag,
# confidence floor, margin floor). Case names describe their role in that
# scenario; see ``build_coding_scenario``'s docstring for the full layout.
class CodingCaseIDs:
    G_DIAG = "CASE-G-DIAG"                    # gold overrides decisive silver
    G_VAGUE = "CASE-G-VAGUE"                  # gold overrides vague silver
    G_NOCANCER = "CASE-G-NOCANCER"            # gold NO_CANCER overrides vague silver
    G_BRONZE = "CASE-G-BRONZE"                # gold overrides low-confidence bronze (no diagnosis)
    DECISIVE_CANCER = "CASE-DECISIVE-CANCER"  # decisive silver, no gold -> diagnosis-sourced
    DECISIVE_NONCANCER = "CASE-DECISIVE-NONCANCER"  # decisive no_signal, no gold -> NO_CANCER
    VAGUE_NOGOLD = "CASE-VAGUE-NOGOLD"        # vague silver, no gold, train -> queued only
    TEST_VAGUE = "CASE-TEST-VAGUE"            # vague silver, no gold, test partition
    BRONZE_HIGH = "CASE-BRONZE-HIGH"          # no diagnosis, high-confidence bronze, 2 codes
    BRONZE_LOW_CONF = "CASE-BRONZE-LOW-CONF"  # no diagnosis, confidence below the floor
    BRONZE_LOW_MARGIN = "CASE-BRONZE-LOW-MARGIN"  # no diagnosis, top1-top2 margin below the floor
    BRONZE_METHOD_FLAG = "CASE-BRONZE-METHOD-FLAG"  # no diagnosis, gate-rejected (blank code)


CODING_SILVER_ID = "coding-test-silver"
CODING_SPLIT_ID = "coding-test-split"
CODING_GENERATION_ID = "coding-test-gen"


def build_coding_scenario(monkeypatch, root: Path) -> dict:
    """Build the full coding/ test scenario: config redirection, a three-way
    split, a silver generation, a gold store and a bronze predictions csv.

    Layout (see ``CodingCaseIDs`` for the case names):
    - G_DIAG / G_VAGUE / G_NOCANCER / G_BRONZE all have review_queue-origin
      gold that must win over whatever silver or bronze says for that case.
    - DECISIVE_CANCER has a decisive silver row AND a disagreeing
      high-confidence bronze row, to prove bronze never overrides it.
    - DECISIVE_NONCANCER is decisive ``no_signal`` (no gold, no bronze).
    - VAGUE_NOGOLD (train) and TEST_VAGUE (test) are vague with no gold, at
      case_presence_prob 0.9 / 0.4 respectively (queue priority ordering).
    - BRONZE_HIGH/LOW_CONF/LOW_MARGIN/METHOD_FLAG have no diagnosis rows at
      all; each exercises one bronze review-gate trigger (or none, for HIGH).

    Returns ``{"split_id", "silver_id", "generation_id", "predictions_csv"}``.
    """
    import config

    point_coding_config_at(monkeypatch, root)
    C = CodingCaseIDs

    make_three_way_split_generation(
        CODING_SPLIT_ID,
        train=[
            C.G_DIAG, C.G_VAGUE, C.G_NOCANCER, C.G_BRONZE, C.DECISIVE_CANCER, C.DECISIVE_NONCANCER,
            C.VAGUE_NOGOLD, C.BRONZE_HIGH, C.BRONZE_LOW_CONF, C.BRONZE_LOW_MARGIN, C.BRONZE_METHOD_FLAG,
        ],
        calibration=[],
        test=[C.TEST_VAGUE],
    )

    silver_rows = [
        (C.G_DIAG, 1, "invented diagnosis text", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3",
         "mast cell tumor", "Exact", 1.0, "tier1_exact"),
        (C.G_VAGUE, 1, "invented diagnosis text", "", "", "", "", "Uncertain", 0.4, "tier3_llm"),
        (C.G_NOCANCER, 1, "invented diagnosis text", "", "", "", "", "No Match", 0.0, "tier3_no_candidates"),
        (C.DECISIVE_CANCER, 1, "invented diagnosis text", "Mast cell sarcoma", "Mast Cell Tumors", "1003/3",
         "mast cell sarcoma", "Exact", 1.0, "tier1_exact"),
        (C.DECISIVE_NONCANCER, 1, "invented diagnosis text", "", "", "", "", "No Match", 0.0, "no_signal"),
        (C.VAGUE_NOGOLD, 1, "invented diagnosis text", "", "", "", "", "Uncertain", 0.4, "tier3_llm"),
        (C.TEST_VAGUE, 1, "invented diagnosis text", "", "", "", "", "No Match", 0.0, "tier3_no_candidates"),
    ]
    make_silver_generation(CODING_SILVER_ID, silver_rows)

    make_gold_store_csv([
        {"case_id": C.G_DIAG, "code": "9990/3", "term": "Gold Term A", "group": "Gold Group A", "origin": "review_queue"},
        {"case_id": C.G_VAGUE, "code": "9991/3", "term": "Gold Term B", "group": "Gold Group B", "origin": "review_queue"},
        {"case_id": C.G_NOCANCER, "code": "NO_CANCER", "term": "", "group": "", "origin": "review_queue"},
        {"case_id": C.G_BRONZE, "code": "9992/3", "term": "Gold Term C", "group": "Gold Group C", "origin": "review_queue"},
    ])

    predictions_rows = [
        # Disagrees with DECISIVE_CANCER's silver code on purpose (never adopted).
        (C.DECISIVE_CANCER, 1, "Wrong Term", "Wrong Group", "9999/0", "0.9500", "0.95", "0.95",
         "label_presence", CODING_GENERATION_ID),
        (C.VAGUE_NOGOLD, 1, "V-Term", "V-Group", "9993/3", "0.9000", "0.90", "0.90",
         "label_presence", CODING_GENERATION_ID),
        (C.TEST_VAGUE, 1, "T-Term", "T-Group", "9994/3", "0.4000", "0.40", "0.40",
         "label_presence", CODING_GENERATION_ID),
        (C.G_VAGUE, 1, "GV-Term", "GV-Group", "9995/3", "0.8000", "0.80", "0.80",
         "label_presence", CODING_GENERATION_ID),
        (C.G_BRONZE, 1, "B-Term-G", "B-Group-G", "B900", "0.0500", "0.05", "0.05",
         "label_presence", CODING_GENERATION_ID),
        (C.BRONZE_HIGH, 1, "B-Term1", "B-Group1", "B001", "0.9000", "0.90", "0.90",
         "label_presence", CODING_GENERATION_ID),
        (C.BRONZE_HIGH, 2, "B-Term2", "B-Group1", "B002", "0.9000", "0.60", "0.50",
         "label_presence", CODING_GENERATION_ID),
        (C.BRONZE_LOW_CONF, 1, "B-Term3", "B-Group3", "B003", "0.1000", "0.10", "0.10",
         "label_presence", CODING_GENERATION_ID),
        (C.BRONZE_LOW_MARGIN, 1, "B-Term4", "B-Group4", "B004", "0.5000", "0.50", "0.50",
         "label_presence", CODING_GENERATION_ID),
        (C.BRONZE_LOW_MARGIN, 2, "B-Term5", "B-Group4", "B005", "0.5000", "0.45", "0.45",
         "label_presence", CODING_GENERATION_ID),
        (C.BRONZE_METHOD_FLAG, 1, "Non-Cancer", "Non-Cancer", "", "0.0500", "0.05", "0.05",
         "rejected_by_case_presence", CODING_GENERATION_ID),
    ]
    predictions_csv = make_bronze_predictions_csv(root / "predictions.csv", predictions_rows)

    return {
        "split_id": CODING_SPLIT_ID, "silver_id": CODING_SILVER_ID,
        "generation_id": CODING_GENERATION_ID, "predictions_csv": predictions_csv,
    }


# ---------------------------------------------------------------------------
# evaluation/ (silver_eval, gold_eval, audit_rates): config redirection and a
# minimal generation directory (manifest + uncommon groups, no weights)
# ---------------------------------------------------------------------------


def point_evaluation_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path evaluation/ reads or writes into ``root``
    (builds on ``point_coding_config_at``: splits, silver, gold, ledger, cause and
    audit stores, adopted codes)."""
    import config

    point_coding_config_at(monkeypatch, root)
    eval_dir = root / "output" / "eval"
    monkeypatch.setattr(config, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(config, "SILVER_EVAL_HISTORY_CSV", eval_dir / "silver_eval_history.csv")
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", root / "output" / "report_mapping" / "current")


# ---------------------------------------------------------------------------
# handoff/ (contracts, imports, exports, worker_format): config redirection
# ---------------------------------------------------------------------------


def point_handoff_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path handoff/ reads or writes into ``root``.

    Builds on ``point_coding_config_at`` (silver dir, gold store, eval-batch
    ledger, adopted codes, review queue) and adds the inbox/outbox dirs and
    the report-mapping generation dirs (``export_bundle``'s source).
    """
    import config

    point_coding_config_at(monkeypatch, root)
    handoff_dir = root / "output" / "handoff"
    monkeypatch.setattr(config, "HANDOFF_DIR", handoff_dir)
    monkeypatch.setattr(config, "HANDOFF_INBOX_DIR", handoff_dir / "inbox")
    monkeypatch.setattr(config, "HANDOFF_OUTBOX_DIR", handoff_dir / "outbox")
    monkeypatch.setattr(config, "HANDOFF_PENDING_DIAGNOSES_CSV", handoff_dir / "inbox" / "pending_diagnoses.csv")
    monkeypatch.setattr(config, "HANDOFF_BUNDLES_DIR", handoff_dir / "outbox" / "bundles")
    report_mapping_dir = root / "output" / "report_mapping"
    monkeypatch.setattr(config, "REPORT_MAPPING_DIR", report_mapping_dir)
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", report_mapping_dir / "current")
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", report_mapping_dir / "candidate")


def make_minimal_generation(directory: Path, generation_id: str, uncommon_groups: list[str]) -> Path:
    """Just what evaluation reads from a generation: manifest.json's generation_id
    and checkpoints/uncommon_groups.txt."""
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    (directory / "checkpoints" / "uncommon_groups.txt").write_text(
        "".join(f"{g}\n" for g in uncommon_groups), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"generation_id": generation_id}), encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# Promotion (generations/promote.py, triggers.py): config redirection
# ---------------------------------------------------------------------------


def point_promotion_config_at(monkeypatch, root: Path) -> None:
    """Everything ``point_training_config_at`` redirects, plus the eval-batch
    ledger, the silver dir and ``ARCHIVE_ROOT``, so promotion tests never touch
    a real ml/output path."""
    import config

    point_training_config_at(monkeypatch, root)
    monkeypatch.setattr(config, "EVAL_BATCH_LEDGER_CSV", root / "output" / "manual_audit" / "eval_batch_ledger.csv")
    monkeypatch.setattr(config, "SILVER_DIR", root / "output" / "silver")
    monkeypatch.setattr(config, "ARCHIVE_ROOT", root / "output" / "archive")
