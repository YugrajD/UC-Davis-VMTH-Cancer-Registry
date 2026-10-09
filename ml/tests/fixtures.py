"""Synthetic fixture factories shared by every ml/ test module.

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

import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

# ---------------------------------------------------------------------------
# Tiny BERT (the tokenizer and starting point for the tiny ModernBERT below)
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
    in well under a second, so tests never need a real checkpoint download.
    Loadable via AutoModel/AutoModelForMaskedLM + AutoTokenizer.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + VOCAB_WORDS
    vocab_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    # transformers 5 takes the vocab itself (a ``vocab_file`` argument is silently ignored).
    tokenizer = BertTokenizer(vocab={token: i for i, token in enumerate(tokens)}, do_lower_case=True)
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
    # Saved as a MaskedLM checkpoint; AutoModel can still load it since transformers strips the "bert." prefix.
    model = BertForMaskedLM(config)
    model.save_pretrained(str(model_dir))
    return model_dir


def build_tiny_modernbert(model_dir: Path) -> Path:
    """A tiny random-init ModernBERT + the tiny word-level tokenizer, in HF format: the offline stand-in for
    BioClinical ModernBERT that ``report_mapping.model.new_encoder`` loads."""
    from transformers import ModernBertConfig, ModernBertModel

    build_tiny_bert(model_dir)
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=TINY_EMB_DIM, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id,
        global_attn_every_n_layers=2, local_attention=8,
    )
    torch.manual_seed(0)
    ModernBertModel(config).save_pretrained(str(model_dir))
    return model_dir


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
# Split generations (generations/): config redirection + legacy flat files
# ---------------------------------------------------------------------------


def point_generations_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path generations/ reads or writes into ``root``."""
    import config

    splits_dir = root / "output" / "splits"
    monkeypatch.setattr(config, "ML_ROOT", root)
    monkeypatch.setattr(config, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(config, "GOLD_STORE_CSV", root / "output" / "manual_audit" / "gold_store.csv")
    monkeypatch.setattr(config, "CORRECTED_ANNOTATIONS_CSV", root / "output" / "coding" / "corrected_annotations.csv")


SPLIT_CASE_IDS = [f"CASE-{i:04d}" for i in range(1, 41)]


# ---------------------------------------------------------------------------
# Silver generations (diagnosis_mapping/): config redirection
# ---------------------------------------------------------------------------


def point_diagnosis_mapping_config_at(monkeypatch, root: Path) -> None:
    """Redirect config.SILVER_DIR into ``root``.

    Mirrors ``point_generations_config_at`` for the silver-generation tests:
    SILVER_DIR is a module-level constant computed once at import, so it needs
    its own monkeypatch even when ML_ROOT is also redirected.
    """
    import config

    monkeypatch.setattr(config, "ML_ROOT", root)
    monkeypatch.setattr(config, "SILVER_DIR", root / "output" / "silver")


# ---------------------------------------------------------------------------
# Manual audit (manual_audit/): config redirection + a stratifiable silver csv
# ---------------------------------------------------------------------------


def point_manual_audit_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path manual_audit/ reads or writes into ``root``.

    Builds on ``point_generations_config_at`` (manual_audit tests also build
    split generations) and additionally redirects the audit/gold/ledger/cause
    stores, the audit directories and list ledger, the eval-batch sheet dir, and the silver dir.
    """
    import config

    point_generations_config_at(monkeypatch, root)
    manual_audit_dir = root / "output" / "manual_audit"
    monkeypatch.setattr(config, "MANUAL_AUDIT_DIR", manual_audit_dir)
    monkeypatch.setattr(config, "AUDIT_STORE_CSV", manual_audit_dir / "audit_store.csv")
    monkeypatch.setattr(config, "GOLD_STORE_CSV", manual_audit_dir / "gold_store.csv")
    monkeypatch.setattr(config, "EVAL_BATCH_LEDGER_CSV", manual_audit_dir / "eval_batch_ledger.csv")
    monkeypatch.setattr(config, "CAUSE_STORE_CSV", manual_audit_dir / "cause_store.csv")
    monkeypatch.setattr(config, "DIAGNOSIS_MAPPING_AUDIT_DIR", manual_audit_dir / "diagnosis_mapping_audit")
    monkeypatch.setattr(config, "DIAGNOSIS_MAPPING_AUDIT_BATCH1_TXT",
                        manual_audit_dir / "diagnosis_mapping_audit" / "diagnosis_mapping_audit_batch1.txt")
    monkeypatch.setattr(config, "REPORT_MAPPING_AUDIT_DIR", manual_audit_dir / "report_mapping_audit")
    monkeypatch.setattr(config, "AUDIT_LIST_LEDGER_CSV", manual_audit_dir / "audit_list_ledger.csv")
    monkeypatch.setattr(config, "EVAL_BATCH_DIR", manual_audit_dir / "eval_batch")
    monkeypatch.setattr(config, "SILVER_DIR", root / "output" / "silver")


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
    store, corrected annotations) and adds ``report.csv``, ``labels.csv``, the
    report-mapping generation/cache directories and the predictions directory."""
    import config

    point_generations_config_at(monkeypatch, root)
    monkeypatch.setattr(config, "PREDICTIONS_DIR", root / "output" / "predictions")
    monkeypatch.setattr(config, "REPORT_CSV", root / "data" / "report.csv")
    monkeypatch.setattr(config, "LABELS_CSV", root / "labels.csv")
    report_mapping_dir = root / "output" / "report_mapping"
    monkeypatch.setattr(config, "REPORT_MAPPING_DIR", report_mapping_dir)
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", report_mapping_dir / "current")
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", report_mapping_dir / "candidate")
    monkeypatch.setattr(config, "OOF_DIR", report_mapping_dir / "oof")


def make_two_way_split_generation(split_id: str, train_ids: list[str], test_ids: list[str] = ()) -> None:
    """Write a minimal, directly-loadable train/test split generation (no
    calibration partition) for tests that need a real two-way split to build
    on (e.g. as ``create_three_way``'s parent) without a flat-file import."""
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


def make_dm_audit_silver_generation(silver_id: str = "dm-audit-test-silver") -> str:
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
# coding/ (rule, combine, corrected, queue): config redirection, a bronze
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
    monkeypatch.setattr(config, "COMBINED_PREDICTIONS_CSV", coding_dir / "combined_predictions.csv")
    monkeypatch.setattr(config, "REVIEW_QUEUE_CSV", coding_dir / "review_queue.csv")
    report_mapping_dir = root / "output" / "report_mapping"
    monkeypatch.setattr(config, "REPORT_MAPPING_DIR", report_mapping_dir)
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", report_mapping_dir / "current")
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", report_mapping_dir / "candidate")


def make_review_generation(directory: Path, generation_id: str, review_cut: float) -> Path:
    """The part of a generation that coding/ reads: a manifest carrying ``generation_id`` and a thresholds.json
    with ``review_cut`` (no weights). Written at ``directory``, e.g. the redirected ``config.REPORT_MAPPING_CURRENT_DIR``."""
    from generations.manifest import write_manifest
    from report_mapping import generation

    directory.mkdir(parents=True, exist_ok=True)
    thresholds = {**TINY_THRESHOLDS, "review_cut": review_cut}
    (directory / generation.THRESHOLDS_FILE).write_text(json.dumps(thresholds), encoding="utf-8")
    write_manifest(directory, {"generation_id": generation_id})
    return directory


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
    BRONZE_LOW_CONF = "CASE-BRONZE-LOW-CONF"  # no diagnosis, rank-1 confidence below review_cut
    BRONZE_TIGHT_MARGIN = "CASE-BRONZE-TIGHT-MARGIN"  # no diagnosis, top1-top2 margin tiny but both above review_cut
    BRONZE_RANK2_LOW = "CASE-BRONZE-RANK2-LOW"  # no diagnosis, confident rank 1, rank 2 below review_cut
    BRONZE_METHOD_FLAG = "CASE-BRONZE-METHOD-FLAG"  # no diagnosis, gate-rejected (blank code)


CODING_SILVER_ID = "coding-test-silver"
CODING_SPLIT_ID = "coding-test-split"
CODING_GENERATION_ID = "coding-test-gen"
CODING_REVIEW_CUT = 0.30


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
    - BRONZE_HIGH/LOW_CONF/TIGHT_MARGIN/RANK2_LOW/METHOD_FLAG have no
      diagnosis rows at all; LOW_CONF (rank 1 below ``review_cut``) and
      METHOD_FLAG (gate-rejected) are queued, the others are not.
    - The current generation (``CODING_GENERATION_ID``) has ``review_cut``
      ``CODING_REVIEW_CUT``; the predictions carry its id.

    Returns ``{"split_id", "silver_id", "generation_id", "predictions_csv"}``.
    """
    import config

    point_coding_config_at(monkeypatch, root)
    make_review_generation(config.REPORT_MAPPING_CURRENT_DIR, CODING_GENERATION_ID, CODING_REVIEW_CUT)
    C = CodingCaseIDs

    make_three_way_split_generation(
        CODING_SPLIT_ID,
        train=[
            C.G_DIAG, C.G_VAGUE, C.G_NOCANCER, C.G_BRONZE, C.DECISIVE_CANCER, C.DECISIVE_NONCANCER,
            C.VAGUE_NOGOLD, C.BRONZE_HIGH, C.BRONZE_LOW_CONF, C.BRONZE_TIGHT_MARGIN, C.BRONZE_RANK2_LOW,
            C.BRONZE_METHOD_FLAG,
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
        # Disagrees with DECISIVE_CANCER's silver code on purpose (never used).
        (C.DECISIVE_CANCER, 1, "Wrong Term", "Wrong Group", "9999/0", "0.9500", "0.95", "0.95",
         "term_head", CODING_GENERATION_ID),
        (C.VAGUE_NOGOLD, 1, "V-Term", "V-Group", "9993/3", "0.9000", "0.90", "0.90",
         "term_head", CODING_GENERATION_ID),
        (C.TEST_VAGUE, 1, "T-Term", "T-Group", "9994/3", "0.4000", "0.40", "0.40",
         "term_head", CODING_GENERATION_ID),
        (C.G_VAGUE, 1, "GV-Term", "GV-Group", "9995/3", "0.8000", "0.80", "0.80",
         "term_head", CODING_GENERATION_ID),
        (C.G_BRONZE, 1, "B-Term-G", "B-Group-G", "B900", "0.0500", "0.05", "0.05",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_HIGH, 1, "B-Term1", "B-Group1", "B001", "0.9000", "0.90", "0.90",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_HIGH, 2, "B-Term2", "B-Group1", "B002", "0.9000", "0.60", "0.50",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_LOW_CONF, 1, "B-Term3", "B-Group3", "B003", "0.1000", "0.10", "0.10",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_TIGHT_MARGIN, 1, "B-Term4", "B-Group4", "B004", "0.5000", "0.50", "0.50",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_TIGHT_MARGIN, 2, "B-Term5", "B-Group4", "B005", "0.5000", "0.49", "0.49",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_RANK2_LOW, 1, "B-Term6", "B-Group6", "B006", "0.9000", "0.90", "0.90",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_RANK2_LOW, 2, "B-Term7", "B-Group6", "B007", "0.9000", "0.10", "0.10",
         "term_head", CODING_GENERATION_ID),
        (C.BRONZE_METHOD_FLAG, 1, "Non-Cancer", "Non-Cancer", "", "0.0500", "0.00", "",
         "rejected_by_case_presence", CODING_GENERATION_ID),
    ]
    predictions_csv = make_bronze_predictions_csv(root / "predictions.csv", predictions_rows)

    return {
        "split_id": CODING_SPLIT_ID, "silver_id": CODING_SILVER_ID,
        "generation_id": CODING_GENERATION_ID, "predictions_csv": predictions_csv,
    }


# ---------------------------------------------------------------------------
# evaluation/ (silver_eval, gold_eval, audit_rates): config redirection and a
# ---------------------------------------------------------------------------


def point_evaluation_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path evaluation/ reads or writes into ``root``
    (builds on ``point_coding_config_at``: splits, silver, gold, ledger, cause and
    audit stores, combined predictions)."""
    import config

    point_coding_config_at(monkeypatch, root)
    eval_dir = root / "output" / "eval"
    monkeypatch.setattr(config, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(config, "SILVER_EVAL_HISTORY_CSV", eval_dir / "silver_eval_history.csv")
    report_mapping_dir = root / "output" / "report_mapping"
    monkeypatch.setattr(config, "REPORT_MAPPING_CURRENT_DIR", report_mapping_dir / "current")
    monkeypatch.setattr(config, "REPORT_MAPPING_CANDIDATE_DIR", report_mapping_dir / "candidate")
    monkeypatch.setattr(config, "UNCOMMON_GROUPS_TXT", root / "uncommon_groups.txt")


# ---------------------------------------------------------------------------
# handoff/ (contracts, imports, exports, worker_format): config redirection
# ---------------------------------------------------------------------------


def point_handoff_config_at(monkeypatch, root: Path) -> None:
    """Redirect every config path handoff/ reads or writes into ``root``.

    Builds on ``point_coding_config_at`` (silver dir, gold store, eval-batch
    ledger, combined predictions, review queue) and adds the inbox/outbox dirs and
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


# ---------------------------------------------------------------------------
# Promotion (generations/promote.py, triggers.py): config redirection
# ---------------------------------------------------------------------------


def point_promotion_config_at(monkeypatch, root: Path) -> None:
    """Everything ``point_training_config_at`` redirects, plus the eval-batch
    ledger, the silver dir, ``ARCHIVE_ROOT``, the uncommon-groups ruler and the history dir (``REPO_ROOT`` is
    ``root``, so an adoption reason is recorded as ``documentation/history/...``), so promotion tests never
    touch a real ml/output path."""
    import config

    point_training_config_at(monkeypatch, root)
    monkeypatch.setattr(config, "REPO_ROOT", root)
    monkeypatch.setattr(config, "HISTORY_DIR", root / "documentation" / "history")
    monkeypatch.setattr(config, "UNCOMMON_GROUPS_TXT", root / "uncommon_groups.txt")
    monkeypatch.setattr(config, "EVAL_BATCH_LEDGER_CSV", root / "output" / "manual_audit" / "eval_batch_ledger.csv")
    monkeypatch.setattr(config, "SILVER_DIR", root / "output" / "silver")
    monkeypatch.setattr(config, "ARCHIVE_ROOT", root / "output" / "archive")


# ---------------------------------------------------------------------------
# New report_mapping generation (joint ModernBERT): a tiny generation directory
# ---------------------------------------------------------------------------

TINY_TERMS = ["Mast cell tumor, malignant", "Mast cell tumor, benign", "Mast cell sarcoma", "Fibrosarcoma, NOS"]
TINY_THRESHOLDS = {"gate_t": 0.5, "tau": 0.1, "rho": 0.25, "review_cut": 0.3, "max_fp_share": 0.0243,
                   "review_target_share": 0.2336}


def build_tiny_generation(root: Path, encoder_dir: Path, calibrated: bool = True, split_id: str = "tiny-split") -> Path:
    """A real generation directory (tiny random ModernBERT, random heads over ``TINY_TERMS``): pending, or
    calibrated with ``TINY_THRESHOLDS``. Built through the same functions training uses."""
    from report_mapping import generation
    from report_mapping.model import JointClassifier, load_tokenizer, new_encoder

    torch.manual_seed(0)
    model = JointClassifier(new_encoder(str(encoder_dir)), list(TINY_TERMS))
    model.save(root, load_tokenizer(str(encoder_dir)))
    groups = {term: (group, code) for group, entries in TAXONOMY_GROUPS.items() for code, term in entries}
    (root / generation.TERM_INFO_FILE).write_text(json.dumps({t: list(groups[t]) for t in TINY_TERMS}), encoding="utf-8")
    generation.write_pending_manifest(root, encoder_id=str(encoder_dir), max_length=64, recipe={},
                                      parents={"split_id": split_id}, training={})
    if calibrated:
        generation.write_calibration(root, TINY_THRESHOLDS, {"partition": "calibration"})
    return root


def make_training_world(root: Path, monkeypatch, split_id: str = "tiny-split") -> tuple[str, str]:
    """Everything a train -> calibrate -> predict smoke run needs, under ``root`` with config redirected
    (``point_promotion_config_at``): report.csv, the taxonomy, an uncommon-groups ruler, a three-way split over
    ``TRAINING_CASE_IDS`` and a labels CSV (cases 1-15 cancer). Returns (labels path, split_id)."""
    import config

    point_promotion_config_at(monkeypatch, root)
    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    make_training_reports_csv(config.REPORT_CSV)
    make_labels_csv(config.LABELS_CSV)
    config.UNCOMMON_GROUPS_TXT.write_text("Rare Sarcomas\nRare Carcinomas\n", encoding="utf-8")
    ids = TRAINING_CASE_IDS
    make_three_way_split_generation(split_id, ids[0:10] + ids[15:25], ids[10:13] + ids[25:28], ids[13:15] + ids[28:30])
    return str(make_training_annotation_csv(root / "labels_table.csv")), split_id
