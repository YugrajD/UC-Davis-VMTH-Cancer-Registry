"""Smoke tests: every fixture builds, and the tiny BERT round-trips."""

from __future__ import annotations

import csv
from pathlib import Path

from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from . import fixtures as fx


def test_tiny_bert_round_trips_through_from_pretrained(tiny_bert_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_bert_dir))
    model = AutoModel.from_pretrained(str(tiny_bert_dir))
    assert model.config.hidden_size == fx.TINY_EMB_DIM
    assert tokenizer.tokenize("mast cell tumor") == ["mast", "cell", "tumor"]

    # Also loadable as a MaskedLM checkpoint.
    mlm_model = AutoModelForMaskedLM.from_pretrained(str(tiny_bert_dir))
    assert mlm_model.base_model.config.hidden_size == fx.TINY_EMB_DIM


def test_labels_csv_matches_real_shape(labels_csv: Path) -> None:
    with open(labels_csv, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    assert rows[1] == fx.LABELS_HEADER[:6] + [""]  # title row (0), real header (1)
    data_rows = rows[2:]
    assert len(data_rows) == sum(len(v) for v in fx.TAXONOMY_GROUPS.values())
    groups_seen = {row[1] for row in data_rows}
    assert "Neoplasms, NOS" in groups_seen


def test_reports_csv_matches_real_column_shape(reports_csv: Path) -> None:
    with open(reports_csv, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0] == fx.REPORT_COLUMNS
    assert len(rows) - 1 == len(fx.CASE_IDS)


def test_diagnoses_csv_matches_real_column_shape(diagnoses_csv: Path) -> None:
    with open(diagnoses_csv, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0] == fx.DIAGNOSES_COLUMNS
    assert len(rows) - 1 == len(fx.CASE_IDS)


def test_annotation_csv_covers_every_decision_stage(annotation_csv: Path) -> None:
    with open(annotation_csv, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    stages = {row["decision_stage"] for row in rows}
    assert stages == {
        "no_signal", "tier1_exact", "tier2_fuzzy", "tier3_llm", "tier3_no_candidates",
    }
    tier3_methods = {row["method"] for row in rows if row["decision_stage"] == "tier3_llm"}
    assert tier3_methods == {"LLM", "Uncertain", "No Match"}


def test_split_files_partition_all_cases(split_files: tuple[Path, Path]) -> None:
    train_path, test_path = split_files
    train_ids = train_path.read_text(encoding="utf-8").split()
    test_ids = test_path.read_text(encoding="utf-8").split()
    assert set(train_ids) | set(test_ids) == set(fx.CASE_IDS)
    assert set(train_ids) & set(test_ids) == set()


def test_tiny_generation_passes_the_spec_check(tiny_generation_dir: Path) -> None:
    from report_mapping.generation import verify_spec

    manifest = verify_spec(tiny_generation_dir)
    assert manifest["model"]["n_terms"] == len(fx.TINY_TERMS) and manifest["calibration"]["status"] == "calibrated"
