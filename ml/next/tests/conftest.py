"""Pytest fixtures wrapping the factories in fixtures.py.

Every fixture here is offline (no network, no HuggingFace download) and
synthetic (no real veterinary text). See fixtures.py for the plain factory
functions these wrap, which later work packages can also call directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import fixtures as fx


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random-init BERT + tokenizer, built once per session."""
    return fx.build_tiny_bert(tmp_path_factory.mktemp("tiny_bert"))


@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    """A synthetic labels.csv-shaped taxonomy file."""
    return fx.make_labels_csv(tmp_path / "labels.csv")


@pytest.fixture
def reports_csv(tmp_path: Path) -> Path:
    """A synthetic report.csv-shaped file."""
    return fx.make_reports_csv(tmp_path / "report.csv")


@pytest.fixture
def diagnoses_csv(tmp_path: Path) -> Path:
    """A synthetic diagnoses.csv-shaped file."""
    return fx.make_diagnoses_csv(tmp_path / "diagnoses.csv")


@pytest.fixture
def annotation_csv(tmp_path: Path) -> Path:
    """A synthetic silver annotation.csv covering every decision_stage."""
    return fx.make_annotation_csv(tmp_path / "annotation.csv")


@pytest.fixture
def split_files(tmp_path: Path) -> tuple[Path, Path]:
    """(train_cases.txt, test_cases.txt) — a legacy-style case-id split."""
    return fx.make_split(tmp_path / "splits")


@pytest.fixture
def case_presence_head():
    """A random-init CasePresenceClassifier at the tiny embedding dimension."""
    return fx.build_case_presence_head()


@pytest.fixture
def group_head():
    """A random-init GroupClassifier over the synthetic taxonomy's groups."""
    return fx.build_group_head()


@pytest.fixture
def label_presence_head():
    """A random-init LabelPresenceClassifier (n_cols=3, concat-3 section spec)."""
    return fx.build_label_presence_head()


@pytest.fixture
def report_mapping_bundle(tmp_path: Path, tiny_bert_dir: Path) -> Path:
    """A full synthetic report-mapping generation directory (report_mapping bundle layout)."""
    return fx.build_report_mapping_bundle(tmp_path / "generation", tiny_bert_dir)
