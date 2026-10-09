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


@pytest.fixture(scope="session")
def tiny_modernbert_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random-init ModernBERT + tokenizer (the offline stand-in for the real encoder), built once per session."""
    return fx.build_tiny_modernbert(tmp_path_factory.mktemp("tiny_modernbert"))


@pytest.fixture(scope="session")
def tiny_generation_dir(tmp_path_factory: pytest.TempPathFactory, tiny_modernbert_dir: Path) -> Path:
    """A calibrated tiny generation (tiny ModernBERT, random heads), built once per session. Tests copy it
    (``shutil.copytree``) and change the copy's manifest fields; never write into it."""
    return fx.build_tiny_generation(tmp_path_factory.mktemp("tiny_generation") / "generation", tiny_modernbert_dir)


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
