"""scripts/handoff.py: thin CLI wiring over handoff.imports/exports.

Minimal coverage, mirroring test_code_cases.py: one happy-path invocation per
subcommand, checking the CLI wires args through correctly — the underlying
behavior is already covered by test_handoff_imports.py / test_handoff_exports.py.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

import config
import io_utils
from coding.combine import COMBINED_CODES_COLUMNS
from coding.queue import REVIEW_QUEUE_COLUMNS

from . import fixtures as fx

_HANDOFF_PY = Path(__file__).resolve().parents[1] / "scripts" / "handoff.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_handoff", _HANDOFF_PY)
handoff_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, handoff_script)
_spec.loader.exec_module(handoff_script)


@pytest.fixture
def handoff_root(monkeypatch, tmp_path):
    fx.point_handoff_config_at(monkeypatch, tmp_path)
    return tmp_path


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["handoff.py"] + argv)
    return handoff_script.main()


def test_import_pending_cli(monkeypatch, handoff_root):
    csv_path = handoff_root / "pending.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "diagnosis_number", "diagnosis"])
        writer.writerow(["CASE-A", 1, "MAST CELL TUMOR"])

    assert _run(monkeypatch, ["import-pending", "--csv", str(csv_path), "--export-id", "e1"]) == 0
    assert config.HANDOFF_PENDING_DIAGNOSES_CSV.is_file()


def _make_stamped_silver_generation(silver_id: str, rows: list[tuple]) -> str:
    """See test_handoff_exports.py's helper of the same name and docstring."""
    df = pd.DataFrame(rows, columns=fx.ANNOTATION_COLUMNS)
    df["silver_generation"] = silver_id
    directory = config.SILVER_DIR / silver_id
    directory.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, directory / "annotation.csv")
    from generations.manifest import write_manifest
    write_manifest(directory, {"silver_id": silver_id, "llm_enabled": True})
    return silver_id


def test_export_silver_cli(monkeypatch, handoff_root):
    _make_stamped_silver_generation("s1", [
        ("CASE-A", 1, "TEXT", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3", "kw", "Exact", 1.0, "tier1_exact"),
    ])
    assert _run(monkeypatch, ["export-silver", "--silver-id", "s1"]) == 0
    assert (config.HANDOFF_OUTBOX_DIR / "silver_codes_s1.csv").is_file()


def test_export_coding_cli(monkeypatch, handoff_root):
    config.COMBINED_CODES_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in COMBINED_CODES_COLUMNS}]), config.COMBINED_CODES_CSV)
    config.REVIEW_QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in REVIEW_QUEUE_COLUMNS}]), config.REVIEW_QUEUE_CSV)

    assert _run(monkeypatch, ["export-coding", "--run-id", "run1"]) == 0
    assert (config.HANDOFF_OUTBOX_DIR / "combined_codes_run1.csv").is_file()
    assert (config.HANDOFF_OUTBOX_DIR / "review_queue_run1.csv").is_file()


def test_export_bundle_cli(monkeypatch, handoff_root, tiny_bert_dir):
    fx.build_report_mapping_bundle(config.REPORT_MAPPING_CURRENT_DIR, tiny_bert_dir)
    assert _run(monkeypatch, ["export-bundle", "--generation", "current"]) == 0
    assert any(config.HANDOFF_BUNDLES_DIR.glob("bundle_*.tar.gz"))
