"""scripts/code_cases.py: the thin CLI entry point over coding.adopt/corrected/queue.

Minimal coverage (WP9 fix 7): one happy-path invocation per subcommand,
checking the CLI wires args through to the right writer and to the right
--out path — the underlying behavior itself is already covered by
test_adopt.py / test_corrected.py / test_queue.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from coding.adopt import ADOPTED_CODES_COLUMNS
from coding.corrected import CORRECTED_COLUMNS
from coding.queue import REVIEW_QUEUE_COLUMNS

from . import fixtures as fx

_CODE_CASES_PY = Path(__file__).resolve().parents[1] / "scripts" / "code_cases.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_code_cases", _CODE_CASES_PY)
code_cases_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, code_cases_script)
_spec.loader.exec_module(code_cases_script)


@pytest.fixture
def scenario(monkeypatch, tmp_path):
    return fx.build_coding_scenario(monkeypatch, tmp_path)


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["code_cases.py"] + argv)
    return code_cases_script.main()


def test_adopt_subcommand_writes_adopted_codes_csv(scenario, monkeypatch, tmp_path):
    out_path = tmp_path / "adopted.csv"
    rc = _run(monkeypatch, [
        "adopt", "--silver", scenario["silver_id"], "--split", scenario["split_id"],
        "--predictions", str(scenario["predictions_csv"]), "--out", str(out_path),
    ])
    assert rc == 0
    df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == ADOPTED_CODES_COLUMNS
    assert fx.CodingCaseIDs.G_DIAG in set(df["case_id"])


def test_queue_subcommand_writes_review_queue_csv(scenario, monkeypatch, tmp_path):
    out_path = tmp_path / "queue.csv"
    rc = _run(monkeypatch, [
        "queue", "--silver", scenario["silver_id"], "--split", scenario["split_id"],
        "--predictions", str(scenario["predictions_csv"]), "--out", str(out_path),
    ])
    assert rc == 0
    df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == REVIEW_QUEUE_COLUMNS
    assert fx.CodingCaseIDs.VAGUE_NOGOLD in set(df["case_id"])


def test_corrected_subcommand_writes_corrected_annotations_csv(scenario, monkeypatch, tmp_path):
    out_path = tmp_path / "corrected.csv"
    rc = _run(monkeypatch, [
        "corrected", "--silver", scenario["silver_id"], "--split", scenario["split_id"],
        "--out", str(out_path),
    ])
    assert rc == 0
    df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == CORRECTED_COLUMNS
    assert fx.CodingCaseIDs.DECISIVE_CANCER in set(df["case_id"])
