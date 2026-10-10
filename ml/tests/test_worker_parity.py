"""Worker parity (R2): the in-memory prediction paths agree with run_predict, and ml-worker's batch_predict.py,
run in-process on a tiny generation, returns the backend's payload for the flat upload text.

app.py calls the same functions as batch_predict.py; its HTTP layer is covered in test_worker_app.py (skipped
when fastapi is not installed, as in the ml venv).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

import config
import io_utils
from handoff import worker_format
from report_mapping import text
from report_mapping.generation import THRESHOLDS_FILE, GenerationError
from report_mapping.inference import decide
from report_mapping.inference.predict import load_generation, predict_frame, predict_texts, run_predict

# Found by walking up, so the path is independent of ml/'s own depth.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ml-worker").is_dir())
_BATCH_PY = _REPO_ROOT / "ml-worker" / "batch_predict.py"

REPORTS = pd.DataFrame({
    "case_id": ["A", "B", "C", "D"],
    "CLINICAL ABSTRACT": ["mass on the flank", "", "", ""],
    "HISTOPATHOLOGICAL SUMMARY": ["mast cell tumor present", "", "no evidence of neoplasia", ""],
    "FINAL COMMENT": ["", "carcinoma malignant", "", ""],                # D has no text anywhere
})


def _load_batch_script():
    spec = importlib.util.spec_from_file_location("ml_worker_batch_predict", _BATCH_PY)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def test_predict_frame_rows_equal_run_predict_rows(tmp_path, monkeypatch, tiny_generation_dir):
    (tmp_path / "data").mkdir()
    REPORTS.to_csv(tmp_path / "data" / "report.csv", index=False, encoding="latin-1")
    monkeypatch.setattr(config, "REPORT_CSV", tmp_path / "data" / "report.csv")
    written = io_utils.read_csv(run_predict(tiny_generation_dir, "cpu", out=tmp_path / "p.csv"), encoding="utf-8",
                                dtype=str, keep_default_na=False)
    rows = predict_frame(text.read_reports(), load_generation(tiny_generation_dir, "cpu"), "cpu")
    assert list(written.columns) == decide.COLUMNS and set(written["case_id"]) == {"A", "B", "C"}
    assert written.to_dict("records") == rows.astype(str).to_dict("records")


def test_predict_texts_on_built_texts_equals_predict_frame(tiny_generation_dir):
    loaded = load_generation(tiny_generation_dir, "cpu")
    via_texts = predict_texts(REPORTS["case_id"].tolist(), text.build_report_texts(REPORTS), loaded, "cpu")
    assert via_texts.equals(predict_frame(REPORTS, loaded, "cpu"))


def test_predict_texts_scores_the_flat_upload_mapping_and_skips_empty_texts(tiny_generation_dir):
    loaded = load_generation(tiny_generation_dir, "cpu")
    upload = pd.DataFrame({"anon_id": ["A", "B", "C"], "Text": ["mast cell tumor present", "   ", "carcinoma malignant"]})
    ids, texts = worker_format.upload_to_texts(upload)
    rows = predict_texts(ids, texts, loaded, "cpu")
    assert list(rows.columns) == decide.COLUMNS
    assert set(rows["case_id"]) == {"A", "C"} and (rows["generation_id"] == loaded.generation_id).all()
    assert set(rows["method"]) <= {"term_head", "rejected_by_case_presence"}
    assert predict_texts([], [], loaded, "cpu").empty
    assert list(predict_texts(["A"], [""], loaded, "cpu").columns) == decide.COLUMNS


def test_batch_worker_returns_the_payload_for_the_flat_text(tmp_path, monkeypatch, tiny_generation_dir):
    upload = pd.DataFrame({"anon_id": ["A", "B", "C"], "Text": ["mast cell tumor present", "", "carcinoma malignant"]})
    io_utils.write_csv(upload, tmp_path / "dataset_a.csv")
    out_dir = tmp_path / "out"
    for var, value in {"JOB_ID": "1", "INPUT_CSV_PATH": str(tmp_path / "dataset_a.csv"), "OUTPUT_DIR": str(out_dir),
                       "MODEL_PATH": str(tiny_generation_dir / "petbert"), "TORCH_NUM_THREADS": "1",
                       "GROUP_CLASSIFIER_PATH": str(tmp_path / "ignored.pt")}.items():
        monkeypatch.setenv(var, value)

    _load_batch_script().main()

    got = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    loaded = load_generation(tiny_generation_dir, "cpu")
    ids, texts = worker_format.upload_to_texts(upload)
    expected_rows = predict_texts(ids, texts, loaded, "cpu")
    assert got == worker_format.response_rows(expected_rows.to_dict("records"), dict(zip(ids, texts)))
    assert [p["anon_id"] for p in got] == ["A", "C"]
    assert {p["source_version"] for p in got} == {loaded.generation_id}
    assert all(p["original_text"] for p in got) and torch.get_num_threads() == 1


def test_batch_worker_refuses_a_bundle_with_a_missing_file(tmp_path, monkeypatch, tiny_generation_dir):
    bundle = shutil.copytree(tiny_generation_dir, tmp_path / "bundle")
    (bundle / THRESHOLDS_FILE).unlink()
    for var, value in {"JOB_ID": "1", "INPUT_CSV_PATH": "unused.csv", "OUTPUT_DIR": str(tmp_path / "out"),
                       "MODEL_PATH": str(bundle / "petbert")}.items():
        monkeypatch.setenv(var, value)
    with pytest.raises(GenerationError, match="missing"):
        _load_batch_script().main()
    assert not (tmp_path / "out").exists()
