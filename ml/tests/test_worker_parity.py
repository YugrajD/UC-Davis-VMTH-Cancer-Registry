"""Worker parity (a WP13 precondition): ml-worker/batch_predict.py, run in-process on a synthetic bundle,
returns exactly what scripts/predict.py's rows give for the same reports, in the backend's payload.

The upload carries the report's own section columns, so both paths embed the same text; Text-only
uploads are covered by test_worker_format.py. app.py calls the same functions but needs fastapi, which
the ml venv does not install.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import io_utils
from handoff import worker_format
from report_mapping.inference.predict import run_predict

from . import fixtures as fx

# Found by walking up, so the path is independent of ml/'s own depth.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ml-worker").is_dir())
_BATCH_PY = _REPO_ROOT / "ml-worker" / "batch_predict.py"


def _load_batch_script():
    spec = importlib.util.spec_from_file_location("ml_worker_batch_predict", _BATCH_PY)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_batch_worker_matches_predict_py(tmp_path, monkeypatch, tiny_bert_dir):
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    bundle = fx.build_report_mapping_bundle(tmp_path / "bundle", tiny_bert_dir)
    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    expected_rows = io_utils.read_csv(
        run_predict(generation_dir=bundle, local_only=True, device_arg="cpu", out_path=tmp_path / "p.csv"),
        encoding="utf-8", dtype=str, keep_default_na=False).to_dict("records")

    upload = io_utils.read_csv(config.REPORT_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    upload = upload.rename(columns={"case_id": worker_format.UPLOAD_ID_COL})
    upload[worker_format.UPLOAD_TEXT_COL] = "echoed back"
    io_utils.write_csv(upload, tmp_path / "dataset_a.csv")
    out_dir = tmp_path / "out"
    for var, value in {"JOB_ID": "1", "INPUT_CSV_PATH": str(tmp_path / "dataset_a.csv"), "OUTPUT_DIR": str(out_dir),
                       "MODEL_PATH": str(bundle / "petbert"),
                       "GROUP_CLASSIFIER_PATH": str(bundle / "checkpoints" / "group_classifier_best.pt")}.items():
        monkeypatch.setenv(var, value)

    _load_batch_script().main()

    got = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    ids = upload[worker_format.UPLOAD_ID_COL].tolist()
    assert got == worker_format.response_rows(expected_rows, dict.fromkeys(ids, "echoed back"))
    assert {p["source_version"] for p in got} == {"test-gen"} and len(got) == len(ids)


def test_batch_worker_refuses_a_bundle_with_a_missing_file(tmp_path, monkeypatch, tiny_bert_dir):
    bundle = fx.build_report_mapping_bundle(tmp_path / "bundle", tiny_bert_dir)
    (bundle / "checkpoints" / "thresholds.json").unlink()
    for var, value in {"JOB_ID": "1", "INPUT_CSV_PATH": "unused.csv", "OUTPUT_DIR": str(tmp_path / "out"),
                       "MODEL_PATH": str(bundle / "petbert")}.items():
        monkeypatch.setenv(var, value)
    with pytest.raises(worker_format.ManifestError, match="missing"):
        _load_batch_script().main()
    assert not (tmp_path / "out").exists()


def test_repeated_upload_ids_keep_each_rows_own_report(tmp_path, report_mapping_bundle):
    # Dataset A may repeat an anon_id; each row must be scored on its own report, not the last one's.
    from report_mapping.inference.predict import predict_frame
    from report_mapping.model.generation import load_generation

    gen = load_generation(report_mapping_bundle)
    reports = io_utils.read_csv(fx.make_training_reports_csv(tmp_path / "r.csv"), encoding="utf-8", dtype=str,
                                keep_default_na=False).iloc[:2]
    alone = [predict_frame(gen, reports.iloc[[i]], ["X"], device_arg="cpu") for i in (0, 1)]
    together = predict_frame(gen, reports, ["X", "X"], device_arg="cpu")
    assert together == alone[0] + alone[1]
