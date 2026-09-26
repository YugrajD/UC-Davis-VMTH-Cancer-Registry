"""End-to-end calibration.status contract: a freshly trained (pending)
candidate refuses real prediction, ``report_mapping.training.calibrate.
calibrate()`` calibrates it, and prediction then works. Exercises the
``calibration.status`` handshake across ``scripts/train.py`` (writes
"pending") -> ``calibrate.py`` (writes "calibrated") -> ``predict.py``
(refuses "pending", allows "calibrated"). Only reads ``calibrate.py``'s
public API; never edits it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generations.manifest import MANIFEST_NAME
from report_mapping.inference.predict import run_predict
from report_mapping.model.generation import GenerationError, load_generation
from report_mapping.training import calibrate as cal

from . import fixtures as fx

SPLIT_ID = "calib-status-flow"
CALIBRATION_IDS = fx.CASE_IDS[:6]


@pytest.fixture
def pending_candidate(tmp_path, monkeypatch, report_mapping_bundle: Path):
    import config

    fx.point_generations_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "REPORT_CSV", fx.make_reports_csv(tmp_path / "report.csv"))
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", tmp_path / "embedding_cache")
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")

    # scripts/train.py's own placeholder status for a freshly trained candidate.
    manifest_path = report_mapping_bundle / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration"] = {"status": "pending"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # embed-only is allowed on a pending candidate (it never reads thresholds) --
    # this is exactly what calibrate.py needs to do before it can calibrate one.
    run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    fx.make_three_way_split_generation(SPLIT_ID, ["CASE-0101", "CASE-0102"], CALIBRATION_IDS, fx.CASE_IDS[6:])
    labels_csv = str(fx.make_calibration_labels_csv(tmp_path / "labels.csv"))
    return report_mapping_bundle, labels_csv


def test_pending_candidate_refuses_real_prediction(pending_candidate):
    generation_dir, _labels = pending_candidate
    with pytest.raises(GenerationError, match="calibration"):
        run_predict(generation_dir=generation_dir, local_only=True, device_arg="cpu")


def test_calibrate_then_predict_succeeds(pending_candidate):
    generation_dir, labels_csv = pending_candidate
    cal.calibrate(generation_dir, labels=labels_csv, split_id=SPLIT_ID, legacy=True)

    gen = load_generation(generation_dir)  # now calibrated -> loads with the default allow_uncalibrated=False
    assert gen.manifest["calibration"]["status"] == "calibrated"

    out_path = run_predict(generation_dir=generation_dir, local_only=True, device_arg="cpu")
    assert out_path is not None and out_path.exists()
