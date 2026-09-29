"""End-to-end prediction on a synthetic bundle: report_mapping.inference.predict.run_predict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from report_mapping.inference.predict import run_predict
from report_mapping.model.generation import GenerationError

from . import fixtures as fx

EXPECTED_COLUMNS = [
    "case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
    "case_presence_prob", "confidence", "group_prob", "method", "generation_id",
]


@pytest.fixture
def wired_config(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "REPORT_CSV", fx.make_reports_csv(tmp_path / "report.csv"))
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", tmp_path / "embedding_cache")
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    return config


def test_end_to_end_prediction_on_synthetic_bundle(wired_config, report_mapping_bundle: Path):
    out_path = run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu")

    assert out_path is not None and out_path.exists()
    table = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert list(table.columns) == EXPECTED_COLUMNS
    assert (table["generation_id"] == "test-gen").all()
    assert set(table["case_id"]) <= set(fx.CASE_IDS)
    # diagnosis_index starts at 1 and is contiguous per case.
    for case_id, group in table.groupby("case_id"):
        ranks = sorted(int(r) for r in group["diagnosis_index"])
        assert ranks == list(range(1, len(ranks) + 1))
    # every row's case_presence_prob is a valid probability.
    probs = table["case_presence_prob"].astype(float)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_embed_only_populates_cache_and_stops(wired_config, report_mapping_bundle: Path):
    cache_dir = wired_config.EMBEDDING_CACHE_DIR
    assert not cache_dir.exists() or not any(cache_dir.iterdir())
    result = run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    assert result is None
    assert cache_dir.exists() and len(list(cache_dir.glob("*.npz"))) == 1


def test_second_run_hits_the_cache_and_matches(wired_config, report_mapping_bundle: Path):
    first = run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu",
                        out_path=Path(wired_config.PREDICTIONS_DIR) / "first.csv")
    cache_dir = wired_config.EMBEDDING_CACHE_DIR
    assert len(list(cache_dir.glob("*.npz"))) == 1

    second = run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu",
                         out_path=Path(wired_config.PREDICTIONS_DIR) / "second.csv")
    assert len(list(cache_dir.glob("*.npz"))) == 1  # still just the one cache entry

    first_table = pd.read_csv(first, dtype=str, keep_default_na=False)
    second_table = pd.read_csv(second, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(first_table, second_table)


def _set_calibration_status(bundle: Path, status: str) -> None:
    import json

    from generations.manifest import MANIFEST_NAME

    manifest_path = bundle / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration"] = {"status": status}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_real_prediction_refused_on_a_pending_candidate(wired_config, report_mapping_bundle: Path):
    _set_calibration_status(report_mapping_bundle, "pending")
    with pytest.raises(GenerationError, match="calibration"):
        run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu")


def test_embed_only_allowed_on_a_pending_candidate(wired_config, report_mapping_bundle: Path):
    # embed-only never reads thresholds, so it's fine on an uncalibrated candidate --
    # this is exactly what calibrate.py's own workflow needs to do first.
    _set_calibration_status(report_mapping_bundle, "pending")
    result = run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    assert result is None
    assert len(list(wired_config.EMBEDDING_CACHE_DIR.glob("*.npz"))) == 1


def test_cache_dir_override_reembeds_into_a_separate_directory(wired_config, report_mapping_bundle: Path):
    # L2b's own need: re-embed into a scratch directory without ever reading
    # from or writing to the real (machine-local) config.EMBEDDING_CACHE_DIR.
    run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    default_files = list(wired_config.EMBEDDING_CACHE_DIR.glob("*.npz"))
    assert len(default_files) == 1

    scratch_dir = wired_config.EMBEDDING_CACHE_DIR.parent / "l2b_scratch"
    assert not scratch_dir.exists()
    run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True,
                cache_dir=scratch_dir)

    # The real cache directory is untouched (still just the one, unchanged file)...
    assert [p.name for p in wired_config.EMBEDDING_CACHE_DIR.glob("*.npz")] == [f.name for f in default_files]
    # ...and the scratch directory got its own copy under the SAME content-hash key
    # (same generation/report/backbone -> same key, just materialized in a new place).
    scratch_files = list(scratch_dir.glob("*.npz"))
    assert len(scratch_files) == 1
    assert scratch_files[0].name == default_files[0].name


def test_cache_dir_override_does_not_hit_the_default_cache(wired_config, report_mapping_bundle: Path):
    # Populate the real cache first, then point at an empty scratch dir with
    # the SAME key on disk elsewhere -- run_predict must still treat the
    # scratch dir as a miss and re-embed into it, never falling back to the
    # default directory's entry.
    run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    scratch_dir = wired_config.EMBEDDING_CACHE_DIR.parent / "l2b_scratch_empty"
    assert not scratch_dir.exists() or not any(scratch_dir.iterdir())

    result = run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu",
                          cache_dir=scratch_dir)
    assert result is not None and result.exists()
    assert len(list(scratch_dir.glob("*.npz"))) == 1
