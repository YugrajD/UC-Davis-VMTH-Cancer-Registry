"""handoff/worker_format.py: the pieces ml-worker shares with the ml suite (bundle root, upload -> flat text,
threads, response payload) and the bundle refusal a worker start-up gets from the generation spec check.
Synthetic values only."""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from generations.manifest import update_manifest
from handoff import worker_format as wf
from report_mapping import generation
from report_mapping.inference.predict import load_generation
from report_mapping.text import build_report_text


def test_bundle_root_is_the_model_path_parent_and_the_path_need_not_exist(tmp_path):
    root = tmp_path / "bundle"
    assert not (root / "petbert").exists()
    assert wf.resolve_bundle_root({"PETBERT_MODEL_PATH": str(root / "petbert")}, "PETBERT_MODEL_PATH") == root
    assert wf.resolve_bundle_root({"MODEL_PATH": str(root / "petbert")}, "MODEL_PATH") == root


def test_both_model_vars_must_name_the_same_bundle(tmp_path):
    root = tmp_path / "bundle"
    env = {"PETBERT_MODEL_PATH": str(root / "petbert"), "MODEL_PATH": str(root / "petbert"), "OTHER": "x"}
    assert wf.resolve_bundle_root(env, "MODEL_PATH") == root
    env["PETBERT_MODEL_PATH"] = str(tmp_path / "other" / "petbert")
    with pytest.raises(wf.BundleError, match="PETBERT_MODEL_PATH"):
        wf.resolve_bundle_root(env, "MODEL_PATH")
    with pytest.raises(wf.BundleError, match="MODEL_PATH"):          # the leaf must be petbert
        wf.resolve_bundle_root({"MODEL_PATH": str(root / "encoder")}, "MODEL_PATH")


def test_the_petbert_era_path_vars_are_ignored_wherever_they_point(tmp_path):
    root = tmp_path / "bundle"
    env = {"MODEL_PATH": str(root / "petbert"), "GROUP_CLASSIFIER_PATH": str(tmp_path / "elsewhere" / "g.pt"),
           "LP_THRESHOLDS_JSON_PATH": str(tmp_path / "lp.json"), "UNCOMMON_GROUPS_PATH": ""}
    assert wf.resolve_bundle_root(env, "MODEL_PATH") == root


def test_torch_threads_default_env_and_refusal(monkeypatch):
    monkeypatch.setattr(wf.os, "cpu_count", lambda: 32)
    assert wf.torch_threads({}) == 8 == wf.DEFAULT_THREADS
    monkeypatch.setattr(wf.os, "cpu_count", lambda: 2)
    assert wf.torch_threads({}) == 2
    assert wf.torch_threads({"TORCH_NUM_THREADS": "3"}) == 3
    for bad in ("0", "-1", "many"):
        with pytest.raises(wf.BundleError, match="TORCH_NUM_THREADS"):
            wf.torch_threads({"TORCH_NUM_THREADS": bad})


def test_upload_text_is_cleaned_and_flat_with_no_heading():
    upload = pd.DataFrame({"anon_id": [" A1 ", "B2", "C3"], "Text": ["  report words ", "", "line1\nline2"],
                           "ANCILLARY TESTS": ["ignored", "ignored", "ignored"]})
    ids, texts = wf.upload_to_texts(upload)
    assert ids == ["A1", "B2", "C3"] and texts == ["report words", "", "line1\nline2"]
    assert ": " not in texts[0]                                       # no "HEADING: " prefix
    # not what build_report_text would give for the same cell under a column
    assert build_report_text(pd.Series({"COMMENT": "report words"})) == "COMMENT: report words" != texts[0]


def test_upload_without_a_text_column_has_empty_texts():
    ids, texts = wf.upload_to_texts(pd.DataFrame({"anon_id": ["A1", "B2"]}))
    assert ids == ["A1", "B2"] and texts == ["", ""]


def _row(case_id, rank, term, conf, method="term_head"):
    return {"case_id": case_id, "diagnosis_index": rank, "predicted_term": term, "predicted_group": "G",
            "predicted_code": "8000/3", "confidence": conf, "method": method, "generation_id": "gen-X"}


def test_response_numbers_multi_row_cases_and_carries_source_version():
    rows = [_row("A1", 2, "T2", "0.60"), _row("A1", 1, "T1", "0.90"), _row("B2", 1, "T3", "0.80")]
    out = wf.response_rows(rows, {"A1": "text a"})
    assert out[0] == {"anon_id": "A1", "original_text": "text a", "predicted_term": "1) T1 2) T2",
                      "predicted_group": "1) G 2) G", "predicted_code": "1) 8000/3 2) 8000/3",
                      "confidence": "1) 0.90 2) 0.60", "method": "1) term_head 2) term_head", "source_version": "gen-X"}
    assert out[1]["predicted_term"] == "T3" and out[1]["original_text"] == "" and out[1]["source_version"] == "gen-X"
    assert list(out[0]) == ["anon_id", "original_text", "predicted_term", "predicted_group", "predicted_code",
                            "confidence", "method", "source_version"]


def test_a_rejected_case_is_one_non_cancer_row_with_confidence_zero():
    (out,) = wf.response_rows([_row("A1", 1, "Non-Cancer", "0.00", "rejected_by_case_presence")], {})
    assert (out["predicted_term"], out["confidence"], out["method"]) == ("Non-Cancer", "0.00", "rejected_by_case_presence")


# --- the bundle refusal: what the worker's load_generation does with a bad generation directory ------------------

@pytest.fixture
def bundle(tmp_path, tiny_generation_dir):
    return shutil.copytree(tiny_generation_dir, tmp_path / "bundle")


def test_the_worker_loads_a_good_generation(bundle):
    assert load_generation(bundle, "cpu").generation_id == generation.verify_spec(bundle)["generation_id"]


def test_the_worker_refuses_a_missing_or_changed_file(bundle):
    (bundle / generation.HEAD_FILE).write_bytes(b"changed")
    with pytest.raises(generation.GenerationError):
        load_generation(bundle, "cpu")


def test_the_worker_refuses_a_missing_manifest_file_entry_or_file(bundle):
    (bundle / generation.THRESHOLDS_FILE).unlink()
    with pytest.raises(generation.GenerationError, match="missing"):
        load_generation(bundle, "cpu")


def test_the_worker_refuses_another_text_spec(bundle):
    manifest = generation.verify_spec(bundle)
    update_manifest(bundle, {"architecture": {**manifest["architecture"], "text_spec_version": 99}})
    with pytest.raises(generation.GenerationError, match="architecture"):
        load_generation(bundle, "cpu")


def test_the_worker_refuses_thresholds_calibrated_for_other_weights(bundle):
    manifest = generation.verify_spec(bundle)
    update_manifest(bundle, {"calibration": {**manifest["calibration"], "weights_sha256": "0" * 64}})
    with pytest.raises(generation.GenerationError, match="calibrated for other weights"):
        load_generation(bundle, "cpu")


def test_the_worker_refuses_an_uncalibrated_generation(bundle):
    update_manifest(bundle, {"calibration": {"status": "pending"}})
    with pytest.raises(generation.GenerationError, match="calibrat"):
        load_generation(bundle, "cpu")
