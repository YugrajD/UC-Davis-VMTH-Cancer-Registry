"""handoff/worker_format.py: the pieces ml-worker shares with the ml suite (bundle root, upload
sections, response payload). Synthetic values only."""

from __future__ import annotations

import pandas as pd
import pytest

from handoff import worker_format as wf
from report_mapping import sections


def test_bundle_root_is_the_model_path_parent_and_other_vars_must_agree(tmp_path):
    root = tmp_path / "bundle"
    env = {"PETBERT_MODEL_PATH": str(root / "petbert"),
           "GROUP_CLASSIFIER_PATH": str(root / "checkpoints" / "group_classifier_best.pt"),
           "LP_THRESHOLDS_JSON_PATH": ""}  # set but empty: ignored, as docker-compose's defaults are
    assert wf.resolve_bundle_root(env, "PETBERT_MODEL_PATH") == root
    assert wf.resolve_bundle_root({"MODEL_PATH": str(root / "petbert")}, "MODEL_PATH") == root


def test_bundle_root_refuses_a_path_outside_the_bundle(tmp_path):
    root = tmp_path / "bundle"
    env = {"MODEL_PATH": str(root / "petbert"),
           "LP_THRESHOLDS_JSON_PATH": str(root / "checkpoints" / "lp_thresholds.json")}  # the pre-WP12 location
    with pytest.raises(wf.BundleError, match="LP_THRESHOLDS_JSON_PATH"):
        wf.resolve_bundle_root(env, "MODEL_PATH")


def test_upload_text_fills_the_first_source_column_of_every_section():
    upload = pd.DataFrame({"anon_id": ["A1"], "Text": ["report words"]})
    frame = sections.build_section_frame(wf.upload_to_reports(upload))
    assert sections.section_texts(frame) == {col: ["report words"] for col in sections.SECTION_COLUMNS}


def test_upload_keeps_section_columns_it_already_has():
    upload = pd.DataFrame({"anon_id": ["A1"], "Text": ["t"], "ANCILLARY TESTS": ["own"]})
    assert wf.upload_to_reports(upload)["ANCILLARY TESTS"].tolist() == ["own"]


def _row(case_id, rank, term, conf):
    return {"case_id": case_id, "diagnosis_index": rank, "predicted_term": term, "predicted_group": "G",
            "predicted_code": "8000/3", "confidence": conf, "method": "m", "generation_id": "gen-X"}


def test_response_numbers_multi_row_cases_and_carries_source_version():
    rows = [_row("A1", 2, "T2", "0.60"), _row("A1", 1, "T1", "0.90"), _row("B2", 1, "T3", "0.80")]
    out = wf.response_rows(rows, {"A1": "text a"})
    assert out[0] == {"anon_id": "A1", "original_text": "text a", "predicted_term": "1) T1 2) T2",
                      "predicted_group": "1) G 2) G", "predicted_code": "1) 8000/3 2) 8000/3",
                      "confidence": "1) 0.90 2) 0.60", "method": "1) m 2) m", "source_version": "gen-X"}
    assert out[1]["predicted_term"] == "T3" and out[1]["original_text"] == "" and out[1]["source_version"] == "gen-X"
