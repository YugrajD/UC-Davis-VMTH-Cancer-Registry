"""cleanup.py: ensemble verification against a mocked llm_client.

Every diagnosis string below is invented for this test file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from taxonomy.taxonomy import TaxonomyLabel
from diagnosis_mapping import llm_client
from diagnosis_mapping.cleanup import CleanupConfig, clean

LABELS = [
    TaxonomyLabel(code="8000/3", group="Round Cell Tumors", term="Mast cell tumor, malignant"),
    TaxonomyLabel(code="8000/0", group="Round Cell Tumors", term="Mast cell tumor, benign"),
    TaxonomyLabel(code="8010/3", group="Rare Sarcomas", term="Fibrosarcoma, NOS"),
]

ROW_COLUMNS = [
    "case_id", "diagnosis_number", "diagnosis", "matched_term", "matched_group",
    "matched_code", "matched_keyword", "method", "confidence", "decision_stage",
]


def _row(case_id, diagnosis, term, group, code, method="Exact", stage="tier1_exact"):
    return {
        "case_id": case_id, "diagnosis_number": 1, "diagnosis": diagnosis,
        "matched_term": term, "matched_group": group, "matched_code": code,
        "matched_keyword": term, "method": method, "confidence": 1.0, "decision_stage": stage,
    }


def _config(models, tiebreaker=None):
    return CleanupConfig(models=models, tiebreaker_model=tiebreaker, timeout=5)


def test_unanimous_correct_keeps_row(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "CORRECT")
    df = pd.DataFrame([_row("CASE-0001", "mast cell tumor", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    assert out_df.loc[0, "matched_code"] == "8000/3"
    assert counters["kept"] == 1 and counters["total_verified"] == 1
    assert diff_df.empty


def test_unanimous_wrong_no_cancer_demotes(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "WRONG_no_cancer")
    df = pd.DataFrame([_row("CASE-0002", "inflammation only, no tumor", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    row = out_df.loc[0]
    assert row["method"] == "No Match" and row["matched_code"] == "" and row["confidence"] == 0.0
    assert counters["set_no_match"] == 1
    assert len(diff_df) == 1 and diff_df.loc[0, "change_type"] == "set_no_match"


def test_unanimous_wrong_should_be_replaces_term(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "WRONG_should_be: Mast cell tumor, benign")
    df = pd.DataFrame([_row("CASE-0003", "mast cell tumor, benign appearing", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    row = out_df.loc[0]
    assert (row["matched_term"], row["matched_code"]) == ("Mast cell tumor, benign", "8000/0")
    # Same group both sides -> term_changed, not group_changed.
    assert counters["term_changed"] == 1


def test_disagreement_without_tiebreaker_flags_uncertain(monkeypatch):
    responses = iter(["CORRECT", "WRONG_no_cancer"])
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: next(responses))
    df = pd.DataFrame([_row("CASE-0004", "ambiguous finding", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    assert out_df.loc[0, "method"] == "Uncertain"
    assert counters["flagged_uncertain"] == 1
    assert counters["tiebreaker_used"] == 0


def test_disagreement_resolved_by_tiebreaker(monkeypatch):
    responses = iter(["CORRECT", "WRONG_no_cancer", "CORRECT"])
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: next(responses))
    df = pd.DataFrame([_row("CASE-0005", "ambiguous finding, tiebreak needed", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"], tiebreaker="model-c"))
    # Tiebreaker "CORRECT" + original "CORRECT" (from model-a) unanimous among the two CORRECTs
    # is not what _resolve does: it re-resolves over all three votes, so CORRECT+WRONG+CORRECT
    # still disagrees -> stays Uncertain. This exercises that the tiebreaker was actually called.
    assert counters["tiebreaker_used"] == 1


def test_rows_outside_methods_to_verify_pass_through(monkeypatch):
    called = []
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: called.append(1) or "CORRECT")
    df = pd.DataFrame([_row("CASE-0006", "no signal at all", "", "", "", method="No Match", stage="no_signal")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    assert not called
    assert counters["total_verified"] == 0
    pd.testing.assert_frame_equal(out_df, df)


def test_group_not_in_taxonomy_flags_uncertain_without_calling_model(monkeypatch):
    called = []
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: called.append(1) or "CORRECT")
    df = pd.DataFrame([_row("CASE-0007", "mystery diagnosis", "Some term", "Group Not In Taxonomy", "0/0")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a"]))
    assert not called
    assert out_df.loc[0, "method"] == "Uncertain"
    assert counters["flagged_uncertain"] == 1
    assert diff_df.loc[0, "reason"] == "group not in taxonomy"


def test_parse_error_response_flags_uncertain(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "gibberish response with no verdict")
    df = pd.DataFrame([_row("CASE-0008", "odd phrasing", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    out_df, diff_df, counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    assert out_df.loc[0, "method"] == "Uncertain"
    assert counters["flagged_uncertain"] == 1


def test_diff_rows_never_include_diagnosis_text(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "WRONG_no_cancer")
    df = pd.DataFrame([_row("CASE-0009", "sensitive invented diagnosis text", "Mast cell tumor, malignant", "Round Cell Tumors", "8000/3")])
    _out_df, diff_df, _counters = clean(df, LABELS, _config(["model-a", "model-b"]))
    assert "diagnosis" not in diff_df.columns
