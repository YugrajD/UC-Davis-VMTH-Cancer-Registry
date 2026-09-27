"""llm_tier.py: candidate assembly (group / -oma suffix), prompt building,
response parsing, and Tier 3 end to end against a mocked llm_client.

Every diagnosis string below is invented for this test file.
"""

from __future__ import annotations

import pytest
import requests

from taxonomy.taxonomy import TaxonomyLabel
from diagnosis_mapping import llm_client, llm_tier
from diagnosis_mapping.keyword_tiers import mask_negation, normalize_llm

LABELS = [
    TaxonomyLabel(code="8000/3", group="Round Cell Tumors", term="Mast cell tumor, malignant"),
    TaxonomyLabel(code="8000/0", group="Round Cell Tumors", term="Mast cell tumor, benign"),
    TaxonomyLabel(code="8010/3", group="Rare Sarcomas", term="Fibrosarcoma, NOS"),
    TaxonomyLabel(code="8020/3", group="Rare Carcinomas", term="Squamous cell carcinoma, NOS"),
]
GROUP_TOKEN_INDEX = llm_tier.build_group_token_index(LABELS)
OMA_INDEX = llm_tier.build_oma_index(LABELS)


def _masked(text: str) -> str:
    return mask_negation(normalize_llm(text))


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------


def test_build_oma_index_maps_oma_suffix_words():
    assert OMA_INDEX == {"fibrosarcoma": 2, "carcinoma": 3}


def test_identify_group_picks_highest_token_overlap():
    masked = _masked("round cell tumor, cannot specify subtype")
    assert llm_tier.identify_group(masked, GROUP_TOKEN_INDEX) == "Round Cell Tumors"


def test_identify_group_returns_none_with_no_overlap():
    masked = _masked("chronic inflammation of the ear canal")
    assert llm_tier.identify_group(masked, GROUP_TOKEN_INDEX) is None


def test_candidates_for_group_match_returns_full_group():
    masked = _masked("round cell tumor, cannot specify subtype")
    candidates = llm_tier._candidates_for(masked, GROUP_TOKEN_INDEX, LABELS, OMA_INDEX)
    assert {c.term for c in candidates} == {"Mast cell tumor, malignant", "Mast cell tumor, benign"}


def test_candidates_for_falls_back_to_oma_suffix_when_no_group():
    masked = _masked("suspicious for a sarcoidoma, group unclear")
    candidates = llm_tier._candidates_for(masked, GROUP_TOKEN_INDEX, LABELS, OMA_INDEX)
    assert candidates == []  # invented word "sarcoidoma" is in no oma index -> no candidates


# ---------------------------------------------------------------------------
# Prompt building + response parsing
# ---------------------------------------------------------------------------


def test_build_llm_prompt_includes_site_hint():
    candidates = LABELS[:2]
    prompt = llm_tier.build_llm_prompt("mass in the subcutaneous tissue", candidates)
    assert "Anatomic site context: skin" in prompt
    assert "1. Mast cell tumor, malignant" in prompt
    assert "2. Mast cell tumor, benign" in prompt


def test_build_llm_prompt_no_site_hint_when_none_found():
    prompt = llm_tier.build_llm_prompt("a mass of uncertain origin", LABELS[:1])
    assert "Anatomic site context" not in prompt


def test_build_llm_prompt_keeps_legacy_em_dashes():
    # The prompt must stay byte-identical to the legacy one that produced silver-0-legacy.
    prompt = llm_tier.build_llm_prompt("mass in the subcutaneous tissue", LABELS[:1])
    assert "negated anywhere in the text — e.g." in prompt
    assert "uncertain or hedged — e.g." in prompt


def test_parse_llm_response_exact_case_insensitive():
    result = llm_tier.parse_llm_response("mast cell tumor, malignant", LABELS[:2])
    assert (result.code, result.method, result.confidence, result.stage) == ("8000/3", "LLM", 1.0, "tier3_llm")


def test_parse_llm_response_difflib_near_match():
    result = llm_tier.parse_llm_response("Mast cell tumor malignant", LABELS[:2])
    assert (result.code, result.confidence) == ("8000/3", 0.9)


def test_parse_llm_response_no_match():
    assert llm_tier.parse_llm_response("no match", LABELS[:2]) is None
    assert llm_tier.parse_llm_response("", LABELS[:2]) is None


def test_parse_llm_response_uncertain():
    result = llm_tier.parse_llm_response("uncertain", LABELS[:2])
    assert (result.method, result.stage) == ("Uncertain", "tier3_llm")


# ---------------------------------------------------------------------------
# Tier 3 end to end (mocked llm_client.chat)
# ---------------------------------------------------------------------------


def _counters() -> dict:
    return {"tier3_calls": 0, "tier3_matched": 0, "tier3_uncertain": 0, "tier3_no_match": 0}


def test_run_tier3_no_candidates_never_calls_the_model(monkeypatch):
    called = []
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: called.append(1) or "should not be reached")
    counters = _counters()
    result = llm_tier.run_tier3(
        "suspicious for a sarcoidoma, group unclear", _masked("suspicious for a sarcoidoma, group unclear"),
        GROUP_TOKEN_INDEX, LABELS, OMA_INDEX, llm_enabled=True, model=None, timeout=5, counters=counters,
    )
    assert result.stage == "tier3_no_candidates" and result.method == "No Match"
    assert not called
    assert counters["tier3_calls"] == 0


def test_run_tier3_answered(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "Mast cell tumor, benign")
    counters = _counters()
    text = "round cell tumor, cannot specify subtype"
    result = llm_tier.run_tier3(
        text, _masked(text), GROUP_TOKEN_INDEX, LABELS, OMA_INDEX,
        llm_enabled=True, model=None, timeout=5, counters=counters,
    )
    assert (result.code, result.method, result.stage) == ("8000/0", "LLM", "tier3_llm")
    assert counters["tier3_calls"] == 1 and counters["tier3_matched"] == 1


def test_run_tier3_uncertain(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "uncertain")
    counters = _counters()
    text = "round cell tumor, cannot specify subtype"
    result = llm_tier.run_tier3(
        text, _masked(text), GROUP_TOKEN_INDEX, LABELS, OMA_INDEX,
        llm_enabled=True, model=None, timeout=5, counters=counters,
    )
    assert (result.method, result.stage) == ("Uncertain", "tier3_llm")
    assert counters["tier3_uncertain"] == 1


def test_run_tier3_declined_no_match(monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "no match")
    counters = _counters()
    text = "round cell tumor, cannot specify subtype"
    result = llm_tier.run_tier3(
        text, _masked(text), GROUP_TOKEN_INDEX, LABELS, OMA_INDEX,
        llm_enabled=True, model=None, timeout=5, counters=counters,
    )
    assert (result.method, result.stage) == ("No Match", "tier3_llm")
    assert counters["tier3_no_match"] == 1


def test_run_tier3_request_failure_counts_as_no_match(monkeypatch):
    def _boom(*a, **k):
        raise requests.RequestException("connection refused")
    monkeypatch.setattr(llm_client, "chat", _boom)
    counters = _counters()
    text = "round cell tumor, cannot specify subtype"
    result = llm_tier.run_tier3(
        text, _masked(text), GROUP_TOKEN_INDEX, LABELS, OMA_INDEX,
        llm_enabled=True, model=None, timeout=5, counters=counters,
    )
    assert (result.method, result.stage) == ("No Match", "tier3_llm")
    assert counters["tier3_no_match"] == 1


def test_run_tier3_llm_disabled_declines_without_calling(monkeypatch):
    called = []
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: called.append(1))
    counters = _counters()
    text = "round cell tumor, cannot specify subtype"
    result = llm_tier.run_tier3(
        text, _masked(text), GROUP_TOKEN_INDEX, LABELS, OMA_INDEX,
        llm_enabled=False, model=None, timeout=5, counters=counters,
    )
    assert (result.method, result.stage) == ("No Match", "tier3_llm")
    assert not called
    assert counters["tier3_calls"] == 0 and counters["tier3_no_match"] == 1
