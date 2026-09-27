"""keyword_tiers.py: normalization, negation masking, Tier 1 (exact) and
Tier 2 (fuzzy) matching, the signal gate, and the cascade fingerprint.

Every diagnosis string below is invented for this test file - none is real
patient data.
"""

from __future__ import annotations

import re

from taxonomy.taxonomy import TaxonomyLabel
from diagnosis_mapping import keyword_tiers as kt
from diagnosis_mapping.keyword_tiers import (
    build_keyword_index,
    cascade_fingerprint,
    extract_site,
    has_signal,
    mask_negation,
    normalize,
    normalize_llm,
    tier1_exact,
    tier2_fuzzy,
)

LABELS = [
    TaxonomyLabel(code="8000/3", group="Round Cell Tumors", term="Mast cell tumor, malignant"),
    TaxonomyLabel(code="8000/0", group="Round Cell Tumors", term="Mast cell tumor, benign"),
    TaxonomyLabel(code="8010/3", group="Rare Sarcomas", term="Fibrosarcoma, NOS"),
    TaxonomyLabel(code="8020/3", group="Rare Carcinomas", term="Squamous cell carcinoma, NOS"),
    TaxonomyLabel(code="9999/0", group="Neoplasms, NOS", term="Neoplasm, benign"),
]

KEYWORD_INDEX = build_keyword_index(LABELS)


def _run(text: str):
    norm = normalize_llm(text)
    masked = mask_negation(norm)
    t1 = tier1_exact(masked, KEYWORD_INDEX, LABELS)
    t2 = None if t1 else tier2_fuzzy(masked, LABELS)
    return masked, t1, t2


# ---------------------------------------------------------------------------
# Normalization / abbreviation expansion
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_collapses_punctuation():
    assert normalize("Mast-Cell/Tumor, (Malignant); Grade: II") == "mast cell tumor malignant grade ii"


def test_normalize_llm_expands_mct_abbreviation():
    assert "mast cell tumor" in normalize_llm("MCT, malignant")


def test_normalize_neoplasia_synonym():
    assert normalize("evidence of neoplasia") == "evidence of neoplasm"


# ---------------------------------------------------------------------------
# Tier 1: exact match
# ---------------------------------------------------------------------------


def test_tier1_exact_match_first_label_wins_shared_core():
    # "Mast cell tumor, malignant" and "..., benign" share the core keyword
    # "mast cell tumor" once qualifiers are stripped; the first taxonomy row
    # to define a keyword wins the dedup, matching the old pipeline.
    masked, t1, _t2 = _run("MAST CELL TUMOR")
    assert t1 is not None
    assert (t1.code, t1.term, t1.method, t1.stage) == ("8000/3", "Mast cell tumor, malignant", "Exact", "tier1_exact")


def test_tier1_exact_match_with_explicit_qualifier():
    _masked, t1, _t2 = _run("Mast cell tumor, benign, well circumscribed")
    assert (t1.code, t1.method, t1.stage) == ("8000/0", "Exact", "tier1_exact")


def test_tier1_exact_match_single_word_core():
    _masked, t1, _t2 = _run("Fibrosarcoma of the skin, margins clear")
    assert (t1.code, t1.term, t1.method, t1.stage) == ("8010/3", "Fibrosarcoma, NOS", "Exact", "tier1_exact")


def test_tier1_no_match_for_unrelated_text():
    _masked, t1, _t2 = _run("Chronic lymphocytic inflammation with fibrosis")
    assert t1 is None


# ---------------------------------------------------------------------------
# Tier 2: fuzzy (token-overlap) match
# ---------------------------------------------------------------------------


def test_tier2_fuzzy_matches_scattered_core_tokens():
    # Core tokens ("squamous", "cell", "carcinoma") all present but not as a
    # contiguous phrase, so Tier 1's literal-substring match cannot fire.
    masked, t1, t2 = _run("Findings show carcinoma; cell morphology squamous in type")
    assert t1 is None
    assert t2 is not None
    assert (t2.code, t2.term, t2.method, t2.stage) == ("8020/3", "Squamous cell carcinoma, NOS", "Fuzzy", "tier2_fuzzy")
    assert t2.confidence >= 0.85


def test_tier2_fuzzy_below_threshold_does_not_match():
    # Only 2 of the 3 core tokens ("squamous", "cell") present -> 0.67 < 0.85.
    masked, t1, t2 = _run("Squamous cell tumor NOS")
    assert t1 is None
    assert t2 is None


def test_tier2_fuzzy_is_behavior_aware():
    # Both candidates share the core "mast cell tumor" once qualifiers are
    # stripped; an explicit "benign" modifier must resolve to the /0 code,
    # not the first-defined /3 code, via the full (unstripped) term compare.
    masked, t1, t2 = _run("benign consistent with cell type tumor of mast")
    assert t1 is None
    assert (t2.code, t2.term, t2.method, t2.stage) == ("8000/0", "Mast cell tumor, benign", "Fuzzy", "tier2_fuzzy")


# ---------------------------------------------------------------------------
# Negation masking + the no_signal gate
# ---------------------------------------------------------------------------


def test_negation_phrase_masks_the_signal_term():
    norm = normalize_llm("No evidence of neoplasia")
    masked = mask_negation(norm)
    assert "neoplasm" not in masked
    assert not has_signal(masked)


def test_non_x_compound_is_masked():
    masked = mask_negation(normalize_llm("non-B cell lymphoma suspected"))
    assert "non" not in masked or "b cell" not in masked


def test_has_signal_true_for_oma_suffix():
    assert has_signal(mask_negation(normalize_llm("a mass suspicious for lymphoma")))


def test_has_signal_false_for_unrelated_text():
    assert not has_signal(mask_negation(normalize_llm("Chronic lymphocytic inflammation with fibrosis")))


def test_no_signal_end_to_end():
    masked, t1, t2 = _run("Chronic lymphocytic inflammation with fibrosis")
    assert t1 is None and t2 is None
    assert not has_signal(masked)


# ---------------------------------------------------------------------------
# Anatomic site extraction
# ---------------------------------------------------------------------------


def test_extract_site_first_match_wins():
    assert extract_site("mass in the subcutaneous tissue of the oral cavity") == "skin"


def test_extract_site_none_when_no_site_keyword():
    assert extract_site("a mass of uncertain origin") is None


# ---------------------------------------------------------------------------
# Cascade fingerprint
# ---------------------------------------------------------------------------


def test_cascade_fingerprint_deterministic():
    assert cascade_fingerprint(LABELS) == cascade_fingerprint(LABELS)


def test_cascade_fingerprint_changes_with_taxonomy():
    other = LABELS + [TaxonomyLabel(code="1/1", group="X", term="Extra term, NOS")]
    assert cascade_fingerprint(LABELS) != cascade_fingerprint(other)


def test_cascade_fingerprint_changes_with_normalize_substitutions(monkeypatch):
    # cascade_fingerprint must move if any of normalize()'s or normalize_llm()'s
    # hardcoded substitutions change, not just the keyword index or thresholds.
    baseline = cascade_fingerprint(LABELS)

    monkeypatch.setattr(kt, "_PUNCT_TO_SPACE_RE", re.compile(r"[-_/,]"))
    assert cascade_fingerprint(LABELS) != baseline
    monkeypatch.undo()

    monkeypatch.setattr(kt, "_NEOPLASIA_EXPANSION", "neoplastic disease")
    assert cascade_fingerprint(LABELS) != baseline
    monkeypatch.undo()

    monkeypatch.setattr(kt, "_PLASMA_CELL_TUMOR_RE", re.compile(r"\bplasma cell mass\b"))
    assert cascade_fingerprint(LABELS) != baseline
    monkeypatch.undo()

    monkeypatch.setattr(kt, "_RE_METASTASIS", re.compile(r"\bmetastases\b"))
    assert cascade_fingerprint(LABELS) != baseline
    monkeypatch.undo()

    monkeypatch.setattr(kt, "_METASTASIS_EXPANSION", "spread")
    assert cascade_fingerprint(LABELS) != baseline
