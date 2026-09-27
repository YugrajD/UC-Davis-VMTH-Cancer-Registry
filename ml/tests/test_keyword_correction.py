"""report_mapping/inference/keyword_correction.py: behavior + subtype filter
composition, and the Lipoma rescue (regex + liposarcoma exclusion)."""

from __future__ import annotations

from report_mapping.inference import keyword_correction as kc
from taxonomy.taxonomy import TaxonomyLabel

LABELS = [
    TaxonomyLabel(code="9120/0", group="Blood vessel tumors", term="Hemangioma, NOS"),      # 0
    TaxonomyLabel(code="9120/3", group="Blood vessel tumors", term="Hemangiosarcoma, NOS"),  # 1
    TaxonomyLabel(code="8005/3", group="Mast cell neoplasms", term="Mast cell sarcoma"),      # 2
    TaxonomyLabel(code="8005/0", group="Mast cell neoplasms", term="Mast cell tumor, benign"),  # 3
    TaxonomyLabel(code="8850/0", group="Lipomatous neoplasms", term="Lipoma, NOS"),           # 4
    TaxonomyLabel(code="8850/3", group="Lipomatous neoplasms", term="Liposarcoma, NOS"),      # 5
]
LABEL_TERMS = [l.term for l in LABELS]


def test_apply_keyword_correction_narrows_by_behavior_digit():
    pool = [0, 1]  # both Blood vessel tumors labels
    result = kc.apply_keyword_correction(
        text="a malignant hemangiosarcoma is present", pool=pool,
        taxonomy_labels=LABELS, labels=LABEL_TERMS, group_name="Blood vessel tumors",
    )
    assert result == [1]  # malignant -> /3 only


def test_apply_keyword_correction_no_behavior_signal_passes_through():
    result = kc.apply_keyword_correction(
        text="a mass was excised from the skin", pool=[0, 1],
        taxonomy_labels=LABELS, labels=LABEL_TERMS, group_name="Blood vessel tumors",
    )
    assert result == [0, 1]


def test_apply_keyword_correction_then_subtype_narrows_further():
    # Both malignant mast-cell labels would tie on behavior alone; here only one
    # behavior-digit survivor exists, so subtype filtering is a pass-through
    # sanity check on the mast-cell subtype rule table (single-element pool).
    result = kc.apply_keyword_correction(
        text="malignant mast cell sarcoma diagnosed on bone marrow aspirate", pool=[2, 3],
        taxonomy_labels=LABELS, labels=LABEL_TERMS, group_name="Mast cell neoplasms",
    )
    assert result == [2]


def test_apply_keyword_correction_empty_pool_short_circuits():
    assert kc.apply_keyword_correction(
        text="malignant", pool=[], taxonomy_labels=LABELS, labels=LABEL_TERMS, group_name="Blood vessel tumors",
    ) == []


def test_find_lipoma_rescue_index():
    assert kc.find_lipoma_rescue_index(LABELS, LABEL_TERMS) == 4


def test_find_lipoma_rescue_index_missing_term_returns_none():
    other_labels = [TaxonomyLabel(code="9999/0", group="Lipomatous neoplasms", term="Something Else")]
    assert kc.find_lipoma_rescue_index(other_labels, ["Something Else"]) is None


def test_lipoma_rescue_applies_on_keyword_and_sufficient_group_prob():
    assert kc.lipoma_rescue_applies(text="a small fatty mass, incidental", lipomatous_group_prob=0.6)
    assert kc.lipoma_rescue_applies(text="adipocyte proliferation noted", lipomatous_group_prob=0.5)


def test_lipoma_rescue_excludes_liposarcoma_even_with_lipoma_word():
    # "liposarcoma" contains no standalone "lipoma" match for \bfatty mass\b etc,
    # but reports mentioning both must still be excluded.
    text = "lipoma-like area but overall consistent with liposarcoma"
    assert not kc.lipoma_rescue_applies(text=text, lipomatous_group_prob=0.9)


def test_lipoma_rescue_requires_minimum_group_prob():
    assert not kc.lipoma_rescue_applies(text="a fatty mass is present", lipomatous_group_prob=0.49)


def test_lipoma_rescue_requires_keyword_match():
    assert not kc.lipoma_rescue_applies(text="no fatty tissue seen", lipomatous_group_prob=0.9)
