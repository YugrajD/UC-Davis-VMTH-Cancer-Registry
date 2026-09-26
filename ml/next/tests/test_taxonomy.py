"""Golden tests: new taxonomy/ against the pre-rewrite ml/ICD_labels/ package.

LEGACY COMPARISON — deleted at cutover (WP13), when the old ml/ tree
(including ml/ICD_labels/) is removed. Everything below that imports
`ICD_labels.*` exists only to prove new/old parity and has no other purpose.

Covers: the full real taxonomy (845 labels, labels.csv is public and safe to
read in full) for load/label-text/lookup parity, plus synthetic behavior and
subtype strings (including negation and the /6-over-/3 tie-break) that never
touch real report or diagnosis text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Old ml/ICD_labels/ is outside the ml/next/ import root (pytest.ini's
# `pythonpath = .` only adds ml/next); this file is deleted at cutover anyway.
_ML_DIR = Path(__file__).resolve().parents[2]
if str(_ML_DIR) not in sys.path:
    sys.path.append(str(_ML_DIR))

from ICD_labels.taxonomy import (
    build_taxonomy_label_texts as old_build_taxonomy_label_texts,
    load_labels_taxonomy as old_load_labels_taxonomy,
)
from ICD_labels.projection import resolve_taxonomy_matches as old_resolve_taxonomy_matches
from ICD_labels.behavior_keywords import (
    best_behavior as old_best_behavior,
    ranked_behaviors as old_ranked_behaviors,
    score_behavior as old_score_behavior,
)
from ICD_labels.subtype_keywords import filter_by_subtype as old_filter_by_subtype

import config
from taxonomy.taxonomy import (
    build_taxonomy_label_texts,
    load_labels_taxonomy,
    resolve_taxonomy_matches,
)
from taxonomy.behavior import best_behavior, ranked_behaviors, score_behavior
from taxonomy.subtype import filter_by_subtype

OLD_LABELS_CSV = str(_ML_DIR / "ICD_labels" / "labels.csv")
NEW_LABELS_CSV = str(config.LABELS_CSV)


# ---------------------------------------------------------------------------
# Taxonomy loading / label texts / lookups — full real taxonomy (845 labels)
# ---------------------------------------------------------------------------

def test_labels_csv_copied_byte_identical():
    assert Path(NEW_LABELS_CSV).read_bytes() == Path(OLD_LABELS_CSV).read_bytes()


def test_load_labels_taxonomy_matches_old_over_all_labels():
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    new_labels = load_labels_taxonomy(NEW_LABELS_CSV)
    assert len(old_labels) == 845  # documented count; catches a parsing regression
    assert [(l.code, l.group, l.term) for l in new_labels] == \
        [(l.code, l.group, l.term) for l in old_labels]


def test_build_taxonomy_label_texts_matches_old_over_all_labels():
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    new_labels = load_labels_taxonomy(NEW_LABELS_CSV)
    old_texts = old_build_taxonomy_label_texts(old_labels)
    new_texts = build_taxonomy_label_texts(new_labels)
    assert new_texts == old_texts
    # Documented format: "{term} {group}".
    assert new_texts[0] == f"{new_labels[0].term} {new_labels[0].group}"


def test_resolve_taxonomy_matches_matches_old_over_all_indices():
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    new_labels = load_labels_taxonomy(NEW_LABELS_CSV)
    plain_labels = [l.term for l in old_labels]
    # Every valid index, plus the empty (-1) and out-of-range sentinels.
    indices = [-1] + list(range(len(old_labels))) + [len(old_labels)]

    old_result = old_resolve_taxonomy_matches(indices, plain_labels, old_labels)
    new_result = resolve_taxonomy_matches(indices, plain_labels, new_labels)
    assert new_result == old_result


def test_resolve_taxonomy_matches_none_taxonomy_fallback_matches_old():
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    plain_labels = [l.term for l in old_labels]
    indices = [-1, 0, 5, len(old_labels)]

    old_result = old_resolve_taxonomy_matches(indices, plain_labels, None)
    new_result = resolve_taxonomy_matches(indices, plain_labels, None)
    assert new_result == old_result


# ---------------------------------------------------------------------------
# Behavior keyword scoring — synthetic strings only, no real report text
# ---------------------------------------------------------------------------

BEHAVIOR_TEST_STRINGS = [
    "Hemangiosarcoma, malignant, with vascular invasion.",
    "Benign lipoma, well-differentiated, encapsulated.",
    "Carcinoma in situ, noninvasive.",
    "Metastatic carcinoma to the lymph node, disseminated disease.",
    "Uncertain malignant potential, borderline features, indeterminate.",
    "Mass of uncertain whether primary or metastatic origin.",
    "No evidence of malignancy; benign appearing mass, not invasive.",
    "This is not a sarcoma; ruled out malignant process.",
    "Highly malignant sarcoma with metastasis to the lung; stage IV disease.",
    "A non-invasive, well differentiated adenoma without atypia.",
    "Plain report text with no behavior keywords at all.",
    "Malignant carcinoma with metastatic spread and mets confirmed; "
    "both malignant and metastatic present.",
    "Malignant tumor with metastatic spread to the lung.",
    "No evidence of malignancy. Not malignant. Benign appearing.",
]


@pytest.mark.parametrize("text", BEHAVIOR_TEST_STRINGS)
def test_score_behavior_matches_old(text):
    assert score_behavior(text) == old_score_behavior(text)


@pytest.mark.parametrize("text", BEHAVIOR_TEST_STRINGS)
def test_ranked_behaviors_matches_old(text):
    assert ranked_behaviors(text) == old_ranked_behaviors(text)


@pytest.mark.parametrize("text", BEHAVIOR_TEST_STRINGS)
def test_best_behavior_matches_old(text):
    assert best_behavior(text) == old_best_behavior(text)


def test_behavior_tie_break_6_over_3():
    # "malignant" scores /3, "metastatic" scores /6 >= 1.0 -> /6 must win,
    # both in the old code and the port.
    text = "Malignant tumor with metastatic spread to the lung."
    assert old_best_behavior(text) == "6"
    assert best_behavior(text) == "6"
    assert old_ranked_behaviors(text)[0] == "6"
    assert ranked_behaviors(text)[0] == "6"
    assert "3" in ranked_behaviors(text)  # /3 still present, just reordered after /6


def test_behavior_negation_suppresses_every_signal():
    # Every keyword in this string sits behind "no evidence of", "not", or
    # "ruled out" within the 50-char negation window -> no signal at all.
    text = "No evidence of malignancy. Not malignant. Benign appearing."
    assert old_score_behavior(text) == {}
    assert score_behavior(text) == {}
    assert old_best_behavior(text) is None
    assert best_behavior(text) is None


def test_behavior_negation_is_local_to_the_window():
    # "malignant" (unnegated) sits far enough past the negated "benign" that
    # its own signal still counts.
    text = "not benign " + ("x " * 40) + "malignant tumor"
    assert old_score_behavior(text) == score_behavior(text)
    assert "3" in score_behavior(text)


# ---------------------------------------------------------------------------
# Subtype keyword filtering — synthetic strings, real term/group names only
# ---------------------------------------------------------------------------

SUBTYPE_TEST_CASES = [
    ("Mast cell neoplasms", "Mast cell leukemia diagnosed on bone marrow aspirate."),
    ("Mast cell neoplasms", "Subcutaneous mast cell tumor, well-circumscribed."),
    ("Mast cell neoplasms", "Visceral mast cell tumor involving the spleen."),
    ("Mast cell neoplasms", "Systemic mastocytosis with extracutaneous involvement."),
    ("Mast cell neoplasms", "Kiupel grade high mast cell tumor."),
    ("Mast cell neoplasms", "Kiupel grade low mast cell tumor."),
    ("Blood vessel tumors", "Hemangiosarcoma of the spleen."),
    ("Blood vessel tumors", "Hemangioendothelioma of the skin."),
    ("Blood vessel tumors", "Hemangioma, cavernous type."),
    ("Melanocytoma and Melanomas", "Amelanotic melanoma of the oral cavity."),
    ("Melanocytoma and Melanomas", "Junctional melanocytoma of the haired skin."),
    ("Meningiomas", "Psammomatous meningioma of the falx."),
    ("Meningiomas", "Atypical meningioma, WHO grade II."),
    ("Lipomatous neoplasms", "Infiltrative lipoma of the thigh."),
    ("Lipomatous neoplasms", "Liposarcoma, well-differentiated."),
    ("Osseous and chondromatous neoplasms", "Osteosarcoma of the distal radius."),
    ("Osseous and chondromatous neoplasms", "Multilobular osteochondrosarcoma of the skull."),
    ("Gliomas", "Glioblastoma multiforme (GBM) of the cerebrum."),
    ("Gliomas", "Pilocytic astrocytoma of the cerebellum."),
    ("Not A Real Group", "Some report text that matches no rule table."),
    ("Mast cell neoplasms", "Report text with no matching subtype keyword at all."),
]


@pytest.mark.parametrize("group_name,text", SUBTYPE_TEST_CASES)
def test_filter_by_subtype_matches_old(group_name, text):
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    terms = [l.term for l in old_labels]
    pool = list(range(len(terms)))
    assert filter_by_subtype(group_name, pool, terms, text) == \
        old_filter_by_subtype(group_name, pool, terms, text)


@pytest.mark.parametrize("group_name,text", SUBTYPE_TEST_CASES)
def test_filter_by_subtype_narrows_or_passes_through(group_name, text):
    # Sanity check the fixture set actually exercises both branches, so the
    # parity assertion above isn't vacuously comparing two no-ops.
    old_labels = old_load_labels_taxonomy(OLD_LABELS_CSV)
    terms = [l.term for l in old_labels]
    pool = list(range(len(terms)))
    result = filter_by_subtype(group_name, pool, terms, text)
    if group_name == "Not A Real Group" or "no matching subtype" in text:
        assert result == pool
    else:
        assert 0 < len(result) < len(pool)


def test_filter_by_subtype_single_element_pool_short_circuits():
    labels = ["Hemangiosarcoma, NOS"]
    pool = [0]
    text = "Hemangioma of the skin."
    assert filter_by_subtype("Blood vessel tumors", pool, labels, text) == pool
    assert old_filter_by_subtype("Blood vessel tumors", pool, labels, text) == pool
