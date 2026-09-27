"""taxonomy/: labels.csv loading, label texts, index resolution, behavior and subtype keywords.

The frozen values below were taken from this package when the pre-cutover golden tests proved it
identical to the old ml/ICD_labels/ (2026-09-27). labels.csv is the public Vet-ICD-O table; every
report-like string here is synthetic.
"""

from __future__ import annotations

import hashlib

import pytest

import config
from taxonomy.behavior import best_behavior, ranked_behaviors, score_behavior
from taxonomy.subtype import filter_by_subtype
from taxonomy.taxonomy import build_taxonomy_label_texts, load_labels_taxonomy, resolve_taxonomy_matches

HEADER = "Vet-ICD-O-canine-1 code,Group,Term,level,Topography,obs"


def _sha(obj) -> str:
    return hashlib.sha256(repr(obj).encode()).hexdigest()


def _write(tmp_path, lines: list[str]):
    path = tmp_path / "labels.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# load_labels_taxonomy on synthetic files
# ---------------------------------------------------------------------------


def test_title_row_is_skipped_and_header_is_row_one(tmp_path):
    path = _write(tmp_path, ["Title row, anything,,,,", HEADER, "8000/0,Group A,Term A,1,,", "8000/3,Group A,Term B,1,,"])
    assert [(l.code, l.group, l.term) for l in load_labels_taxonomy(path)] == [
        ("8000/0", "Group A", "Term A"), ("8000/3", "Group A", "Term B")]


def test_repeat_blank_and_incomplete_rows_are_dropped(tmp_path):
    path = _write(tmp_path, ["Title", HEADER, "8000/0,Group A,Term A,,,", ",,,,,", "8000/0,Group A,Term A,2,x,",
                             "8001/0,,Term C,,,", "8002/0,Group B,Term D"])
    assert [l.term for l in load_labels_taxonomy(path)] == ["Term A", "Term D"]


def test_unexpected_header_or_empty_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unexpected labels header"):
        load_labels_taxonomy(_write(tmp_path, ["Title", "code,Group,Term", "8000/0,G,T"]))
    with pytest.raises(ValueError, match="empty or malformed"):
        load_labels_taxonomy(_write(tmp_path, ["Title", HEADER]))


# ---------------------------------------------------------------------------
# The real taxonomy, frozen
# ---------------------------------------------------------------------------


def test_real_taxonomy_is_frozen():
    assert hashlib.sha256(config.LABELS_CSV.read_bytes()).hexdigest() == \
        "5e660ee677d57577c0f0be90abeb4b0e53d2e946bac39737fe2449e14d999604"
    labels = load_labels_taxonomy(str(config.LABELS_CSV))
    assert (len(labels), len({l.group for l in labels}), len({l.code for l in labels})) == (845, 52, 534)
    assert _sha([(l.code, l.group, l.term) for l in labels]) == \
        "8c0b936cbb3389c01250ede248847989668a1fb29ca64d4237bdd54a4635a161"
    texts = build_taxonomy_label_texts(labels)
    assert texts[0] == f"{labels[0].term} {labels[0].group}"
    assert _sha(texts) == "dc777d76a6413338f85ac5f69b3914c1e287c22a958ca393f2a99eee7f19b2fb"


def test_resolve_taxonomy_matches_sentinels_and_fallback():
    labels = load_labels_taxonomy(str(config.LABELS_CSV))
    terms = [l.term for l in labels]
    indices = [-1, 0, 5, len(labels)]  # empty sentinel, two real labels, out of range
    assert resolve_taxonomy_matches(indices, terms, labels) == (
        ["", "Neoplasm, benign", "Tumor cells, benign", ""],
        ["", "Neoplasms, NOS", "Neoplasms, NOS", ""],
        ["", "8000/0", "8001/0", ""],
    )
    assert resolve_taxonomy_matches(indices, terms, None) == (
        ["", "Neoplasm, benign", "Tumor cells, benign", ""], [""] * 4, [""] * 4)


# ---------------------------------------------------------------------------
# Behavior keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,ranked", [
    ("Hemangiosarcoma, malignant, with vascular invasion.", ["3"]),
    ("Benign lipoma, well-differentiated, encapsulated.", ["0"]),
    ("Carcinoma in situ, noninvasive.", ["2", "3", "0"]),
    ("Metastatic carcinoma to the lymph node, disseminated disease.", ["6", "3"]),
    ("Uncertain malignant potential, borderline features, indeterminate.", ["1", "3"]),
    ("A non-invasive, well differentiated adenoma without atypia.", ["0"]),
    ("Plain report text with no behavior keywords at all.", []),
])
def test_ranked_and_best_behavior(text, ranked):
    assert ranked_behaviors(text) == ranked
    assert best_behavior(text) == (ranked[0] if ranked else None)


def test_behavior_tie_break_6_over_3():
    text = "Malignant tumor with metastatic spread to the lung."
    assert best_behavior(text) == "6"
    assert ranked_behaviors(text)[0] == "6" and "3" in ranked_behaviors(text)


def test_behavior_negation_suppresses_every_signal_in_its_window():
    assert score_behavior("No evidence of malignancy. Not malignant. Benign appearing.") == {}
    assert best_behavior("No evidence of malignancy. Not malignant. Benign appearing.") is None
    # An unnegated keyword beyond the 50-char window still counts.
    assert score_behavior("not benign " + ("x " * 40) + "malignant tumor") == {"3": 1.0}


# ---------------------------------------------------------------------------
# Subtype keywords (the pool is the whole taxonomy, so counts pin each rule table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group,text,count,first", [
    ("Mast cell neoplasms", "Mast cell leukemia diagnosed on bone marrow aspirate.", 39, "Mast cell leukemia"),
    ("Mast cell neoplasms", "Kiupel grade high mast cell tumor.", 3, "Cutaneous mast cell tumor grade Kiupel high"),
    ("Blood vessel tumors", "Hemangiosarcoma of the spleen.", 10, "Hemangiosarcoma, NOS"),
    ("Blood vessel tumors", "Hemangioma, cavernous type.", 16, "Hemangioma, NOS"),
    ("Melanocytoma and Melanomas", "Amelanotic melanoma of the oral cavity.", 1, "Amelanotic melanoma"),
    ("Meningiomas", "Atypical meningioma, WHO grade II.", 3, "Atypical choroid plexus papilloma"),
    ("Lipomatous neoplasms", "Liposarcoma, well-differentiated.", 5, "Liposarcoma, NOS"),
    ("Lipomatous neoplasms", "Infiltrative lipoma of the thigh.", 16,
     "Nonpapillary urothelial carcinoma, non infiltrating"),
    ("Osseous and chondromatous neoplasms", "Osteosarcoma of the distal radius.", 18, "Osteosarcoma, NOS"),
    ("Osseous and chondromatous neoplasms", "Multilobular osteochondrosarcoma of the skull.", 6,
     "Multilobular tumor of bone"),
    ("Gliomas", "Glioblastoma multiforme (GBM) of the cerebrum.", 4, "Glioblastoma"),
    ("Gliomas", "Pilocytic astrocytoma of the cerebellum.", 1, "Pilocytic astrocytoma"),
])
def test_filter_by_subtype_narrows(group, text, count, first):
    terms = [l.term for l in load_labels_taxonomy(str(config.LABELS_CSV))]
    result = filter_by_subtype(group, list(range(len(terms))), terms, text)
    assert (len(result), terms[result[0]]) == (count, first)


def test_filter_by_subtype_passes_through_without_a_rule_or_a_match():
    terms = [l.term for l in load_labels_taxonomy(str(config.LABELS_CSV))]
    pool = list(range(len(terms)))
    assert filter_by_subtype("Not A Real Group", pool, terms, "Some text matching no rule table.") == pool
    assert filter_by_subtype("Mast cell neoplasms", pool, terms, "Text with no subtype keyword at all.") == pool
    assert filter_by_subtype("Blood vessel tumors", [0], ["Hemangiosarcoma, NOS"], "Hemangioma of the skin.") == [0]
