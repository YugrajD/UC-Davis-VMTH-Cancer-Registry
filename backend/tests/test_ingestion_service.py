"""Unit tests for ingestion_service pure functions.

Covers normalize_anon_id, split_numbered, and parse_predictions — the
parsing layer that converts raw ML output into per-patient diagnosis dicts.
"""

import pytest

from app.services.ingestion_service import normalize_anon_id, parse_predictions, split_numbered


# ---------------------------------------------------------------------------
# normalize_anon_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("37",        "ID_37"),    # plain integer string
    ("37.0",      "ID_37"),    # Excel float export
    ("37.9",      "ID_37"),    # non-whole float → truncated
    ("ID_37",     "ID_37"),    # already normalised
    ("ID_37.0",   "ID_37"),    # ID_ prefix with float suffix
    ("id_37",     "ID_37"),    # lowercase prefix
    ("ID_  37",   "ID_37"),    # whitespace inside suffix
    ("nan",       ""),         # pandas NaN sentinel
    ("NaN",       ""),
    ("",          ""),         # empty
    ("lymphoma",  "lymphoma"), # non-numeric, non-ID_ passthrough
])
def test_normalize_anon_id(raw, expected):
    assert normalize_anon_id(raw) == expected


def test_normalize_anon_id_strips_outer_whitespace():
    assert normalize_anon_id("  42  ") == "ID_42"


# ---------------------------------------------------------------------------
# split_numbered
# ---------------------------------------------------------------------------


def test_split_numbered_two_items():
    assert split_numbered("1) Lymphoma 2) MCT") == ["Lymphoma", "MCT"]


def test_split_numbered_three_items():
    result = split_numbered("1) foo 2) bar 3) baz")
    assert result == ["foo", "bar", "baz"]


def test_split_numbered_single_item_with_marker():
    assert split_numbered("1) Lymphoma") == ["Lymphoma"]


def test_split_numbered_no_marker_returns_whole_string():
    assert split_numbered("Lymphoma") == ["Lymphoma"]


def test_split_numbered_empty_string():
    assert split_numbered("") == []


def test_split_numbered_confidence_values():
    result = split_numbered("1) 0.85 2) 0.60")
    assert result == ["0.85", "0.60"]


def test_split_numbered_strips_parts():
    result = split_numbered("1)  Mast cell  2)  Lymphoma ")
    assert result[0] == "Mast cell"
    assert result[1] == "Lymphoma"


# ---------------------------------------------------------------------------
# parse_predictions
# ---------------------------------------------------------------------------


def _row(anon_id="ID_1", original_text="report text",
         predicted_term="Lymphoma", predicted_group="Lymphoma",
         predicted_code="9590/3", confidence="0.85", method="embedding"):
    return {
        "anon_id": anon_id,
        "original_text": original_text,
        "predicted_term": predicted_term,
        "predicted_group": predicted_group,
        "predicted_code": predicted_code,
        "confidence": confidence,
        "method": method,
    }


def test_parse_predictions_single_diagnosis():
    result = parse_predictions([_row()])
    assert "ID_1" in result
    diags = result["ID_1"]
    assert len(diags) == 1
    d = diags[0]
    assert d["diagnosis_index"] == 1
    assert d["predicted_group"] == "Lymphoma"
    assert d["confidence"] == pytest.approx(0.85)
    assert d["original_text"] == "report text"


def test_parse_predictions_numbered_format_splits_into_two():
    row = _row(
        predicted_term="1) Lymphoma 2) MCT",
        predicted_group="1) Lymphoma 2) MCT",
        predicted_code="1) 9590/3 2) 8720/3",
        confidence="1) 0.85 2) 0.60",
        method="1) embedding 2) embedding",
    )
    result = parse_predictions([row])
    diags = result["ID_1"]
    assert len(diags) == 2
    assert diags[0]["diagnosis_index"] == 1
    assert diags[0]["predicted_group"] == "Lymphoma"
    assert diags[0]["confidence"] == pytest.approx(0.85)
    assert diags[1]["diagnosis_index"] == 2
    assert diags[1]["predicted_group"] == "MCT"
    assert diags[1]["confidence"] == pytest.approx(0.60)


def test_parse_predictions_original_text_preserved_on_all_ranks():
    row = _row(
        original_text="full report",
        predicted_term="1) Lymphoma 2) MCT",
        predicted_group="1) Lymphoma 2) MCT",
        predicted_code="",
        confidence="1) 0.85 2) 0.60",
        method="1) embedding 2) embedding",
    )
    result = parse_predictions([row])
    for d in result["ID_1"]:
        assert d["original_text"] == "full report"


def test_parse_predictions_skips_method_empty():
    rows = [
        _row(anon_id="ID_1", method="embedding"),
        _row(anon_id="ID_2", method="empty"),
    ]
    result = parse_predictions(rows)
    assert "ID_1" in result
    assert "ID_2" not in result


def test_parse_predictions_skips_blank_anon_id():
    result = parse_predictions([_row(anon_id="")])
    assert result == {}


def test_parse_predictions_normalizes_anon_id():
    result = parse_predictions([_row(anon_id="37.0")])
    assert "ID_37" in result
    assert "37.0" not in result


def test_parse_predictions_multiple_patients():
    rows = [_row(anon_id="ID_1"), _row(anon_id="ID_2")]
    result = parse_predictions(rows)
    assert set(result.keys()) == {"ID_1", "ID_2"}


def test_parse_predictions_invalid_confidence_defaults_to_zero():
    row = _row(confidence="not-a-number")
    result = parse_predictions([row])
    assert result["ID_1"][0]["confidence"] == pytest.approx(0.0)


def test_parse_predictions_low_confidence_method_preserved():
    row = _row(method="low_confidence", confidence="0.20")
    result = parse_predictions([row])
    assert result["ID_1"][0]["method"] == "low_confidence"
