"""Unit tests for ingestion_service pure functions.

Covers normalize_anon_id, split_numbered, parse_predictions, and
parse_dataset_a_demographics — the parsing layer that converts raw ML output
and demographics CSVs into per-patient dicts.
"""

import csv
import io

import pytest

from app.services.ingestion_service import (
    normalize_anon_id,
    parse_dataset_a_demographics,
    parse_predictions,
    split_numbered,
)


# ---------------------------------------------------------------------------
# normalize_anon_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("37",         "ID_37"),     # plain integer string
    ("37.0",       "ID_37"),     # Excel float export
    ("37.9",       "ID_37"),     # non-whole float → truncated
    ("ID_37",      "ID_37"),     # already normalised
    ("ID_37.0",    "ID_37"),     # ID_ prefix with float suffix
    ("id_37",      "ID_37"),     # lowercase prefix
    ("ID_  37",    "ID_37"),     # whitespace inside suffix
    ("nan",        ""),          # pandas NaN sentinel
    ("NaN",        ""),
    ("",           ""),          # empty
    ("lymphoma",   "lymphoma"),  # non-numeric, non-ID_ passthrough
    ("CASE-0001",  "CASE-0001"), # CASE-prefixed IDs pass through unchanged
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


def test_parse_predictions_case_id_alias():
    """case_id column is accepted as an alias for anon_id."""
    row = {
        "case_id": "CASE-0001",
        "predicted_term": "Lymphoma",
        "predicted_group": "Lymphoma",
        "predicted_code": "9590/3",
        "confidence": "0.90",
        "method": "embedding",
    }
    result = parse_predictions([row])
    assert "CASE-0001" in result
    assert result["CASE-0001"][0]["predicted_group"] == "Lymphoma"


def test_parse_predictions_per_row_format():
    """Explicit integer diagnosis_index column → per-row format, one diag per row."""
    rows = [
        {
            "case_id": "CASE-0001",
            "diagnosis_index": "1",
            "predicted_term": "Diffuse large B-cell lymphoma",
            "predicted_group": "Lymphoma",
            "predicted_code": "9680/3",
            "confidence": "0.92",
            "method": "embedding",
        },
        {
            "case_id": "CASE-0001",
            "diagnosis_index": "2",
            "predicted_term": "Mast cell tumor",
            "predicted_group": "MCT",
            "predicted_code": "8720/3",
            "confidence": "0.55",
            "method": "embedding",
        },
    ]
    result = parse_predictions(rows)
    assert "CASE-0001" in result
    diags = result["CASE-0001"]
    assert len(diags) == 2
    assert diags[0]["diagnosis_index"] == 1
    assert diags[0]["predicted_group"] == "Lymphoma"
    assert diags[0]["confidence"] == pytest.approx(0.92)
    assert diags[1]["diagnosis_index"] == 2
    assert diags[1]["predicted_group"] == "MCT"


def test_parse_predictions_per_row_no_original_text():
    """Per-row format without original_text column stores empty string."""
    row = {
        "case_id": "CASE-0002",
        "diagnosis_index": "1",
        "predicted_term": "Lymphoma",
        "predicted_group": "Lymphoma",
        "predicted_code": "",
        "confidence": "0.80",
        "method": "embedding",
    }
    result = parse_predictions([row])
    assert result["CASE-0002"][0]["original_text"] == ""


# ---------------------------------------------------------------------------
# parse_dataset_a_demographics
# ---------------------------------------------------------------------------


def _make_demo_csv(rows: list[dict], fieldnames: list[str] | None = None) -> bytes:
    fields = fieldnames or list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def test_demographics_case_id_alias():
    """case_id column is accepted as alias for anon_id."""
    csv_bytes = _make_demo_csv([{
        "case_id": "CASE-0001",
        "DtOfRq": "15-Mar-24",
        "Sex": "FS",
        "Species": "Dog",
        "Breed": "Labrador",
        "Zipcode": "95616",
        "RfrrVtrnZipcode": "",
    }])
    result = parse_dataset_a_demographics(csv_bytes)
    assert "CASE-0001" in result
    assert result["CASE-0001"]["sex"] == "Spayed Female"
    assert result["CASE-0001"]["breed"] == "Labrador"
    assert result["CASE-0001"]["zip"] == "95616"


def test_demographics_short_zip_column_names():
    """'Zipcode' and 'RfrrVtrnZipcode' are accepted as short-form aliases."""
    csv_bytes = _make_demo_csv([{
        "case_id": "CASE-0002",
        "DtOfRq": "",
        "Sex": "M",
        "Species": "Dog",
        "Breed": "",
        "Zipcode": "",
        "RfrrVtrnZipcode": "94103",
    }])
    result = parse_dataset_a_demographics(csv_bytes)
    assert result["CASE-0002"]["zip"] == "94103"


def test_demographics_long_zip_column_names():
    """'Zipcode Zipcode' and 'RfrrVtrn Zipcode Zipcode' still work."""
    csv_bytes = _make_demo_csv([{
        "anon_id": "ID_10",
        "DtOfRq": "",
        "Sex": "MC",
        "Species": "Dog",
        "Breed": "Poodle",
        "Zipcode Zipcode": "90210",
        "RfrrVtrn Zipcode Zipcode": "94103",
    }])
    result = parse_dataset_a_demographics(csv_bytes)
    assert result["ID_10"]["zip"] == "90210"  # primary takes priority


def test_demographics_zip_fallback_to_referral():
    """Falls back to referral zip when primary is blank."""
    csv_bytes = _make_demo_csv([{
        "case_id": "CASE-0003",
        "DtOfRq": "",
        "Sex": "F",
        "Species": "Dog",
        "Breed": "",
        "Zipcode": "NA",
        "RfrrVtrnZipcode": "93711",
    }])
    result = parse_dataset_a_demographics(csv_bytes)
    assert result["CASE-0003"]["zip"] == "93711"


def test_demographics_unknown_sex_returns_none():
    """Sex codes not in SEX_MAP (U, X) resolve to None."""
    for sex_code in ("U", "X"):
        csv_bytes = _make_demo_csv([{
            "case_id": "CASE-0004",
            "DtOfRq": "",
            "Sex": sex_code,
            "Species": "Dog",
            "Breed": "",
            "Zipcode": "",
            "RfrrVtrnZipcode": "",
        }])
        result = parse_dataset_a_demographics(csv_bytes)
        assert result["CASE-0004"]["sex"] is None, f"expected None for sex code {sex_code!r}"


def test_demographics_first_value_wins_on_duplicate_rows():
    """When the same case appears twice, the first non-empty value is kept."""
    rows = [
        {"case_id": "CASE-0005", "DtOfRq": "", "Sex": "M", "Species": "Dog", "Breed": "Lab",
         "Zipcode": "95616", "RfrrVtrnZipcode": ""},
        {"case_id": "CASE-0005", "DtOfRq": "", "Sex": "F", "Species": "Cat", "Breed": "Siamese",
         "Zipcode": "90210", "RfrrVtrnZipcode": ""},
    ]
    csv_bytes = _make_demo_csv(rows)
    result = parse_dataset_a_demographics(csv_bytes)
    assert result["CASE-0005"]["sex"] == "Male"
    assert result["CASE-0005"]["breed"] == "Lab"
    assert result["CASE-0005"]["zip"] == "95616"
