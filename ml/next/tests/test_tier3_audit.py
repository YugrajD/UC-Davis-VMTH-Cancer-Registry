"""manual_audit/tier3_audit.py: sample, pilot, ingest.

The ingest fixtures below are built from the REAL legacy sheet headers only
(read once, by column name, never by value) — every case id, diagnosis and
term is invented. This exercises "the existing tier3_audit_* pilot, remainder
and full sheets, with their key file, in their current format, unchanged."
"""

from __future__ import annotations

import pytest

import io_utils
from manual_audit import sheets, tier3_audit

from . import fixtures as fx

# The real tier3_audit_review.csv / tier3_audit_key.csv headers, verified once
# against the on-disk legacy sheets (headers and row counts only — never their
# content). Kept as literal lists here so a header drift in the legacy files
# would be caught by a failing assertion, not silently ignored.
REAL_REVIEW_HEADER = ["row_id", "Clinical Diagnosis", "Predicted Match", "Actual Diagnosis"]
REAL_KEY_HEADER = [
    "row_id", "case_id", "diagnosis_number", "diagnosis",
    "cascade_matched_term", "cascade_matched_group", "cascade_matched_code",
    "cascade_method", "decision_stage", "sample_stratum", "sample_weight",
]


@pytest.fixture
def silver_env(tmp_path, monkeypatch):
    fx.point_manual_audit_config_at(monkeypatch, tmp_path)
    return fx.make_tier3_silver_generation()


def _invented_key_row(row_id, case_id, diag_no, term, group, code, method, stage, stratum, weight):
    return {
        "row_id": row_id, "case_id": case_id, "diagnosis_number": str(diag_no),
        "diagnosis": "invented diagnosis text", "cascade_matched_term": term,
        "cascade_matched_group": group, "cascade_matched_code": code,
        "cascade_method": method, "decision_stage": stage,
        "sample_stratum": stratum, "sample_weight": f"{weight:.2f}",
    }


def _write_invented_sheets(tmp_path):
    """Four rows covering correct / no_cancer / uncertain / wrong, real headers only."""
    key_rows = [
        _invented_key_row("1", "CASE-A", 1, "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3",
                          "Exact", "tier1_exact", "tier3_llm_answered", 4.0),
        _invented_key_row("2", "CASE-B", 1, "", "", "", "No Match", "tier3_llm",
                          "tier3_llm_no_match", 3.4),
        _invented_key_row("3", "CASE-C", 1, "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3",
                          "Exact", "tier1_exact", "tier3_llm_answered", 4.0),
        _invented_key_row("4", "CASE-D", 1, "", "", "", "No Match", "tier3_llm",
                          "tier3_llm_no_match", 3.4),
    ]
    review_rows = [
        {"row_id": "1", "Clinical Diagnosis": "invented text a", "Predicted Match": "Mast cell tumor, malignant",
         "Actual Diagnosis": ""},
        {"row_id": "2", "Clinical Diagnosis": "invented text b", "Predicted Match": "(none)",
         "Actual Diagnosis": ""},
        {"row_id": "3", "Clinical Diagnosis": "invented text c", "Predicted Match": "Mast cell tumor, malignant",
         "Actual Diagnosis": "unclear"},
        {"row_id": "4", "Clinical Diagnosis": "invented text d", "Predicted Match": "(none)",
         "Actual Diagnosis": "Fibrosarcoma, NOS"},
    ]
    assert list(review_rows[0].keys()) == REAL_REVIEW_HEADER
    assert list(key_rows[0].keys()) == REAL_KEY_HEADER
    review_csv = tmp_path / "tier3_audit_review.csv"
    key_csv = tmp_path / "tier3_audit_key.csv"
    sheets.write_csv(review_csv, REAL_REVIEW_HEADER, review_rows)
    sheets.write_csv(key_csv, REAL_KEY_HEADER, key_rows)
    return review_csv, key_csv


def test_ingest_from_real_sheet_shape_never_touches_gold(tmp_path, labels_csv):
    review_csv, key_csv = _write_invented_sheets(tmp_path)
    audit_store = tmp_path / "audit_store.csv"
    gold_store = tmp_path / "gold_store.csv"  # never written by tier3_audit

    result = tier3_audit.ingest(
        review_csv=review_csv, key_csv=key_csv, batch=1, reviewer="Dr. Test",
        labels_csv=labels_csv, out_csv=audit_store,
    )
    assert result == {"ingested": 4, "added": 4, "replaced": 0, "total_rows": 4}
    assert not gold_store.exists()

    store = io_utils.read_csv(audit_store, encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(store.columns) == tier3_audit.AUDIT_STORE_FIELDS
    by_case = store.set_index("case_id")
    assert by_case.loc["CASE-A", "verdict"] == "correct"
    assert by_case.loc["CASE-B", "verdict"] == "no_cancer"
    assert by_case.loc["CASE-C", "verdict"] == "uncertain"
    assert by_case.loc["CASE-D", "verdict"] == "wrong"
    assert by_case.loc["CASE-D", "corrected_code"] == "2001/3"
    assert (store["match_strength"] == "gold").all()
    assert (store["batch"] == "1").all()


def test_ingest_resolves_ambiguous_term_with_group_prefix(tmp_path, labels_csv):
    # Extend the synthetic taxonomy with a second term sharing a group + a
    # colliding term name in another group, mirroring the real taxonomy's one
    # ambiguous term ("Papillary adenocarcinoma").
    with open(labels_csv, "a", encoding="utf-8", newline="") as file:
        file.write("4001/3,Rare Sarcomas,Ambiguous Term,Preferred,,\n")
        file.write("4002/3,Rare Carcinomas,Ambiguous Term,Preferred,,\n")
    review_csv, key_csv = _write_invented_sheets(tmp_path)
    rows = sheets.read_csv(review_csv)
    rows[3]["Actual Diagnosis"] = "Rare Sarcomas: Ambiguous Term"
    sheets.write_csv(review_csv, REAL_REVIEW_HEADER, rows)
    audit_store = tmp_path / "audit_store.csv"

    tier3_audit.ingest(review_csv, key_csv, batch=1, reviewer="Dr. Test",
                       labels_csv=labels_csv, out_csv=audit_store)
    store = io_utils.read_csv(audit_store, encoding="utf-8", dtype=str, keep_default_na=False)
    assert store.set_index("case_id").loc["CASE-D", "corrected_code"] == "4001/3"


def test_ingest_aborts_on_unknown_term_writes_nothing(tmp_path, labels_csv):
    review_csv, key_csv = _write_invented_sheets(tmp_path)
    rows = sheets.read_csv(review_csv)
    rows[3]["Actual Diagnosis"] = "Not A Real Taxonomy Term"
    sheets.write_csv(review_csv, REAL_REVIEW_HEADER, rows)
    audit_store = tmp_path / "audit_store.csv"

    with pytest.raises(tier3_audit.Tier3AuditError, match="Not A Real Taxonomy Term"):
        tier3_audit.ingest(review_csv, key_csv, batch=1, reviewer="Dr. Test",
                           labels_csv=labels_csv, out_csv=audit_store)
    assert not audit_store.exists()


def test_reingest_replaces_not_duplicates(tmp_path, labels_csv):
    review_csv, key_csv = _write_invented_sheets(tmp_path)
    audit_store = tmp_path / "audit_store.csv"
    tier3_audit.ingest(review_csv, key_csv, batch=1, reviewer="Dr. Test",
                       labels_csv=labels_csv, out_csv=audit_store)

    # Re-review row 1 as "wrong" and re-ingest the same batch.
    rows = sheets.read_csv(review_csv)
    rows[0]["Actual Diagnosis"] = "Fibrosarcoma, NOS"
    sheets.write_csv(review_csv, REAL_REVIEW_HEADER, rows)
    result = tier3_audit.ingest(review_csv, key_csv, batch=1, reviewer="Dr. Test",
                                labels_csv=labels_csv, out_csv=audit_store)
    assert result == {"ingested": 4, "added": 0, "replaced": 4, "total_rows": 4}

    store = io_utils.read_csv(audit_store, encoding="utf-8", dtype=str, keep_default_na=False)
    assert len(store) == 4  # no duplicates
    assert store.set_index("case_id").loc["CASE-A", "verdict"] == "wrong"


def test_ingest_refuses_unknown_columns_in_existing_store(tmp_path, labels_csv):
    import pandas as pd

    review_csv, key_csv = _write_invented_sheets(tmp_path)
    audit_store = tmp_path / "audit_store.csv"
    io_utils.write_csv(pd.DataFrame([{"case_id": "X", "unexpected_column": "y"}]), audit_store)
    with pytest.raises(tier3_audit.Tier3AuditError, match="unexpected"):
        tier3_audit.ingest(review_csv, key_csv, batch=1, reviewer="Dr. Test",
                           labels_csv=labels_csv, out_csv=audit_store)


def test_sample_selects_only_auditable_stages(silver_env, labels_csv, tmp_path):
    all_cases = {f"CASE-{i:04d}" for i in range(1, 8)}
    result = tier3_audit.sample(
        silver_id=silver_env, test_case_ids=all_cases, labels_csv=labels_csv,
        batch=1, n_rows=20, seed=1, out_dir=tmp_path / "sheets",
    )
    # fixtures.make_tier3_silver_generation has exactly one row per auditable
    # stratum (tier2_fuzzy, tier3_llm x3 outcomes, tier3_no_candidates) = 5 rows.
    assert result["n_rows"] == 5
    review_rows = sheets.read_csv(result["review_csv"])
    assert {r["row_id"] for r in review_rows} == {str(i) for i in range(1, 6)}
    key_rows = sheets.read_csv(result["key_csv"])
    assert all(r["decision_stage"] not in ("no_signal", "tier1_exact") for r in key_rows)


def test_sample_excludes_given_cases(silver_env, labels_csv, tmp_path):
    all_cases = {f"CASE-{i:04d}" for i in range(1, 8)}
    result = tier3_audit.sample(
        silver_id=silver_env, test_case_ids=all_cases, labels_csv=labels_csv,
        batch=1, n_rows=20, seed=1, exclude_case_ids={"CASE-0004"}, out_dir=tmp_path / "sheets",
    )
    key_rows = sheets.read_csv(result["key_csv"])
    assert "CASE-0004" not in {r["case_id"] for r in key_rows}


def test_pilot_splits_stratified_without_loss(tmp_path, labels_csv):
    review_csv, key_csv = _write_invented_sheets(tmp_path)
    # pilot() requires an unfilled sheet; blank out the Actual Diagnosis column
    # the way `sample` would have left it before anyone reviewed it.
    rows = sheets.read_csv(review_csv)
    for row in rows:
        row["Actual Diagnosis"] = ""
    sheets.write_csv(review_csv, REAL_REVIEW_HEADER, rows)
    pilot_csv = tmp_path / "pilot.csv"
    remainder_csv = tmp_path / "remainder.csv"
    result = tier3_audit.pilot(
        review_csv, key_csv, pilot_csv, remainder_csv, n_rows=2, min_per_stratum=1,
    )
    assert result["pilot_rows"] == 2
    assert result["remainder_rows"] == 2
    pilot_rows = sheets.read_csv(pilot_csv)
    remainder_rows = sheets.read_csv(remainder_csv)
    assert len(pilot_rows) + len(remainder_rows) == 4
    assert {r["row_id"] for r in pilot_rows} | {r["row_id"] for r in remainder_rows} == {"1", "2", "3", "4"}


def test_pilot_preamble_matches_legacy_closing_sentence(tmp_path, labels_csv):
    review_csv, key_csv = _write_invented_sheets(tmp_path)
    rows = sheets.read_csv(review_csv)
    for row in rows:
        row["Actual Diagnosis"] = ""
    sheets.write_csv(review_csv, REAL_REVIEW_HEADER, rows)
    instructions_md = tmp_path / "pilot_instructions.md"
    tier3_audit.pilot(
        review_csv, key_csv, tmp_path / "pilot.csv", tmp_path / "remainder.csv",
        n_rows=2, min_per_stratum=1, out_instructions_md=instructions_md,
    )
    text = instructions_md.read_text(encoding="utf-8")
    assert "that is a useful result, not a complaint" in text


def test_sample_refuses_to_overwrite_an_existing_case_ledger(silver_env, labels_csv, tmp_path):
    out_dir = tmp_path / "sheets"
    all_cases = {f"CASE-{i:04d}" for i in range(1, 8)}
    tier3_audit.sample(
        silver_id=silver_env, test_case_ids=all_cases, labels_csv=labels_csv,
        batch=1, n_rows=20, seed=1, out_dir=out_dir,
    )
    with pytest.raises(tier3_audit.Tier3AuditError, match="already exists"):
        tier3_audit.sample(
            silver_id=silver_env, test_case_ids=all_cases, labels_csv=labels_csv,
            batch=1, n_rows=20, seed=2, out_dir=out_dir,
        )


def test_sample_refuses_to_overwrite_an_existing_review_or_key_sheet(silver_env, labels_csv, tmp_path):
    """WP7 fix 9: the ledger isn't the only write-once artifact — a
    pre-existing review/key sheet (e.g. a reviewer's in-progress edits) must
    also block sample(), even when no ledger exists yet for that batch."""
    out_dir = tmp_path / "sheets"
    all_cases = {f"CASE-{i:04d}" for i in range(1, 8)}
    out_dir.mkdir(parents=True)
    (out_dir / "tier3_audit_review.csv").write_text("row_id\n1\n", encoding="utf-8")
    assert not tier3_audit.batch_ledger_path(1, out_dir).is_file()
    with pytest.raises(tier3_audit.Tier3AuditError, match="already exist"):
        tier3_audit.sample(
            silver_id=silver_env, test_case_ids=all_cases, labels_csv=labels_csv,
            batch=1, n_rows=20, seed=1, out_dir=out_dir,
        )
