"""manual_audit/diagnosis_mapping_audit.py: stratified row sample -> key CSV + case-ID list; pending cases."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import io_utils
from manual_audit import diagnosis_mapping_audit as dm
from manual_audit import sheets

from . import fixtures as fx

SPLIT_ID = "dm-audit-split"
EVAL_SIDE = [f"CASE-{i:04d}" for i in range(1, 8)]


@pytest.fixture
def silver_env(tmp_path, monkeypatch):
    fx.point_manual_audit_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation(SPLIT_ID, ["CASE-0901"], EVAL_SIDE[:3], EVAL_SIDE[3:])
    return fx.make_dm_audit_silver_generation()


def test_sample_selects_only_auditable_stages_and_writes_the_case_list(silver_env):
    result = dm.sample(silver_env, SPLIT_ID, batch=1, n_rows=20, seed=1)
    # fixtures.make_dm_audit_silver_generation has exactly one row per auditable
    # stratum (tier2_fuzzy, tier3_llm x3 outcomes, tier3_no_candidates) = 5 rows.
    assert result["n_rows"] == 5
    key_rows = sheets.read_csv(dm.key_path(1))
    assert list(key_rows[0]) == dm.KEY_COLS
    assert all(r["decision_stage"] not in ("no_signal", "tier1_exact") for r in key_rows)
    listed = dm.case_list_path(1).read_text(encoding="utf-8").splitlines()
    assert listed == sorted({r["case_id"] for r in key_rows})
    assert dm.case_list_path(1) == config.DIAGNOSIS_MAPPING_AUDIT_BATCH1_TXT  # what eval_batch excludes


def test_sample_draws_only_eval_side_cases_and_skips_earlier_batches(silver_env, monkeypatch):
    first = dm.sample(silver_env, SPLIT_ID, batch=1, n_rows=20, seed=1)
    assert set(dm.case_list_path(1).read_text(encoding="utf-8").split()) <= set(EVAL_SIDE)
    second = dm.sample(silver_env, SPLIT_ID, batch=2, n_rows=20, seed=1)
    assert first["n_rows"] == 5 and second["n_rows"] == 0  # every auditable row was drawn in batch 1

    # A case in train is never drawn.
    fx.make_three_way_split_generation("train-only", ["CASE-0004"], [], [c for c in EVAL_SIDE if c != "CASE-0004"])
    monkeypatch.setattr(config, "DIAGNOSIS_MAPPING_AUDIT_DIR", config.DIAGNOSIS_MAPPING_AUDIT_DIR / "other")
    dm.sample(silver_env, "train-only", batch=1, n_rows=20, seed=1)
    assert "CASE-0004" not in dm.case_list_path(1).read_text(encoding="utf-8").split()


def test_sample_refuses_an_existing_batch(silver_env):
    dm.sample(silver_env, SPLIT_ID, batch=1, n_rows=20, seed=1)
    with pytest.raises(dm.DiagnosisMappingAuditError, match="already exists"):
        dm.sample(silver_env, SPLIT_ID, batch=1, n_rows=20, seed=2)
    dm.key_path(1).unlink()
    with pytest.raises(dm.DiagnosisMappingAuditError, match="already exists"):  # the case list alone blocks too
        dm.sample(silver_env, SPLIT_ID, batch=1, n_rows=20, seed=2)


def test_pending_skips_row_reviewed_and_gold_cases(silver_env):
    dm.sample(silver_env, SPLIT_ID, batch=1, n_rows=20, seed=1)
    key_rows = sheets.read_csv(dm.key_path(1))
    all_cases = [r["case_id"] for r in key_rows]
    assert dm.pending_case_ids() == list(dict.fromkeys(all_cases))

    reviewed, with_gold = key_rows[0], key_rows[1]
    io_utils.write_csv(pd.DataFrame([{**{f: "" for f in dm.AUDIT_STORE_FIELDS}, "case_id": reviewed["case_id"],
                                      "diagnosis_number": reviewed["diagnosis_number"], "batch": "1"}],
                                    columns=dm.AUDIT_STORE_FIELDS), config.AUDIT_STORE_CSV)
    io_utils.write_csv(pd.DataFrame([{"case_id": with_gold["case_id"], "code": "NO_CANCER",
                                      "origin": "diagnosis_mapping_audit"}]), config.GOLD_STORE_CSV)
    assert dm.pending_case_ids() == [c for c in dict.fromkeys(all_cases)
                                     if c not in (reviewed["case_id"], with_gold["case_id"])]
