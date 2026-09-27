"""manual_audit/report_mapping_audit.py: contradicted-label + random draw from gate OOF scores."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import io_utils
from manual_audit import report_mapping_audit as rm

from . import fixtures as fx

SPLIT_ID = "rm-audit-split"
# Labelled cancer with p < 0.2: C1 (0.01, strongest) .. C3 (0.15). Labelled no cancer with p > 0.8: N1 (0.99) .. N2.
OOF = [
    ("C1", "1", "0.01"), ("C2", "1", "0.1"), ("C3", "1", "0.15"),
    ("N1", "0", "0.99"), ("N2", "0", "0.85"),
    *[(f"R{i:02d}", str(i % 2), "0.5") for i in range(20)],  # ordinary, uncontradicted labels
]


@pytest.fixture
def oof_csv(tmp_path, monkeypatch):
    fx.point_manual_audit_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation(SPLIT_ID, [c for c, _, _ in OOF], ["CAL-1"], ["TEST-1"])
    path = tmp_path / "oof.csv"
    io_utils.write_csv(pd.DataFrame(OOF, columns=["case_id", "target", "prob"]), path)
    return path


def _ledger(batch):
    return io_utils.read_csv(rm.ledger_path(batch), encoding="utf-8", dtype=str, keep_default_na=False)


def test_splits_contradicted_evenly_strongest_first_then_draws_random_rest(oof_csv):
    result = rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=4, n_random=5, seed=1)
    ledger = _ledger(1)
    assert list(ledger.columns) == rm.LEDGER_COLS
    by_reason = ledger.groupby("reason")["case_id"].apply(list).to_dict()
    assert by_reason[rm.CONTRADICTED_CANCER] == ["C1", "C2"]
    assert by_reason[rm.CONTRADICTED_NO_CANCER] == ["N1", "N2"]
    assert len(by_reason[rm.RANDOM]) == 5 and not set(by_reason[rm.RANDOM]) & {"C1", "C2", "N1", "N2"}
    assert ledger.loc[ledger["case_id"] == "C1", "oof_prob"].item() == "0.01"
    assert rm.case_list_path(1).read_text(encoding="utf-8").splitlines() == list(ledger["case_id"])
    assert result["pools"] == {rm.CONTRADICTED_CANCER: 3, rm.CONTRADICTED_NO_CANCER: 2, rm.RANDOM: 21}


def test_a_short_side_hands_its_remainder_to_the_other(oof_csv):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=5, n_random=0)
    reasons = _ledger(1)["reason"].value_counts().to_dict()
    assert reasons == {rm.CONTRADICTED_CANCER: 3, rm.CONTRADICTED_NO_CANCER: 2}


def test_later_batches_skip_earlier_batches_and_gold(oof_csv):
    config.GOLD_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{"case_id": "C2", "code": "NO_CANCER", "origin": "report_mapping_audit"}]),
                       config.GOLD_STORE_CSV)
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=2, n_random=3, seed=1)
    rm.sample(oof_csv, SPLIT_ID, batch=2, n_contradicted=2, n_random=3, seed=1)
    first, second = _ledger(1), _ledger(2)
    assert not set(first["case_id"]) & set(second["case_id"])
    assert "C2" not in set(first["case_id"]) | set(second["case_id"])
    assert list(first.loc[first["reason"] == rm.CONTRADICTED_CANCER, "case_id"]) == ["C1"]
    assert list(second.loc[second["reason"] == rm.CONTRADICTED_CANCER, "case_id"]) == ["C3"]
    assert rm.pending_case_ids() == list(first["case_id"]) + list(second["case_id"])


def test_refuses_an_existing_batch_and_oof_cases_outside_train(oof_csv, tmp_path):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=2, n_random=2)
    with pytest.raises(rm.ReportMappingAuditError, match="already exists"):
        rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=2, n_random=2)
    leaky = tmp_path / "leaky.csv"
    io_utils.write_csv(pd.DataFrame([("TEST-1", "1", "0.01")], columns=["case_id", "target", "prob"]), leaky)
    with pytest.raises(rm.ReportMappingAuditError, match="TEST-1"):
        rm.sample(leaky, SPLIT_ID, batch=2)
