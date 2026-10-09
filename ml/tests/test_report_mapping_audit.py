"""manual_audit/report_mapping_audit.py: contradicted-label + random draw from the cross-fit scores."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import io_utils
from manual_audit import report_mapping_audit as rm
from report_mapping.training.crossfit import COLUMNS as CROSSFIT_COLUMNS

from . import fixtures as fx

SPLIT_ID = "rm-audit-split"


def _cancer(case_id, gate, *term_probs):
    return [(case_id, "0", "1", gate, f"Term {i}", p) for i, p in enumerate(term_probs)]


def _no_cancer(case_id, gate):
    return [(case_id, "0", "0", gate, "", "")]


# Gate contradictions: labelled cancer with gate p < 0.2 (C1 strongest .. C3); labelled no cancer with p > 0.8 (N1, N2).
# Term contradictions: labelled cancer, gate p >= 0.8, every labelled term below 0.2 (empty = outside the fold vocabulary):
#   T1 (best 0.01), T5 (best 0.19), T2 (nothing scored, comes last). Near misses: T3 has one term at 0.9, T4 has gate 0.79,
#   T6 sits exactly on the 0.2 term bound, T7 on the 0.8 gate bound with a confident term.
ROWS = [
    *_cancer("C1", "0.01", "0.5"), *_cancer("C2", "0.1", "0.5"), *_cancer("C3", "0.15", "0.5"),
    *_no_cancer("N1", "0.99"), *_no_cancer("N2", "0.85"),
    *_cancer("T1", "0.95", "0.01"), *_cancer("T2", "0.9", ""), *_cancer("T5", "0.99", "0.19", "0.05"),
    *_cancer("T3", "0.85", "0.05", "0.9"), *_cancer("T4", "0.79", "0.01"), *_cancer("T6", "0.9", "0.2"),
    *_cancer("T7", "0.8", "0.5"),
    *[row for i in range(20) for row in (_cancer(f"R{i:02d}", "0.5", "0.5") if i % 2 else _no_cancer(f"R{i:02d}", "0.5"))],
]
CASES = sorted({r[0] for r in ROWS})


def _write_oof(path, rows):
    io_utils.write_csv(pd.DataFrame(rows, columns=CROSSFIT_COLUMNS), path)
    return path


@pytest.fixture
def oof_csv(tmp_path, monkeypatch):
    fx.point_manual_audit_config_at(monkeypatch, tmp_path)
    fx.make_three_way_split_generation(SPLIT_ID, CASES, ["CAL-1"], ["TEST-1"])
    return _write_oof(tmp_path / "oof.csv", ROWS)


def _ledger(batch):
    return io_utils.read_csv(rm.ledger_path(batch), encoding="utf-8", dtype=str, keep_default_na=False)


def _by_reason(ledger):
    return ledger.groupby("reason")["case_id"].apply(list).to_dict()


def test_splits_contradicted_evenly_across_reasons_strongest_first_then_draws_random_rest(oof_csv):
    result = rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=6, n_random=5, seed=1)
    ledger = _ledger(1)
    assert list(ledger.columns) == rm.LEDGER_COLS
    by_reason = _by_reason(ledger)
    assert by_reason[rm.CONTRADICTED_CANCER] == ["C1", "C2"]
    assert by_reason[rm.CONTRADICTED_NO_CANCER] == ["N1", "N2"]
    assert by_reason[rm.CONTRADICTED_TERM] == ["T1", "T5"]
    drawn = set(by_reason[rm.RANDOM])
    assert len(drawn) == 5 and not drawn & {"C1", "C2", "N1", "N2", "T1", "T5"}
    assert ledger.loc[ledger["case_id"] == "C1", "oof_prob"].item() == "0.01"
    assert rm.case_list_path(1).read_text(encoding="utf-8").splitlines() == list(ledger["case_id"])
    assert result["pools"] == {rm.CONTRADICTED_CANCER: 3, rm.CONTRADICTED_NO_CANCER: 2, rm.CONTRADICTED_TERM: 3,
                               rm.RANDOM: len(CASES) - 6}


def test_term_contradiction_needs_a_confident_gate_and_every_term_below_the_bound(oof_csv):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=100, n_random=0)
    by_reason = _by_reason(_ledger(1))
    # Strongest first; T2 (no term scored) last. T3 (one term at 0.9), T4 (gate 0.79), T6 (term 0.2), T7 (confident term) are out.
    assert by_reason[rm.CONTRADICTED_TERM] == ["T1", "T5", "T2"]


def test_ledger_records_the_lowest_scored_term_probability_and_blank_for_no_cancer(oof_csv):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=100, n_random=100, seed=3)
    term_prob = dict(zip(_ledger(1)["case_id"], _ledger(1)["term_prob"]))
    assert term_prob["T1"] == "0.01" and term_prob["T5"] == "0.05"      # lowest of "0.19", "0.05"
    assert term_prob["T2"] == ""                                         # nothing scored
    assert term_prob["T3"] == "0.05"                                     # random draw, same rule
    assert term_prob["N1"] == "" and term_prob["R00"] == ""              # no-cancer cases carry no term
    assert term_prob["C1"] == "0.5"


def test_ledger_has_no_text_columns(oof_csv):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=4, n_random=4)
    ledger = _ledger(1)
    assert list(ledger.columns) == ["case_id", "reason", "target", "oof_prob", "term_prob"]
    assert not {"term", "text", "diagnosis", "report"} & set(ledger.columns)
    assert ledger["case_id"].str.fullmatch(r"[A-Z]\d+").all()


def test_a_short_reason_hands_its_turns_to_the_others(oof_csv):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=8, n_random=0)
    reasons = _ledger(1)["reason"].value_counts().to_dict()
    assert reasons == {rm.CONTRADICTED_CANCER: 3, rm.CONTRADICTED_NO_CANCER: 2, rm.CONTRADICTED_TERM: 3}
    rm.sample(oof_csv, SPLIT_ID, batch=2, n_contradicted=100, n_random=0)
    assert _ledger(2).empty    # everything contradicted was taken by batch 1


def test_later_batches_skip_earlier_batches_and_gold(oof_csv):
    config.GOLD_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{"case_id": "C2", "code": "NO_CANCER", "origin": "report_mapping_audit"}]),
                       config.GOLD_STORE_CSV)
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=3, n_random=3, seed=1)
    rm.sample(oof_csv, SPLIT_ID, batch=2, n_contradicted=3, n_random=3, seed=1)
    first, second = _ledger(1), _ledger(2)
    assert not set(first["case_id"]) & set(second["case_id"])
    assert "C2" not in set(first["case_id"]) | set(second["case_id"])
    assert _by_reason(first)[rm.CONTRADICTED_CANCER] == ["C1"] and _by_reason(first)[rm.CONTRADICTED_TERM] == ["T1"]
    assert _by_reason(second)[rm.CONTRADICTED_CANCER] == ["C3"] and _by_reason(second)[rm.CONTRADICTED_TERM] == ["T5"]
    assert rm.pending_case_ids() == list(first["case_id"]) + list(second["case_id"])


def test_refuses_an_existing_batch_oof_cases_outside_train_and_the_old_gate_only_file(oof_csv, tmp_path):
    rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=2, n_random=2)
    with pytest.raises(rm.ReportMappingAuditError, match="already exists"):
        rm.sample(oof_csv, SPLIT_ID, batch=1, n_contradicted=2, n_random=2)
    leaky = _write_oof(tmp_path / "leaky.csv", _cancer("TEST-1", "0.01", "0.5"))
    with pytest.raises(rm.ReportMappingAuditError, match="TEST-1"):
        rm.sample(leaky, SPLIT_ID, batch=2)
    old = tmp_path / "old.csv"
    io_utils.write_csv(pd.DataFrame([("C1", "1", "0.01")], columns=["case_id", "target", "prob"]), old)
    with pytest.raises(rm.ReportMappingAuditError, match="cross-fit"):
        rm.sample(old, SPLIT_ID, batch=2)
