"""evaluation/audit_rates.py: weighted Tier-3 audit rates and Wilson/Kish CIs, against hand computation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import config
import io_utils
from evaluation import audit_rates, intervals
from manual_audit.tier3_audit import AUDIT_STORE_FIELDS

from . import fixtures as fx


def _rows(stratum, batch, weight, verdicts):
    return [{"case_id": f"{stratum}-{batch}-{i}", "diagnosis_number": "1", "sample_stratum": stratum,
             "sample_weight": f"{weight:.2f}", "batch": str(batch), "verdict": v} for i, v in enumerate(verdicts)]


@pytest.fixture
def result(tmp_path, monkeypatch):
    fx.point_evaluation_config_at(monkeypatch, tmp_path)
    rows = (
        _rows("tier3_llm_no_match", 1, 10, ["wrong", "no_cancer", "no_cancer", "uncertain"])
        + _rows("tier3_llm_no_match", 2, 5, ["wrong", "no_cancer"])
        + _rows("tier2_fuzzy", 1, 2, ["correct", "correct", "correct", "correct", "wrong"])
        + _rows("tier3_no_candidates", 1, 3, ["wrong", "no_cancer"])
    )
    config.AUDIT_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame(rows, columns=AUDIT_STORE_FIELDS).fillna(""), config.AUDIT_STORE_CSV)
    return audit_rates.audit_rates()


def test_declined_missed_cancer_rate_and_ci(result):
    r = result["declined_missed_cancer"]
    # weighted: (10 + 5) / (4*10 + 2*5) = 0.3; Kish n_eff = 50² / (4*100 + 2*25)
    n_eff = 50 ** 2 / 450
    assert (r["n"], r["hits"], r["sample_share"]) == (6, 2, pytest.approx(2 / 6))
    assert r["weighted_share"] == pytest.approx(0.3) and r["represents"] == pytest.approx(50)
    assert (r["lo"], r["hi"]) == pytest.approx(intervals.wilson(0.3 * n_eff, n_eff))
    assert result["declined_uncertain"]["weighted_share"] == pytest.approx(10 / 50)


def test_tier2_fuzzy_error_rate(result):
    r = result["tier2_fuzzy_error"]
    assert r["weighted_share"] == pytest.approx(0.2)
    assert (r["lo"], r["hi"]) == pytest.approx(intervals.wilson(1, 5))


def test_reservoir_is_stratified(result):
    r = result["reservoir"]
    # no_match: p=0.3, W=50/56, n_eff=50²/450; no_candidates: p=0.5, W=6/56, n_eff=2
    w1, w2, n1 = 50 / 56, 6 / 56, 50 ** 2 / 450
    p = w1 * 0.3 + w2 * 0.5
    var = w1 ** 2 * 0.21 / n1 + w2 ** 2 * 0.25 / 2
    n_eff = p * (1 - p) / var
    assert r["weighted_share"] == pytest.approx(18 / 56) == pytest.approx(p)
    assert (r["lo"], r["hi"]) == pytest.approx(intervals.wilson(p * n_eff, n_eff))


def test_stratum_table_and_warnings(result):
    table = result["by_stratum"]
    assert list(dict.fromkeys(table["stratum"])) == ["tier3_llm_no_match", "tier2_fuzzy", "tier3_no_candidates"]
    headline = table[table["headline"]].set_index("stratum")["verdict"].to_dict()
    assert headline == {"tier3_llm_no_match": "wrong", "tier2_fuzzy": "correct", "tier3_no_candidates": "wrong"}
    correct = table[(table["stratum"] == "tier2_fuzzy") & (table["verdict"] == "correct")].iloc[0]
    assert math.isclose(correct["weighted_share"], 0.8)
    assert any("PARTIAL" in w for w in result["warnings"])
    assert any("2 batches" in w for w in result["warnings"])


def test_no_audit_rows_fails_cleanly(tmp_path, monkeypatch):
    fx.point_evaluation_config_at(monkeypatch, tmp_path)
    with pytest.raises(audit_rates.AuditRatesError, match="no audit rows"):
        audit_rates.audit_rates()
