"""manual_audit/audit_list.py + handoff export/import: one case-ID worklist, origins kept locally."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import io_utils
from handoff import contracts, exports
from handoff import imports as handoff_imports
from manual_audit import audit_list, sheets
from manual_audit import diagnosis_mapping_audit as dm
from manual_audit import report_mapping_audit as rm

from . import fixtures as fx


def _write(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, path)


@pytest.fixture
def sources(tmp_path, monkeypatch, labels_csv):
    """E1, E2 drawn by an eval batch; D1 (and E2 again) in a DM audit batch; R1 (and D1 again) in an RM audit
    batch; Q1, R1, Q2 in the review queue. Q2 already has gold."""
    fx.point_handoff_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "LABELS_CSV", labels_csv)
    _write(pd.DataFrame({"case_id": ["E1", "E2"]}), config.EVAL_BATCH_LEDGER_CSV)
    dm.key_path(1).parent.mkdir(parents=True, exist_ok=True)
    sheets.write_csv(dm.key_path(1), dm.KEY_COLS, [{**{c: "" for c in dm.KEY_COLS}, "case_id": c, "diagnosis_number": "1"}
                                                   for c in ("D1", "E2")])
    _write(pd.DataFrame({"case_id": ["R1", "D1"], "reason": "random", "target": "1", "oof_prob": "0.5"}),
           rm.ledger_path(1))
    _write(pd.DataFrame({"case_id": ["Q1", "R1", "Q2"]}), config.REVIEW_QUEUE_CSV)
    _write(pd.DataFrame([{"case_id": "Q2", "code": "NO_CANCER", "origin": "review_queue"}]), config.GOLD_STORE_CSV)


def test_build_orders_by_source_lists_each_case_once_and_skips_gold(sources):
    built = audit_list.build("L1")
    assert built["case_ids"] == ["E1", "E2", "D1", "R1", "Q1"]
    assert built["origin_of"] == {"E1": "eval_batch", "E2": "eval_batch", "D1": "diagnosis_mapping_audit",
                                  "R1": "report_mapping_audit", "Q1": "review_queue"}
    assert built["new_cases"] == 5


def test_export_writes_ids_only_records_origins_and_keeps_them_across_lists(sources):
    result = exports.export_audit_list("L1")
    assert result["path"].read_text(encoding="utf-8") == "E1\nE2\nD1\nR1\nQ1\n"
    contracts.verify_sidecar(result["path"], expected_kind=contracts.AUDIT_LIST_EXPORT_KIND)
    with pytest.raises(audit_list.AuditListError, match="L1"):
        exports.export_audit_list("L1")

    # E1 leaves the eval-batch ledger; if it resurfaces from another source it keeps its first origin.
    _write(pd.DataFrame({"case_id": ["E2"]}), config.EVAL_BATCH_LEDGER_CSV)
    _write(pd.DataFrame({"case_id": ["Q1", "E1"]}), config.REVIEW_QUEUE_CSV)
    second = audit_list.build("L2")
    assert second["origin_of"]["E1"] == "eval_batch" and second["new_cases"] == 0
    ledger = io_utils.read_csv(config.AUDIT_LIST_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(ledger.columns) == audit_list.LEDGER_COLS and len(ledger) == 5


def _gold_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_import_gold_fills_a_blank_origin_from_the_list_and_refuses_unlisted_or_conflicting(sources, tmp_path):
    exports.export_audit_list("L1")
    ok = _gold_csv(tmp_path / "g1.csv", [{"case_id": "R1", "term": "Fibrosarcoma, NOS", "origin": ""},
                                         {"case_id": "D1", "term": "Mast cell tumor, malignant", "origin": ""}])
    handoff_imports.import_gold(ok, "export-1", reviewer="Dr. Test")
    stored = io_utils.read_csv(config.GOLD_STORE_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    assert dict(zip(stored["case_id"], stored["origin"]))["R1"] == "report_mapping_audit"
    assert dict(zip(stored["case_id"], stored["origin"]))["D1"] == "diagnosis_mapping_audit"
    assert "R1" not in audit_list.build("L2")["case_ids"]  # reviewed cases leave the list

    no_origin_column = tmp_path / "g2.csv"
    no_origin_column.write_text("case_id,term\nUNLISTED,Fibrosarcoma, NOS\n", encoding="utf-8")
    with pytest.raises(handoff_imports.HandoffImportError, match="never on an audit list"):
        handoff_imports.import_gold(no_origin_column, "export-2", reviewer="Dr. Test")
    conflicting = _gold_csv(tmp_path / "g3.csv", [{"case_id": "Q1", "term": "Fibrosarcoma, NOS",
                                                   "origin": "report_mapping_audit"}])
    with pytest.raises(handoff_imports.HandoffImportError, match="Q1"):
        handoff_imports.import_gold(conflicting, "export-3", reviewer="Dr. Test")
    assert not (config.HANDOFF_INBOX_DIR / "gold_export-3.csv").exists()
