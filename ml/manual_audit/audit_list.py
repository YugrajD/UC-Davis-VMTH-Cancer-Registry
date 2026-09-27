"""The universal audit list: every case awaiting specialist review, for the dashboard.

One list gathers every review source, each case once, in this order (a case
listed by two sources keeps the first):

1. ``eval_batch`` — drawn eval-batch cases (gold-eval).
2. ``diagnosis_mapping_audit`` — Diagnosis-Mapping audit cases.
3. ``report_mapping_audit`` — Report-Mapping audit cases.
4. ``review_queue`` — ``config.REVIEW_QUEUE_CSV``, in its own priority order.

Cases that already have gold are left off. The backend receives only case IDs
(``handoff.exports.export_audit_list``). The gold origin each case must come
back under is kept here, in ``config.AUDIT_LIST_LEDGER_CSV`` — one row per case,
first list wins, so a case never changes origin — and ``origin_of`` hands it to
the gold import when the backend's rows carry no origin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
import io_utils
from manual_audit import diagnosis_mapping_audit, eval_batch, report_mapping_audit
from manual_audit.gold import load_gold

LEDGER_COLS = ["case_id", "origin", "list_id"]


class AuditListError(Exception):
    """An audit list was refused."""


def _ledger() -> pd.DataFrame:
    path = config.AUDIT_LIST_LEDGER_CSV
    if not path.is_file():
        return pd.DataFrame(columns=LEDGER_COLS)
    return io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def _review_queue_case_ids() -> list[str]:
    path = config.REVIEW_QUEUE_CSV
    if not path.is_file():
        return []
    return list(io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)["case_id"])


def build(list_id: str, *, include_review_queue: bool = True) -> dict:
    """The list's case IDs in order, with each case's origin. Writes nothing; ``record`` does.

    ``include_review_queue=False`` leaves the review queue off, e.g. while a cascade fix is expected to
    shrink it."""
    ledger = _ledger()
    if list_id in set(ledger["list_id"]):
        raise AuditListError(f"list_id {list_id!r} is already in {config.AUDIT_LIST_LEDGER_CSV}")
    known_origin = dict(zip(ledger["case_id"], ledger["origin"]))
    gold_cases = set(load_gold()["case_id"])
    sources = [
        ("eval_batch", eval_batch.pending_case_ids()),
        ("diagnosis_mapping_audit", diagnosis_mapping_audit.pending_case_ids()),
        ("report_mapping_audit", report_mapping_audit.pending_case_ids()),
        ("review_queue", _review_queue_case_ids() if include_review_queue else []),
    ]
    origin_of_case: dict[str, str] = {}
    for origin, case_ids in sources:
        for case_id in case_ids:
            if case_id not in gold_cases and case_id not in origin_of_case:
                origin_of_case[case_id] = known_origin.get(case_id, origin)
    by_origin = pd.Series(origin_of_case, dtype=str).value_counts().to_dict()
    return {"list_id": list_id, "case_ids": list(origin_of_case), "origin_of": origin_of_case,
            "by_origin": by_origin, "new_cases": len(set(origin_of_case) - set(known_origin))}


def record(built: dict) -> Path:
    """Append the cases ``built`` lists for the first time to the ledger, with their origin."""
    ledger = _ledger()
    new = [{"case_id": c, "origin": o, "list_id": built["list_id"]}
           for c, o in built["origin_of"].items() if c not in set(ledger["case_id"])]
    path = config.AUDIT_LIST_LEDGER_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.concat([ledger, pd.DataFrame(new, columns=LEDGER_COLS)], ignore_index=True), path)
    return path


def origin_of(case_ids) -> dict[str, str]:
    """The recorded gold origin of each of ``case_ids`` that was ever on an audit list."""
    ledger = _ledger()
    ledger = ledger[ledger["case_id"].isin(set(case_ids))]
    return dict(zip(ledger["case_id"], ledger["origin"]))
