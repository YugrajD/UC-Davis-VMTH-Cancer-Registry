"""The combination rule: gold > silver > bronze, one combined prediction per case.

Per icd-mapping-strategy.md ("Coding a case") and ml-rewrite-plan.md's
Artefacts:

- A case with any gold row (any origin) takes gold — ``code_source=manual``,
  ``review_status=confirmed`` (gold is already a specialist's review).
- Else, a case with diagnosis (silver) rows where every row is decisive
  takes silver — ``code_source=diagnosis``, ``review_status=auto_accepted``.
- Else, a case with diagnosis rows where any row is vague gets **no** combined
  code here — it is only in the review queue (``coding.queue``) until gold
  resolves it.
- Else (no diagnosis rows at all), the case takes bronze —
  ``code_source=report``, ``review_status`` = ``queued`` if bronze is
  low-confidence, else ``auto_accepted``. **Bronze never overrides silver**:
  bronze is only ever used for a case with zero diagnosis rows.
  **Exception:** an ``unidentified_cancer`` bronze case (the case-presence
  gate passed but no label resolved) gets **no** combined row either — it is
  not a confident non-cancer call, so it must not take ``NO_CANCER``; it is
  always low-confidence (score 0.0), so it lands in the review queue instead
  (see ``_bronze_case_codes``).

A non-cancer result (silver's ``no_signal``/declined-LLM rows, or a bronze
row rejected by the case-presence gate) is written as one ``NO_CANCER`` row —
the same sentinel ``manual_audit.gold.NO_CANCER`` uses, so there is exactly
one sentinel across the codebase.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
import io_utils
from coding.rule import case_is_vague
from diagnosis_mapping.silver import load_silver
from manual_audit.gold import NO_CANCER, load_gold

COMBINED_PREDICTIONS_COLUMNS = [
    "case_id", "code", "term", "group", "code_source",
    "source_version", "source_confidence", "review_status",
]

# ---------------------------------------------------------------------------
# The bronze review gate. Default = the backend's *current* bronze review
# gate (backend/app/config.py REVIEW_AUTO_ACCEPT_CONFIDENCE/MARGIN, applied in
# backend/app/services/ingestion_service.py:512-526): a row is queued when its
# method is the pipeline's low-confidence rejection, OR its confidence is
# below the threshold, OR (rank-1 only, and only when both ranks exist) the
# top1-top2 margin is too tight. All three numbers live here together so they
# can be recalibrated in one place once the random slice (icd-mapping-
# strategy.md "Improving the methods") gives a real basis — they are backend
# defaults carried over, not yet validated against this pipeline's own data.
# ---------------------------------------------------------------------------
BRONZE_LOW_CONFIDENCE_METHOD = "rejected_by_case_presence"  # this pipeline's name for the backend's "low_confidence"
BRONZE_LOW_CONFIDENCE_THRESHOLD = 0.23
BRONZE_LOW_MARGIN_THRESHOLD = 0.15


def load_bronze_predictions(predictions_csv: str | Path) -> pd.DataFrame:
    return io_utils.read_csv(predictions_csv, encoding="utf-8", dtype=str, keep_default_na=False)


def resolve_generation_id(bronze: pd.DataFrame, generation_id: str | None = None) -> str:
    """The bronze generation_id for this run: the explicit override if given,
    else the predictions file's own (single) ``generation_id`` column value."""
    if generation_id is not None:
        return generation_id
    ids = bronze["generation_id"].unique()
    if len(ids) != 1:
        raise ValueError(f"predictions carry {len(ids)} distinct generation_id value(s); pass --generation-id")
    return ids[0]


def _float_or_zero(value: str) -> float:
    return float(value) if value else 0.0


def bronze_case_is_low_confidence(case_rows: pd.DataFrame) -> bool:
    """Same gate as the backend's ingestion_service, applied to one case's
    bronze prediction rows (``diagnosis_index`` 1, 2, ...).

    The backend computes ``needs_review`` per row (``ingestion_service.py``:
    "method == low_confidence OR conf < threshold OR margin_too_tight"), and
    the margin term is only ever attached to the rank-1 row. Mirrored here at
    case granularity: the method/confidence checks run over EVERY row (a
    low-confidence row at rank 2+ must still queue the case), while the
    margin check stays rank-1-only, since a margin between ranks below 1 was
    never computed by the backend in the first place.
    """
    rank1 = case_rows[case_rows["diagnosis_index"] == "1"]
    if rank1.empty:
        return True  # no rank-1 row is unexpected for a case bronze ran on; queue defensively
    for row in case_rows.to_dict("records"):
        if row["method"] == BRONZE_LOW_CONFIDENCE_METHOD:
            return True
        if _float_or_zero(row["confidence"]) < BRONZE_LOW_CONFIDENCE_THRESHOLD:
            return True
    top1_conf = _float_or_zero(rank1.iloc[0]["confidence"])
    rank2 = case_rows[case_rows["diagnosis_index"] == "2"]
    if not rank2.empty:
        top2_conf = _float_or_zero(rank2.iloc[0]["confidence"])
        # Mirrors the backend: a margin is only meaningful when both ranks
        # actually carry a nonzero confidence.
        if top1_conf and top2_conf and (top1_conf - top2_conf) < BRONZE_LOW_MARGIN_THRESHOLD:
            return True
    return False


def _dedup_code_rows(rows: list[dict]) -> list[dict]:
    """Distinct (code) rows, first occurrence wins; order preserved."""
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["code"] in seen:
            continue
        seen.add(row["code"])
        out.append(row)
    return out


def _gold_source_version(origin: str, batch_or_export_id: str) -> str:
    """``origin`` (plus ``batch_or_export_id`` when there is one) identifies exactly
    which gold ingest produced a row. ``gold_snapshot_hash`` (a hash of gold-train
    only) can't serve this: an eval_batch/random_slice row is never in gold-train,
    so stamping it with that hash would silently misattribute its provenance."""
    return f"{origin}:{batch_or_export_id}" if batch_or_export_id else origin


def _gold_case_codes(case_id: str, rows: pd.DataFrame) -> list[dict]:
    return [{
        "case_id": case_id, "code": r["code"], "term": r["term"], "group": r["group"],
        "code_source": "manual",
        "source_version": _gold_source_version(r["origin"], r["batch_or_export_id"]),
        "source_confidence": "", "review_status": "confirmed",
    } for r in rows.to_dict("records")]


def _silver_case_codes(case_id: str, rows: pd.DataFrame, silver_id: str) -> list[dict]:
    records = rows.to_dict("records")
    cancer_rows = [
        {"case_id": case_id, "code": r["matched_code"], "term": r["matched_term"], "group": r["matched_group"],
         "code_source": "diagnosis", "source_version": silver_id, "source_confidence": r["decision_stage"],
         "review_status": "auto_accepted"}
        for r in records if r["matched_code"]
    ]
    if cancer_rows:
        return _dedup_code_rows(cancer_rows)
    first = records[0]
    return [{
        "case_id": case_id, "code": NO_CANCER, "term": "", "group": "",
        "code_source": "diagnosis", "source_version": silver_id, "source_confidence": first["decision_stage"],
        "review_status": "auto_accepted",
    }]


UNIDENTIFIED_CANCER_METHOD = "unidentified_cancer"  # gate passed (likely cancer), but no label resolved


def _bronze_case_codes(case_id: str, rows: pd.DataFrame, generation_id: str) -> list[dict] | None:
    """Adopted rows for one bronze-only case, or ``None`` if it must not be
    coded at all (an ``unidentified_cancer`` case — see below)."""
    # diagnosis_index is read as str (dtype=str throughout); sorting it as a
    # string would put rank 10 before rank 2, so sort by its int value.
    records = rows.sort_values("diagnosis_index", key=lambda s: s.astype(int)).to_dict("records")
    cancer_rows = [
        {"case_id": case_id, "code": r["predicted_code"], "term": r["predicted_term"], "group": r["predicted_group"],
         "code_source": "report", "source_version": generation_id, "source_confidence": r["confidence"],
         "review_status": ""}
        for r in records if r["predicted_code"]
    ]
    if cancer_rows:
        combined = _dedup_code_rows(cancer_rows)
    else:
        first = records[0]
        if first["method"] == UNIDENTIFIED_CANCER_METHOD:
            # The case-presence gate passed (the case likely has cancer) but no
            # label was ever resolved — this is not a confident non-cancer
            # call, so it must not take the NO_CANCER sentinel. Its score is
            # always 0.0 (report_mapping.inference.stages), so
            # bronze_case_is_low_confidence already queues it (coding.queue);
            # here it simply gets no combined row at all, like a vague silver case.
            return None
        combined = [{
            "case_id": case_id, "code": NO_CANCER, "term": "", "group": "",
            "code_source": "report", "source_version": generation_id, "source_confidence": first["confidence"],
            "review_status": "",
        }]
    status = "queued" if bronze_case_is_low_confidence(rows) else "auto_accepted"
    for row in combined:
        row["review_status"] = status
    return combined


def combine_predictions(
    silver_id: str,
    split_id: str,
    predictions_csv: str | Path,
    *,
    generation_id: str | None = None,
) -> pd.DataFrame:
    """Build the combined-predictions table for every case in silver ∪ bronze ∪ gold.

    Gold beats silver beats bronze; a vague silver case contributes no row
    here at all (see the module docstring). ``split_id`` is accepted (and
    required) for CLI/API symmetry with ``coding.corrected``/``coding.queue``,
    which both need a split to resolve partitions; ``combine_predictions`` itself
    doesn't currently need one (gold rows are now provenance-stamped from
    their own origin, not a split-scoped gold snapshot — see
    ``_gold_source_version``).
    """
    silver = load_silver(silver_id)
    gold = load_gold()
    bronze = load_bronze_predictions(predictions_csv)

    gold_by_case = {cid: g for cid, g in gold.groupby("case_id")}
    silver_by_case = {cid: g for cid, g in silver.groupby("case_id")}
    bronze_by_case = {cid: g for cid, g in bronze.groupby("case_id")}

    # Resolved lazily (only if a case actually needs bronze coding), so a
    # run with no bronze-only cases never has to make sense of predictions_csv's
    # generation_id column at all.
    bronze_generation_id: str | None = None

    all_cases = sorted(set(gold_by_case) | set(silver_by_case) | set(bronze_by_case))
    rows: list[dict] = []
    for case_id in all_cases:
        if case_id in gold_by_case:
            rows.extend(_gold_case_codes(case_id, gold_by_case[case_id]))
        elif case_id in silver_by_case:
            case_rows = silver_by_case[case_id]
            if case_is_vague(case_rows):
                continue  # queued, not coded (coding.queue)
            rows.extend(_silver_case_codes(case_id, case_rows, silver_id))
        elif case_id in bronze_by_case:
            if bronze_generation_id is None:
                bronze_generation_id = resolve_generation_id(bronze, generation_id)
            bronze_rows = _bronze_case_codes(case_id, bronze_by_case[case_id], bronze_generation_id)
            if bronze_rows is not None:
                rows.extend(bronze_rows)
            # else: an unidentified_cancer bronze-only case — no combined row;
            # it is queued instead (coding.queue).

    return pd.DataFrame(rows, columns=COMBINED_PREDICTIONS_COLUMNS)


def write_combined_predictions(
    silver_id: str,
    split_id: str,
    predictions_csv: str | Path,
    *,
    generation_id: str | None = None,
    out_csv: str | Path | None = None,
) -> Path:
    out_path = Path(out_csv) if out_csv is not None else config.COMBINED_PREDICTIONS_CSV
    df = combine_predictions(silver_id, split_id, predictions_csv, generation_id=generation_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, out_path)
    return out_path
