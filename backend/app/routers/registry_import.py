"""Loads ML's combined-predictions and review-queue files as the registry's
code of record.

Pairs with database/migrations/035_combined_predictions.sql and
ml/documentation/audit-list-change-request.md, section 1 ("Combined
predictions: the registry's code of record"). Every dashboard/incidence/
trends/geo/export query already filters on review_status IN ('confirmed',
'corrected') and excludes the Non-Cancer cancer type (backend/app/services/
review_filter.py) — mapping ML's review_status onto ours (confirmed/
auto_accepted -> confirmed, queued -> pending) and never creating a
case_diagnoses row for a NO_CANCER case means none of those six consumers
need any changes.

Endpoints:
  POST /api/v1/registry/combined-predictions/import
"""

import csv
import hashlib
import io
import json
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, require_admin
from app.cache import clear_all_caches
from app.config import settings
from app.database import get_db
from app.models.models import CancerType, CaseDiagnosis, Patient, PathologyReport
from app.rate_limit import limiter
from app.services.ingestion_service import normalize_anon_id

router = APIRouter(prefix="/api/v1/registry", tags=["registry-import"])

# Matches ml/handoff/contracts.py's kind/schema_version constants —
# duplicated here since production has no /ml to import them from.
_COMBINED_PREDICTIONS_KIND = "combined_predictions"
_COMBINED_PREDICTIONS_SCHEMA_VERSION = 2
_REVIEW_QUEUE_KIND = "review_queue"
_REVIEW_QUEUE_SCHEMA_VERSION = 1

_CODE_SOURCES = frozenset({"manual", "diagnosis", "report"})
_ML_REVIEW_STATUSES = frozenset({"confirmed", "auto_accepted", "queued"})
_NO_CANCER = "NO_CANCER"

_COMBINED_PREDICTIONS_REQUIRED_COLUMNS = frozenset({
    "case_id", "code", "term", "group", "code_source",
    "source_version", "source_confidence", "review_status", "n_codes",
})


class CombinedPredictionsImportSummary(BaseModel):
    cases_coded: int
    cases_no_cancer: int
    cases_awaiting_review: int
    not_found: list[str]


def _verify_manifest(
    raw_bytes: bytes, manifest_bytes: bytes, *, expected_kind: str, expected_schema_version: int, label: str,
) -> None:
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"{label} manifest is not valid JSON")
    if manifest.get("kind") != expected_kind:
        raise HTTPException(status_code=400, detail=f"{label} manifest kind {manifest.get('kind')!r} != {expected_kind!r}")
    if manifest.get("schema_version") != expected_schema_version:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} manifest schema_version {manifest.get('schema_version')!r} != "
                f"{expected_schema_version!r} (this backend doesn't know that shape)"
            ),
        )
    expected_sha256 = manifest.get("sha256")
    if not expected_sha256:
        raise HTTPException(status_code=400, detail=f"{label} manifest is missing sha256")
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: sha256 {actual_sha256} does not match its manifest ({expected_sha256})",
        )


def _parse_combined_predictions(raw_bytes: bytes) -> dict[str, list[dict]]:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="combined_predictions file must be UTF-8 encoded")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = set(reader.fieldnames or [])
    missing_cols = _COMBINED_PREDICTIONS_REQUIRED_COLUMNS - fieldnames
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"combined_predictions missing column(s): {', '.join(sorted(missing_cols))}")

    by_case: dict[str, list[dict]] = {}
    for row in reader:
        case_id = (row.get("case_id") or "").strip()
        if not case_id:
            continue
        by_case.setdefault(case_id, []).append(row)
    if not by_case:
        raise HTTPException(status_code=400, detail="combined_predictions has no rows")

    errors: list[str] = []
    for case_id, rows in by_case.items():
        n_codes_values = {(row.get("n_codes") or "").strip() for row in rows}
        if n_codes_values != {str(len(rows))}:
            errors.append(f"{case_id}: n_codes mismatch (file says {sorted(n_codes_values)}, got {len(rows)} row(s))")
        is_no_cancer = any((row.get("code") or "").strip() == _NO_CANCER for row in rows)
        if is_no_cancer and len(rows) != 1:
            errors.append(f"{case_id}: NO_CANCER mixed with other codes")
        for row in rows:
            code_source = (row.get("code_source") or "").strip()
            if code_source not in _CODE_SOURCES:
                errors.append(f"{case_id}: invalid code_source {code_source!r}")
            review_status = (row.get("review_status") or "").strip()
            if review_status not in _ML_REVIEW_STATUSES:
                errors.append(f"{case_id}: invalid review_status {review_status!r}")
    if errors:
        shown = errors[:20]
        more = f" (+{len(errors) - 20} more)" if len(errors) > 20 else ""
        raise HTTPException(status_code=400, detail=f"combined_predictions validation failed: {'; '.join(shown)}{more}")

    return by_case


def _parse_review_queue_case_ids(raw_bytes: bytes) -> set[str]:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="review_queue file must be UTF-8 encoded")
    reader = csv.DictReader(io.StringIO(text))
    if "case_id" not in (reader.fieldnames or []):
        raise HTTPException(status_code=400, detail="review_queue missing case_id column")
    ids = {(row.get("case_id") or "").strip() for row in reader}
    ids.discard("")
    return ids


def _ml_review_status_to_ours(ml_status: str) -> str:
    return "confirmed" if ml_status in ("confirmed", "auto_accepted") else "pending"


@router.post("/combined-predictions/import")
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def import_combined_predictions(
    request: Request,
    predictions_file: UploadFile = File(...),
    predictions_manifest_file: UploadFile = File(...),
    queue_file: UploadFile = File(...),
    queue_manifest_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_admin),
) -> CombinedPredictionsImportSummary:
    """Admin-only. Loads combined_predictions_<run>.csv as the registry's code
    of record and review_queue_<run>.csv to flag cases awaiting review (a case
    in neither file hasn't reached ML yet and is left untouched). Verifies both
    files against their .manifest.json sidecars first.

    For every case in combined_predictions, deletes and replaces that
    patient's case_diagnoses rows outright — this does cascade-delete any
    DiagnosisReviewEvent history for that case. That's a deliberate, accepted
    tradeoff now that the Review Queue is retired, not an oversight."""
    predictions_bytes = await predictions_file.read()
    predictions_manifest_bytes = await predictions_manifest_file.read()
    _verify_manifest(
        predictions_bytes, predictions_manifest_bytes,
        expected_kind=_COMBINED_PREDICTIONS_KIND, expected_schema_version=_COMBINED_PREDICTIONS_SCHEMA_VERSION,
        label="combined_predictions",
    )

    queue_bytes = await queue_file.read()
    queue_manifest_bytes = await queue_manifest_file.read()
    _verify_manifest(
        queue_bytes, queue_manifest_bytes,
        expected_kind=_REVIEW_QUEUE_KIND, expected_schema_version=_REVIEW_QUEUE_SCHEMA_VERSION,
        label="review_queue",
    )

    by_case = _parse_combined_predictions(predictions_bytes)
    queue_case_ids = _parse_review_queue_case_ids(queue_bytes)

    # Resolve every case_id (from both files) to a patient, in one query.
    all_raw_ids = set(by_case.keys()) | queue_case_ids
    normalized_to_raw: dict[str, list[str]] = {}
    for raw_id in all_raw_ids:
        normalized = normalize_anon_id(raw_id)
        if normalized:
            normalized_to_raw.setdefault(normalized, []).append(raw_id)

    patient_rows = (
        await db.execute(select(Patient.id, Patient.anon_id).where(Patient.anon_id.in_(normalized_to_raw.keys())))
    ).all() if normalized_to_raw else []
    patient_id_by_normalized = {row.anon_id: row.id for row in patient_rows}

    not_found = sorted({
        raw_id
        for normalized, raw_ids in normalized_to_raw.items()
        if normalized not in patient_id_by_normalized
        for raw_id in raw_ids
    })

    # Pre-load the full CancerType map (small table), create any new groups.
    cancer_type_rows = (await db.execute(select(CancerType.id, CancerType.name))).all()
    cancer_type_map = {row.name: row.id for row in cancer_type_rows}
    needed_groups = {
        (row.get("group") or "").strip()
        for rows in by_case.values() for row in rows
        if (row.get("code") or "").strip() != _NO_CANCER and (row.get("group") or "").strip()
    }
    new_groups = needed_groups - set(cancer_type_map.keys())
    if new_groups:
        await db.execute(
            pg_insert(CancerType.__table__)
            .values([{"name": g} for g in new_groups])
            .on_conflict_do_nothing(index_elements=["name"])
        )
        cancer_type_rows = (await db.execute(select(CancerType.id, CancerType.name))).all()
        cancer_type_map = {row.name: row.id for row in cancer_type_rows}

    resolved_patient_ids = list(patient_id_by_normalized.values())
    report_rows = (
        await db.execute(select(PathologyReport.patient_id, PathologyReport.id).where(PathologyReport.patient_id.in_(resolved_patient_ids)))
    ).all() if resolved_patient_ids else []
    report_id_by_patient = {row.patient_id: row.id for row in report_rows}

    coded_patient_ids: list[int] = []
    no_cancer_patient_ids: list[int] = []
    no_cancer_source_version_by_patient: dict[int, Optional[str]] = {}
    new_rows: list[dict] = []

    for case_id, rows in by_case.items():
        normalized = normalize_anon_id(case_id)
        patient_id = patient_id_by_normalized.get(normalized) if normalized else None
        if patient_id is None:
            continue
        coded_patient_ids.append(patient_id)
        first_code = (rows[0].get("code") or "").strip()
        if first_code == _NO_CANCER:
            no_cancer_patient_ids.append(patient_id)
            no_cancer_source_version_by_patient[patient_id] = (rows[0].get("source_version") or "").strip() or None
            continue
        report_id = report_id_by_patient.get(patient_id)
        for i, row in enumerate(rows, start=1):
            group = (row.get("group") or "").strip()
            ml_status = (row.get("review_status") or "").strip()
            new_rows.append({
                "patient_id": patient_id,
                "cancer_type_id": cancer_type_map[group],
                "icd_o_code": (row.get("code") or "").strip() or None,
                "predicted_term": (row.get("term") or "").strip() or None,
                "pathology_report_id": report_id,
                "diagnosis_index": i,
                "source_version": (row.get("source_version") or "").strip() or None,
                "code_source": (row.get("code_source") or "").strip(),
                "source_confidence": (row.get("source_confidence") or "").strip() or None,
                "ml_review_status": ml_status,
                "review_status": _ml_review_status_to_ours(ml_status),
            })

    # Cases in the queue file but not combined_predictions: flag awaiting
    # review, leave any existing case_diagnoses/registry_no_cancer state as-is.
    awaiting_patient_ids: list[int] = []
    for case_id in queue_case_ids - set(by_case.keys()):
        normalized = normalize_anon_id(case_id)
        patient_id = patient_id_by_normalized.get(normalized) if normalized else None
        if patient_id is not None:
            awaiting_patient_ids.append(patient_id)

    if coded_patient_ids:
        await db.execute(delete(CaseDiagnosis).where(CaseDiagnosis.patient_id.in_(coded_patient_ids)))
    if new_rows:
        await db.execute(insert(CaseDiagnosis.__table__), new_rows)

    if no_cancer_patient_ids:
        by_source_version: dict[Optional[str], list[int]] = {}
        for pid in no_cancer_patient_ids:
            by_source_version.setdefault(no_cancer_source_version_by_patient[pid], []).append(pid)
        for source_version, pids in by_source_version.items():
            await db.execute(
                update(Patient)
                .where(Patient.id.in_(pids))
                .values(registry_no_cancer=True, registry_no_cancer_source_version=source_version, registry_awaiting_review=False)
            )
    coded_non_no_cancer = [pid for pid in coded_patient_ids if pid not in no_cancer_patient_ids]
    if coded_non_no_cancer:
        await db.execute(
            update(Patient)
            .where(Patient.id.in_(coded_non_no_cancer))
            .values(registry_no_cancer=False, registry_no_cancer_source_version=None, registry_awaiting_review=False)
        )
    if awaiting_patient_ids:
        await db.execute(
            update(Patient).where(Patient.id.in_(awaiting_patient_ids)).values(registry_awaiting_review=True)
        )

    if coded_patient_ids or awaiting_patient_ids:
        # Mirrors app.services.ingestion_service's post-write refresh — the
        # dashboards read these views, not the tables directly.
        for view in ("mv_county_cancer_incidence", "mv_yearly_trends"):
            await db.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))

    await db.commit()
    clear_all_caches()

    return CombinedPredictionsImportSummary(
        cases_coded=len(coded_non_no_cancer),
        cases_no_cancer=len(no_cancer_patient_ids),
        cases_awaiting_review=len(awaiting_patient_ids),
        not_found=not_found,
    )
