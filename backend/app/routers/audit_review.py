"""The dashboard review worklist (audit list) and gold export.

Pairs with database/migrations/033_gold_review.sql and
ml/documentation/audit-list-change-request.md. Import and worklist are
implemented; review-screen/export endpoints land in later commits.

Endpoints:
  POST /api/v1/audit-review/lists/import   - import audit_list_<id>.txt + its
                                              .manifest.json sidecar, replacing
                                              the active worklist
  GET  /api/v1/audit-review/worklist       - the active list's cases in
                                              order, with review status
  GET  /api/v1/audit-review/taxonomy-terms - the code picker's source data
  GET  /api/v1/audit-review/cases/{case_id} - one case's full record: report
                                              text, predicted codes, and its
                                              existing gold review if any
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, require_admin, require_reviewer
from app.config import settings
from app.database import get_db
from app.models.models import (
    AuditList,
    AuditListCase,
    CancerType,
    CaseDiagnosis,
    CaseReview,
    CaseReviewCode,
    Patient,
    PathologyReport,
    TaxonomyTerm,
)
from app.rate_limit import limiter
from app.services.ingestion_service import normalize_anon_id

router = APIRouter(prefix="/api/v1/audit-review", tags=["audit-review"])

# Matches ml/handoff/contracts.py's AUDIT_LIST_EXPORT_KIND / SCHEMA_VERSION —
# duplicated here since production has no /ml to import them from. Bump the
# version only in lockstep with a matching change on ML's side.
_AUDIT_LIST_KIND = "audit_list"
_AUDIT_LIST_SCHEMA_VERSION = 1
_FILENAME_PREFIX = "audit_list_"
_FILENAME_SUFFIX = ".txt"


class AuditListImportSummary(BaseModel):
    list_id: str
    case_count: int
    replaced_list_id: str | None
    not_found: list[str]


def _parse_list_id(filename: str) -> str:
    name = (filename or "").strip()
    if not (name.startswith(_FILENAME_PREFIX) and name.endswith(_FILENAME_SUFFIX)):
        raise HTTPException(
            status_code=400,
            detail=f"File name must look like {_FILENAME_PREFIX}<list_id>{_FILENAME_SUFFIX}, got {filename!r}",
        )
    list_id = name[len(_FILENAME_PREFIX):-len(_FILENAME_SUFFIX)]
    if not list_id:
        raise HTTPException(status_code=400, detail="File name is missing a list_id")
    return list_id


def _parse_case_ids(list_bytes: bytes) -> list[str]:
    try:
        text = list_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Audit list must be UTF-8 encoded")

    case_ids = [line.strip() for line in text.split("\n") if line.strip()]
    if not case_ids:
        raise HTTPException(status_code=400, detail="Audit list has no case IDs")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"Audit list repeats case_id(s): {', '.join(sorted(duplicates)[:20])}",
        )
    return case_ids


@router.post("/lists/import")
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def import_audit_list(
    request: Request,
    list_file: UploadFile = File(...),
    manifest_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_admin),
) -> AuditListImportSummary:
    """Admin-only. Loads audit_list_<list_id>.txt as the specialist's worklist,
    replacing whichever list is currently active. Every list is the complete
    current worklist, not an increment — a case that drops off a newer list
    simply stops appearing, though its audit_list_cases row (and any review)
    is kept: a case is still safe to export as long as it was ever on some
    list, since ML refuses the whole gold file for a case_id that never was."""
    list_id = _parse_list_id(list_file.filename or "")

    list_bytes = await list_file.read()
    if not list_bytes:
        raise HTTPException(status_code=400, detail="Audit list file is empty")

    manifest_bytes = await manifest_file.read()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Manifest sidecar is not valid JSON")

    if manifest.get("kind") != _AUDIT_LIST_KIND:
        raise HTTPException(
            status_code=400,
            detail=f"Manifest kind {manifest.get('kind')!r} != {_AUDIT_LIST_KIND!r}",
        )
    if manifest.get("schema_version") != _AUDIT_LIST_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Manifest schema_version {manifest.get('schema_version')!r} != "
                f"{_AUDIT_LIST_SCHEMA_VERSION!r} (this backend doesn't know that shape)"
            ),
        )
    expected_sha256 = manifest.get("sha256")
    if not expected_sha256:
        raise HTTPException(status_code=400, detail="Manifest is missing sha256")
    actual_sha256 = hashlib.sha256(list_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HTTPException(
            status_code=400,
            detail=f"sha256 {actual_sha256} does not match the manifest's recorded {expected_sha256}",
        )

    case_ids = _parse_case_ids(list_bytes)

    existing = (
        await db.execute(select(AuditList.id).where(AuditList.list_id == list_id))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"list_id {list_id!r} was already imported")

    previous_active = (
        await db.execute(select(AuditList.list_id).where(AuditList.is_active.is_(True)))
    ).scalar_one_or_none()

    await db.execute(update(AuditList).where(AuditList.is_active.is_(True)).values(is_active=False))

    new_list = AuditList(
        list_id=list_id,
        imported_by_email=user.email,
        sha256=actual_sha256,
        case_count=len(case_ids),
        is_active=True,
    )
    db.add(new_list)
    await db.flush()

    db.add_all(
        AuditListCase(audit_list_id=new_list.id, case_id=case_id, position=position)
        for position, case_id in enumerate(case_ids)
    )

    # Informational only — an unmatched case still gets imported (shown in the
    # worklist flagged "no record found"), never dropped from the list.
    normalized_ids = {normalize_anon_id(case_id) for case_id in case_ids}
    normalized_ids.discard("")
    found = set(
        (await db.execute(select(Patient.anon_id).where(Patient.anon_id.in_(normalized_ids)))).scalars()
    )
    not_found = [case_id for case_id in case_ids if normalize_anon_id(case_id) not in found]

    await db.commit()

    return AuditListImportSummary(
        list_id=list_id,
        case_count=len(case_ids),
        replaced_list_id=previous_active,
        not_found=not_found,
    )


# --- Worklist ----------------------------------------------------------


class WorklistCase(BaseModel):
    case_id: str
    position: int
    patient_found: bool
    review_status: Literal["unreviewed", "reviewed", "locked"]
    no_cancer: Optional[bool]
    code_count: int
    reviewed_by_email: Optional[str]
    reviewed_at: Optional[datetime]


class WorklistResponse(BaseModel):
    list_id: Optional[str]
    imported_at: Optional[datetime]
    case_count: int
    cases: list[WorklistCase]


@router.get("/worklist")
async def get_worklist(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
) -> WorklistResponse:
    """The active audit list's cases in review order, each flagged with
    whether a matching patient exists and its review status. No active list
    (never imported, or the DB is otherwise empty) returns an empty worklist,
    not an error."""
    active_list = (
        await db.execute(select(AuditList).where(AuditList.is_active.is_(True)))
    ).scalar_one_or_none()
    if active_list is None:
        return WorklistResponse(list_id=None, imported_at=None, case_count=0, cases=[])

    entries = (
        await db.execute(
            select(AuditListCase)
            .where(AuditListCase.audit_list_id == active_list.id)
            .order_by(AuditListCase.position)
        )
    ).scalars().all()
    case_ids = [entry.case_id for entry in entries]

    reviews: dict[str, CaseReview] = {}
    if case_ids:
        rows = (
            await db.execute(select(CaseReview).where(CaseReview.case_id.in_(case_ids)))
        ).scalars().all()
        reviews = {r.case_id: r for r in rows}

    code_counts: dict[int, int] = {}
    review_ids = [r.id for r in reviews.values()]
    if review_ids:
        count_rows = (
            await db.execute(
                select(CaseReviewCode.case_review_id, func.count(CaseReviewCode.id))
                .where(CaseReviewCode.case_review_id.in_(review_ids))
                .group_by(CaseReviewCode.case_review_id)
            )
        ).all()
        code_counts = dict(count_rows)

    normalized_ids = {normalize_anon_id(case_id) for case_id in case_ids}
    normalized_ids.discard("")
    found_patients: set[str] = set()
    if normalized_ids:
        found_patients = set(
            (await db.execute(select(Patient.anon_id).where(Patient.anon_id.in_(normalized_ids)))).scalars()
        )

    cases = []
    for entry in entries:
        review = reviews.get(entry.case_id)
        if review is None:
            status: Literal["unreviewed", "reviewed", "locked"] = "unreviewed"
        elif review.locked:
            status = "locked"
        else:
            status = "reviewed"
        cases.append(WorklistCase(
            case_id=entry.case_id,
            position=entry.position,
            patient_found=normalize_anon_id(entry.case_id) in found_patients,
            review_status=status,
            no_cancer=review.no_cancer if review else None,
            code_count=code_counts.get(review.id, 0) if review else 0,
            reviewed_by_email=review.reviewed_by_email if review else None,
            reviewed_at=review.reviewed_at if review else None,
        ))

    return WorklistResponse(
        list_id=active_list.list_id,
        imported_at=active_list.imported_at,
        case_count=active_list.case_count,
        cases=cases,
    )


# --- Taxonomy ------------------------------------------------------------


class TaxonomyTermOut(BaseModel):
    vet_icd_o_code: Optional[str]
    taxonomy_group: str
    taxonomy_term: str


@router.get("/taxonomy-terms")
async def list_taxonomy_terms(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
) -> list[TaxonomyTermOut]:
    """The Vet-ICD-O-Canine-1 (group, term) pairs the review screen's code
    picker is built from — seeded once via database/seed/seed_taxonomy_terms.py,
    never read live from ml/taxonomy/labels.csv (production has no /ml)."""
    rows = (
        await db.execute(
            select(TaxonomyTerm).order_by(TaxonomyTerm.taxonomy_group, TaxonomyTerm.taxonomy_term)
        )
    ).scalars().all()
    return [
        TaxonomyTermOut(
            vet_icd_o_code=r.vet_icd_o_code,
            taxonomy_group=r.taxonomy_group,
            taxonomy_term=r.taxonomy_term,
        )
        for r in rows
    ]


# --- Case detail -----------------------------------------------------------


class PredictedCode(BaseModel):
    diagnosis_index: Optional[int]
    cancer_type_name: str
    icd_o_code: Optional[str]
    predicted_term: Optional[str]
    confidence: Optional[float]
    prediction_method: Optional[str]


class ExistingReviewCode(BaseModel):
    taxonomy_group: str
    taxonomy_term: str


class CaseDetail(BaseModel):
    case_id: str
    patient_found: bool
    patient_anon_id: Optional[str]
    # The clinic's short "Clinical Diagnoses" text, from pathology_reports.
    source_diagnosis: Optional[str]
    # The full pathology report, fetched from GCS. None if unavailable
    # (GCS not configured, or the fetch failed) — never an error.
    report_text: Optional[str]
    predicted_codes: list[PredictedCode]
    # The case's existing gold review, if a specialist has already recorded
    # one — lets the review screen pre-fill (editable) or show it read-only
    # (locked).
    review_exists: bool
    review_no_cancer: Optional[bool]
    review_codes: list[ExistingReviewCode]
    review_locked: Optional[bool]
    reviewed_by_email: Optional[str]
    reviewed_at: Optional[datetime]


async def _fetch_report_text(report: PathologyReport) -> Optional[str]:
    """Fetch a pathology report's full text from GCS. Mirrors
    diagnoses_review._fetch_report_text, adapted to take the report
    directly rather than through a CaseDiagnosis's relationship — never
    raises; a missing bucket/path/fetch failure just means no text."""
    if not report.gcs_path or not settings.GCS_BUCKET:
        return None
    try:
        from app.services.gcp_batch_service import download_report_text_from_gcs
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, download_report_text_from_gcs, report.gcs_path)
    except Exception:
        return None


@router.get("/cases/{case_id}")
async def get_case_detail(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
) -> CaseDetail:
    """One case's full record for the review screen: the clinical diagnosis
    text, the full pathology report, every current predicted code (seeing
    predictions is fine per the change request), and the case's existing
    gold review if one has already been recorded. Scoped to case IDs that
    have appeared on some audit list — this isn't a general patient lookup."""
    on_a_list = (
        await db.execute(select(AuditListCase.id).where(AuditListCase.case_id == case_id).limit(1))
    ).scalar_one_or_none()
    if on_a_list is None:
        raise HTTPException(status_code=404, detail=f"case_id {case_id!r} was never on an audit list")

    normalized = normalize_anon_id(case_id)
    patient = None
    if normalized:
        patient = (
            await db.execute(select(Patient).where(Patient.anon_id == normalized))
        ).scalar_one_or_none()

    source_diagnosis: Optional[str] = None
    report_text: Optional[str] = None
    predicted_codes: list[PredictedCode] = []
    if patient is not None:
        report = (
            await db.execute(select(PathologyReport).where(PathologyReport.patient_id == patient.id))
        ).scalar_one_or_none()
        if report is not None:
            source_diagnosis = report.source_diagnosis
            report_text = await _fetch_report_text(report)

        diag_rows = (
            await db.execute(
                select(CaseDiagnosis, CancerType.name)
                .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
                .where(CaseDiagnosis.patient_id == patient.id)
                .order_by(CaseDiagnosis.diagnosis_index)
            )
        ).all()
        predicted_codes = [
            PredictedCode(
                diagnosis_index=diag.diagnosis_index,
                cancer_type_name=name,
                icd_o_code=diag.icd_o_code,
                predicted_term=diag.predicted_term,
                confidence=float(diag.confidence) if diag.confidence is not None else None,
                prediction_method=diag.prediction_method,
            )
            for diag, name in diag_rows
        ]

    review = (
        await db.execute(select(CaseReview).where(CaseReview.case_id == case_id))
    ).scalar_one_or_none()
    review_codes: list[ExistingReviewCode] = []
    if review is not None:
        code_rows = (
            await db.execute(select(CaseReviewCode).where(CaseReviewCode.case_review_id == review.id))
        ).scalars().all()
        review_codes = [
            ExistingReviewCode(taxonomy_group=c.taxonomy_group, taxonomy_term=c.taxonomy_term)
            for c in code_rows
        ]

    return CaseDetail(
        case_id=case_id,
        patient_found=patient is not None,
        patient_anon_id=patient.anon_id if patient else None,
        source_diagnosis=source_diagnosis,
        report_text=report_text,
        predicted_codes=predicted_codes,
        review_exists=review is not None,
        review_no_cancer=review.no_cancer if review else None,
        review_codes=review_codes,
        review_locked=review.locked if review else None,
        reviewed_by_email=review.reviewed_by_email if review else None,
        reviewed_at=review.reviewed_at if review else None,
    )
