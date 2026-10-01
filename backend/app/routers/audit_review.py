"""The dashboard review worklist (audit list) and gold export.

Pairs with database/migrations/033_gold_review.sql and
ml/documentation/audit-list-change-request.md.

Endpoints:
  POST /api/v1/audit-review/lists/import    - import audit_list_<id>.txt + its
                                               .manifest.json sidecar, replacing
                                               the active worklist
  GET  /api/v1/audit-review/worklist        - the active list's cases in
                                               order, with review status
  GET  /api/v1/audit-review/taxonomy-terms  - the code picker's source data
  GET  /api/v1/audit-review/cases/{case_id} - one case's full record: report
                                               text, predicted codes, and its
                                               existing gold review if any
  POST /api/v1/audit-review/cases/{case_id}/review - record/replace a case's
                                               complete gold review
  POST /api/v1/audit-review/cases/{case_id}/reopen - admin: unlock an
                                               exported review for editing
  POST /api/v1/audit-review/gold-exports    - admin: lock one reviewer's
                                               unlocked reviews into a batch
  GET  /api/v1/audit-review/gold-exports/{export_id} - re-download that
                                               batch's gold_<export_id>.csv
"""

import asyncio
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
    GoldExport,
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
    term_level: Optional[str]


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
            term_level=r.term_level,
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
    # Provenance from a combined_predictions load (database/migrations/
    # 035_combined_predictions.sql) — None for a code that predates that
    # pipeline. code_source: 'manual' (gold) / 'diagnosis' (silver) /
    # 'report' (bronze). source_confidence is free text (a decision-stage
    # name for silver, a numeric string for bronze). ml_review_status is
    # ML's own raw value (confirmed/auto_accepted/queued).
    code_source: Optional[str]
    source_confidence: Optional[str]
    ml_review_status: Optional[str]


class ExistingReviewCode(BaseModel):
    taxonomy_group: str
    taxonomy_term: str


class CaseDetail(BaseModel):
    case_id: str
    patient_found: bool
    patient_anon_id: Optional[str]
    # Demographics, so the review screen shows the full picture per
    # audit-list-change-request.md section 2 — not just the report.
    patient_species: Optional[str]
    patient_breed: Optional[str]
    patient_sex: Optional[str]
    # Calendar-year difference between diagnosis_date and birth_date — same
    # definition used for the age_group dimension in trends.py/incidence.py,
    # kept consistent rather than computing a calendar-exact age here.
    patient_age: Optional[int]
    # The clinic's short "Clinical Diagnoses" text, from pathology_reports.
    source_diagnosis: Optional[str]
    # The full pathology report, fetched from GCS. None if unavailable
    # (GCS not configured, or the fetch failed) — never an error.
    report_text: Optional[str]
    predicted_codes: list[PredictedCode]
    # True when a combined_predictions load coded this case NO_CANCER (case-
    # level, no code rows — see migration 035). Distinct from review_no_cancer
    # below, which is a specialist's own gold judgement.
    registry_no_cancer: bool
    # The case's existing gold review, if a specialist has already recorded
    # one — lets the review screen pre-fill (editable) or show it read-only
    # (locked). When no review exists yet, the frontend pre-fills from
    # predicted_codes/registry_no_cancer instead (approve-or-correct, per
    # ML: "pre-filling is fine... no bulk approve keeps that effect small").
    review_exists: bool
    review_no_cancer: Optional[bool]
    review_codes: list[ExistingReviewCode]
    review_locked: Optional[bool]
    reviewed_by_email: Optional[str]
    reviewed_at: Optional[datetime]


async def _fetch_report_text(report: PathologyReport) -> Optional[str]:
    """Fetch a pathology report's full text from GCS or S3, whichever backend
    wrote it — never raises; a missing bucket/path/fetch failure just means
    no text."""
    if not report.storage_path:
        return None
    try:
        loop = asyncio.get_running_loop()
        if settings.USE_ECS_ML:
            from app.services.s3_service import download_report_text
            return await loop.run_in_executor(None, download_report_text, report.storage_path)
        if settings.GCS_BUCKET:
            from app.services.gcp_batch_service import download_report_text_from_gcs
            return await loop.run_in_executor(None, download_report_text_from_gcs, report.storage_path)
        return None
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
            await db.execute(
                select(Patient)
                .options(selectinload(Patient.breed), selectinload(Patient.species))
                .where(Patient.anon_id == normalized)
            )
        ).scalar_one_or_none()

    source_diagnosis: Optional[str] = None
    report_text: Optional[str] = None
    predicted_codes: list[PredictedCode] = []
    patient_breed: Optional[str] = None
    patient_species: Optional[str] = None
    patient_age: Optional[int] = None
    if patient is not None:
        patient_breed = patient.breed.name if patient.breed else None
        patient_species = patient.species.name if patient.species else None
        if patient.birth_date and patient.diagnosis_date:
            patient_age = patient.diagnosis_date.year - patient.birth_date.year

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
                code_source=diag.code_source,
                source_confidence=diag.source_confidence,
                ml_review_status=diag.ml_review_status,
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
        patient_species=patient_species,
        patient_breed=patient_breed,
        patient_sex=patient.sex if patient else None,
        patient_age=patient_age,
        source_diagnosis=source_diagnosis,
        report_text=report_text,
        predicted_codes=predicted_codes,
        registry_no_cancer=bool(patient.registry_no_cancer) if patient else False,
        review_exists=review is not None,
        review_no_cancer=review.no_cancer if review else None,
        review_codes=review_codes,
        review_locked=review.locked if review else None,
        reviewed_by_email=review.reviewed_by_email if review else None,
        reviewed_at=review.reviewed_at if review else None,
    )


# --- Save review -----------------------------------------------------------


class ReviewCodeIn(BaseModel):
    taxonomy_group: str = Field(..., max_length=255)
    taxonomy_term: str = Field(..., max_length=255)


class ReviewSaveRequest(BaseModel):
    no_cancer: bool
    codes: list[ReviewCodeIn] = Field(default_factory=list)


class ReviewSaveResult(BaseModel):
    case_id: str
    no_cancer: bool
    code_count: int
    reviewed_by_email: str
    reviewed_at: datetime
    locked: bool


async def _invalid_taxonomy_pairs(db: AsyncSession, pairs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """The subset of (group, term) `pairs` not present in taxonomy_terms."""
    if not pairs:
        return set()
    groups = {g for g, _ in pairs}
    terms = {t for _, t in pairs}
    rows = (
        await db.execute(
            select(TaxonomyTerm.taxonomy_group, TaxonomyTerm.taxonomy_term)
            .where(TaxonomyTerm.taxonomy_group.in_(groups))
            .where(TaxonomyTerm.taxonomy_term.in_(terms))
        )
    ).all()
    valid = {(g, t) for g, t in rows}
    return pairs - valid


@router.post("/cases/{case_id}/review")
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def save_case_review(
    request: Request,
    case_id: str,
    payload: ReviewSaveRequest,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
) -> ReviewSaveResult:
    """Record or replace a case's complete gold review: either no_cancer or a
    complete code set, never both — each save fully replaces the prior code
    set, it doesn't patch it. Any reviewer/admin can edit an unlocked review;
    the last saver becomes reviewed_by_email. Refuses to edit a locked
    (exported, not yet reopened) review."""
    on_a_list = (
        await db.execute(select(AuditListCase.id).where(AuditListCase.case_id == case_id).limit(1))
    ).scalar_one_or_none()
    if on_a_list is None:
        raise HTTPException(status_code=404, detail=f"case_id {case_id!r} was never on an audit list")

    if payload.no_cancer:
        if payload.codes:
            raise HTTPException(status_code=400, detail="no_cancer=true cannot carry any codes")
    elif not payload.codes:
        raise HTTPException(status_code=400, detail="At least one code is required unless no_cancer=true")

    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for code in payload.codes:
        key = (code.taxonomy_group, code.taxonomy_term)
        if key in seen:
            duplicates.append(f"{code.taxonomy_group}: {code.taxonomy_term}")
        seen.add(key)
    if duplicates:
        raise HTTPException(status_code=400, detail=f"Duplicate code(s): {', '.join(duplicates)}")

    invalid = await _invalid_taxonomy_pairs(db, seen)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Not in the taxonomy: {', '.join(f'{g}: {t}' for g, t in sorted(invalid))}",
        )

    existing = (
        await db.execute(select(CaseReview).where(CaseReview.case_id == case_id))
    ).scalar_one_or_none()
    if existing is not None and existing.locked:
        raise HTTPException(status_code=409, detail="This review is locked (already exported) — reopen it first")

    now = datetime.now(timezone.utc)
    if existing is None:
        review = CaseReview(
            case_id=case_id, no_cancer=payload.no_cancer,
            reviewed_by_email=user.email, reviewed_at=now, locked=False,
        )
        db.add(review)
        await db.flush()
    else:
        review = existing
        review.no_cancer = payload.no_cancer
        review.reviewed_by_email = user.email
        review.reviewed_at = now
        await db.execute(delete(CaseReviewCode).where(CaseReviewCode.case_review_id == review.id))

    if not payload.no_cancer:
        db.add_all(
            CaseReviewCode(
                case_review_id=review.id,
                taxonomy_group=code.taxonomy_group,
                taxonomy_term=code.taxonomy_term,
            )
            for code in payload.codes
        )

    await db.commit()

    return ReviewSaveResult(
        case_id=case_id,
        no_cancer=review.no_cancer,
        code_count=0 if payload.no_cancer else len(payload.codes),
        reviewed_by_email=review.reviewed_by_email,
        reviewed_at=review.reviewed_at,
        locked=review.locked,
    )


# --- Reopen ------------------------------------------------------------


class ReopenResult(BaseModel):
    case_id: str
    locked: bool


@router.post("/cases/{case_id}/reopen")
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def reopen_case_review(
    request: Request,
    case_id: str,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_admin),
) -> ReopenResult:
    """Admin-only. Clears locked on an exported review so it can be edited
    again. gold_export_id/exported_at are left as-is — history of the last
    export, not cleared — the next save-and-export replaces ML's copy."""
    review = (
        await db.execute(select(CaseReview).where(CaseReview.case_id == case_id))
    ).scalar_one_or_none()
    if review is None:
        raise HTTPException(status_code=404, detail=f"No review recorded for case_id {case_id!r}")
    if not review.locked:
        raise HTTPException(status_code=400, detail="This review is not locked")

    review.locked = False
    await db.commit()

    return ReopenResult(case_id=case_id, locked=False)


# --- Gold export -------------------------------------------------------


class GoldExportRequest(BaseModel):
    reviewer_email: str = Field(..., max_length=255)


class GoldExportSummary(BaseModel):
    export_id: str
    reviewer_email: str
    case_count: int


def _build_gold_csv(reviews_with_codes: list[tuple[CaseReview, list[CaseReviewCode]]]) -> str:
    """case_id,term rows — one per code, or exactly one NO_CANCER row.
    Exact strings, standard quoting; deliberately not the existing
    admin-export's _safe_csv_value, whose formula-injection tab-prefix would
    corrupt a term and get the whole file refused by ML."""
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(["case_id", "term"])
    for review, codes in reviews_with_codes:
        if review.no_cancer:
            writer.writerow([review.case_id, "NO_CANCER"])
        else:
            for code in codes:
                writer.writerow([review.case_id, f"{code.taxonomy_group}: {code.taxonomy_term}"])
    return output.getvalue()


async def _next_export_id(db: AsyncSession) -> str:
    """<today's date>-<counter>, e.g. 2026-09-27-1, then -2, ..."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = (
        await db.execute(select(GoldExport.export_id).where(GoldExport.export_id.like(f"{today}-%")))
    ).scalars().all()
    max_n = 0
    for export_id in existing:
        suffix = export_id.rsplit("-", 1)[-1]
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return f"{today}-{max_n + 1}"


@router.post("/gold-exports")
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def create_gold_export(
    request: Request,
    payload: GoldExportRequest,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_admin),
) -> GoldExportSummary:
    """Admin-only. Locks one reviewer's every unlocked review (first-time
    exports and reopened-then-corrected ones alike) into one new gold export
    batch, under SELECT ... FOR UPDATE so a concurrent edit can't slip in
    mid-export. Re-validates every code against the taxonomy first — like ML,
    refuses the whole export for one bad row rather than exporting the rest,
    naming the case IDs."""
    eligible = (
        await db.execute(
            select(CaseReview)
            .where(CaseReview.reviewed_by_email == payload.reviewer_email)
            .where(CaseReview.locked.is_(False))
            .with_for_update()
        )
    ).scalars().all()
    if not eligible:
        raise HTTPException(
            status_code=400,
            detail=f"No unlocked reviews for {payload.reviewer_email!r} to export",
        )

    review_ids = [r.id for r in eligible]
    code_rows = (
        await db.execute(select(CaseReviewCode).where(CaseReviewCode.case_review_id.in_(review_ids)))
    ).scalars().all()
    codes_by_review: dict[int, list[CaseReviewCode]] = {}
    for code in code_rows:
        codes_by_review.setdefault(code.case_review_id, []).append(code)

    all_pairs = {(code.taxonomy_group, code.taxonomy_term) for code in code_rows}
    invalid_pairs = await _invalid_taxonomy_pairs(db, all_pairs)
    if invalid_pairs:
        bad_case_ids = sorted({
            review.case_id
            for review in eligible
            for code in codes_by_review.get(review.id, [])
            if (code.taxonomy_group, code.taxonomy_term) in invalid_pairs
        })
        raise HTTPException(
            status_code=400,
            detail=f"Code(s) no longer in the taxonomy — case_id(s): {', '.join(bad_case_ids)}",
        )

    export_id = await _next_export_id(db)
    export = GoldExport(export_id=export_id, reviewer_email=payload.reviewer_email, case_count=len(eligible))
    db.add(export)
    await db.flush()

    now = datetime.now(timezone.utc)
    for review in eligible:
        review.locked = True
        review.gold_export_id = export.id
        review.exported_at = now

    await db.commit()

    return GoldExportSummary(export_id=export_id, reviewer_email=payload.reviewer_email, case_count=len(eligible))


@router.get("/gold-exports/{export_id}")
async def download_gold_export(
    export_id: str,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_admin),
) -> StreamingResponse:
    """Admin-only. Re-download a past export batch's gold CSV, rebuilt from
    the case_reviews rows still carrying its gold_export_id — never from a
    file on disk, so it's always exactly what was recorded at export time."""
    export = (
        await db.execute(select(GoldExport).where(GoldExport.export_id == export_id))
    ).scalar_one_or_none()
    if export is None:
        raise HTTPException(status_code=404, detail=f"export_id {export_id!r} not found")

    reviews = (
        await db.execute(
            select(CaseReview).where(CaseReview.gold_export_id == export.id).order_by(CaseReview.case_id)
        )
    ).scalars().all()

    codes_by_review: dict[int, list[CaseReviewCode]] = {}
    review_ids = [r.id for r in reviews]
    if review_ids:
        code_rows = (
            await db.execute(select(CaseReviewCode).where(CaseReviewCode.case_review_id.in_(review_ids)))
        ).scalars().all()
        for code in code_rows:
            codes_by_review.setdefault(code.case_review_id, []).append(code)

    csv_text = _build_gold_csv([(review, codes_by_review.get(review.id, [])) for review in reviews])

    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=gold_{export_id}.csv"},
    )
