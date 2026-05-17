"""Per-diagnosis manual review workflow.

Pairs with database/migrations/010_diagnosis_review.sql. Admin-only.

Endpoints:
  GET  /api/v1/diagnoses/pending          - paginated triage queue
  GET  /api/v1/diagnoses/{id}             - full detail incl. event log
  POST /api/v1/diagnoses/{id}/review      - confirm | correct | reject
  GET  /api/v1/diagnoses/pending/count    - badge counter

The router writes to diagnosis_review_events on every state change so
multiple reviewers can collaborate without losing history. When a
reviewer corrects to a brand-new cancer type, the type is auto-created
with confirmed=False and surfaces for admin sign-off elsewhere.
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import CurrentUser, require_reviewer
from app.database import get_db
from app.models.models import (
    CancerType,
    CaseDiagnosis,
    DiagnosisReviewEvent,
    IngestionJob,
    Patient,
)


router = APIRouter(prefix="/api/v1/diagnoses", tags=["diagnoses-review"])


# --- Schemas --------------------------------------------------------------


class PendingDiagnosis(BaseModel):
    id: int
    patient_anon_id: Optional[str]
    cancer_type_id: int
    cancer_type_name: str
    icd_o_code: Optional[str]
    predicted_term: Optional[str]
    confidence: Optional[float]
    top2_margin: Optional[float]
    prediction_method: Optional[str]
    diagnosis_index: Optional[int]
    review_status: str
    ingestion_job_id: Optional[int] = None
    job_filename: Optional[str] = None
    job_created_at: Optional[datetime] = None


class ReviewEventOut(BaseModel):
    id: int
    actor_email: str
    action: str
    from_status: Optional[str]
    to_status: str
    cancer_type_id_before: Optional[int]
    cancer_type_id_after: Optional[int]
    icd_o_code_before: Optional[str]
    icd_o_code_after: Optional[str]
    notes: Optional[str]
    created_at: datetime


class DiagnosisDetail(PendingDiagnosis):
    original_cancer_type_id: Optional[int]
    original_icd_o_code: Optional[str]
    original_predicted_term: Optional[str]
    reviewed_by_email: Optional[str]
    reviewed_at: Optional[datetime]
    reviewer_notes: Optional[str]
    events: list[ReviewEventOut]


class ReviewAction(BaseModel):
    action: Literal["confirm", "correct", "reject"]
    cancer_type_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Required for 'correct'. If unknown, will be auto-created with confirmed=False.",
    )
    icd_o_code: Optional[str] = Field(default=None, max_length=20)
    predicted_term: Optional[str] = Field(default=None, max_length=500)
    notes: Optional[str] = Field(default=None, max_length=5000)


# --- Helpers --------------------------------------------------------------


async def _get_or_404(db: AsyncSession, diagnosis_id: int) -> CaseDiagnosis:
    result = await db.execute(
        select(CaseDiagnosis)
        .options(
            selectinload(CaseDiagnosis.cancer_type),
            selectinload(CaseDiagnosis.patient),
            selectinload(CaseDiagnosis.review_events),
        )
        .where(CaseDiagnosis.id == diagnosis_id)
    )
    diag = result.scalar_one_or_none()
    if diag is None:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return diag


async def _resolve_or_create_cancer_type(
    db: AsyncSession, name: str
) -> tuple[int, bool]:
    """Return (cancer_type_id, was_created). New types are inserted with
    confirmed=False so the admin sign-off list can pick them up."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="cancer_type_name is empty")

    existing = await db.execute(
        select(CancerType.id).where(CancerType.name == name)
    )
    row = existing.scalar_one_or_none()
    if row is not None:
        return row, False

    new_type = CancerType(name=name, confirmed=False)
    db.add(new_type)
    await db.flush()
    return new_type.id, True


def _to_detail(diag: CaseDiagnosis) -> DiagnosisDetail:
    return DiagnosisDetail(
        id=diag.id,
        patient_anon_id=diag.patient.anon_id if diag.patient else None,
        cancer_type_id=diag.cancer_type_id,
        cancer_type_name=diag.cancer_type.name if diag.cancer_type else "",
        icd_o_code=diag.icd_o_code,
        predicted_term=diag.predicted_term,
        confidence=float(diag.confidence) if diag.confidence is not None else None,
        top2_margin=float(diag.top2_margin) if diag.top2_margin is not None else None,
        prediction_method=diag.prediction_method,
        diagnosis_index=diag.diagnosis_index,
        review_status=diag.review_status,
        original_cancer_type_id=diag.original_cancer_type_id,
        original_icd_o_code=diag.original_icd_o_code,
        original_predicted_term=diag.original_predicted_term,
        reviewed_by_email=diag.reviewed_by_email,
        reviewed_at=diag.reviewed_at,
        reviewer_notes=diag.reviewer_notes,
        events=[
            ReviewEventOut(
                id=e.id,
                actor_email=e.actor_email,
                action=e.action,
                from_status=e.from_status,
                to_status=e.to_status,
                cancer_type_id_before=e.cancer_type_id_before,
                cancer_type_id_after=e.cancer_type_id_after,
                icd_o_code_before=e.icd_o_code_before,
                icd_o_code_after=e.icd_o_code_after,
                notes=e.notes,
                created_at=e.created_at,
            )
            for e in diag.review_events
        ],
    )


# --- Endpoints ------------------------------------------------------------


@router.get("/pending/count")
async def pending_count(
    db: AsyncSession = Depends(get_db),
    _reviewer: CurrentUser = Depends(require_reviewer),
) -> dict:
    """Cheap counter for the nav badge."""
    result = await db.execute(
        select(func.count(CaseDiagnosis.id)).where(
            CaseDiagnosis.review_status == "pending"
        )
    )
    return {"count": result.scalar() or 0}


@router.get("/pending")
async def list_pending(
    db: AsyncSession = Depends(get_db),
    _reviewer: CurrentUser = Depends(require_reviewer),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    cancer_type_id: Optional[int] = None,
    method: Optional[str] = Query(default=None, max_length=50),
    max_confidence: Optional[float] = None,
    ingestion_job_id: Optional[int] = None,
) -> list[PendingDiagnosis]:
    """Paginated review queue, optionally filtered."""
    query = (
        select(
            CaseDiagnosis,
            CancerType.name,
            Patient.anon_id,
            IngestionJob.dataset_a_filename,
            IngestionJob.created_at,
        )
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .outerjoin(IngestionJob, IngestionJob.id == CaseDiagnosis.ingestion_job_id)
        .where(CaseDiagnosis.review_status == "pending")
    )
    if cancer_type_id is not None:
        query = query.where(CaseDiagnosis.cancer_type_id == cancer_type_id)
    if method is not None:
        query = query.where(CaseDiagnosis.prediction_method == method)
    if max_confidence is not None:
        query = query.where(CaseDiagnosis.confidence <= max_confidence)
    if ingestion_job_id is not None:
        query = query.where(CaseDiagnosis.ingestion_job_id == ingestion_job_id)

    query = (
        query.order_by(
            CaseDiagnosis.ingestion_job_id.desc().nulls_last(),
            CaseDiagnosis.confidence.asc().nulls_first(),
        )
        .limit(limit)
        .offset(offset)
    )
    rows = (await db.execute(query)).all()
    return [
        PendingDiagnosis(
            id=d.id,
            patient_anon_id=anon_id,
            cancer_type_id=d.cancer_type_id,
            cancer_type_name=ct_name,
            icd_o_code=d.icd_o_code,
            predicted_term=d.predicted_term,
            confidence=float(d.confidence) if d.confidence is not None else None,
            top2_margin=float(d.top2_margin) if d.top2_margin is not None else None,
            prediction_method=d.prediction_method,
            diagnosis_index=d.diagnosis_index,
            review_status=d.review_status,
            ingestion_job_id=d.ingestion_job_id,
            job_filename=job_filename,
            job_created_at=job_created_at,
        )
        for d, ct_name, anon_id, job_filename, job_created_at in rows
    ]


@router.get("/{diagnosis_id}")
async def get_diagnosis(
    diagnosis_id: int,
    db: AsyncSession = Depends(get_db),
    _reviewer: CurrentUser = Depends(require_reviewer),
) -> DiagnosisDetail:
    diag = await _get_or_404(db, diagnosis_id)
    return _to_detail(diag)


@router.post("/{diagnosis_id}/review")
async def review_diagnosis(
    diagnosis_id: int,
    body: ReviewAction,
    db: AsyncSession = Depends(get_db),
    reviewer: CurrentUser = Depends(require_reviewer),
) -> DiagnosisDetail:
    # Lock the row before reading status to prevent concurrent double-reviews.
    locked = await db.execute(
        select(CaseDiagnosis)
        .where(CaseDiagnosis.id == diagnosis_id)
        .with_for_update()
    )
    diag = locked.scalar_one_or_none()
    if diag is None:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    if diag.review_status not in ("pending", "confirmed", "corrected"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot review a diagnosis in status '{diag.review_status}'",
        )

    from_status = diag.review_status
    ct_before = diag.cancer_type_id
    code_before = diag.icd_o_code

    if body.action == "confirm":
        diag.review_status = "confirmed"
    elif body.action == "reject":
        diag.review_status = "rejected"
    elif body.action == "correct":
        if not body.cancer_type_name:
            raise HTTPException(
                status_code=400,
                detail="'correct' requires cancer_type_name",
            )
        new_ct_id, _created = await _resolve_or_create_cancer_type(
            db, body.cancer_type_name
        )
        if diag.original_cancer_type_id is None:
            # Preserve PetBERT's original prediction the first time we
            # overwrite it. Subsequent corrections leave original_* intact.
            diag.original_cancer_type_id = ct_before
            diag.original_icd_o_code = code_before
            diag.original_predicted_term = diag.predicted_term
        diag.cancer_type_id = new_ct_id
        diag.icd_o_code = body.icd_o_code or None
        if body.predicted_term is not None:
            diag.predicted_term = body.predicted_term
        diag.review_status = "corrected"

    diag.reviewed_by_email = reviewer.email
    diag.reviewed_at = datetime.now(timezone.utc)
    if body.notes:
        diag.reviewer_notes = body.notes

    db.add(DiagnosisReviewEvent(
        case_diagnosis_id=diag.id,
        actor_email=reviewer.email,
        action=body.action,
        from_status=from_status,
        to_status=diag.review_status,
        cancer_type_id_before=ct_before,
        cancer_type_id_after=diag.cancer_type_id,
        icd_o_code_before=code_before,
        icd_o_code_after=diag.icd_o_code,
        notes=body.notes,
    ))

    await db.commit()

    refreshed = await _get_or_404(db, diagnosis_id)
    return _to_detail(refreshed)
