"""BERT search and pathology report endpoints."""

from fastapi import APIRouter, Depends, Query, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from typing import Optional

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.database import get_db
from app.models.models import PathologyReport, CancerType
from app.rate_limit import limiter
from app.schemas.schemas import (
    ClassifyRequest, ClassifyResult, ReportOut, ReportSearchResponse
)
from app.services.bert_service import BertClassifier

router = APIRouter(prefix="/api/v1/search", tags=["search"])

classifier = BertClassifier()


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcard characters in user input."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.post("/classify", response_model=ClassifyResult)
@limiter.limit(settings.RATE_LIMIT_EXPENSIVE)
async def classify_report(
    body: ClassifyRequest,
    request: Request,
    _user: CurrentUser = Depends(get_current_user),
):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Report text is required")

    result = classifier.classify(body.text)
    return result


@router.get("/reports", response_model=ReportSearchResponse)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def search_reports(
    request: Request,
    keyword: Optional[str] = Query(default=None, max_length=500),
    classification: Optional[str] = Query(default=None, max_length=200),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    _user: CurrentUser = Depends(get_current_user),
):
    stmt = select(PathologyReport)

    if keyword:
        escaped = _escape_like(keyword)
        stmt = stmt.where(PathologyReport.report_text.ilike(f"%{escaped}%"))

    if classification:
        stmt = stmt.where(PathologyReport.classification == classification)

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Paginate
    stmt = stmt.order_by(PathologyReport.report_date.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    reports = result.scalars().all()

    return ReportSearchResponse(
        reports=[
            ReportOut(
                id=r.id,
                patient_id=r.patient_id,
                report_text=r.report_text,
                classification=r.classification,
                confidence_score=float(r.confidence_score) if r.confidence_score else None,
                report_date=r.report_date,
            )
            for r in reports
        ],
        total=total,
    )
