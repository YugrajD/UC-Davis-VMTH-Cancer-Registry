"""BERT search and pathology report endpoints."""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from typing import Optional

from app.database import get_db
from app.models.models import PathologyReport, CancerType
from app.schemas.schemas import (
    ClassifyRequest, ClassifyResult, ReportOut, ReportSearchResponse
)
from app.services.bert_service import BertClassifier

router = APIRouter(prefix="/api/v1/search", tags=["search"])

classifier = BertClassifier()


@router.post("/classify", response_model=ClassifyResult)
async def classify_report(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Report text is required")

    result = classifier.classify(request.text)
    return result


@router.get("/reports", response_model=ReportSearchResponse)
async def search_reports(
    keyword: Optional[str] = Query(default=None, max_length=500),
    classification: Optional[str] = Query(default=None, max_length=200),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(PathologyReport)

    if keyword:
        stmt = stmt.where(PathologyReport.report_text.ilike(f"%{keyword}%"))

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
