"""Export request endpoints — users request data export access, admins resolve."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, get_current_user, require_admin
from app.database import get_db
from app.models.models import ExportRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export-requests", tags=["export-requests"])


# --- Schemas ----------------------------------------------------------------


class ExportRequestCreate(BaseModel):
    reason: Optional[str] = None


class ExportRequestResolve(BaseModel):
    action: str  # "approve" or "deny"


class ExportRequestOut(BaseModel):
    id: int
    email: str
    status: str
    reason: Optional[str] = None
    resolved_by_email: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime


class PendingCountOut(BaseModel):
    count: int


# --- Endpoints --------------------------------------------------------------


@router.post("/", response_model=ExportRequestOut)
async def submit_export_request(
    body: ExportRequestCreate,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """Submit a request for data export access."""
    # Check for existing pending request
    existing = await db.execute(
        select(ExportRequest).where(
            func.lower(ExportRequest.email) == user.email.lower(),
            ExportRequest.status == "pending",
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="You already have a pending export request")

    req = ExportRequest(
        email=user.email.lower(),
        status="pending",
        reason=body.reason,
        created_at=datetime.now(timezone.utc),
    )
    db.add(req)
    await db.commit()
    await db.refresh(req)

    return _to_out(req)


@router.get("/mine", response_model=list[ExportRequestOut])
async def my_export_requests(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """List the current user's export requests."""
    result = await db.execute(
        select(ExportRequest)
        .where(func.lower(ExportRequest.email) == user.email.lower())
        .order_by(ExportRequest.created_at.desc())
    )
    return [_to_out(r) for r in result.scalars().all()]


@router.get("/pending", response_model=list[ExportRequestOut])
async def pending_export_requests(
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """List all pending export requests (admin-only)."""
    result = await db.execute(
        select(ExportRequest)
        .where(ExportRequest.status == "pending")
        .order_by(ExportRequest.created_at.asc())
    )
    return [_to_out(r) for r in result.scalars().all()]


@router.get("/pending/count", response_model=PendingCountOut)
async def pending_export_request_count(
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """Count pending export requests (admin-only, used for badge)."""
    result = await db.execute(
        select(func.count(ExportRequest.id)).where(ExportRequest.status == "pending")
    )
    return PendingCountOut(count=result.scalar() or 0)


@router.post("/{request_id}/resolve", response_model=ExportRequestOut)
async def resolve_export_request(
    request_id: int,
    body: ExportRequestResolve,
    db: AsyncSession = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
):
    """Approve or deny an export request (admin-only)."""
    if body.action not in ("approve", "deny"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'deny'")

    result = await db.execute(
        select(ExportRequest).where(ExportRequest.id == request_id)
    )
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Export request not found")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already '{req.status}'")

    now = datetime.now(timezone.utc)
    req.resolved_by_email = admin.email
    req.resolved_at = now
    req.status = "approved" if body.action == "approve" else "denied"

    await db.commit()
    await db.refresh(req)
    return _to_out(req)


@router.get("/download")
async def download_export_csv(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """Download county export CSV. Admins always allowed; others need an approved request."""
    if not user.is_admin:
        result = await db.execute(
            select(ExportRequest).where(
                func.lower(ExportRequest.email) == user.email.lower(),
                ExportRequest.status == "approved",
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=403,
                detail="You need an approved export request to download data",
            )

    from app.services.export_service import generate_county_export_csv

    csv_content = await generate_county_export_csv(db)

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vmth_county_export.csv"},
    )


# --- Helpers ----------------------------------------------------------------


def _to_out(req: ExportRequest) -> ExportRequestOut:
    return ExportRequestOut(
        id=req.id,
        email=req.email,
        status=req.status,
        reason=req.reason,
        resolved_by_email=req.resolved_by_email,
        resolved_at=req.resolved_at,
        created_at=req.created_at,
    )
