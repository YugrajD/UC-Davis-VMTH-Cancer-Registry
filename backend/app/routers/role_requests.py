"""Role request endpoints — users request uploader/reviewer roles, admins resolve."""

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, get_current_user, require_admin
from app.database import get_db
from app.models.models import RoleRequest, UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/role-requests", tags=["role-requests"])


# --- Schemas ----------------------------------------------------------------


class RoleRequestCreate(BaseModel):
    requested_role: Literal["uploader", "reviewer"]
    reason: Optional[str] = Field(default=None, max_length=2000)


class RoleRequestResolve(BaseModel):
    action: Literal["approve", "deny"]


class RoleRequestOut(BaseModel):
    id: int
    email: str
    requested_role: str
    status: str
    reason: Optional[str] = None
    resolved_by_email: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime


class PendingCountOut(BaseModel):
    count: int


# --- Endpoints --------------------------------------------------------------


@router.post("/", response_model=RoleRequestOut)
async def submit_role_request(
    body: RoleRequestCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """Submit a request for the uploader or reviewer role."""
    # Check if user already has the role
    if body.requested_role == "uploader" and user.is_uploader:
        raise HTTPException(status_code=400, detail="You already have the uploader role")
    if body.requested_role == "reviewer" and user.is_reviewer:
        raise HTTPException(status_code=400, detail="You already have the reviewer role")

    # Check for existing pending request
    existing = await db.execute(
        select(RoleRequest).where(
            func.lower(RoleRequest.email) == user.email.lower(),
            RoleRequest.requested_role == body.requested_role,
            RoleRequest.status == "pending",
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="You already have a pending request for this role")

    req = RoleRequest(
        email=user.email.lower(),
        requested_role=body.requested_role,
        status="pending",
        reason=body.reason,
        created_at=datetime.now(timezone.utc),
    )
    db.add(req)
    await db.commit()
    await db.refresh(req)

    # Best-effort email notification to admins (don't pass the request-scoped
    # db session — background tasks run after dependency cleanup closes it).
    background_tasks.add_background_task(_notify_admins, user.email, body.requested_role)

    return _to_out(req)


@router.get("/mine", response_model=list[RoleRequestOut])
async def my_role_requests(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """List the current user's role requests."""
    result = await db.execute(
        select(RoleRequest)
        .where(func.lower(RoleRequest.email) == user.email.lower())
        .order_by(RoleRequest.created_at.desc())
    )
    return [_to_out(r) for r in result.scalars().all()]


@router.get("/pending", response_model=list[RoleRequestOut])
async def pending_role_requests(
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """List all pending role requests (admin-only)."""
    result = await db.execute(
        select(RoleRequest)
        .where(RoleRequest.status == "pending")
        .order_by(RoleRequest.created_at.asc())
    )
    return [_to_out(r) for r in result.scalars().all()]


@router.get("/pending/count", response_model=PendingCountOut)
async def pending_role_request_count(
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """Count pending role requests (admin-only, used for badge)."""
    result = await db.execute(
        select(func.count(RoleRequest.id)).where(RoleRequest.status == "pending")
    )
    return PendingCountOut(count=result.scalar() or 0)


@router.post("/{request_id}/resolve", response_model=RoleRequestOut)
async def resolve_role_request(
    request_id: int,
    body: RoleRequestResolve,
    db: AsyncSession = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
):
    """Approve or deny a role request (admin-only)."""
    result = await db.execute(
        select(RoleRequest).where(RoleRequest.id == request_id)
    )
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(status_code=404, detail="Role request not found")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already '{req.status}'")

    now = datetime.now(timezone.utc)
    req.resolved_by_email = admin.email
    req.resolved_at = now

    if body.action == "deny":
        req.status = "denied"
        await db.commit()
        await db.refresh(req)
        return _to_out(req)

    # Approve — grant the role via user_roles table
    req.status = "approved"

    user_email = req.email.lower()
    role_result = await db.execute(
        select(UserRole).where(func.lower(UserRole.email) == user_email)
    )
    role_row = role_result.scalar_one_or_none()

    if role_row is None:
        role_row = UserRole(
            email=user_email,
            is_admin=False,
            is_uploader=req.requested_role == "uploader",
            is_reviewer=req.requested_role == "reviewer",
            updated_by_email=admin.email,
            updated_at=now,
        )
        db.add(role_row)
    else:
        if req.requested_role == "uploader":
            role_row.is_uploader = True
        elif req.requested_role == "reviewer":
            role_row.is_reviewer = True
        role_row.updated_by_email = admin.email
        role_row.updated_at = now

    await db.commit()
    await db.refresh(req)
    return _to_out(req)


# --- Helpers ----------------------------------------------------------------


def _to_out(req: RoleRequest) -> RoleRequestOut:
    return RoleRequestOut(
        id=req.id,
        email=req.email,
        requested_role=req.requested_role,
        status=req.status,
        reason=req.reason,
        resolved_by_email=req.resolved_by_email,
        resolved_at=req.resolved_at,
        created_at=req.created_at,
    )


async def _notify_admins(requester_email: str, requested_role: str):
    """Fetch admin emails and send notification. Best-effort."""
    try:
        from app.database import async_session
        from app.services.email import send_role_request_email

        async with async_session() as session:
            result = await session.execute(
                select(UserRole.email).where(UserRole.is_admin == True)  # noqa: E712
            )
            admin_emails = [r[0] for r in result.all()]

        send_role_request_email(requester_email, requested_role, admin_emails)
    except Exception:
        logger.exception("Failed to send role-request admin notification")
