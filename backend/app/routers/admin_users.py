"""Admin panel — manage per-user role assignments.

All endpoints require the admin role.

Endpoints:
  GET  /api/v1/admin/users/{email}/roles  - lookup current roles
  PUT  /api/v1/admin/users/{email}/roles  - set roles (creates row if absent)
  GET  /api/v1/admin/users/roles          - list all rows with non-default roles
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, require_admin
from app.database import get_db
from app.models.models import UserRole

router = APIRouter(prefix="/api/v1/admin/users", tags=["admin-users"])


# --- Schemas --------------------------------------------------------------


class UserRolesOut(BaseModel):
    email: str
    is_admin: bool
    is_uploader: bool
    is_reviewer: bool
    updated_by_email: Optional[str] = None
    updated_at: Optional[datetime] = None
    # True when no DB row exists yet — caller is seeing env-fallback values
    # or zeroed defaults.
    persisted: bool


class UserRolesIn(BaseModel):
    is_admin: bool = False
    is_uploader: bool = False
    is_reviewer: bool = False


# --- Helpers --------------------------------------------------------------


def _normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if "@" not in e or len(e) > 255:
        raise HTTPException(status_code=400, detail="Invalid email")
    return e


async def _lookup(db: AsyncSession, email: str) -> Optional[UserRole]:
    result = await db.execute(
        select(UserRole).where(func.lower(UserRole.email) == email)
    )
    return result.scalar_one_or_none()


# --- Endpoints ------------------------------------------------------------


@router.get("/{email}/roles", response_model=UserRolesOut)
async def get_user_roles(
    email: str,
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """Look up the role assignment for an email.

    Returns the DB row when one exists. When no row is found, returns the
    env-fallback values (or all-false defaults) with persisted=False so
    the caller can show "no record yet".
    """
    normalized = _normalize_email(email)
    row = await _lookup(db, normalized)
    if row is None:
        # Mirror the resolution that get_current_user would compute.
        from app.auth import _resolve_roles_from_env  # late import avoids cycle
        is_admin, is_uploader, is_reviewer = _resolve_roles_from_env(normalized)
        return UserRolesOut(
            email=normalized,
            is_admin=is_admin,
            is_uploader=is_uploader,
            is_reviewer=is_reviewer,
            updated_by_email=None,
            updated_at=None,
            persisted=False,
        )
    return UserRolesOut(
        email=row.email,
        is_admin=row.is_admin,
        is_uploader=row.is_uploader,
        is_reviewer=row.is_reviewer,
        updated_by_email=row.updated_by_email,
        updated_at=row.updated_at,
        persisted=True,
    )


@router.put("/{email}/roles", response_model=UserRolesOut)
async def set_user_roles(
    email: str,
    body: UserRolesIn,
    db: AsyncSession = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
):
    """Upsert role assignments for an email.

    Admins implicitly hold lower-privilege roles, so is_admin=True forces
    is_uploader and is_reviewer to True regardless of the body.

    Self-demotion guardrail: an admin cannot remove their own admin flag
    (would lock the system out if they were the last admin).
    """
    normalized = _normalize_email(email)

    is_admin = body.is_admin
    is_uploader = is_admin or body.is_uploader
    is_reviewer = is_admin or body.is_reviewer

    if normalized == admin.email.lower() and not is_admin:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove your own admin role",
        )

    row = await _lookup(db, normalized)
    now = datetime.now(timezone.utc)
    if row is None:
        row = UserRole(
            email=normalized,
            is_admin=is_admin,
            is_uploader=is_uploader,
            is_reviewer=is_reviewer,
            updated_by_email=admin.email,
            updated_at=now,
        )
        db.add(row)
    else:
        row.is_admin = is_admin
        row.is_uploader = is_uploader
        row.is_reviewer = is_reviewer
        row.updated_by_email = admin.email
        row.updated_at = now

    await db.commit()

    return UserRolesOut(
        email=row.email,
        is_admin=row.is_admin,
        is_uploader=row.is_uploader,
        is_reviewer=row.is_reviewer,
        updated_by_email=row.updated_by_email,
        updated_at=row.updated_at,
        persisted=True,
    )


@router.get("/roles", response_model=list[UserRolesOut])
async def list_user_roles(
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """List every user_roles row, ordered by email."""
    result = await db.execute(select(UserRole).order_by(UserRole.email))
    rows = result.scalars().all()
    return [
        UserRolesOut(
            email=r.email,
            is_admin=r.is_admin,
            is_uploader=r.is_uploader,
            is_reviewer=r.is_reviewer,
            updated_by_email=r.updated_by_email,
            updated_at=r.updated_at,
            persisted=True,
        )
        for r in rows
    ]
