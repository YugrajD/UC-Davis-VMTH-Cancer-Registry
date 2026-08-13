"""Auth router — current user info and self-service account deletion."""

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.database import get_db
from app.models.models import UserRole
from app.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Arbitrary fixed key for a Postgres session-level advisory lock (scoped to
# the enclosing transaction via pg_advisory_xact_lock). Serializes concurrent
# "am I the last admin?" checks so two admins deleting their accounts at the
# same instant can't both pass the check and leave zero admins.
_ADMIN_DELETE_LOCK_KEY = 72738491


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)):
    """Return current user's email and role flags."""
    return {
        "email": user.email,
        "is_admin": user.is_admin,
        "is_uploader": user.is_uploader,
        "is_reviewer": user.is_reviewer,
    }


async def _count_effective_admins(db: AsyncSession) -> int:
    """Count distinct admin accounts (DB rows plus un-overridden env fallback)."""
    result = await db.execute(select(UserRole.email, UserRole.is_admin))
    rows = result.all()
    db_emails = {email.lower() for email, _ in rows}
    db_admin_count = sum(1 for _, is_admin in rows if is_admin)
    env_admin_extra = sum(
        1 for e in settings.admin_emails_list if e.lower() not in db_emails
    )
    return db_admin_count + env_admin_extra


async def _delete_supabase_user(user_id: str) -> None:
    """Permanently delete the Supabase auth user via the Admin API."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        logger.error("Account deletion requested but Supabase admin credentials are not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion is not available right now. Please contact an administrator.",
        )
    url = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/admin/users/{user_id}"
    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(url, headers=headers)
    if resp.status_code not in (200, 204):
        logger.error("Supabase admin user-delete failed: %s %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to delete account with the authentication provider",
        )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def delete_me(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """Permanently delete the current user's own account (self-service only).

    Identity is taken entirely from the verified JWT (sub/email) — the
    caller cannot target any account but their own. Patient and diagnosis
    records are untouched: they have no FK to user accounts, only
    audit-trail email strings (uploaded_by_email etc).

    Ordering: the local user_roles row is deleted (uncommitted) before the
    Supabase Admin API call, and rolled back if that call fails. This way
    the harder-to-reverse external delete happens last and a failure there
    never leaves an orphaned, still-privileged role row behind — worst
    case is a no-op that the client can safely retry.
    """
    if user.is_admin:
        # Hold the lock until commit/rollback so two admins deleting at the
        # same instant can't both read a count that hasn't reflected the
        # other's (pending) deletion yet.
        await db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ADMIN_DELETE_LOCK_KEY})
        admin_count = await _count_effective_admins(db)
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the only admin account. Promote another user to admin first.",
            )

    await db.execute(delete(UserRole).where(func.lower(UserRole.email) == user.email.lower()))

    try:
        await _delete_supabase_user(user.sub)
    except HTTPException:
        await db.rollback()
        raise

    await db.commit()

    logger.warning("Account deleted: %s (was_admin=%s)", user.email, user.is_admin)
