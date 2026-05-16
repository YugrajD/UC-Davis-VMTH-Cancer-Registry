"""Admin maintenance endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, require_admin
from app.cache import clear_all_caches
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

_MATERIALIZED_VIEWS = [
    "mv_county_cancer_incidence",
    "mv_yearly_trends",
]


@router.post("/refresh-views")
async def refresh_materialized_views(
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """Refresh all materialized views and clear the in-memory cache.

    Safe to call while the app is serving traffic — uses CONCURRENTLY so
    reads are not blocked during the refresh.  Call after manual data edits
    or whenever the dashboard shows stale aggregates.
    """
    errors = []
    for view in _MATERIALIZED_VIEWS:
        try:
            await db.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
        except Exception as e:
            logger.warning("Could not refresh %s: %s", view, e)
            errors.append(f"{view}: {e}")

    await db.commit()
    clear_all_caches()

    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"Partial refresh failure: {errors}",
        )

    return {"refreshed": _MATERIALIZED_VIEWS}
