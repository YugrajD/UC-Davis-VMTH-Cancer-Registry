"""Unit tests for /api/v1/admin/refresh-views."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, require_admin
from app.cache import _caches, get_cache
from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _admin_user() -> CurrentUser:
    return CurrentUser(
        sub="sub-admin",
        email="admin@ucdavis.edu",
        is_admin=True,
        is_uploader=True,
        is_reviewer=True,
    )


def _non_admin_user() -> CurrentUser:
    return CurrentUser(
        sub="sub-user",
        email="user@ucdavis.edu",
        is_admin=False,
        is_uploader=False,
        is_reviewer=False,
    )


def _override_admin(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[require_admin] = _f


def _override_db(execute_side_effects):
    mock_db = AsyncMock()
    mock_db.execute.side_effect = execute_side_effects

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override
    return mock_db


def _cleanup():
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/v1/admin/refresh-views
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_views_runs_both_views_and_returns_list():
    """Happy path: REFRESH succeeds for both views, response lists them."""
    _override_admin(_admin_user())
    db = _override_db([AsyncMock(), AsyncMock()])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/admin/refresh-views")

        assert r.status_code == 200
        data = r.json()
        assert "refreshed" in data
        assert "mv_county_cancer_incidence" in data["refreshed"]
        assert "mv_yearly_trends" in data["refreshed"]
        # Two REFRESH statements were issued.
        assert db.execute.call_count == 2
        # The transaction was committed.
        db.commit.assert_awaited_once()
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_refresh_views_uses_concurrently_to_avoid_blocking_reads():
    """The REFRESH must use CONCURRENTLY so reads aren't blocked."""
    _override_admin(_admin_user())
    db = _override_db([AsyncMock(), AsyncMock()])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/api/v1/admin/refresh-views")

        # Inspect the executed SQL — each call's first arg is a TextClause.
        sqls = [call.args[0].text for call in db.execute.call_args_list]
        assert all("CONCURRENTLY" in s for s in sqls), \
            f"All refreshes must use CONCURRENTLY; got: {sqls}"
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_refresh_views_clears_in_memory_caches():
    """After a successful refresh, all registered caches should be empty."""
    _override_admin(_admin_user())
    _override_db([AsyncMock(), AsyncMock()])
    # Seed a cache so we can confirm it's cleared.
    cache = get_cache("test_ns_refresh", maxsize=8, ttl=60)
    cache["key"] = {"value": 42}
    assert "key" in cache
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/admin/refresh-views")
        assert r.status_code == 200
        assert "key" not in cache, "Cache should be cleared after refresh"
    finally:
        _caches.pop("test_ns_refresh", None)
        _cleanup()


@pytest.mark.asyncio
async def test_refresh_views_returns_500_if_a_view_fails():
    """If any view fails to refresh, return 500 with the partial-failure detail."""
    _override_admin(_admin_user())
    # First REFRESH succeeds, second raises.
    _override_db([AsyncMock(), RuntimeError("relation does not exist")])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/admin/refresh-views")
        assert r.status_code == 500
        assert "Partial refresh failure" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_refresh_views_requires_admin():
    """Non-admin users must get 403."""
    # Override require_admin to raise 403 like the real dep would.
    from fastapi import HTTPException, status

    async def _deny():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    app.dependency_overrides[require_admin] = _deny
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/admin/refresh-views")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_refresh_views_does_not_leak_non_admin_access():
    """When require_admin is satisfied, the endpoint accepts the request.

    Smoke check that the dependency is wired correctly — without the admin
    override the request would be rejected by the bearer scheme (401/403).
    """
    _override_admin(_admin_user())
    _override_db([AsyncMock(), AsyncMock()])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/admin/refresh-views")
        assert r.status_code == 200
    finally:
        _cleanup()
