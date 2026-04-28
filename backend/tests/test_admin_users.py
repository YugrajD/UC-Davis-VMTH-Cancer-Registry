"""Unit tests for /api/v1/admin/users — role management."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, require_admin
from app.database import get_db
from app.main import app
from app.services import role_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _admin_user(email: str = "admin@ucdavis.edu") -> CurrentUser:
    return CurrentUser(
        sub="sub-admin",
        email=email,
        is_admin=True,
        is_uploader=True,
        is_reviewer=True,
    )


def _scalar_one_or_none_result(value):
    """Mock execute() result whose .scalar_one_or_none() returns value."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _row(
    email="user@ucdavis.edu",
    is_admin=False,
    is_uploader=False,
    is_reviewer=False,
):
    return SimpleNamespace(
        email=email,
        is_admin=is_admin,
        is_uploader=is_uploader,
        is_reviewer=is_reviewer,
        updated_by_email="prior-admin@ucdavis.edu",
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _override_admin(user: CurrentUser):
    """Replace require_admin with a fixture that returns the given user."""
    async def _f():
        return user
    app.dependency_overrides[require_admin] = _f


def _override_db(execute_results):
    """Override get_db with an AsyncMock whose execute() returns given values in order.

    `execute_results` is a list of mock result objects (or a callable that
    returns one, for stateful mocks).
    """
    mock_db = AsyncMock()
    if callable(execute_results):
        mock_db.execute.side_effect = execute_results
    else:
        mock_db.execute.side_effect = execute_results

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override
    return mock_db


def _cleanup():
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /{email}/roles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_roles_returns_db_row_when_present():
    _override_admin(_admin_user())
    _override_db([
        _scalar_one_or_none_result(
            _row(email="alice@ucdavis.edu", is_reviewer=True)
        ),
    ])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/admin/users/alice@ucdavis.edu/roles")
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == "alice@ucdavis.edu"
        assert data["is_reviewer"] is True
        assert data["persisted"] is True
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_get_roles_falls_back_to_env_when_no_row(monkeypatch):
    _override_admin(_admin_user())
    _override_db([_scalar_one_or_none_result(None)])
    # Pretend the env says alice is a reviewer.
    from app.config import settings
    monkeypatch.setattr(settings, "REVIEWER_EMAILS", "alice@ucdavis.edu")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/admin/users/alice@ucdavis.edu/roles")
        data = r.json()
        assert r.status_code == 200
        assert data["persisted"] is False
        assert data["is_reviewer"] is True
        assert data["is_admin"] is False
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_get_roles_normalizes_email_case():
    _override_admin(_admin_user())
    _override_db([_scalar_one_or_none_result(None)])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/admin/users/Mixed.Case@UCDavis.edu/roles")
        data = r.json()
        assert data["email"] == "mixed.case@ucdavis.edu"
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_get_roles_rejects_invalid_email():
    _override_admin(_admin_user())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/admin/users/not-an-email/roles")
        assert r.status_code == 400
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# PUT /{email}/roles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_roles_inserts_when_no_row():
    _override_admin(_admin_user())
    mock_db = _override_db([_scalar_one_or_none_result(None)])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.put(
                "/api/v1/admin/users/new@ucdavis.edu/roles",
                json={"is_admin": False, "is_uploader": True, "is_reviewer": False},
            )
        assert r.status_code == 200
        # The endpoint should have queued an insert and committed.
        assert mock_db.add.called
        assert mock_db.commit.called
        data = r.json()
        assert data["is_uploader"] is True
        assert data["persisted"] is True
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_put_roles_updates_existing_row():
    _override_admin(_admin_user())
    existing = _row(email="user@ucdavis.edu")
    _override_db([_scalar_one_or_none_result(existing)])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.put(
                "/api/v1/admin/users/user@ucdavis.edu/roles",
                json={"is_admin": False, "is_uploader": False, "is_reviewer": True},
            )
        assert r.status_code == 200
        # Row mutated in place — verify the new values landed on the model.
        assert existing.is_reviewer is True
        assert existing.is_uploader is False
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_put_roles_admin_implies_uploader_and_reviewer():
    _override_admin(_admin_user())
    _override_db([_scalar_one_or_none_result(None)])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.put(
                "/api/v1/admin/users/promoted@ucdavis.edu/roles",
                json={"is_admin": True, "is_uploader": False, "is_reviewer": False},
            )
        data = r.json()
        # Server must force the lower roles on whenever is_admin is true.
        assert data["is_admin"] is True
        assert data["is_uploader"] is True
        assert data["is_reviewer"] is True
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_put_roles_blocks_self_demotion():
    me = _admin_user(email="me@ucdavis.edu")
    _override_admin(me)
    _override_db([])  # should reject before any DB call
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.put(
                "/api/v1/admin/users/me@ucdavis.edu/roles",
                json={"is_admin": False, "is_uploader": True, "is_reviewer": True},
            )
        assert r.status_code == 400
        assert "own admin role" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_put_roles_self_demotion_check_is_case_insensitive():
    me = _admin_user(email="Me@UCDavis.edu")
    _override_admin(me)
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.put(
                "/api/v1/admin/users/me@ucdavis.edu/roles",
                json={"is_admin": False, "is_uploader": True, "is_reviewer": True},
            )
        assert r.status_code == 400
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_put_roles_lets_admin_keep_their_admin_role():
    """Self-edit is fine as long as is_admin stays true."""
    me = _admin_user(email="me@ucdavis.edu")
    _override_admin(me)
    _override_db([_scalar_one_or_none_result(None)])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.put(
                "/api/v1/admin/users/me@ucdavis.edu/roles",
                json={"is_admin": True, "is_uploader": True, "is_reviewer": True},
            )
        assert r.status_code == 200
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_roles_requires_admin():
    """Without overriding require_admin, the call should not 200."""
    # Don't override require_admin — let the real one run. Without a JWT
    # we expect a 401/403 from the bearer scheme.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/admin/users/alice@ucdavis.edu/roles")
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_put_roles_requires_admin():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.put(
            "/api/v1/admin/users/alice@ucdavis.edu/roles",
            json={"is_admin": False, "is_uploader": True, "is_reviewer": False},
        )
    assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# role_seed.seed_user_roles_from_env — pure logic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_inserts_only_missing_emails(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_EMAILS", "a@x.com")
    monkeypatch.setattr(settings, "UPLOADER_EMAILS", "u@x.com")
    monkeypatch.setattr(settings, "REVIEWER_EMAILS", "r@x.com")

    # Pretend a@x.com is already in the DB; u and r are not.
    existing = MagicMock()
    existing.all.return_value = [("a@x.com",)]

    db = AsyncMock()
    db.execute.return_value = existing
    db.add = MagicMock()

    inserted = await role_seed.seed_user_roles_from_env(db)
    assert inserted == 2  # u and r, not a
    # add() called twice — for the two new emails
    assert db.add.call_count == 2


@pytest.mark.asyncio
async def test_seed_admin_implies_lower_roles(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_EMAILS", "a@x.com")
    monkeypatch.setattr(settings, "UPLOADER_EMAILS", "")
    monkeypatch.setattr(settings, "REVIEWER_EMAILS", "")

    existing = MagicMock()
    existing.all.return_value = []

    db = AsyncMock()
    db.execute.return_value = existing
    # `db.add` is invoked synchronously by SQLAlchemy code under test, so
    # override the AsyncMock attr with a regular MagicMock to capture args.
    db.add = MagicMock()

    await role_seed.seed_user_roles_from_env(db)
    assert db.add.call_count == 1
    row = db.add.call_args.args[0]
    assert row.is_admin is True
    assert row.is_uploader is True
    assert row.is_reviewer is True


@pytest.mark.asyncio
async def test_seed_returns_zero_when_env_empty(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_EMAILS", "")
    monkeypatch.setattr(settings, "UPLOADER_EMAILS", "")
    monkeypatch.setattr(settings, "REVIEWER_EMAILS", "")

    db = AsyncMock()
    inserted = await role_seed.seed_user_roles_from_env(db)
    assert inserted == 0
    db.execute.assert_not_called()
