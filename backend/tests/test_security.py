"""Unit tests for security hardening — input validation, auth gates, and sanitization."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, get_current_user, require_reviewer
from app.database import get_db
from app.main import app
from app.routers.search import _escape_like
from app.services.job_processor import _safe_error_message
from tests.conftest import scalar_result, scalars_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _user(email: str = "user@ucdavis.edu") -> CurrentUser:
    return CurrentUser(
        sub="sub-user",
        email=email,
        is_admin=False,
        is_uploader=False,
        is_reviewer=False,
    )


def _override_auth(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[get_current_user] = _f


def _override_db(execute_results):
    mock_db = AsyncMock()
    mock_db.execute.side_effect = execute_results

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override
    return mock_db


def _cleanup():
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Body size middleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_body_returns_413():
    """Content-Length > 10 MB on a non-upload path should return 413."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/search/classify",
            content=b"x",
            headers={"Content-Length": str(11 * 1024 * 1024)},
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_path_allows_larger_body():
    """The upload path should NOT return 413 for a body under 50 MB.

    We send a request with Content-Length just under 50 MB. It won't
    succeed (no auth, no file), but it should NOT be 413.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingest/upload",
            content=b"x",
            headers={"Content-Length": str(40 * 1024 * 1024)},
        )
    # Should be 401/403 (auth required), not 413
    assert response.status_code != 413


# ---------------------------------------------------------------------------
# Search endpoint auth requirements
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_requires_auth():
    """POST /api/v1/search/classify should reject unauthenticated requests."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/search/classify",
            json={"text": "some report text"},
        )
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# ClassifyRequest schema validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_rejects_empty_text():
    """ClassifyRequest with empty text should return 422 (min_length=1)."""
    _override_auth(_user())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search/classify",
                json={"text": ""},
            )
        assert response.status_code == 422
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_classify_rejects_oversized_text():
    """ClassifyRequest with text > 50,000 chars should return 422."""
    _override_auth(_user())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search/classify",
                json={"text": "x" * 50_001},
            )
        assert response.status_code == 422
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# IngestionJobReview schema validation
# ---------------------------------------------------------------------------


def _reviewer(email: str = "reviewer@ucdavis.edu") -> CurrentUser:
    return CurrentUser(
        sub="sub-reviewer",
        email=email,
        is_admin=False,
        is_uploader=False,
        is_reviewer=True,
    )


def _override_reviewer(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[require_reviewer] = _f


@pytest.mark.asyncio
async def test_review_rejects_invalid_action():
    """IngestionJobReview with action != approve|reject should return 422."""
    _override_reviewer(_reviewer())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest/jobs/1/review",
                json={"action": "invalid_action"},
            )
        assert response.status_code == 422
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_review_rejects_oversized_rejection_reason():
    """IngestionJobReview with rejection_reason > 2000 chars should return 422."""
    _override_reviewer(_reviewer())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest/jobs/1/review",
                json={"action": "reject", "rejection_reason": "x" * 2001},
            )
        assert response.status_code == 422
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# LIKE wildcard escaping (unit test — no HTTP)
# ---------------------------------------------------------------------------


def test_escape_like_percent():
    assert _escape_like("100%") == "100\\%"


def test_escape_like_underscore():
    assert _escape_like("test_value") == "test\\_value"


def test_escape_like_backslash():
    assert _escape_like("path\\to") == "path\\\\to"


def test_escape_like_no_special_chars():
    assert _escape_like("normal text") == "normal text"


def test_escape_like_combined():
    assert _escape_like("50%_off\\sale") == "50\\%\\_off\\\\sale"


# ---------------------------------------------------------------------------
# Error message sanitization (unit test — no HTTP)
# ---------------------------------------------------------------------------


def test_safe_error_message_runtime_error():
    """RuntimeError messages (which we control) are passed through with truncation."""
    e = RuntimeError("ML worker returned 500: internal error")
    assert _safe_error_message(e) == "ML worker returned 500: internal error"


def test_safe_error_message_runtime_error_truncation():
    """RuntimeError messages are truncated to 500 chars."""
    long_msg = "x" * 600
    e = RuntimeError(long_msg)
    result = _safe_error_message(e)
    assert len(result) == 500


def test_safe_error_message_other_exception():
    """Non-RuntimeError exceptions only expose the class name."""
    e = ConnectionError("postgresql://user:pass@host/db refused")
    assert _safe_error_message(e) == "ConnectionError"


def test_safe_error_message_value_error():
    """ValueError should not leak its message."""
    e = ValueError("/app/uploads/../../etc/passwd")
    assert _safe_error_message(e) == "ValueError"
