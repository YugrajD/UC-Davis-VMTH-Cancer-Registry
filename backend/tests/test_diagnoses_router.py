"""Unit tests for the diagnoses-review router.

Covers _fetch_report_text fallback behaviour and the list_diagnoses
endpoint (status filter, role-based scoping).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, get_current_user
from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uploader(sub: str = "sub-u", email: str = "uploader@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub=sub, email=email, is_admin=False, is_uploader=True, is_reviewer=False)


def _reviewer(email: str = "reviewer@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub="sub-r", email=email, is_admin=False, is_uploader=True, is_reviewer=True)


def _admin(email: str = "admin@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub="sub-a", email=email, is_admin=True, is_uploader=True, is_reviewer=True)


def _non_uploader() -> CurrentUser:
    return CurrentUser(sub="sub-n", email="nobody@ucdavis.edu", is_admin=False, is_uploader=False, is_reviewer=False)


def _override_user(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[get_current_user] = _f


def _override_db(rows):
    mock_db = AsyncMock()
    result = MagicMock()
    result.all.return_value = rows
    mock_db.execute.return_value = result

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override
    return mock_db


def _cleanup():
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# _fetch_report_text — unit tests (import the helper directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_report_text_returns_none_when_no_pathology_report():
    from app.routers.diagnoses_review import _fetch_report_text

    diag = SimpleNamespace(pathology_report=None)
    result = await _fetch_report_text(diag)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_report_text_returns_none_when_gcs_path_missing():
    from app.routers.diagnoses_review import _fetch_report_text

    diag = SimpleNamespace(pathology_report=SimpleNamespace(gcs_path=None))
    result = await _fetch_report_text(diag)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_report_text_returns_none_when_gcs_bucket_not_configured(monkeypatch):
    from app.routers.diagnoses_review import _fetch_report_text
    from app.config import settings

    monkeypatch.setattr(settings, "GCS_BUCKET", "")
    diag = SimpleNamespace(pathology_report=SimpleNamespace(gcs_path="reports/1/ID_1.txt"))
    result = await _fetch_report_text(diag)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_report_text_downloads_from_gcs(monkeypatch):
    from app.routers.diagnoses_review import _fetch_report_text
    from app.config import settings

    monkeypatch.setattr(settings, "GCS_BUCKET", "my-bucket")
    diag = SimpleNamespace(pathology_report=SimpleNamespace(gcs_path="reports/1/ID_1.txt"))

    with patch("app.services.gcp_batch_service.download_report_text_from_gcs", return_value="report body") as mock_dl:
        result = await _fetch_report_text(diag)

    assert result == "report body"
    mock_dl.assert_called_once_with("reports/1/ID_1.txt")


@pytest.mark.asyncio
async def test_fetch_report_text_returns_none_on_gcs_error(monkeypatch):
    from app.routers.diagnoses_review import _fetch_report_text
    from app.config import settings

    monkeypatch.setattr(settings, "GCS_BUCKET", "my-bucket")
    diag = SimpleNamespace(pathology_report=SimpleNamespace(gcs_path="reports/1/ID_1.txt"))

    with patch("app.services.gcp_batch_service.download_report_text_from_gcs", side_effect=Exception("GCS error")):
        result = await _fetch_report_text(diag)

    assert result is None


# ---------------------------------------------------------------------------
# GET /api/v1/diagnoses — list_diagnoses endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_diagnoses_requires_uploader_role():
    _override_user(_non_uploader())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_list_diagnoses_returns_200_for_uploader():
    _override_user(_uploader())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses")
        assert r.status_code == 200
        assert r.json() == []
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_list_diagnoses_ignores_invalid_status_filter():
    """An unrecognised status value should not be applied as a filter (silently ignored)."""
    _override_user(_uploader())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses?status=invalid_status")
        # Should still return 200 — the invalid status is dropped, not rejected.
        assert r.status_code == 200
    finally:
        _cleanup()


@pytest.mark.parametrize("status", ["pending", "confirmed", "corrected", "rejected"])
@pytest.mark.asyncio
async def test_list_diagnoses_accepts_valid_status_filters(status):
    _override_user(_uploader())
    _override_db([])
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(f"/api/v1/diagnoses?status={status}")
        assert r.status_code == 200
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_list_diagnoses_requires_auth():
    """Without overriding get_current_user the call should fail auth."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/diagnoses")
    assert r.status_code in (401, 403)
