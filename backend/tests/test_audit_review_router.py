"""Unit tests for the audit-review router's audit-list import endpoint.

Covers filename parsing, manifest/sha256 verification, and the happy path
(replacing the active list + reporting unmatched case IDs) without making
real network or DB calls.
"""

import hashlib
import json

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, get_current_user
from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admin(email: str = "admin@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub="sub-a", email=email, is_admin=True, is_uploader=True, is_reviewer=True)


def _reviewer(email: str = "reviewer@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub="sub-r", email=email, is_admin=False, is_uploader=True, is_reviewer=True)


def _override_user(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[get_current_user] = _f


def _cleanup():
    app.dependency_overrides.clear()


def _manifest(sha256: str, *, kind: str = "audit_list", schema_version: int = 1) -> bytes:
    return json.dumps({
        "kind": kind,
        "schema_version": schema_version,
        "sha256": sha256,
        "written_at": "2026-09-27T00:00:00+00:00",
    }).encode("utf-8")


def _files(list_bytes: bytes, manifest_bytes: bytes, *, filename: str = "audit_list_2026-09-27-2.txt"):
    return {
        "list_file": (filename, list_bytes, "text/plain"),
        "manifest_file": (filename + ".manifest.json", manifest_bytes, "application/json"),
    }


# ---------------------------------------------------------------------------
# POST /api/v1/audit-review/lists/import
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_audit_list_requires_admin():
    _override_user(_reviewer())
    list_bytes = b"CASE-0001\nCASE-0002\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256)),
            )
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_bad_filename():
    _override_user(_admin())
    list_bytes = b"CASE-0001\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256), filename="cases.txt"),
            )
        assert r.status_code == 400
        assert "audit_list_" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_sha256_mismatch():
    _override_user(_admin())
    list_bytes = b"CASE-0001\n"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest("0" * 64)),
            )
        assert r.status_code == 400
        assert "sha256" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_wrong_manifest_kind():
    _override_user(_admin())
    list_bytes = b"CASE-0001\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256, kind="gold")),
            )
        assert r.status_code == 400
        assert "kind" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_wrong_schema_version():
    _override_user(_admin())
    list_bytes = b"CASE-0001\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256, schema_version=2)),
            )
        assert r.status_code == 400
        assert "schema_version" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_duplicate_case_ids():
    _override_user(_admin())
    list_bytes = b"CASE-0001\nCASE-0002\nCASE-0001\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256)),
            )
        assert r.status_code == 400
        assert "CASE-0001" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_empty_list():
    _override_user(_admin())
    list_bytes = b"\n\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256)),
            )
        assert r.status_code == 400
        assert "no case IDs" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_rejects_existing_list_id():
    _override_user(_admin())
    list_bytes = b"CASE-0001\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()

    mock_db = AsyncMock()
    existing_result = MagicMock()
    existing_result.scalar_one_or_none.return_value = 1  # a row already exists
    mock_db.execute.side_effect = [existing_result]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256)),
            )
        assert r.status_code == 409
        assert "already imported" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_audit_list_replaces_active_list_and_reports_not_found():
    _override_user(_admin())
    list_bytes = b"CASE-0001\nCASE-9999\n"
    sha256 = hashlib.sha256(list_bytes).hexdigest()

    mock_db = AsyncMock()
    existing_result = MagicMock()
    existing_result.scalar_one_or_none.return_value = None
    previous_active_result = MagicMock()
    previous_active_result.scalar_one_or_none.return_value = "2026-09-20-1"
    deactivate_result = MagicMock()
    found_result = MagicMock()
    found_result.scalars.return_value = ["CASE-0001"]
    mock_db.execute.side_effect = [
        existing_result, previous_active_result, deactivate_result, found_result,
    ]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/lists/import",
                files=_files(list_bytes, _manifest(sha256), filename="audit_list_2026-09-27-2.txt"),
            )
        assert r.status_code == 200
        body = r.json()
        assert body["list_id"] == "2026-09-27-2"
        assert body["case_count"] == 2
        assert body["replaced_list_id"] == "2026-09-20-1"
        assert body["not_found"] == ["CASE-9999"]
        mock_db.commit.assert_awaited_once()
        mock_db.flush.assert_awaited_once()
    finally:
        _cleanup()
