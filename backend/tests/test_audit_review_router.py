"""Unit tests for the audit-review router: audit-list import and the worklist.

Covers filename parsing, manifest/sha256 verification, the import happy path
(replacing the active list + reporting unmatched case IDs), and the worklist's
status merge (unreviewed/reviewed/locked, patient_found, code_count) — all
without making real network or DB calls.
"""

import hashlib
import json
from datetime import datetime, timezone

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, get_current_user
from app.database import get_db
from app.main import app
from app.models.models import AuditList, AuditListCase, CaseReview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admin(email: str = "admin@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub="sub-a", email=email, is_admin=True, is_uploader=True, is_reviewer=True)


def _reviewer(email: str = "reviewer@ucdavis.edu") -> CurrentUser:
    return CurrentUser(sub="sub-r", email=email, is_admin=False, is_uploader=True, is_reviewer=True)


def _non_reviewer() -> CurrentUser:
    return CurrentUser(sub="sub-n", email="nobody@ucdavis.edu", is_admin=False, is_uploader=True, is_reviewer=False)


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


# ---------------------------------------------------------------------------
# GET /api/v1/audit-review/worklist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worklist_requires_reviewer():
    _override_user(_non_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/worklist")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_worklist_no_active_list_returns_empty():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    no_active_result = MagicMock()
    no_active_result.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [no_active_result]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/worklist")
        assert r.status_code == 200
        body = r.json()
        assert body == {"list_id": None, "imported_at": None, "case_count": 0, "cases": []}
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_worklist_merges_review_status_patient_match_and_code_count():
    _override_user(_reviewer())

    active_list = AuditList(
        id=1, list_id="2026-09-27-2", imported_by_email="admin@ucdavis.edu",
        sha256="a" * 64, case_count=3, is_active=True,
        imported_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
    )
    entries = [
        AuditListCase(id=1, audit_list_id=1, case_id="CASE-0001", position=0),
        AuditListCase(id=2, audit_list_id=1, case_id="CASE-0002", position=1),
        AuditListCase(id=3, audit_list_id=1, case_id="CASE-9999", position=2),
    ]
    reviewed_unlocked = CaseReview(
        id=10, case_id="CASE-0002", no_cancer=False, reviewed_by_email="dr.smith@ucdavis.edu",
        reviewed_at=datetime(2026, 9, 27, tzinfo=timezone.utc), locked=False,
    )
    reviewed_locked = CaseReview(
        id=20, case_id="CASE-9999", no_cancer=True, reviewed_by_email="dr.jones@ucdavis.edu",
        reviewed_at=datetime(2026, 9, 27, tzinfo=timezone.utc), locked=True,
    )

    mock_db = AsyncMock()
    active_list_result = MagicMock()
    active_list_result.scalar_one_or_none.return_value = active_list
    entries_result = MagicMock()
    entries_result.scalars.return_value.all.return_value = entries
    reviews_result = MagicMock()
    reviews_result.scalars.return_value.all.return_value = [reviewed_unlocked, reviewed_locked]
    code_counts_result = MagicMock()
    code_counts_result.all.return_value = [(10, 2)]  # reviewed_locked (no_cancer) has zero codes
    patients_result = MagicMock()
    patients_result.scalars.return_value = ["CASE-0001", "CASE-0002"]  # CASE-9999 has no matching patient
    mock_db.execute.side_effect = [
        active_list_result, entries_result, reviews_result, code_counts_result, patients_result,
    ]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/worklist")
        assert r.status_code == 200
        body = r.json()
        assert body["list_id"] == "2026-09-27-2"
        assert body["case_count"] == 3
        assert len(body["cases"]) == 3

        unreviewed = body["cases"][0]
        assert unreviewed["case_id"] == "CASE-0001"
        assert unreviewed["review_status"] == "unreviewed"
        assert unreviewed["patient_found"] is True
        assert unreviewed["no_cancer"] is None
        assert unreviewed["code_count"] == 0

        reviewed = body["cases"][1]
        assert reviewed["case_id"] == "CASE-0002"
        assert reviewed["review_status"] == "reviewed"
        assert reviewed["patient_found"] is True
        assert reviewed["no_cancer"] is False
        assert reviewed["code_count"] == 2
        assert reviewed["reviewed_by_email"] == "dr.smith@ucdavis.edu"

        locked = body["cases"][2]
        assert locked["case_id"] == "CASE-9999"
        assert locked["review_status"] == "locked"
        assert locked["patient_found"] is False
        assert locked["no_cancer"] is True
        assert locked["code_count"] == 0
    finally:
        _cleanup()
