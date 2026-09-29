"""Unit tests for the registry-import router: loading combined_predictions +
review_queue as the registry's code of record.

Covers manifest/sha256 verification, row-level validation (n_codes, NO_CANCER
exclusivity, code_source/review_status enums), and the happy path (coded
cases, a NO_CANCER case, an awaiting-review-only case, an unmatched case) —
all without making real network or DB calls.
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


def _reviewer() -> CurrentUser:
    return CurrentUser(sub="sub-r", email="reviewer@ucdavis.edu", is_admin=False, is_uploader=True, is_reviewer=True)


def _override_user(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[get_current_user] = _f


def _cleanup():
    app.dependency_overrides.clear()


def _manifest(sha256: str, *, kind: str, schema_version: int) -> bytes:
    return json.dumps({
        "kind": kind, "schema_version": schema_version, "sha256": sha256,
        "written_at": "2026-09-27T00:00:00+00:00",
    }).encode("utf-8")


_PREDICTIONS_HEADER = "case_id,code,term,group,code_source,source_version,source_confidence,review_status,n_codes\n"


def _predictions_manifest(raw: bytes, *, kind: str = "combined_predictions", schema_version: int = 2) -> bytes:
    return _manifest(hashlib.sha256(raw).hexdigest(), kind=kind, schema_version=schema_version)


def _queue_manifest(raw: bytes, *, kind: str = "review_queue", schema_version: int = 1) -> bytes:
    return _manifest(hashlib.sha256(raw).hexdigest(), kind=kind, schema_version=schema_version)


def _files(predictions: bytes, queue: bytes, *, predictions_manifest: bytes = None, queue_manifest: bytes = None):
    return {
        "predictions_file": ("combined_predictions_2026-09-27.csv", predictions, "text/csv"),
        "predictions_manifest_file": (
            "combined_predictions_2026-09-27.csv.manifest.json",
            predictions_manifest if predictions_manifest is not None else _predictions_manifest(predictions),
            "application/json",
        ),
        "queue_file": ("review_queue_2026-09-27.csv", queue, "text/csv"),
        "queue_manifest_file": (
            "review_queue_2026-09-27.csv.manifest.json",
            queue_manifest if queue_manifest is not None else _queue_manifest(queue),
            "application/json",
        ),
    }


_EMPTY_QUEUE = b"case_id\n"


# ---------------------------------------------------------------------------
# POST /api/v1/registry/combined-predictions/import
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_requires_admin():
    _override_user(_reviewer())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,NO_CANCER,,,diagnosis,silver-0,no_signal,auto_accepted,1\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE),
            )
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_predictions_sha256_mismatch():
    _override_user(_admin())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,NO_CANCER,,,diagnosis,silver-0,no_signal,auto_accepted,1\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE, predictions_manifest=_manifest("0" * 64, kind="combined_predictions", schema_version=2)),
            )
        assert r.status_code == 400
        assert "sha256" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_queue_sha256_mismatch():
    _override_user(_admin())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,NO_CANCER,,,diagnosis,silver-0,no_signal,auto_accepted,1\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE, queue_manifest=_manifest("0" * 64, kind="review_queue", schema_version=1)),
            )
        assert r.status_code == 400
        assert "sha256" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_wrong_predictions_schema_version():
    _override_user(_admin())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,NO_CANCER,,,diagnosis,silver-0,no_signal,auto_accepted,1\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE, predictions_manifest=_predictions_manifest(predictions, schema_version=1)),
            )
        assert r.status_code == 400
        assert "schema_version" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_missing_columns():
    _override_user(_admin())
    predictions = b"case_id,code\nCASE-0001,NO_CANCER\n"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE),
            )
        assert r.status_code == 400
        assert "missing column" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_n_codes_mismatch():
    _override_user(_admin())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,9740/3,MCT,Mast cell neoplasms,report,gen-0,0.8,auto_accepted,2\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE),
            )
        assert r.status_code == 400
        assert "n_codes mismatch" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_no_cancer_mixed_with_codes():
    _override_user(_admin())
    predictions = (
        _PREDICTIONS_HEADER
        + "CASE-0001,NO_CANCER,,,diagnosis,silver-0,no_signal,auto_accepted,2\n"
        + "CASE-0001,9740/3,MCT,Mast cell neoplasms,diagnosis,silver-0,tier1_exact,auto_accepted,2\n"
    ).encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE),
            )
        assert r.status_code == 400
        assert "NO_CANCER mixed" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_invalid_code_source():
    _override_user(_admin())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,9740/3,MCT,Mast cell neoplasms,bogus,gen-0,0.8,auto_accepted,1\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE),
            )
        assert r.status_code == 400
        assert "code_source" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_rejects_invalid_review_status():
    _override_user(_admin())
    predictions = (_PREDICTIONS_HEADER + "CASE-0001,9740/3,MCT,Mast cell neoplasms,report,gen-0,0.8,bogus,1\n").encode()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, _EMPTY_QUEUE),
            )
        assert r.status_code == 400
        assert "review_status" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_import_happy_path_codes_no_cancer_awaiting_and_not_found():
    _override_user(_admin())
    predictions = (
        _PREDICTIONS_HEADER
        # CASE-0001: two codes, both diagnosis/silver, confirmed
        + "CASE-0001,9740/3,\"Mast cell tumor, NOS\",Mast cell neoplasms,diagnosis,silver-0,tier1_exact,confirmed,2\n"
        + "CASE-0001,8810/3,\"Fibrosarcoma, NOS\",Fibromatous neoplasms,diagnosis,silver-0,tier1_exact,confirmed,2\n"
        # CASE-0002: NO_CANCER
        + "CASE-0002,NO_CANCER,,,diagnosis,silver-0,no_signal,auto_accepted,1\n"
        # CASE-9999: not a real patient
        + "CASE-9999,9740/3,MCT,Mast cell neoplasms,report,gen-0,0.8,queued,1\n"
    ).encode()
    queue = b"case_id\nCASE-0003\n"  # awaiting review, no combined-predictions row

    mock_db = AsyncMock()
    patients_result = MagicMock()
    patients_result.all.return_value = [
        MagicMock(id=1, anon_id="CASE-0001"),
        MagicMock(id=2, anon_id="CASE-0002"),
        MagicMock(id=3, anon_id="CASE-0003"),
    ]
    mast_cell_type = MagicMock(id=10)
    mast_cell_type.name = "Mast cell neoplasms"
    fibromatous_type = MagicMock(id=11)
    fibromatous_type.name = "Fibromatous neoplasms"
    cancer_types_result = MagicMock()
    cancer_types_result.all.return_value = [mast_cell_type, fibromatous_type]
    reports_result = MagicMock()
    reports_result.all.return_value = [MagicMock(patient_id=1, id=100)]
    delete_result = MagicMock()
    insert_result = MagicMock()
    no_cancer_update_result = MagicMock()
    coded_update_result = MagicMock()
    awaiting_update_result = MagicMock()
    refresh_result = MagicMock()
    mock_db.execute.side_effect = [
        patients_result, cancer_types_result, reports_result,
        delete_result, insert_result,
        no_cancer_update_result, coded_update_result, awaiting_update_result,
        refresh_result, refresh_result,
    ]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/registry/combined-predictions/import",
                files=_files(predictions, queue),
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cases_coded"] == 1  # CASE-0001 only (CASE-9999 unmatched)
        assert body["cases_no_cancer"] == 1  # CASE-0002
        assert body["cases_awaiting_review"] == 1  # CASE-0003
        assert body["not_found"] == ["CASE-9999"]
        mock_db.commit.assert_awaited_once()
    finally:
        _cleanup()
