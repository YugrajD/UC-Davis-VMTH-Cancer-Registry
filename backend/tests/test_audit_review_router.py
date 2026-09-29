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
from app.models.models import (
    AuditList,
    AuditListCase,
    CancerType,
    CaseDiagnosis,
    CaseReview,
    CaseReviewCode,
    GoldExport,
    Patient,
    PathologyReport,
    TaxonomyTerm,
)


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


# ---------------------------------------------------------------------------
# GET /api/v1/audit-review/taxonomy-terms
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_taxonomy_terms_requires_reviewer():
    _override_user(_non_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/taxonomy-terms")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_taxonomy_terms_returns_rows():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    terms_result = MagicMock()
    terms_result.scalars.return_value.all.return_value = [
        TaxonomyTerm(
            id=1, vet_icd_o_code="9590/3", taxonomy_group="Malignant lymphomas",
            taxonomy_term="Malignant lymphoma, NOS", term_level="Preferred",
        ),
        TaxonomyTerm(
            id=2, vet_icd_o_code="8050/3", taxonomy_group="Epithelial neoplasms, NOS",
            taxonomy_term="Papillary adenocarcinoma", term_level="Preferred",
        ),
    ]
    mock_db.execute.side_effect = [terms_result]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/taxonomy-terms")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        assert body[0] == {
            "vet_icd_o_code": "9590/3", "taxonomy_group": "Malignant lymphomas",
            "taxonomy_term": "Malignant lymphoma, NOS", "term_level": "Preferred",
        }
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# GET /api/v1/audit-review/cases/{case_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_case_detail_requires_reviewer():
    _override_user(_non_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/cases/CASE-0001")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_case_detail_404_when_never_on_a_list():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    not_on_list_result = MagicMock()
    not_on_list_result.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [not_on_list_result]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/cases/CASE-0001")
        assert r.status_code == 404
        assert "never on an audit list" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_case_detail_patient_not_found_still_returns_200():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    on_list_result = MagicMock()
    on_list_result.scalar_one_or_none.return_value = 1
    patient_result = MagicMock()
    patient_result.scalar_one_or_none.return_value = None
    review_result = MagicMock()
    review_result.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [on_list_result, patient_result, review_result]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/cases/CASE-9999")
        assert r.status_code == 200
        body = r.json()
        assert body["patient_found"] is False
        assert body["predicted_codes"] == []
        assert body["source_diagnosis"] is None
        assert body["review_exists"] is False
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_case_detail_happy_path_with_predictions_and_existing_review():
    _override_user(_reviewer())

    patient = Patient(id=1, anon_id="CASE-0001", data_source="petbert")
    report = PathologyReport(id=1, patient_id=1, gcs_path=None, source_diagnosis="Mast cell tumor, skin mass")
    diag = CaseDiagnosis(
        id=1, patient_id=1, cancer_type_id=1, icd_o_code="9740/3", predicted_term="Mast cell tumor, NOS",
        confidence=0.91, prediction_method="embedding", diagnosis_index=1,
    )
    review = CaseReview(
        id=10, case_id="CASE-0001", no_cancer=False, reviewed_by_email="dr.smith@ucdavis.edu",
        locked=False,
    )
    review_code = CaseReviewCode(
        id=1, case_review_id=10, taxonomy_group="Mast cell neoplasms",
        taxonomy_term="Cutaneous mast cell tumor grade Patnaik II",
    )

    mock_db = AsyncMock()
    on_list_result = MagicMock()
    on_list_result.scalar_one_or_none.return_value = 1
    patient_result = MagicMock()
    patient_result.scalar_one_or_none.return_value = patient
    report_result = MagicMock()
    report_result.scalar_one_or_none.return_value = report
    diag_result = MagicMock()
    diag_result.all.return_value = [(diag, "Mast cell neoplasms")]
    review_result = MagicMock()
    review_result.scalar_one_or_none.return_value = review
    review_codes_result = MagicMock()
    review_codes_result.scalars.return_value.all.return_value = [review_code]
    mock_db.execute.side_effect = [
        on_list_result, patient_result, report_result, diag_result, review_result, review_codes_result,
    ]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/cases/CASE-0001")
        assert r.status_code == 200
        body = r.json()
        assert body["patient_found"] is True
        assert body["source_diagnosis"] == "Mast cell tumor, skin mass"
        assert body["report_text"] is None  # gcs_path unset — no GCS call attempted
        assert len(body["predicted_codes"]) == 1
        assert body["predicted_codes"][0]["cancer_type_name"] == "Mast cell neoplasms"
        assert body["predicted_codes"][0]["icd_o_code"] == "9740/3"
        assert body["review_exists"] is True
        assert body["review_no_cancer"] is False
        assert body["review_locked"] is False
        assert len(body["review_codes"]) == 1
        assert body["review_codes"][0]["taxonomy_term"] == "Cutaneous mast cell tumor grade Patnaik II"
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# POST /api/v1/audit-review/cases/{case_id}/review
# ---------------------------------------------------------------------------


def _one_code():
    return {"taxonomy_group": "Mast cell neoplasms", "taxonomy_term": "Cutaneous mast cell tumor grade Patnaik II"}


@pytest.mark.asyncio
async def test_save_review_requires_reviewer():
    _override_user(_non_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": True, "codes": []},
            )
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_404_when_never_on_a_list():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    not_on_list = MagicMock()
    not_on_list.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [not_on_list]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": True, "codes": []},
            )
        assert r.status_code == 404
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_rejects_no_cancer_with_codes():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    mock_db.execute.side_effect = [on_list]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": True, "codes": [_one_code()]},
            )
        assert r.status_code == 400
        assert "cannot carry any codes" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_rejects_empty_codes_without_no_cancer():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    mock_db.execute.side_effect = [on_list]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": False, "codes": []},
            )
        assert r.status_code == 400
        assert "At least one code" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_rejects_duplicate_codes():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    mock_db.execute.side_effect = [on_list]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": False, "codes": [_one_code(), _one_code()]},
            )
        assert r.status_code == 400
        assert "Duplicate code" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_rejects_codes_not_in_taxonomy():
    _override_user(_reviewer())
    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    taxonomy_check = MagicMock()
    taxonomy_check.all.return_value = []  # nothing matches — the submitted pair is invalid
    mock_db.execute.side_effect = [on_list, taxonomy_check]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": False, "codes": [{"taxonomy_group": "Bogus", "taxonomy_term": "Not real"}]},
            )
        assert r.status_code == 400
        assert "Not in the taxonomy" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_rejects_locked_review():
    _override_user(_reviewer())
    locked_review = CaseReview(id=1, case_id="CASE-0001", no_cancer=False, locked=True, reviewed_by_email="dr.smith@ucdavis.edu")

    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    taxonomy_check = MagicMock()
    taxonomy_check.all.return_value = [("Mast cell neoplasms", "Cutaneous mast cell tumor grade Patnaik II")]
    existing_review = MagicMock()
    existing_review.scalar_one_or_none.return_value = locked_review
    mock_db.execute.side_effect = [on_list, taxonomy_check, existing_review]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": False, "codes": [_one_code()]},
            )
        assert r.status_code == 409
        assert "locked" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_creates_new_no_cancer_review():
    _override_user(_reviewer(email="dr.jones@ucdavis.edu"))
    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    no_existing_review = MagicMock()
    no_existing_review.scalar_one_or_none.return_value = None
    # No taxonomy check — codes is empty, _invalid_taxonomy_pairs short-circuits.
    mock_db.execute.side_effect = [on_list, no_existing_review]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": True, "codes": []},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["no_cancer"] is True
        assert body["code_count"] == 0
        assert body["reviewed_by_email"] == "dr.jones@ucdavis.edu"
        mock_db.commit.assert_awaited_once()
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_save_review_replaces_existing_unlocked_review():
    _override_user(_reviewer(email="dr.jones@ucdavis.edu"))
    existing = CaseReview(id=1, case_id="CASE-0001", no_cancer=True, locked=False, reviewed_by_email="dr.smith@ucdavis.edu")

    mock_db = AsyncMock()
    on_list = MagicMock()
    on_list.scalar_one_or_none.return_value = 1
    taxonomy_check = MagicMock()
    taxonomy_check.all.return_value = [("Mast cell neoplasms", "Cutaneous mast cell tumor grade Patnaik II")]
    existing_review = MagicMock()
    existing_review.scalar_one_or_none.return_value = existing
    delete_codes = MagicMock()
    mock_db.execute.side_effect = [on_list, taxonomy_check, existing_review, delete_codes]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/cases/CASE-0001/review",
                json={"no_cancer": False, "codes": [_one_code()]},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["no_cancer"] is False
        assert body["code_count"] == 1
        assert body["reviewed_by_email"] == "dr.jones@ucdavis.edu"  # last editor wins
        mock_db.commit.assert_awaited_once()
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# POST /api/v1/audit-review/cases/{case_id}/reopen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reopen_requires_admin():
    _override_user(_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/audit-review/cases/CASE-0001/reopen")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_reopen_404_when_no_review():
    _override_user(_admin())
    mock_db = AsyncMock()
    no_review = MagicMock()
    no_review.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [no_review]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/audit-review/cases/CASE-0001/reopen")
        assert r.status_code == 404
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_reopen_400_when_not_locked():
    _override_user(_admin())
    review = CaseReview(id=1, case_id="CASE-0001", locked=False)
    mock_db = AsyncMock()
    found = MagicMock()
    found.scalar_one_or_none.return_value = review
    mock_db.execute.side_effect = [found]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/audit-review/cases/CASE-0001/reopen")
        assert r.status_code == 400
        assert "not locked" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_reopen_success_clears_lock():
    _override_user(_admin())
    review = CaseReview(id=1, case_id="CASE-0001", locked=True)
    mock_db = AsyncMock()
    found = MagicMock()
    found.scalar_one_or_none.return_value = review
    mock_db.execute.side_effect = [found]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/audit-review/cases/CASE-0001/reopen")
        assert r.status_code == 200
        assert r.json() == {"case_id": "CASE-0001", "locked": False}
        assert review.locked is False
        mock_db.commit.assert_awaited_once()
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# POST /api/v1/audit-review/gold-exports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_gold_export_requires_admin():
    _override_user(_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/gold-exports", json={"reviewer_email": "dr.smith@ucdavis.edu"}
            )
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_create_gold_export_400_when_nothing_eligible():
    _override_user(_admin())
    mock_db = AsyncMock()
    eligible = MagicMock()
    eligible.scalars.return_value.all.return_value = []
    mock_db.execute.side_effect = [eligible]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/gold-exports", json={"reviewer_email": "dr.smith@ucdavis.edu"}
            )
        assert r.status_code == 400
        assert "No unlocked reviews" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_create_gold_export_rejects_stale_taxonomy_codes():
    _override_user(_admin())
    review = CaseReview(id=1, case_id="CASE-0001", no_cancer=False, locked=False, reviewed_by_email="dr.smith@ucdavis.edu")
    code = CaseReviewCode(id=1, case_review_id=1, taxonomy_group="Retired Group", taxonomy_term="Retired Term")

    mock_db = AsyncMock()
    eligible = MagicMock()
    eligible.scalars.return_value.all.return_value = [review]
    codes_result = MagicMock()
    codes_result.scalars.return_value.all.return_value = [code]
    taxonomy_check = MagicMock()
    taxonomy_check.all.return_value = []  # the (group, term) no longer exists
    mock_db.execute.side_effect = [eligible, codes_result, taxonomy_check]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/gold-exports", json={"reviewer_email": "dr.smith@ucdavis.edu"}
            )
        assert r.status_code == 400
        assert "CASE-0001" in r.json()["detail"]
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_create_gold_export_locks_reviews_and_returns_summary():
    _override_user(_admin())
    review1 = CaseReview(id=1, case_id="CASE-0001", no_cancer=False, locked=False, reviewed_by_email="dr.smith@ucdavis.edu")
    review2 = CaseReview(id=2, case_id="CASE-0002", no_cancer=True, locked=False, reviewed_by_email="dr.smith@ucdavis.edu")
    code = CaseReviewCode(id=1, case_review_id=1, taxonomy_group="Mast cell neoplasms", taxonomy_term="Cutaneous mast cell tumor grade Patnaik II")

    mock_db = AsyncMock()
    eligible = MagicMock()
    eligible.scalars.return_value.all.return_value = [review1, review2]
    codes_result = MagicMock()
    codes_result.scalars.return_value.all.return_value = [code]
    taxonomy_check = MagicMock()
    taxonomy_check.all.return_value = [("Mast cell neoplasms", "Cutaneous mast cell tumor grade Patnaik II")]
    existing_export_ids = MagicMock()
    existing_export_ids.scalars.return_value.all.return_value = []
    mock_db.execute.side_effect = [eligible, codes_result, taxonomy_check, existing_export_ids]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/audit-review/gold-exports", json={"reviewer_email": "dr.smith@ucdavis.edu"}
            )
        assert r.status_code == 200
        body = r.json()
        assert body["reviewer_email"] == "dr.smith@ucdavis.edu"
        assert body["case_count"] == 2
        assert review1.locked is True
        assert review2.locked is True
        mock_db.commit.assert_awaited_once()
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# GET /api/v1/audit-review/gold-exports/{export_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_gold_export_requires_admin():
    _override_user(_reviewer())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/gold-exports/2026-09-27-1")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_download_gold_export_404_when_missing():
    _override_user(_admin())
    mock_db = AsyncMock()
    not_found = MagicMock()
    not_found.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [not_found]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/gold-exports/2026-09-27-1")
        assert r.status_code == 404
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_download_gold_export_builds_expected_csv():
    _override_user(_admin())
    export = GoldExport(id=1, export_id="2026-09-27-1", reviewer_email="dr.smith@ucdavis.edu", case_count=2)
    review1 = CaseReview(id=1, case_id="CASE-0001", no_cancer=False, locked=True)
    review2 = CaseReview(id=2, case_id="CASE-0002", no_cancer=True, locked=True)
    code1 = CaseReviewCode(id=1, case_review_id=1, taxonomy_group="Mast cell neoplasms", taxonomy_term="Cutaneous mast cell tumor grade Patnaik II")
    code2 = CaseReviewCode(id=2, case_review_id=1, taxonomy_group="Fibromatous neoplasms", taxonomy_term="Fibrosarcoma, NOS")

    mock_db = AsyncMock()
    export_result = MagicMock()
    export_result.scalar_one_or_none.return_value = export
    reviews_result = MagicMock()
    reviews_result.scalars.return_value.all.return_value = [review1, review2]
    codes_result = MagicMock()
    codes_result.scalars.return_value.all.return_value = [code1, code2]
    mock_db.execute.side_effect = [export_result, reviews_result, codes_result]

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/audit-review/gold-exports/2026-09-27-1")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "gold_2026-09-27-1.csv" in r.headers["content-disposition"]
        lines = r.text.strip("\n").split("\n")
        assert lines[0] == "case_id,term"
        assert lines[1] == "CASE-0001,Mast cell neoplasms: Cutaneous mast cell tumor grade Patnaik II"
        assert lines[2] == 'CASE-0001,"Fibromatous neoplasms: Fibrosarcoma, NOS"'
        assert lines[3] == "CASE-0002,NO_CANCER"
        assert "origin" not in lines[0]
    finally:
        _cleanup()
