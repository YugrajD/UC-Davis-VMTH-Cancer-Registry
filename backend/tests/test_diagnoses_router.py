"""Unit tests for the retraining CSV exports — the only thing left in
diagnoses_review.py now that the Review Queue is retired (the Audit Worklist
replaces it; see backend/app/routers/audit_review.py).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import CurrentUser, get_current_user
from app.database import get_db
from app.main import app


def _admin() -> CurrentUser:
    return CurrentUser(sub="sub-a", email="admin@ucdavis.edu", is_admin=True, is_uploader=True, is_reviewer=True)


def _uploader() -> CurrentUser:
    return CurrentUser(sub="sub-u", email="uploader@ucdavis.edu", is_admin=False, is_uploader=True, is_reviewer=False)


def _override_user(user: CurrentUser):
    async def _f():
        return user
    app.dependency_overrides[get_current_user] = _f


def _cleanup():
    app.dependency_overrides.clear()


def _row(anon_id="CASE-0001", diagnosis_index=1, source_diagnosis="Skin mass", cancer_type="Mast cell neoplasms", icd_o_code="9740/3"):
    return MagicMock(
        anon_id=anon_id, diagnosis_index=diagnosis_index, source_diagnosis=source_diagnosis,
        cancer_type=cancer_type, icd_o_code=icd_o_code,
    )


@pytest.mark.asyncio
async def test_export_audited_requires_admin():
    _override_user(_uploader())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses/export/audited.csv")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_export_all_requires_admin():
    _override_user(_uploader())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses/export/all.csv")
        assert r.status_code == 403
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_export_audited_returns_expected_csv():
    _override_user(_admin())
    mock_db = AsyncMock()
    result = MagicMock()
    result.all.return_value = [_row()]
    mock_db.execute.return_value = result

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses/export/audited.csv")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "audited_diagnoses.csv" in r.headers["content-disposition"]
        lines = r.text.strip("\r\n").split("\r\n")
        assert lines[0] == "case_id,diagnosis_index,clinical_diagnosis,cancer_type,icd_o_code"
        assert lines[1] == "CASE-0001,1,Skin mass,Mast cell neoplasms,9740/3"
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_export_all_returns_expected_csv():
    _override_user(_admin())
    mock_db = AsyncMock()
    result = MagicMock()
    result.all.return_value = [_row(anon_id="CASE-0002")]
    mock_db.execute.return_value = result

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses/export/all.csv")
        assert r.status_code == 200
        assert "all_diagnoses.csv" in r.headers["content-disposition"]
        assert "CASE-0002" in r.text
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_export_defuses_csv_formula_injection():
    _override_user(_admin())
    mock_db = AsyncMock()
    result = MagicMock()
    result.all.return_value = [_row(source_diagnosis="=cmd|'/c calc'!A1")]
    mock_db.execute.return_value = result

    async def override():
        yield mock_db
    app.dependency_overrides[get_db] = override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/diagnoses/export/all.csv")
        assert r.status_code == 200
        assert "\t=cmd" in r.text
    finally:
        _cleanup()
