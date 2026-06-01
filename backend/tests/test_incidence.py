"""Unit tests for GET /api/v1/incidence/* endpoints."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database import get_db
from app.routers.incidence import SEX_MAP
from tests.conftest import all_result, scalar_result


# ---------------------------------------------------------------------------
# Row factory
# ---------------------------------------------------------------------------

def row(**kwargs):
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# /incidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_incidence_returns_200():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/incidence")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_incidence_schema():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([
        row(cancer_type="Lymphoma", county="Yolo", species="Dog", year=2022.0, count=10),
    ])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/incidence")).json()

    app.dependency_overrides.clear()

    assert "data" in data
    assert "total" in data
    assert "filters_applied" in data
    assert len(data["data"]) == 1
    record = data["data"][0]
    assert record["cancer_type"] == "Lymphoma"
    assert record["county"] == "Yolo"
    assert record["year"] == 2022


@pytest.mark.asyncio
async def test_incidence_total_is_sum_of_counts():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([
        row(cancer_type="Lymphoma", county="Yolo", species="Dog", year=2022.0, count=10),
        row(cancer_type="Osteosarcoma", county="Sacramento", species="Dog", year=2021.0, count=5),
    ])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/incidence")).json()

    app.dependency_overrides.clear()
    assert data["total"] == 15


@pytest.mark.asyncio
async def test_incidence_empty_result():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/incidence")).json()

    app.dependency_overrides.clear()
    assert data["data"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_incidence_filters_echoed_in_response():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get(
            "/api/v1/incidence?species=Dog&year_start=2020&year_end=2023"
        )).json()

    app.dependency_overrides.clear()

    filters = data["filters_applied"]
    assert filters["species"] == ["Dog"]
    assert filters["year_start"] == 2020
    assert filters["year_end"] == 2023


# ---------------------------------------------------------------------------
# /incidence/by-cancer-type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_by_cancer_type_returns_200():
    mock_db = AsyncMock()
    # Two DB calls: (1) scalar denominator, (2) per-type numerator rows
    mock_db.execute.side_effect = [
        scalar_result(0),
        all_result([]),
    ]

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/incidence/by-cancer-type")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_by_cancer_type_schema():
    mock_db = AsyncMock()
    # Two DB calls: (1) scalar denominator = 100 patients, (2) per-type rows
    mock_db.execute.side_effect = [
        scalar_result(100),
        all_result([
            row(cancer_type="Lymphoma", count=50),
            row(cancer_type="Mast Cell Tumor", count=30),
        ]),
    ]

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/incidence/by-cancer-type")).json()

    app.dependency_overrides.clear()

    assert data["total"] == 100  # total_patients (denominator)
    assert data["data"][0]["cancer_type"] == "Lymphoma"
    assert data["data"][0]["count"] == 50
    assert data["data"][0]["pccp"] == 50.0  # 50/100 * 100


# ---------------------------------------------------------------------------
# /incidence/by-species
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_by_species_sets_all_as_cancer_type():
    """The by-species endpoint sets cancer_type='All' for each record."""
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([
        row(species="Dog", count=300),
    ])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/incidence/by-species")).json()

    app.dependency_overrides.clear()

    assert data["data"][0]["cancer_type"] == "All"
    assert data["data"][0]["species"] == "Dog"


# ---------------------------------------------------------------------------
# /incidence/by-breed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_by_breed_returns_200():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/incidence/by-breed")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_by_breed_schema():
    mock_db = AsyncMock()
    mock_db.execute.return_value = all_result([
        row(breed="Labrador Retriever", species="Dog", count=25),
    ])

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/incidence/by-breed")).json()

    app.dependency_overrides.clear()

    record = data["data"][0]
    assert record["breed"] == "Labrador Retriever"
    assert record["species"] == "Dog"
    assert record["count"] == 25


# ---------------------------------------------------------------------------
# /incidence/breed-detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_breed_detail_requires_breed_param():
    mock_db = AsyncMock()

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/incidence/breed-detail")

    app.dependency_overrides.clear()
    assert response.status_code == 422   # missing required query param


@pytest.mark.asyncio
async def test_breed_detail_schema():
    mock_db = AsyncMock()
    mock_db.execute.side_effect = [
        scalar_result(10),
        all_result([row(sex="Male", count=6), row(sex="Female", count=4)]),
        all_result([row(cancer_type="Lymphoma", count=10)]),
        all_result([row(county_name="Yolo", fips_code="06113", count=5)]),
    ]

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get(
            "/api/v1/incidence/breed-detail?breed=Labrador+Retriever"
        )).json()

    app.dependency_overrides.clear()

    assert data["breed"] == "Labrador Retriever"
    assert data["total_cases"] == 10
    assert len(data["sex_breakdown"]) == 2
    assert data["cancer_types"][0]["cancer_type"] == "Lymphoma"
    assert data["county_cases"][0]["fips_code"] == "06113"


# ---------------------------------------------------------------------------
# SEX_MAP unit tests (no DB required)
# ---------------------------------------------------------------------------

def test_sex_map_covers_all_frontend_values():
    """All frontend sex filter values map to valid DB values."""
    frontend_values = {"male_intact", "male_neutered", "female_intact", "female_spayed"}
    assert set(SEX_MAP.keys()) == frontend_values


def test_sex_map_db_values():
    assert SEX_MAP["male_intact"] == "Male"
    assert SEX_MAP["male_neutered"] == "Neutered Male"
    assert SEX_MAP["female_intact"] == "Female"
    assert SEX_MAP["female_spayed"] == "Spayed Female"
