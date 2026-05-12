"""Unit tests for GET /api/v1/dashboard/summary and /filters."""

import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database import get_db
from tests.conftest import (
    scalar_result, one_result, all_result, first_result, scalars_result,
    make_species, make_cancer_type, make_county, make_breed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary_db(
    total_cases=395,
    total_counties=16,
    year_min=2018.0,
    year_max=2024.0,
    species_rows=None,
    cancer_rows=None,
    top_county_row=("Yolo", 80),
):
    """Return a mock DB session pre-configured for the summary endpoint."""
    if species_rows is None:
        species_rows = [("Dog", total_cases)]
    if cancer_rows is None:
        cancer_rows = [("Lymphoma", 150), ("Mast Cell Tumor", 100)]

    mock_db = AsyncMock()
    mock_db.execute.side_effect = [
        scalar_result(total_cases),             # total_cases
        scalar_result(total_counties),          # total_counties
        one_result((year_min, year_max)),       # year_range
        all_result(species_rows),               # species_breakdown
        all_result(cancer_rows),                # top_cancers
        first_result(top_county_row),           # top_county
    ]
    return mock_db


def _filters_db():
    """Return a mock DB session pre-configured for the filters endpoint."""
    mock_db = AsyncMock()
    mock_db.execute.side_effect = [
        scalars_result([make_species()]),
        scalars_result([make_cancer_type()]),
        scalars_result([make_county()]),
        scalars_result([make_breed()]),
        one_result((2018.0, 2024.0)),           # year_range
    ]
    return mock_db


# ---------------------------------------------------------------------------
# /summary tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_returns_200():
    app.dependency_overrides[get_db] = lambda: _summary_db().__aiter__()

    async def override():
        yield _summary_db()

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/dashboard/summary")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_summary_has_all_required_fields():
    async def override():
        yield _summary_db()

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/summary")).json()

    app.dependency_overrides.clear()

    assert "total_cases" in data
    assert "total_patients" in data
    assert "total_counties" in data
    assert "year_range" in data
    assert "species_breakdown" in data
    assert "top_cancers" in data
    assert "top_county" in data
    assert "top_county_cases" in data


@pytest.mark.asyncio
async def test_summary_values_match_db():
    async def override():
        yield _summary_db(total_cases=100, total_counties=5, top_county_row=("Sacramento", 30))

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/summary")).json()

    app.dependency_overrides.clear()

    assert data["total_cases"] == 100
    assert data["total_counties"] == 5
    assert data["top_county"] == "Sacramento"
    assert data["top_county_cases"] == 30


@pytest.mark.asyncio
async def test_summary_year_range_is_list_of_two_ints():
    async def override():
        yield _summary_db(year_min=2019.0, year_max=2023.0)

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/summary")).json()

    app.dependency_overrides.clear()

    assert data["year_range"] == [2019, 2023]


@pytest.mark.asyncio
async def test_summary_species_percentage_sums_to_100():
    species_rows = [("Dog", 300), ("Cat", 100)]

    async def override():
        yield _summary_db(total_cases=400, species_rows=species_rows)

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/summary")).json()

    app.dependency_overrides.clear()

    total_pct = sum(s["percentage"] for s in data["species_breakdown"])
    assert abs(total_pct - 100.0) < 0.2   # allow small rounding error


@pytest.mark.asyncio
async def test_summary_empty_db_returns_defaults():
    """When DB has no data, endpoint returns zero counts and default year range."""
    async def override():
        yield _summary_db(
            total_cases=0,
            total_counties=0,
            year_min=None,
            year_max=None,
            species_rows=[],
            cancer_rows=[],
            top_county_row=None,
        )

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/summary")).json()

    app.dependency_overrides.clear()

    assert data["total_cases"] == 0
    assert data["total_counties"] == 0
    assert data["top_county"] == "Unknown"
    assert data["top_county_cases"] == 0
    assert data["year_range"] == [2015, 2024]   # fallback defaults


@pytest.mark.asyncio
async def test_summary_top_cancers_ordered():
    cancer_rows = [("Lymphoma", 200), ("Mast Cell Tumor", 100), ("Osteosarcoma", 50)]

    async def override():
        yield _summary_db(cancer_rows=cancer_rows)

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/summary")).json()

    app.dependency_overrides.clear()

    counts = [ct["count"] for ct in data["top_cancers"]]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# /filters tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filters_returns_200():
    async def override():
        yield _filters_db()

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/dashboard/filters")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_filters_has_all_required_keys():
    async def override():
        yield _filters_db()

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/filters")).json()

    app.dependency_overrides.clear()

    assert "species" in data
    assert "cancer_types" in data
    assert "counties" in data
    assert "breeds" in data
    assert "year_range" in data


@pytest.mark.asyncio
async def test_filters_year_range_fallback_when_null():
    mock_db = AsyncMock()
    mock_db.execute.side_effect = [
        scalars_result([]),
        scalars_result([]),
        scalars_result([]),
        scalars_result([]),
        one_result((None, None)),   # no patients with dates
    ]

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/filters")).json()

    app.dependency_overrides.clear()

    assert data["year_range"] == [2015, 2024]


@pytest.mark.asyncio
async def test_filters_species_list_contents():
    mock_db = AsyncMock()
    mock_db.execute.side_effect = [
        scalars_result([make_species(1, "Dog"), make_species(2, "Cat")]),
        scalars_result([]),
        scalars_result([]),
        scalars_result([]),
        one_result((2020.0, 2024.0)),
    ]

    async def override():
        yield mock_db

    app.dependency_overrides[get_db] = override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/v1/dashboard/filters")).json()

    app.dependency_overrides.clear()

    names = [s["name"] for s in data["species"]]
    assert "Dog" in names
    assert "Cat" in names
