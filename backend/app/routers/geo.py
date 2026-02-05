"""GeoJSON map endpoints using PostGIS spatial queries."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func
from typing import Optional, List
import json

from app.database import get_db
from app.models.models import County, CancerCase, CancerType, Patient, Species
from app.schemas.schemas import (
    GeoJSONResponse, GeoJSONFeature, GeoJSONFeatureProperties,
    CountyDetail, CountyOut, TopCancer, SpeciesBreakdown
)

router = APIRouter(prefix="/api/v1/geo", tags=["geo"])


@router.get("/counties", response_model=GeoJSONResponse)
async def get_counties_geojson(
    species: Optional[List[str]] = Query(None),
    cancer_type: Optional[List[str]] = Query(None),
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    # Build dynamic WHERE clause
    conditions = []
    params = {}
    if species:
        conditions.append("s.name = ANY(:species)")
        params["species"] = species
    if cancer_type:
        conditions.append("ct.name = ANY(:cancer_type)")
        params["cancer_type"] = cancer_type
    if year_start:
        conditions.append("EXTRACT(YEAR FROM cc.diagnosis_date) >= :year_start")
        params["year_start"] = year_start
    if year_end:
        conditions.append("EXTRACT(YEAR FROM cc.diagnosis_date) <= :year_end")
        params["year_end"] = year_end
    if sex and sex != "All":
        conditions.append("p.sex ILIKE :sex")
        params["sex"] = f"%{sex}%"

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = text(f"""
        SELECT
            c.id,
            c.name,
            c.fips_code,
            c.population,
            ST_AsGeoJSON(c.geom)::json AS geometry,
            COALESCE(case_counts.total, 0) AS total_cases,
            case_counts.top_cancer
        FROM counties c
        LEFT JOIN (
            SELECT
                cc.county_id,
                COUNT(*) AS total,
                (SELECT ct2.name
                 FROM cancer_cases cc2
                 JOIN cancer_types ct2 ON cc2.cancer_type_id = ct2.id
                 WHERE cc2.county_id = cc.county_id
                 GROUP BY ct2.name
                 ORDER BY COUNT(*) DESC
                 LIMIT 1
                ) AS top_cancer
            FROM cancer_cases cc
            JOIN patients p ON cc.patient_id = p.id
            JOIN species s ON p.species_id = s.id
            JOIN cancer_types ct ON cc.cancer_type_id = ct.id
            WHERE {where_clause}
            GROUP BY cc.county_id
        ) case_counts ON c.id = case_counts.county_id
        WHERE c.geom IS NOT NULL
        ORDER BY c.name
    """)

    result = await db.execute(query, params)
    rows = result.all()

    features = []
    for row in rows:
        geometry = row.geometry if row.geometry else {"type": "MultiPolygon", "coordinates": []}
        cases_per_capita = None
        if row.population and row.population > 0 and row.total_cases > 0:
            cases_per_capita = round(row.total_cases / row.population * 100000, 2)

        features.append(GeoJSONFeature(
            geometry=geometry,
            properties=GeoJSONFeatureProperties(
                name=row.name,
                fips_code=row.fips_code,
                population=row.population,
                total_cases=row.total_cases,
                cases_per_capita=cases_per_capita,
                top_cancer=row.top_cancer,
            )
        ))

    return GeoJSONResponse(features=features)


@router.get("/counties/{county_id}", response_model=CountyDetail)
async def get_county_detail(
    county_id: int,
    db: AsyncSession = Depends(get_db),
):
    # Get county info
    county_result = await db.execute(select(County).where(County.id == county_id))
    county = county_result.scalar_one_or_none()
    if not county:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="County not found")

    # Total cases
    result = await db.execute(
        select(func.count(CancerCase.id)).where(CancerCase.county_id == county_id)
    )
    total_cases = result.scalar() or 0

    # Cancer breakdown
    result = await db.execute(
        select(CancerType.name, func.count(CancerCase.id).label("cnt"))
        .join(CancerCase, CancerCase.cancer_type_id == CancerType.id)
        .where(CancerCase.county_id == county_id)
        .group_by(CancerType.name)
        .order_by(func.count(CancerCase.id).desc())
    )
    cancer_breakdown = [TopCancer(cancer_type=name, count=cnt) for name, cnt in result.all()]

    # Species breakdown
    result = await db.execute(
        select(Species.name, func.count(CancerCase.id).label("cnt"))
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .where(CancerCase.county_id == county_id)
        .group_by(Species.name)
        .order_by(func.count(CancerCase.id).desc())
    )
    species_rows = result.all()
    species_breakdown = [
        SpeciesBreakdown(
            species=name, count=cnt,
            percentage=round(cnt / total_cases * 100, 1) if total_cases > 0 else 0
        )
        for name, cnt in species_rows
    ]

    # Yearly trend
    result = await db.execute(
        select(
            func.extract("year", CancerCase.diagnosis_date).label("year"),
            func.count(CancerCase.id).label("count")
        )
        .where(CancerCase.county_id == county_id)
        .group_by(func.extract("year", CancerCase.diagnosis_date))
        .order_by(func.extract("year", CancerCase.diagnosis_date))
    )
    yearly_trend = [{"year": int(r.year), "count": r.count} for r in result.all()]

    return CountyDetail(
        county=CountyOut(
            id=county.id, name=county.name, fips_code=county.fips_code,
            population=county.population,
            area_sq_miles=float(county.area_sq_miles) if county.area_sq_miles else None,
        ),
        total_cases=total_cases,
        cancer_breakdown=cancer_breakdown,
        species_breakdown=species_breakdown,
        yearly_trend=yearly_trend,
    )
