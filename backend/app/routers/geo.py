"""GeoJSON map endpoints using PostGIS spatial queries."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func
from typing import Optional, List
import json

from app.database import get_db
from app.models.models import County, CancerCase, CancerType, Patient, Species, CaseDiagnosis
from app.schemas.schemas import (
    GeoJSONResponse, GeoJSONFeature, GeoJSONFeatureProperties,
    CountyDetail, CountyOut, TopCancer, SpeciesBreakdown
)

router = APIRouter(prefix="/api/v1/geo", tags=["geo"])

# Map frontend sex filter values to DB values
SEX_MAP = {
    "male_intact": "Male",
    "male_neutered": "Neutered Male",
    "female_intact": "Female",
    "female_spayed": "Spayed Female",
}


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
    if year_start:
        conditions.append("EXTRACT(YEAR FROM cc.diagnosis_date) >= :year_start")
        params["year_start"] = year_start
    if year_end:
        conditions.append("EXTRACT(YEAR FROM cc.diagnosis_date) <= :year_end")
        params["year_end"] = year_end
    if sex and sex != "All" and sex != "all":
        mapped_sex = SEX_MAP.get(sex, sex)
        conditions.append("p.sex = :sex")
        params["sex"] = mapped_sex

    conditions.append("p.data_source = 'petbert'")
    # If filtering by cancer_type, restrict to cases that have that diagnosis
    if cancer_type:
        conditions.append(
            "cc.id IN (SELECT cd.case_id FROM case_diagnoses cd "
            "JOIN cancer_types ct ON ct.id = cd.cancer_type_id WHERE ct.name = ANY(:cancer_type))"
        )
        params["cancer_type"] = cancer_type
    where_clause = " AND ".join(conditions)

    query = text(f"""
        SELECT
            c.id,
            c.name,
            c.fips_code,
            ST_AsGeoJSON(c.geom)::json AS geometry,
            COALESCE(case_counts.total, 0) AS total_cases,
            case_counts.top_cancer
        FROM counties c
        LEFT JOIN (
            SELECT
                cc.county_id,
                COUNT(DISTINCT cc.id) AS total,
                (SELECT ct2.name
                 FROM case_diagnoses cd
                 JOIN cancer_cases cc2 ON cc2.id = cd.case_id
                 JOIN patients p2 ON p2.id = cc2.patient_id AND p2.data_source = 'petbert'
                 JOIN cancer_types ct2 ON ct2.id = cd.cancer_type_id
                 WHERE cc2.county_id = cc.county_id
                 GROUP BY ct2.name
                 ORDER BY COUNT(*) DESC
                 LIMIT 1
                ) AS top_cancer
            FROM cancer_cases cc
            JOIN patients p ON cc.patient_id = p.id
            JOIN species s ON p.species_id = s.id
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
        features.append(GeoJSONFeature(
            geometry=geometry,
            properties=GeoJSONFeatureProperties(
                name=row.name,
                fips_code=row.fips_code,
                total_cases=row.total_cases,
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

    # Total cases (ingested only)
    result = await db.execute(
        select(func.count(CancerCase.id))
        .join(Patient, Patient.id == CancerCase.patient_id)
        .where(CancerCase.county_id == county_id, Patient.data_source == "petbert")
    )
    total_cases = result.scalar() or 0

    # Cancer breakdown (ingested only; from case_diagnoses)
    result = await db.execute(
        select(CancerType.name, func.count(CaseDiagnosis.id).label("cnt"))
        .select_from(CaseDiagnosis)
        .join(CancerCase, CancerCase.id == CaseDiagnosis.case_id)
        .join(Patient, Patient.id == CancerCase.patient_id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(CancerCase.county_id == county_id, Patient.data_source == "petbert")
        .group_by(CancerType.name)
        .order_by(func.count(CaseDiagnosis.id).desc())
    )
    cancer_breakdown = [TopCancer(cancer_type=name, count=cnt) for name, cnt in result.all()]

    # Species breakdown (ingested only)
    result = await db.execute(
        select(Species.name, func.count(CancerCase.id).label("cnt"))
        .select_from(CancerCase)
        .join(Patient, Patient.id == CancerCase.patient_id)
        .join(Species, Species.id == Patient.species_id)
        .where(CancerCase.county_id == county_id, Patient.data_source == "petbert")
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

    # Yearly trend (ingested only)
    result = await db.execute(
        select(
            func.extract("year", CancerCase.diagnosis_date).label("year"),
            func.count(CancerCase.id).label("count")
        )
        .select_from(CancerCase)
        .join(Patient, Patient.id == CancerCase.patient_id)
        .where(CancerCase.county_id == county_id, Patient.data_source == "petbert")
        .group_by(func.extract("year", CancerCase.diagnosis_date))
        .order_by(func.extract("year", CancerCase.diagnosis_date))
    )
    yearly_trend = [{"year": int(r.year), "count": r.count} for r in result.all()]

    return CountyDetail(
        county=CountyOut(
            id=county.id, name=county.name, fips_code=county.fips_code,
            area_sq_miles=float(county.area_sq_miles) if county.area_sq_miles else None,
        ),
        total_cases=total_cases,
        cancer_breakdown=cancer_breakdown,
        species_breakdown=species_breakdown,
        yearly_trend=yearly_trend,
    )
