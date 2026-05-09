"""Time series trend endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List

from app.database import get_db
from app.models.models import CancerType, Patient, Species, County, CaseDiagnosis
from app.schemas.schemas import TrendsResponse, TrendSeries, TrendPoint
from app.services.review_filter import apply_review_filter

router = APIRouter(prefix="/api/v1/trends", tags=["trends"])

# Map frontend sex filter values to DB values
SEX_MAP = {
    "male_intact": "Male",
    "male_neutered": "Neutered Male",
    "female_intact": "Female",
    "female_spayed": "Spayed Female",
}


@router.get("/yearly", response_model=TrendsResponse)
async def get_yearly_trends(
    species: Optional[List[str]] = Query(None, max_length=50),
    cancer_type: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            func.extract("year", Patient.diagnosis_date).label("year"),
            func.count(func.distinct(Patient.id)).label("count"),
            func.count(func.distinct(Patient.id)).filter(Patient.outcome == "deceased").label("deceased"),
            func.count(func.distinct(Patient.id)).filter(Patient.outcome == "alive").label("alive"),
        )
        .select_from(Patient)
        .join(Species, Patient.species_id == Species.id)
        .join(County, Patient.county_id == County.id)
        .where(Patient.data_source == "petbert")
    )
    if cancer_type:
        stmt = (
            stmt.join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
            .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
            .where(CancerType.name.in_(cancer_type))
        )
        stmt = apply_review_filter(stmt)
    if species:
        stmt = stmt.where(Species.name.in_(species))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        mapped_sex = SEX_MAP.get(sex, sex)
        stmt = stmt.where(Patient.sex == mapped_sex)

    stmt = stmt.group_by(func.extract("year", Patient.diagnosis_date)).order_by(
        func.extract("year", Patient.diagnosis_date)
    )

    result = await db.execute(stmt)
    rows = result.all()

    data = [
        TrendPoint(year=int(r.year), count=r.count, deceased=r.deceased, alive=r.alive)
        for r in rows
    ]

    return TrendsResponse(series=[TrendSeries(name="All Cases", data=data)])


@router.get("/by-cancer-type", response_model=TrendsResponse)
async def get_trends_by_cancer_type(
    species: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            CancerType.name.label("cancer_type"),
            func.extract("year", Patient.diagnosis_date).label("year"),
            func.count(CaseDiagnosis.id).label("count"),
            func.count(CaseDiagnosis.id).filter(Patient.outcome == "deceased").label("deceased"),
            func.count(CaseDiagnosis.id).filter(Patient.outcome == "alive").label("alive"),
        )
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .join(Species, Patient.species_id == Species.id)
        .join(County, Patient.county_id == County.id)
        .where(Patient.data_source == "petbert")
    )
    stmt = apply_review_filter(stmt)
    if species:
        stmt = stmt.where(Species.name.in_(species))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        mapped_sex = SEX_MAP.get(sex, sex)
        stmt = stmt.where(Patient.sex == mapped_sex)

    stmt = stmt.group_by(
        CancerType.name, func.extract("year", Patient.diagnosis_date)
    ).order_by(CancerType.name, func.extract("year", Patient.diagnosis_date))

    result = await db.execute(stmt)
    rows = result.all()

    # Group by cancer type
    series_map: dict[str, list[TrendPoint]] = {}
    for r in rows:
        ct = r.cancer_type
        if ct not in series_map:
            series_map[ct] = []
        series_map[ct].append(
            TrendPoint(year=int(r.year), count=r.count, deceased=r.deceased, alive=r.alive)
        )

    series = [TrendSeries(name=name, data=pts) for name, pts in series_map.items()]
    return TrendsResponse(series=series)
