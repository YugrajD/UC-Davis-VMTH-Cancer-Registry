"""Time series trend endpoints."""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List

from app.cache import cached_response
from app.config import settings
from app.database import get_db
from app.rate_limit import limiter
from app.models.models import CancerType, Patient, Species, County, CaseDiagnosis
from app.models.views import mv_yearly_trends
from app.schemas.schemas import TrendsResponse, TrendSeries, TrendPoint
from app.services.review_filter import apply_review_filter, CALIFORNIA_PATIENT_FILTER

router = APIRouter(prefix="/api/v1/trends", tags=["trends"])

# Map frontend sex filter values to DB values
SEX_MAP = {
    "male_intact": "Male",
    "male_neutered": "Neutered Male",
    "female_intact": "Female",
    "female_spayed": "Spayed Female",
}


@router.get("/yearly", response_model=TrendsResponse)
@limiter.limit("60/minute")
@cached_response("trends_yearly", ttl=settings.CACHE_TTL_TRENDS)
async def get_yearly_trends(
    request: Request,
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
        .outerjoin(County, Patient.county_id == County.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
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

    stmt = stmt.where(Patient.diagnosis_date.is_not(None))
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
@limiter.limit("60/minute")
@cached_response("trends_by_cancer", ttl=settings.CACHE_TTL_TRENDS)
async def get_trends_by_cancer_type(
    request: Request,
    species: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    mv = mv_yearly_trends.c
    stmt = select(
        mv.cancer_type_name.label("cancer_type"),
        mv.year,
        func.sum(mv.case_count).label("count"),
        func.sum(mv.deceased_count).label("deceased"),
        func.sum(mv.alive_count).label("alive"),
    ).select_from(mv_yearly_trends)

    if species:
        stmt = stmt.where(mv.species_name.in_(species))
    if county:
        stmt = stmt.where(mv.county_name.in_(county))
    if sex and sex not in ("All", "all"):
        stmt = stmt.where(mv.sex == SEX_MAP.get(sex, sex))

    stmt = stmt.where(mv.year.is_not(None))
    stmt = stmt.group_by(mv.cancer_type_name, mv.year)
    stmt = stmt.order_by(mv.cancer_type_name, mv.year)

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
