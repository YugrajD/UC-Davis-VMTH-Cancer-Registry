"""Time series trend endpoints."""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case as sa_case, literal
from typing import Optional, List

from app.cache import cached_response
from app.config import settings
from app.database import get_db
from app.rate_limit import limiter
from app.models.models import CancerType, Patient, Species, County, CaseDiagnosis
from app.models.views import mv_yearly_trends
from app.schemas.schemas import TrendsResponse, TrendSeries, TrendPoint
from app.services.review_filter import apply_review_filter, CALIFORNIA_PATIENT_FILTER, NON_CANCER_TYPE_NAME

router = APIRouter(prefix="/api/v1/trends", tags=["trends"])

# Map frontend sex filter values to DB values
SEX_MAP = {
    "male_intact": "Male",
    "male_neutered": "Neutered Male",
    "female_intact": "Female",
    "female_spayed": "Spayed Female",
}

VALID_AGE_GROUPS = {"young", "juvenile", "adult", "old", "senior"}


def _age_group_case(diagnosis_date_col, birth_date_col):
    age_expr = func.extract("year", diagnosis_date_col) - func.extract("year", birth_date_col)
    return sa_case(
        (birth_date_col.is_(None), literal("Unknown")),
        (diagnosis_date_col.is_(None), literal("Unknown")),
        (age_expr.between(0, 2), literal("young")),
        (age_expr.between(3, 5), literal("juvenile")),
        (age_expr.between(6, 8), literal("adult")),
        (age_expr.between(9, 11), literal("old")),
        (age_expr >= 12, literal("senior")),
        else_=literal("Unknown"),
    )


@router.get("/yearly", response_model=TrendsResponse)
@limiter.limit("60/minute")
@cached_response("trends_yearly", ttl=settings.CACHE_TTL_TRENDS)
async def get_yearly_trends(
    request: Request,
    species: Optional[List[str]] = Query(None, max_length=50),
    cancer_type: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    sex: Optional[str] = Query(None, max_length=50),
    age_group: Optional[str] = Query(None, max_length=20),
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
            .where(CancerType.name != NON_CANCER_TYPE_NAME)
        )
        stmt = apply_review_filter(stmt)
    if species:
        stmt = stmt.where(Species.name.in_(species))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        mapped_sex = SEX_MAP.get(sex, sex)
        stmt = stmt.where(Patient.sex == mapped_sex)
    if age_group and age_group in VALID_AGE_GROUPS:
        stmt = stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)

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
    age_group: Optional[str] = Query(None, max_length=20),
    db: AsyncSession = Depends(get_db),
):
    """Patient-level trends by cancer type with PCCP per year (Eq 4).

    Denominator per year: distinct petbert patients with any confirmed/corrected diagnosis in that year.
    Numerator per year per cancer type: distinct petbert patients with that cancer type in that year.
    """
    # Denominator by year: patients with any confirmed/corrected diagnosis
    denom_stmt = (
        select(
            func.extract("year", Patient.diagnosis_date).label("year"),
            func.count(func.distinct(Patient.id)).label("total"),
        )
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(Patient.diagnosis_date.is_not(None))
    )
    denom_stmt = apply_review_filter(denom_stmt)
    if species:
        denom_stmt = denom_stmt.join(Species, Patient.species_id == Species.id).where(Species.name.in_(species))
    if county:
        denom_stmt = denom_stmt.join(County, Patient.county_id == County.id).where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        denom_stmt = denom_stmt.where(Patient.sex == SEX_MAP.get(sex, sex))
    if age_group and age_group in VALID_AGE_GROUPS:
        denom_stmt = denom_stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)
    denom_stmt = denom_stmt.group_by(func.extract("year", Patient.diagnosis_date))
    denom_by_year = {int(r.year): r.total for r in (await db.execute(denom_stmt)).all()}

    # Numerator: distinct patients per cancer type per year
    num_stmt = (
        select(
            CancerType.name.label("cancer_type"),
            func.extract("year", Patient.diagnosis_date).label("year"),
            func.count(func.distinct(Patient.id)).label("count"),
        )
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(Patient.diagnosis_date.is_not(None))
    )
    num_stmt = apply_review_filter(num_stmt)
    if species:
        num_stmt = num_stmt.join(Species, Patient.species_id == Species.id).where(Species.name.in_(species))
    if county:
        num_stmt = num_stmt.join(County, Patient.county_id == County.id).where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        num_stmt = num_stmt.where(Patient.sex == SEX_MAP.get(sex, sex))
    if age_group and age_group in VALID_AGE_GROUPS:
        num_stmt = num_stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)
    num_stmt = num_stmt.group_by(CancerType.name, func.extract("year", Patient.diagnosis_date))
    num_stmt = num_stmt.order_by(CancerType.name, func.extract("year", Patient.diagnosis_date))

    rows = (await db.execute(num_stmt)).all()

    # Group by cancer type, attach PCCP per year
    series_map: dict[str, list[TrendPoint]] = {}
    for r in rows:
        ct = r.cancer_type
        year = int(r.year)
        if ct not in series_map:
            series_map[ct] = []
        total = denom_by_year.get(year, 0)
        pccp = round(r.count / total * 100, 2) if total > 0 else None
        series_map[ct].append(
            TrendPoint(year=year, count=r.count, pccp=pccp, total_patients=total)
        )

    series = [TrendSeries(name=name, data=pts) for name, pts in series_map.items()]
    return TrendsResponse(series=series)
