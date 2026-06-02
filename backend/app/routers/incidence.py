"""Incidence and mortality endpoints with filter support."""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List

from app.cache import cached_response
from app.config import settings
from app.database import get_db
from app.rate_limit import limiter
from app.models.models import CancerType, Patient, Species, Breed, County, CaseDiagnosis
from app.models.views import mv_county_cancer
from app.schemas.schemas import IncidenceRecord, IncidenceResponse, BreedDetailOut, BreedCancerTypeCount, BreedCountyCount, BreedSexCount
from app.services.review_filter import apply_review_filter, CALIFORNIA_PATIENT_FILTER, NON_CANCER_TYPE_NAME

router = APIRouter(prefix="/api/v1/incidence", tags=["incidence"])

# Map frontend sex filter values to DB values
SEX_MAP = {
    "male_intact": "Male",
    "male_neutered": "Neutered Male",
    "female_intact": "Female",
    "female_spayed": "Spayed Female",
}


def _apply_mv_filters(stmt, species: Optional[List[str]], cancer_type: Optional[List[str]],
                      county: Optional[List[str]], year_start: Optional[int],
                      year_end: Optional[int], sex: Optional[str]):
    """Apply filters to a query on mv_county_cancer_incidence."""
    mv = mv_county_cancer.c
    if species:
        stmt = stmt.where(mv.species_name.in_(species))
    if cancer_type:
        stmt = stmt.where(mv.cancer_type_name.in_(cancer_type))
    if county:
        stmt = stmt.where(mv.county_name.in_(county))
    if year_start:
        stmt = stmt.where(mv.year >= year_start)
    if year_end:
        stmt = stmt.where(mv.year <= year_end)
    if sex and sex not in ("All", "all"):
        mapped_sex = SEX_MAP.get(sex, sex)
        stmt = stmt.where(mv.sex == mapped_sex)
    return stmt


def _apply_filters(stmt, species: Optional[List[str]], cancer_type: Optional[List[str]],
                   county: Optional[List[str]], year_start: Optional[int],
                   year_end: Optional[int], sex: Optional[str]):
    """Apply common filters to a query statement (ingested data only).

    Used by breed endpoints that still require live table joins.
    """
    stmt = stmt.where(Patient.data_source == "petbert")
    stmt = stmt.where(CALIFORNIA_PATIENT_FILTER)
    stmt = apply_review_filter(stmt)
    stmt = stmt.where(CancerType.name != NON_CANCER_TYPE_NAME)
    if species:
        stmt = stmt.where(Species.name.in_(species))
    if cancer_type:
        stmt = stmt.where(CancerType.name.in_(cancer_type))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if year_start:
        stmt = stmt.where(func.extract("year", Patient.diagnosis_date) >= year_start)
    if year_end:
        stmt = stmt.where(func.extract("year", Patient.diagnosis_date) <= year_end)
    if sex and sex not in ("All", "all"):
        mapped_sex = SEX_MAP.get(sex, sex)
        stmt = stmt.where(Patient.sex == mapped_sex)
    return stmt


@router.get("", response_model=IncidenceResponse)
@limiter.limit("60/minute")
@cached_response("incidence", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_incidence(
    request: Request,
    species: Optional[List[str]] = Query(None, max_length=50),
    cancer_type: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    mv = mv_county_cancer.c
    stmt = select(
        mv.cancer_type_name.label("cancer_type"),
        mv.county_name.label("county"),
        mv.species_name.label("species"),
        mv.year,
        func.sum(mv.case_count).label("count"),
    ).select_from(mv_county_cancer)

    stmt = _apply_mv_filters(stmt, species, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.where(mv.year.is_not(None))
    stmt = stmt.group_by(mv.cancer_type_name, mv.county_name, mv.species_name, mv.year)
    stmt = stmt.order_by(func.sum(mv.case_count).desc())

    result = await db.execute(stmt)
    rows = result.all()

    data = [
        IncidenceRecord(
            cancer_type=r.cancer_type, county=r.county,
            species=r.species, year=int(r.year), count=r.count
        )
        for r in rows
    ]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={
            "species": species, "cancer_type": cancer_type,
            "county": county, "year_start": year_start,
            "year_end": year_end, "sex": sex
        }
    )


@router.get("/by-cancer-type", response_model=IncidenceResponse)
@limiter.limit("60/minute")
@cached_response("incidence_by_cancer", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_incidence_by_cancer_type(
    request: Request,
    species: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    mv = mv_county_cancer.c
    stmt = select(
        mv.cancer_type_name.label("cancer_type"),
        func.sum(mv.case_count).label("count"),
    ).select_from(mv_county_cancer)

    stmt = _apply_mv_filters(stmt, species, None, county, year_start, year_end, sex)
    stmt = stmt.where(mv.cancer_type_name != NON_CANCER_TYPE_NAME)
    stmt = stmt.group_by(mv.cancer_type_name).order_by(func.sum(mv.case_count).desc())

    result = await db.execute(stmt)
    data = [IncidenceRecord(cancer_type=r.cancer_type, count=r.count) for r in result.all()]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={"species": species, "county": county,
                         "year_start": year_start, "year_end": year_end, "sex": sex}
    )


@router.get("/by-species", response_model=IncidenceResponse)
@limiter.limit("60/minute")
@cached_response("incidence_by_species", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_incidence_by_species(
    request: Request,
    cancer_type: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    mv = mv_county_cancer.c
    stmt = select(
        mv.species_name.label("species"),
        func.sum(mv.case_count).label("count"),
    ).select_from(mv_county_cancer)

    stmt = _apply_mv_filters(stmt, None, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.group_by(mv.species_name).order_by(func.sum(mv.case_count).desc())

    result = await db.execute(stmt)
    data = [IncidenceRecord(species=r.species, count=r.count, cancer_type="All") for r in result.all()]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={"cancer_type": cancer_type, "county": county,
                         "year_start": year_start, "year_end": year_end, "sex": sex}
    )


@router.get("/by-breed", response_model=IncidenceResponse)
@limiter.limit("60/minute")
@cached_response("incidence_by_breed", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_incidence_by_breed(
    request: Request,
    species: Optional[List[str]] = Query(None, max_length=50),
    cancer_type: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            Breed.name.label("breed"),
            Species.name.label("species"),
            func.count(CaseDiagnosis.id).label("count"),
        )
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(Species, Patient.species_id == Species.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .outerjoin(County, Patient.county_id == County.id)
    )
    stmt = _apply_filters(stmt, species, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.group_by(Breed.name, Species.name).order_by(func.count(CaseDiagnosis.id).desc())

    result = await db.execute(stmt)
    data = [
        IncidenceRecord(breed=r.breed, species=r.species, count=r.count, cancer_type="All")
        for r in result.all()
    ]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={"species": species, "cancer_type": cancer_type,
                         "county": county, "year_start": year_start,
                         "year_end": year_end, "sex": sex}
    )


@router.get("/by-zip", response_model=IncidenceResponse)
@limiter.limit("60/minute")
@cached_response("incidence_by_zip", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_incidence_by_zip(
    request: Request,
    species: Optional[List[str]] = Query(None, max_length=50),
    cancer_type: Optional[List[str]] = Query(None, max_length=50),
    county: Optional[List[str]] = Query(None, max_length=100),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    sex: Optional[str] = Query(None, max_length=50),
    db: AsyncSession = Depends(get_db),
):
    """Return reviewed California cancer diagnosis counts grouped by 5-digit ZIP."""
    zip_expr = func.substring(func.trim(Patient.zip_code), 1, 5).label("zip_code")
    stmt = (
        select(
            zip_expr,
            func.count(CaseDiagnosis.id).label("count"),
        )
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .outerjoin(Species, Patient.species_id == Species.id)
        .outerjoin(County, Patient.county_id == County.id)
        .where(Patient.zip_code.is_not(None))
        .where(Patient.county_id.is_not(None))
        .where(func.length(func.trim(Patient.zip_code)) >= 5)
    )
    stmt = _apply_filters(stmt, species, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.group_by(zip_expr).order_by(func.count(CaseDiagnosis.id).desc())

    result = await db.execute(stmt)
    data = [
        IncidenceRecord(cancer_type="All", zip_code=r.zip_code, count=r.count)
        for r in result.all()
    ]

    return IncidenceResponse(
        data=data,
        total=sum(r.count for r in data),
        filters_applied={
            "species": species, "cancer_type": cancer_type,
            "county": county, "year_start": year_start,
            "year_end": year_end, "sex": sex
        }
    )


@router.get("/breed-detail", response_model=BreedDetailOut)
@limiter.limit("60/minute")
@cached_response("breed_detail", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_breed_detail(
    request: Request,
    breed: str = Query(..., max_length=200, description="Breed name to look up"),
    db: AsyncSession = Depends(get_db),
):
    """Return cancer-type breakdown, sex breakdown, and county distribution for a single breed.

    Includes ALL data where breed_id IS NOT NULL (mock + real).
    """
    # --- total cases ---
    total_stmt = apply_review_filter(
        select(func.count(CaseDiagnosis.id))
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
    )
    total_cases = (await db.execute(total_stmt)).scalar() or 0

    # --- sex breakdown ---
    sex_stmt = apply_review_filter(
        select(Patient.sex.label("sex"), func.count(CaseDiagnosis.id).label("count"))
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
        .group_by(Patient.sex)
        .order_by(func.count(CaseDiagnosis.id).desc())
    )
    sex_rows = (await db.execute(sex_stmt)).all()
    sex_breakdown = [BreedSexCount(sex=r.sex or "Unknown", count=r.count) for r in sex_rows]

    # --- cancer types ---
    ct_stmt = apply_review_filter(
        select(CancerType.name.label("cancer_type"), func.count(CaseDiagnosis.id).label("count"))
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
        .group_by(CancerType.name)
        .order_by(func.count(CaseDiagnosis.id).desc())
    )
    ct_rows = (await db.execute(ct_stmt)).all()
    cancer_types = [BreedCancerTypeCount(cancer_type=r.cancer_type, count=r.count) for r in ct_rows]

    # --- county distribution ---
    county_stmt = apply_review_filter(
        select(
            County.name.label("county_name"),
            County.fips_code.label("fips_code"),
            func.count(CaseDiagnosis.id).label("count"),
        )
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .join(County, Patient.county_id == County.id)
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .group_by(County.name, County.fips_code)
        .order_by(func.count(CaseDiagnosis.id).desc())
    )
    county_rows = (await db.execute(county_stmt)).all()
    county_cases = [BreedCountyCount(county_name=r.county_name, fips_code=r.fips_code, count=r.count) for r in county_rows]

    return BreedDetailOut(
        breed=breed,
        total_cases=total_cases,
        sex_breakdown=sex_breakdown,
        cancer_types=cancer_types,
        county_cases=county_cases,
    )
