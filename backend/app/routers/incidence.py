"""Incidence and mortality endpoints with filter support."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case as sa_case, literal, distinct
from typing import Optional, List

from app.cache import cached_response
from app.config import settings
from app.database import get_db
from app.rate_limit import limiter
from app.models.models import CancerType, Patient, Species, Breed, County, CaseDiagnosis
from app.models.views import mv_county_cancer
from app.schemas.schemas import (
    IncidenceRecord, IncidenceResponse,
    PCCPCountyRecord, PCCPResponse,
    BreedDetailOut, BreedCancerTypeCount, BreedCountyCount, BreedSexCount,
    AgeDetailOut, AgeCancerTypeCount, AgeCountyCount, AgeSexCount,
)
from app.services.review_filter import apply_review_filter, CALIFORNIA_PATIENT_FILTER, NON_CANCER_TYPE_NAME

router = APIRouter(prefix="/api/v1/incidence", tags=["incidence"])

# Map frontend sex filter values to DB values
SEX_MAP = {
    "male_intact": "Male",
    "male_neutered": "Neutered Male",
    "female_intact": "Female",
    "female_spayed": "Spayed Female",
}

# Valid age group values (frontend and DB both use these)
AGE_GROUPS = {"young", "juvenile", "adult", "old", "senior"}

def _age_group_case(diagnosis_date_col, birth_date_col):
    """SQLAlchemy CASE expression mapping birth/diagnosis dates to an age group string."""
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


def _apply_mv_filters(stmt, species: Optional[List[str]], cancer_type: Optional[List[str]],
                      county: Optional[List[str]], year_start: Optional[int],
                      year_end: Optional[int], sex: Optional[str],
                      age_group: Optional[str] = None):
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
    if age_group and age_group in AGE_GROUPS:
        stmt = stmt.where(mv.age_group == age_group)
    return stmt


def _apply_filters(stmt, species: Optional[List[str]], cancer_type: Optional[List[str]],
                   county: Optional[List[str]], year_start: Optional[int],
                   year_end: Optional[int], sex: Optional[str],
                   age_group: Optional[str] = None):
    """Apply common filters to a query statement (ingested data only).

    Used by breed/age endpoints that still require live table joins.
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
    if age_group and age_group in AGE_GROUPS:
        stmt = stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)
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
    age_group: Optional[str] = Query(None, max_length=20),
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

    stmt = _apply_mv_filters(stmt, species, cancer_type, county, year_start, year_end, sex, age_group)
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
            "year_end": year_end, "sex": sex, "age_group": age_group
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
    age_group: Optional[str] = Query(None, max_length=20),
    db: AsyncSession = Depends(get_db),
):
    """Patient-level per-cancer-type counts with PCCP (per 100 tested).

    Denominator: distinct petbert patients with any confirmed/corrected diagnosis.
    Numerator: distinct petbert patients per cancer type (non-Non-Cancer).
    """
    # Denominator: all patients with any confirmed/corrected diagnosis
    denom_stmt = (
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
    )
    denom_stmt = apply_review_filter(denom_stmt)
    if species:
        denom_stmt = denom_stmt.join(Species, Patient.species_id == Species.id).where(Species.name.in_(species))
    if county:
        denom_stmt = denom_stmt.join(County, Patient.county_id == County.id).where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        denom_stmt = denom_stmt.where(Patient.sex == SEX_MAP.get(sex, sex))
    if age_group and age_group in AGE_GROUPS:
        denom_stmt = denom_stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)
    if year_start:
        denom_stmt = denom_stmt.where(func.extract("year", Patient.diagnosis_date) >= year_start)
    if year_end:
        denom_stmt = denom_stmt.where(func.extract("year", Patient.diagnosis_date) <= year_end)
    total_patients = (await db.execute(denom_stmt)).scalar() or 0

    # Numerator: distinct patients per cancer type (cancer only)
    num_stmt = (
        select(
            CancerType.name.label("cancer_type"),
            func.count(distinct(Patient.id)).label("count"),
        )
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
    )
    num_stmt = apply_review_filter(num_stmt)
    if species:
        num_stmt = num_stmt.join(Species, Patient.species_id == Species.id).where(Species.name.in_(species))
    if county:
        num_stmt = num_stmt.join(County, Patient.county_id == County.id).where(County.name.in_(county))
    if sex and sex not in ("All", "all"):
        num_stmt = num_stmt.where(Patient.sex == SEX_MAP.get(sex, sex))
    if age_group and age_group in AGE_GROUPS:
        num_stmt = num_stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)
    if year_start:
        num_stmt = num_stmt.where(func.extract("year", Patient.diagnosis_date) >= year_start)
    if year_end:
        num_stmt = num_stmt.where(func.extract("year", Patient.diagnosis_date) <= year_end)
    num_stmt = num_stmt.group_by(CancerType.name).order_by(func.count(distinct(Patient.id)).desc())

    rows = (await db.execute(num_stmt)).all()
    data = [
        IncidenceRecord(
            cancer_type=r.cancer_type,
            count=r.count,
            pccp=round(r.count / total_patients * 100, 2) if total_patients > 0 else None,
            total_patients=total_patients,
        )
        for r in rows
    ]

    return IncidenceResponse(
        data=data,
        total=total_patients,
        filters_applied={"species": species, "county": county,
                         "year_start": year_start, "year_end": year_end,
                         "sex": sex, "age_group": age_group},
    )


@router.get("/pccp", response_model=PCCPResponse)
@limiter.limit("60/minute")
@cached_response("incidence_pccp", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_pccp_by_county(
    request: Request,
    sex: Optional[str] = Query(None, max_length=50),
    age_group: Optional[str] = Query(None, max_length=20),
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    cancer_type: Optional[str] = Query(None, max_length=200),
    db: AsyncSession = Depends(get_db),
):
    """Per-county PCCP: cancer patients / tested patients × 100.

    Denominator: petbert patients with any confirmed/corrected diagnosis, county_id IS NOT NULL.
      - Demographic filters (sex, age_group, year) apply to the denominator.
      - cancer_type does NOT apply to the denominator.
    Numerator: subset with a matching cancer diagnosis.
      - All filters including cancer_type apply to the numerator.
    """
    def _add_demo(stmt):
        if sex and sex not in ("All", "all"):
            stmt = stmt.where(Patient.sex == SEX_MAP.get(sex, sex))
        if age_group and age_group in AGE_GROUPS:
            stmt = stmt.where(_age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group)
        if year_start:
            stmt = stmt.where(func.extract("year", Patient.diagnosis_date) >= year_start)
        if year_end:
            stmt = stmt.where(func.extract("year", Patient.diagnosis_date) <= year_end)
        return stmt

    # Denominator: patients with any confirmed/corrected diagnosis, grouped by county.
    # cancer_type intentionally excluded — denominator is always all tested animals.
    denom_stmt = apply_review_filter(
        _add_demo(
            select(Patient.county_id, func.count(distinct(Patient.id)).label("n"))
            .select_from(Patient)
            .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
            .where(Patient.data_source == "petbert")
            .where(Patient.county_id.is_not(None))
            .group_by(Patient.county_id)
        )
    )
    denom_rows = {r.county_id: r.n for r in (await db.execute(denom_stmt)).all()}

    # Numerator: patients with a matching cancer diagnosis, grouped by county.
    # cancer_type narrows the numerator when provided.
    num_base = (
        select(Patient.county_id, func.count(distinct(Patient.id)).label("n"))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(Patient.county_id.is_not(None))
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .group_by(Patient.county_id)
    )
    if cancer_type and cancer_type not in ("All Types", "all"):
        num_base = num_base.where(CancerType.name == cancer_type)
    num_stmt = apply_review_filter(_add_demo(num_base))
    num_rows = {r.county_id: r.n for r in (await db.execute(num_stmt)).all()}

    county_ids = set(denom_rows.keys()) | set(num_rows.keys())
    county_map: dict[int, str] = {}
    if county_ids:
        county_stmt = select(County.id, County.name).where(County.id.in_(county_ids))
        county_map = {r.id: r.name for r in (await db.execute(county_stmt)).all()}

    data = []
    for county_id, total in denom_rows.items():
        county_name = county_map.get(county_id)
        if county_name is None:
            continue
        cancer = num_rows.get(county_id, 0)
        pccp = round(cancer / total * 100, 2) if total > 0 else 0.0
        data.append(PCCPCountyRecord(
            county=county_name,
            cancer_patients=cancer,
            total_patients=total,
            pccp=pccp,
        ))

    overall_total = sum(denom_rows.values())
    overall_cancer = sum(r.cancer_patients for r in data)
    overall_pccp = round(overall_cancer / overall_total * 100, 2) if overall_total > 0 else 0.0

    return PCCPResponse(
        data=sorted(data, key=lambda r: r.pccp, reverse=True),
        overall_cancer_patients=overall_cancer,
        overall_total_patients=overall_total,
        overall_pccp=overall_pccp,
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
    age_group: Optional[str] = Query(None, max_length=20),
    db: AsyncSession = Depends(get_db),
):
    mv = mv_county_cancer.c
    stmt = select(
        mv.species_name.label("species"),
        func.sum(mv.case_count).label("count"),
    ).select_from(mv_county_cancer)

    stmt = _apply_mv_filters(stmt, None, cancer_type, county, year_start, year_end, sex, age_group)
    stmt = stmt.group_by(mv.species_name).order_by(func.sum(mv.case_count).desc())

    result = await db.execute(stmt)
    data = [IncidenceRecord(species=r.species, count=r.count, cancer_type="All") for r in result.all()]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={"cancer_type": cancer_type, "county": county,
                         "year_start": year_start, "year_end": year_end,
                         "sex": sex, "age_group": age_group}
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
    age_group: Optional[str] = Query(None, max_length=20),
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
    stmt = _apply_filters(stmt, species, cancer_type, county, year_start, year_end, sex, age_group)
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
                         "year_end": year_end, "sex": sex, "age_group": age_group}
    )


@router.get("/breed-detail", response_model=BreedDetailOut)
@limiter.limit("60/minute")
@cached_response("breed_detail", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_breed_detail(
    request: Request,
    breed: str = Query(..., max_length=200, description="Breed name to look up"),
    db: AsyncSession = Depends(get_db),
):
    """Return PCCP-based cancer-type breakdown, sex breakdown, and county distribution for a breed.

    Two PCCP denominators:
    - Eq 5 (pccp_of_all): numerator / all petbert tested dogs
    - Eq 6 (pccp_within_breed): numerator / tested dogs of this breed
    """
    # --- Eq 5 denominator: all petbert tested dogs ---
    global_denom_stmt = apply_review_filter(
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
    )
    global_total_patients = (await db.execute(global_denom_stmt)).scalar() or 0

    # --- Eq 6 denominator: tested dogs of this breed ---
    breed_denom_stmt = apply_review_filter(
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .where(Patient.data_source == "petbert")
        .where(Breed.name == breed)
        .where(CALIFORNIA_PATIENT_FILTER)
    )
    breed_total_patients = (await db.execute(breed_denom_stmt)).scalar() or 0

    # --- total cancer patients for this breed ---
    total_stmt = apply_review_filter(
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
    )
    total_cases = (await db.execute(total_stmt)).scalar() or 0
    overall_pccp_within_breed = round(total_cases / breed_total_patients * 100, 2) if breed_total_patients > 0 else None
    overall_pccp_of_all = round(total_cases / global_total_patients * 100, 2) if global_total_patients > 0 else None

    # --- sex breakdown (distinct cancer patients per sex) ---
    sex_stmt = apply_review_filter(
        select(Patient.sex.label("sex"), func.count(distinct(Patient.id)).label("count"))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
        .group_by(Patient.sex)
        .order_by(func.count(distinct(Patient.id)).desc())
    )
    sex_rows = (await db.execute(sex_stmt)).all()
    sex_breakdown = [BreedSexCount(sex=r.sex or "Unknown", count=r.count) for r in sex_rows]

    # --- cancer types with dual PCCP ---
    ct_stmt = apply_review_filter(
        select(CancerType.name.label("cancer_type"), func.count(distinct(Patient.id)).label("count"))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
        .group_by(CancerType.name)
        .order_by(func.count(distinct(Patient.id)).desc())
    )
    ct_rows = (await db.execute(ct_stmt)).all()
    cancer_types = [
        BreedCancerTypeCount(
            cancer_type=r.cancer_type,
            count=r.count,
            pccp_within_breed=round(r.count / breed_total_patients * 100, 2) if breed_total_patients > 0 else None,
            pccp_of_all=round(r.count / global_total_patients * 100, 2) if global_total_patients > 0 else None,
        )
        for r in ct_rows
    ]

    # --- county distribution (cancer patient counts for geographic distribution) ---
    county_stmt = apply_review_filter(
        select(
            County.name.label("county_name"),
            County.fips_code.label("fips_code"),
            func.count(distinct(Patient.id)).label("count"),
        )
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .join(County, Patient.county_id == County.id)
        .where(Patient.data_source == "petbert")
        .where(Breed.name == breed)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
        .group_by(County.name, County.fips_code)
        .order_by(func.count(distinct(Patient.id)).desc())
    )
    county_rows = (await db.execute(county_stmt)).all()
    county_cases = [BreedCountyCount(county_name=r.county_name, fips_code=r.fips_code, count=r.count) for r in county_rows]

    return BreedDetailOut(
        breed=breed,
        total_cases=total_cases,
        breed_total_patients=breed_total_patients,
        global_total_patients=global_total_patients,
        pccp_within_breed=overall_pccp_within_breed,
        pccp_of_all=overall_pccp_of_all,
        sex_breakdown=sex_breakdown,
        cancer_types=cancer_types,
        county_cases=county_cases,
    )


@router.get("/age-detail", response_model=AgeDetailOut)
@limiter.limit("60/minute")
@cached_response("age_detail", ttl=settings.CACHE_TTL_INCIDENCE)
async def get_age_detail(
    request: Request,
    age_group: str = Query(..., max_length=20, description="Age group to look up"),
    db: AsyncSession = Depends(get_db),
):
    """Return PCCP-based cancer-type breakdown, sex breakdown, and county distribution for an age group.

    Two PCCP denominators:
    - Eq 5 equiv (pccp_of_all): numerator / all petbert tested dogs
    - Eq 6 equiv (pccp_within_age): numerator / tested dogs of this age group
    """
    if age_group not in AGE_GROUPS:
        raise HTTPException(status_code=400, detail=f"Invalid age_group. Must be one of: {', '.join(sorted(AGE_GROUPS))}")

    age_filter = _age_group_case(Patient.diagnosis_date, Patient.birth_date) == age_group

    # --- Eq 5 denominator: all petbert tested dogs ---
    global_denom_stmt = apply_review_filter(
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
    )
    global_total_patients = (await db.execute(global_denom_stmt)).scalar() or 0

    # --- Eq 6 denominator: tested dogs of this age group ---
    age_denom_stmt = apply_review_filter(
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(age_filter)
    )
    age_total_patients = (await db.execute(age_denom_stmt)).scalar() or 0

    # --- total cancer patients in this age group ---
    total_stmt = apply_review_filter(
        select(func.count(distinct(Patient.id)))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(age_filter)
    )
    total_cases = (await db.execute(total_stmt)).scalar() or 0
    overall_pccp_within_age = round(total_cases / age_total_patients * 100, 2) if age_total_patients > 0 else None
    overall_pccp_of_all = round(total_cases / global_total_patients * 100, 2) if global_total_patients > 0 else None

    # --- sex breakdown (distinct cancer patients per sex) ---
    sex_stmt = apply_review_filter(
        select(Patient.sex.label("sex"), func.count(distinct(Patient.id)).label("count"))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(age_filter)
        .group_by(Patient.sex)
        .order_by(func.count(distinct(Patient.id)).desc())
    )
    sex_rows = (await db.execute(sex_stmt)).all()
    sex_breakdown = [AgeSexCount(sex=r.sex or "Unknown", count=r.count) for r in sex_rows]

    # --- cancer types with dual PCCP ---
    ct_stmt = apply_review_filter(
        select(CancerType.name.label("cancer_type"), func.count(distinct(Patient.id)).label("count"))
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(age_filter)
        .group_by(CancerType.name)
        .order_by(func.count(distinct(Patient.id)).desc())
    )
    ct_rows = (await db.execute(ct_stmt)).all()
    cancer_types = [
        AgeCancerTypeCount(
            cancer_type=r.cancer_type,
            count=r.count,
            pccp_within_age=round(r.count / age_total_patients * 100, 2) if age_total_patients > 0 else None,
            pccp_of_all=round(r.count / global_total_patients * 100, 2) if global_total_patients > 0 else None,
        )
        for r in ct_rows
    ]

    # --- county distribution (cancer patient counts for geographic distribution) ---
    county_stmt = apply_review_filter(
        select(
            County.name.label("county_name"),
            County.fips_code.label("fips_code"),
            func.count(distinct(Patient.id)).label("count"),
        )
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .join(County, Patient.county_id == County.id)
        .where(Patient.data_source == "petbert")
        .where(CALIFORNIA_PATIENT_FILTER)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(age_filter)
        .group_by(County.name, County.fips_code)
        .order_by(func.count(distinct(Patient.id)).desc())
    )
    county_rows = (await db.execute(county_stmt)).all()
    county_cases = [AgeCountyCount(county_name=r.county_name, fips_code=r.fips_code, count=r.count) for r in county_rows]

    return AgeDetailOut(
        age_group=age_group,
        total_cases=total_cases,
        age_total_patients=age_total_patients,
        global_total_patients=global_total_patients,
        pccp_within_age=overall_pccp_within_age,
        pccp_of_all=overall_pccp_of_all,
        sex_breakdown=sex_breakdown,
        cancer_types=cancer_types,
        county_cases=county_cases,
    )
