"""Dashboard summary endpoints."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func

from app.cache import cached_response
from app.config import settings
from app.database import get_db
from app.rate_limit import limiter
from app.schemas.schemas import DashboardSummary, SpeciesBreakdown, TopCancer, FilterOptions
from app.models.models import (
    Species, Breed, CancerType, County, Patient, CaseDiagnosis
)
from app.services.review_filter import apply_review_filter, CALIFORNIA_PATIENT_FILTER, NON_CANCER_TYPE_NAME

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


# Only count ingested (PetBERT) data from California zip codes.
# Patients with a non-CA zip have county_id = NULL — they are excluded from all stats.
# Patients with no zip at all have zip_code = NULL — included (we can't confirm non-CA).
_PETBERT_FILTER = (Patient.data_source == "petbert") & CALIFORNIA_PATIENT_FILTER


@router.get("/summary", response_model=DashboardSummary)
@limiter.limit("60/minute")
@cached_response("dashboard_summary", ttl=settings.CACHE_TTL_DASHBOARD)
async def get_summary(request: Request, db: AsyncSession = Depends(get_db)):
    # Total cases (ingested only) — count distinct patients with petbert data
    result = await db.execute(
        select(func.count(Patient.id))
        .where(_PETBERT_FILTER)
    )
    total_cases = result.scalar() or 0

    # Total patients (ingested only)
    total_patients = total_cases

    # Total counties with cases (ingested only)
    result = await db.execute(
        select(func.count(func.distinct(Patient.county_id)))
        .where(_PETBERT_FILTER)
    )
    total_counties = result.scalar() or 0

    # Year range (ingested only; diagnosis_date may be null)
    result = await db.execute(
        select(
            func.min(func.extract("year", Patient.diagnosis_date)),
            func.max(func.extract("year", Patient.diagnosis_date))
        )
        .where(_PETBERT_FILTER)
    )
    row = result.one()
    year_range = [int(row[0] or 2015), int(row[1] or 2024)]

    # Species breakdown (ingested only)
    result = await db.execute(
        select(Species.name, func.count(Patient.id).label("cnt"))
        .select_from(Patient)
        .join(Species, Species.id == Patient.species_id)
        .where(_PETBERT_FILTER)
        .group_by(Species.name)
        .order_by(func.count(Patient.id).desc())
    )
    species_rows = result.all()
    species_breakdown = [
        SpeciesBreakdown(
            species=name,
            count=cnt,
            percentage=round(cnt / total_cases * 100, 1) if total_cases > 0 else 0
        )
        for name, cnt in species_rows
    ]

    # Top cancers (ingested only; count confirmed/corrected diagnoses)
    top_cancers_query = (
        select(CancerType.name, func.count(CaseDiagnosis.id).label("cnt"))
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(_PETBERT_FILTER)
        .group_by(CancerType.name)
        .order_by(func.count(CaseDiagnosis.id).desc())
        .limit(8)
    )
    result = await db.execute(apply_review_filter(
        top_cancers_query.where(CancerType.name != NON_CANCER_TYPE_NAME)
    ))
    top_cancers = [TopCancer(cancer_type=name, count=cnt) for name, cnt in result.all()]

    # Top county (ingested only)
    result = await db.execute(
        select(County.name, func.count(Patient.id).label("cnt"))
        .select_from(Patient)
        .join(County, County.id == Patient.county_id)
        .where(_PETBERT_FILTER)
        .group_by(County.name)
        .order_by(func.count(Patient.id).desc())
        .limit(1)
    )
    top_county_row = result.first()
    top_county = top_county_row[0] if top_county_row else "Unknown"
    top_county_cases = top_county_row[1] if top_county_row else 0

    return DashboardSummary(
        total_cases=total_cases,
        total_patients=total_patients,
        total_counties=total_counties,
        year_range=year_range,
        species_breakdown=species_breakdown,
        top_cancers=top_cancers,
        top_county=top_county,
        top_county_cases=top_county_cases,
    )


@router.get("/filters", response_model=FilterOptions)
@limiter.limit("60/minute")
@cached_response("dashboard_filters", ttl=settings.CACHE_TTL_CALENVIRO)
async def get_filter_options(request: Request, db: AsyncSession = Depends(get_db)):
    species = (await db.execute(select(Species).order_by(Species.name))).scalars().all()
    cancer_types = (await db.execute(
        select(CancerType).where(CancerType.name != NON_CANCER_TYPE_NAME).order_by(CancerType.name)
    )).scalars().all()
    counties = (await db.execute(select(County).order_by(County.name))).scalars().all()
    breeds = (await db.execute(
        select(Breed)
        .join(Patient, Patient.breed_id == Breed.id)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(CancerType.name != NON_CANCER_TYPE_NAME)
        .where(CALIFORNIA_PATIENT_FILTER)
        .group_by(Breed.id)
        .order_by(Breed.name)
    )).scalars().all()

    result = await db.execute(
        select(
            func.min(func.extract("year", Patient.diagnosis_date)),
            func.max(func.extract("year", Patient.diagnosis_date))
        )
    )
    row = result.one()
    year_range = [int(row[0] or 2015), int(row[1] or 2024)]

    return FilterOptions(
        species=species,
        cancer_types=cancer_types,
        counties=counties,
        breeds=breeds,
        year_range=year_range,
    )
