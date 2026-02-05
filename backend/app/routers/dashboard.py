"""Dashboard summary endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func

from app.database import get_db
from app.schemas.schemas import DashboardSummary, SpeciesBreakdown, TopCancer, FilterOptions
from app.models.models import (
    Species, Breed, CancerType, County, Patient, CancerCase
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def get_summary(db: AsyncSession = Depends(get_db)):
    # Total cases
    result = await db.execute(select(func.count(CancerCase.id)))
    total_cases = result.scalar() or 0

    # Total patients
    result = await db.execute(select(func.count(Patient.id)))
    total_patients = result.scalar() or 0

    # Total counties with cases
    result = await db.execute(
        select(func.count(func.distinct(CancerCase.county_id)))
    )
    total_counties = result.scalar() or 0

    # Year range
    result = await db.execute(
        select(
            func.min(func.extract("year", CancerCase.diagnosis_date)),
            func.max(func.extract("year", CancerCase.diagnosis_date))
        )
    )
    row = result.one()
    year_range = [int(row[0] or 2015), int(row[1] or 2024)]

    # Species breakdown
    result = await db.execute(
        select(
            Species.name,
            func.count(CancerCase.id).label("cnt")
        )
        .join(Patient, Patient.id == CancerCase.patient_id)
        .join(Species, Species.id == Patient.species_id)
        .group_by(Species.name)
        .order_by(func.count(CancerCase.id).desc())
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

    # Top cancers
    result = await db.execute(
        select(
            CancerType.name,
            func.count(CancerCase.id).label("cnt")
        )
        .join(CancerCase, CancerCase.cancer_type_id == CancerType.id)
        .group_by(CancerType.name)
        .order_by(func.count(CancerCase.id).desc())
        .limit(8)
    )
    top_cancers = [TopCancer(cancer_type=name, count=cnt) for name, cnt in result.all()]

    # Top county
    result = await db.execute(
        select(
            County.name,
            func.count(CancerCase.id).label("cnt")
        )
        .join(CancerCase, CancerCase.county_id == County.id)
        .group_by(County.name)
        .order_by(func.count(CancerCase.id).desc())
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
async def get_filter_options(db: AsyncSession = Depends(get_db)):
    species = (await db.execute(select(Species).order_by(Species.name))).scalars().all()
    cancer_types = (await db.execute(select(CancerType).order_by(CancerType.name))).scalars().all()
    counties = (await db.execute(select(County).order_by(County.name))).scalars().all()
    breeds = (await db.execute(select(Breed).order_by(Breed.name))).scalars().all()

    result = await db.execute(
        select(
            func.min(func.extract("year", CancerCase.diagnosis_date)),
            func.max(func.extract("year", CancerCase.diagnosis_date))
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
