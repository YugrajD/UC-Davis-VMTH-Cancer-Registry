"""Aggregation and statistics service."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.models import Patient, Species, CancerType, County, CaseDiagnosis


async def get_species_distribution(db: AsyncSession) -> list[dict]:
    """Get patient counts by species."""
    stmt = (
        select(Species.name, func.count(Patient.id).label("count"))
        .select_from(Patient)
        .join(Species, Patient.species_id == Species.id)
        .group_by(Species.name)
        .order_by(func.count(Patient.id).desc())
    )
    result = await db.execute(stmt)
    return [{"species": name, "count": cnt} for name, cnt in result.all()]


async def get_cancer_type_distribution(db: AsyncSession) -> list[dict]:
    """Get diagnosis counts by cancer type."""
    stmt = (
        select(CancerType.name, func.count(CaseDiagnosis.id).label("count"))
        .select_from(CaseDiagnosis)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .group_by(CancerType.name)
        .order_by(func.count(CaseDiagnosis.id).desc())
    )
    result = await db.execute(stmt)
    return [{"cancer_type": name, "count": cnt} for name, cnt in result.all()]


async def get_county_distribution(db: AsyncSession) -> list[dict]:
    """Get patient counts by county."""
    stmt = (
        select(County.name, func.count(Patient.id).label("count"))
        .select_from(Patient)
        .join(County, County.id == Patient.county_id)
        .group_by(County.name)
        .order_by(func.count(Patient.id).desc())
    )
    result = await db.execute(stmt)
    return [{"county": name, "count": cnt} for name, cnt in result.all()]
