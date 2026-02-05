"""Aggregation and statistics service."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.models import CancerCase, Patient, Species, CancerType, County


async def get_species_distribution(db: AsyncSession) -> list[dict]:
    """Get case counts by species."""
    stmt = (
        select(Species.name, func.count(CancerCase.id).label("count"))
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .group_by(Species.name)
        .order_by(func.count(CancerCase.id).desc())
    )
    result = await db.execute(stmt)
    return [{"species": name, "count": cnt} for name, cnt in result.all()]


async def get_cancer_type_distribution(db: AsyncSession) -> list[dict]:
    """Get case counts by cancer type."""
    stmt = (
        select(CancerType.name, func.count(CancerCase.id).label("count"))
        .join(CancerCase, CancerCase.cancer_type_id == CancerType.id)
        .group_by(CancerType.name)
        .order_by(func.count(CancerCase.id).desc())
    )
    result = await db.execute(stmt)
    return [{"cancer_type": name, "count": cnt} for name, cnt in result.all()]


async def get_county_distribution(db: AsyncSession) -> list[dict]:
    """Get case counts by county."""
    stmt = (
        select(County.name, func.count(CancerCase.id).label("count"))
        .join(CancerCase, CancerCase.county_id == County.id)
        .group_by(County.name)
        .order_by(func.count(CancerCase.id).desc())
    )
    result = await db.execute(stmt)
    return [{"county": name, "count": cnt} for name, cnt in result.all()]
