"""Time series trend endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List

from app.database import get_db
from app.models.models import CancerCase, CancerType, Patient, Species, County
from app.schemas.schemas import TrendsResponse, TrendSeries, TrendPoint

router = APIRouter(prefix="/api/v1/trends", tags=["trends"])


@router.get("/yearly", response_model=TrendsResponse)
async def get_yearly_trends(
    species: Optional[List[str]] = Query(None),
    cancer_type: Optional[List[str]] = Query(None),
    county: Optional[List[str]] = Query(None),
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            func.extract("year", CancerCase.diagnosis_date).label("year"),
            func.count(CancerCase.id).label("count"),
            func.count(CancerCase.id).filter(CancerCase.outcome == "deceased").label("deceased"),
            func.count(CancerCase.id).filter(CancerCase.outcome == "alive").label("alive"),
        )
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .join(CancerType, CancerCase.cancer_type_id == CancerType.id)
        .join(County, CancerCase.county_id == County.id)
    )

    if species:
        stmt = stmt.where(Species.name.in_(species))
    if cancer_type:
        stmt = stmt.where(CancerType.name.in_(cancer_type))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if sex and sex != "All":
        stmt = stmt.where(Patient.sex.ilike(f"%{sex}%"))

    stmt = stmt.group_by(func.extract("year", CancerCase.diagnosis_date)).order_by(
        func.extract("year", CancerCase.diagnosis_date)
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
    species: Optional[List[str]] = Query(None),
    county: Optional[List[str]] = Query(None),
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            CancerType.name.label("cancer_type"),
            func.extract("year", CancerCase.diagnosis_date).label("year"),
            func.count(CancerCase.id).label("count"),
            func.count(CancerCase.id).filter(CancerCase.outcome == "deceased").label("deceased"),
            func.count(CancerCase.id).filter(CancerCase.outcome == "alive").label("alive"),
        )
        .join(CancerType, CancerCase.cancer_type_id == CancerType.id)
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .join(County, CancerCase.county_id == County.id)
    )

    if species:
        stmt = stmt.where(Species.name.in_(species))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if sex and sex != "All":
        stmt = stmt.where(Patient.sex.ilike(f"%{sex}%"))

    stmt = stmt.group_by(
        CancerType.name, func.extract("year", CancerCase.diagnosis_date)
    ).order_by(CancerType.name, func.extract("year", CancerCase.diagnosis_date))

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
