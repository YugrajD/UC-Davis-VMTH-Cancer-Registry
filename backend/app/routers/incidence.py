"""Incidence and mortality endpoints with filter support."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List

from app.database import get_db
from app.models.models import CancerCase, CancerType, Patient, Species, Breed, County
from app.schemas.schemas import IncidenceRecord, IncidenceResponse

router = APIRouter(prefix="/api/v1/incidence", tags=["incidence"])


def _apply_filters(stmt, species: Optional[List[str]], cancer_type: Optional[List[str]],
                   county: Optional[List[str]], year_start: Optional[int],
                   year_end: Optional[int], sex: Optional[str]):
    """Apply common filters to a query statement."""
    if species:
        stmt = stmt.where(Species.name.in_(species))
    if cancer_type:
        stmt = stmt.where(CancerType.name.in_(cancer_type))
    if county:
        stmt = stmt.where(County.name.in_(county))
    if year_start:
        stmt = stmt.where(func.extract("year", CancerCase.diagnosis_date) >= year_start)
    if year_end:
        stmt = stmt.where(func.extract("year", CancerCase.diagnosis_date) <= year_end)
    if sex and sex != "All":
        stmt = stmt.where(Patient.sex.ilike(f"%{sex}%"))
    return stmt


@router.get("", response_model=IncidenceResponse)
async def get_incidence(
    species: Optional[List[str]] = Query(None),
    cancer_type: Optional[List[str]] = Query(None),
    county: Optional[List[str]] = Query(None),
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            CancerType.name.label("cancer_type"),
            County.name.label("county"),
            Species.name.label("species"),
            func.extract("year", CancerCase.diagnosis_date).label("year"),
            func.count(CancerCase.id).label("count"),
        )
        .join(CancerType, CancerCase.cancer_type_id == CancerType.id)
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .join(County, CancerCase.county_id == County.id)
    )
    stmt = _apply_filters(stmt, species, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.group_by(
        CancerType.name, County.name, Species.name,
        func.extract("year", CancerCase.diagnosis_date)
    ).order_by(func.count(CancerCase.id).desc())

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
async def get_incidence_by_cancer_type(
    species: Optional[List[str]] = Query(None),
    county: Optional[List[str]] = Query(None),
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            CancerType.name.label("cancer_type"),
            func.count(CancerCase.id).label("count"),
        )
        .join(CancerType, CancerCase.cancer_type_id == CancerType.id)
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .join(County, CancerCase.county_id == County.id)
    )
    stmt = _apply_filters(stmt, species, None, county, year_start, year_end, sex)
    stmt = stmt.group_by(CancerType.name).order_by(func.count(CancerCase.id).desc())

    result = await db.execute(stmt)
    data = [IncidenceRecord(cancer_type=r.cancer_type, count=r.count) for r in result.all()]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={"species": species, "county": county,
                         "year_start": year_start, "year_end": year_end, "sex": sex}
    )


@router.get("/by-species", response_model=IncidenceResponse)
async def get_incidence_by_species(
    cancer_type: Optional[List[str]] = Query(None),
    county: Optional[List[str]] = Query(None),
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            Species.name.label("species"),
            func.count(CancerCase.id).label("count"),
        )
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .join(CancerType, CancerCase.cancer_type_id == CancerType.id)
        .join(County, CancerCase.county_id == County.id)
    )
    stmt = _apply_filters(stmt, None, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.group_by(Species.name).order_by(func.count(CancerCase.id).desc())

    result = await db.execute(stmt)
    data = [IncidenceRecord(species=r.species, count=r.count, cancer_type="All") for r in result.all()]

    return IncidenceResponse(
        data=data, total=sum(r.count for r in data),
        filters_applied={"cancer_type": cancer_type, "county": county,
                         "year_start": year_start, "year_end": year_end, "sex": sex}
    )


@router.get("/by-breed", response_model=IncidenceResponse)
async def get_incidence_by_breed(
    species: Optional[List[str]] = Query(None),
    cancer_type: Optional[List[str]] = Query(None),
    county: Optional[List[str]] = Query(None),
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    sex: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            Breed.name.label("breed"),
            Species.name.label("species"),
            func.count(CancerCase.id).label("count"),
        )
        .join(Patient, CancerCase.patient_id == Patient.id)
        .join(Species, Patient.species_id == Species.id)
        .join(Breed, Patient.breed_id == Breed.id)
        .join(CancerType, CancerCase.cancer_type_id == CancerType.id)
        .join(County, CancerCase.county_id == County.id)
    )
    stmt = _apply_filters(stmt, species, cancer_type, county, year_start, year_end, sex)
    stmt = stmt.group_by(Breed.name, Species.name).order_by(func.count(CancerCase.id).desc())

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
