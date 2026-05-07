"""CSV export generator for patient-level disease and demographic data."""

import csv
import io

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Breed,
    CancerType,
    CaseDiagnosis,
    County,
    Patient,
    Species,
)
from app.services.review_filter import review_status_sql_in

CSV_COLUMNS = [
    "species",
    "breed",
    "sex",
    "county",
    "zip_code",
    "diagnosis_date",
    "outcome",
    "cancer_type",
    "icd_o_code",
    "confidence",
    "review_status",
]


async def generate_patient_export_csv(db: AsyncSession) -> str:
    """Build a CSV string with one row per patient-diagnosis (confirmed/corrected only)."""

    visible = review_status_sql_in()

    # Join patients → case_diagnoses → species/breed/county/cancer_type
    stmt = (
        select(
            Species.name.label("species"),
            Breed.name.label("breed"),
            Patient.sex,
            County.name.label("county"),
            Patient.zip_code,
            Patient.diagnosis_date,
            Patient.outcome,
            CancerType.name.label("cancer_type"),
            CaseDiagnosis.icd_o_code,
            CaseDiagnosis.confidence,
            CaseDiagnosis.review_status,
        )
        .select_from(Patient)
        .join(CaseDiagnosis, CaseDiagnosis.patient_id == Patient.id)
        .outerjoin(Species, Species.id == Patient.species_id)
        .outerjoin(Breed, Breed.id == Patient.breed_id)
        .outerjoin(County, County.id == Patient.county_id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CaseDiagnosis.review_status.in_(["confirmed", "corrected"]))
        .order_by(Patient.diagnosis_date, Patient.id)
    )

    result = await db.execute(stmt)
    rows = result.all()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "species": row.species or "",
            "breed": row.breed or "",
            "sex": row.sex or "",
            "county": row.county or "",
            "zip_code": row.zip_code or "",
            "diagnosis_date": str(row.diagnosis_date) if row.diagnosis_date else "",
            "outcome": row.outcome or "",
            "cancer_type": row.cancer_type or "",
            "icd_o_code": row.icd_o_code or "",
            "confidence": float(row.confidence) if row.confidence is not None else "",
            "review_status": row.review_status or "",
        })

    return output.getvalue()
