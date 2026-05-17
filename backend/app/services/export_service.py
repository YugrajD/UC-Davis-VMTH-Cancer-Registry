"""CSV export generator for patient-level disease and demographic data."""

import csv
import io

from sqlalchemy import func, select
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

# Characters that spreadsheet applications (Excel, Google Sheets) interpret
# as formula starters.  Prefixing with a tab prevents execution.
_FORMULA_CHARS = frozenset("=+-@\t")


def _safe_csv_value(value: str) -> str:
    """Defuse potential CSV formula injection by tab-prefixing dangerous values.

    Checks the first non-whitespace character so that space-padded formulas
    like ' =SUM(...)' are also caught.
    """
    if value and value.lstrip()[0:1] in _FORMULA_CHARS:
        return "\t" + value
    return value



async def generate_patient_export_csv(
    db: AsyncSession,
    cancer_type: str | None = None,
    county: str | None = None,
    zip_code: str | None = None,
    sex: str | None = None,
    breed: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> str:
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

    if cancer_type:
        stmt = stmt.where(CancerType.name == cancer_type)
    if county:
        stmt = stmt.where(County.name == county)
    if zip_code:
        stmt = stmt.where(Patient.zip_code == zip_code)
    if sex:
        stmt = stmt.where(Patient.sex == sex)
    if breed:
        stmt = stmt.where(Breed.name == breed)
    if year_start is not None:
        stmt = stmt.where(func.extract("year", Patient.diagnosis_date) >= year_start)
    if year_end is not None:
        stmt = stmt.where(func.extract("year", Patient.diagnosis_date) <= year_end)

    result = await db.execute(stmt)
    rows = result.all()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "species": _safe_csv_value(row.species or ""),
            "breed": _safe_csv_value(row.breed or ""),
            "sex": _safe_csv_value(row.sex or ""),
            "county": _safe_csv_value(row.county or ""),
            "zip_code": _safe_csv_value(row.zip_code or ""),
            "diagnosis_date": str(row.diagnosis_date) if row.diagnosis_date else "",
            "outcome": _safe_csv_value(row.outcome or ""),
            "cancer_type": _safe_csv_value(row.cancer_type or ""),
            "icd_o_code": _safe_csv_value(row.icd_o_code or ""),
            "confidence": float(row.confidence) if row.confidence is not None else "",
            "review_status": _safe_csv_value(row.review_status or ""),
        })

    return output.getvalue()
