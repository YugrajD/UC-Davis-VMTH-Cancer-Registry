"""Retraining CSV exports over the registry's finalized diagnoses.

The per-row Review Queue this router used to also serve (confirm/correct/
reject) is retired — the Audit Worklist (backend/app/routers/audit_review.py)
replaces it entirely, per ml/documentation/audit-list-change-request.md. This
export feature is kept: it's a bulk, ongoing feed over the full
case_diagnoses table (tens of thousands of rows), not the curated few-hundred
-case sample the Audit Worklist's gold export covers — the two don't overlap.

Endpoints:
  GET  /api/v1/diagnoses/export/audited.csv  - manually audited diagnoses only
  GET  /api/v1/diagnoses/export/all.csv      - every finalized diagnosis
"""

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.rate_limit import limiter
from app.database import get_db
from app.models.models import CancerType, CaseDiagnosis, Patient, PathologyReport

router = APIRouter(prefix="/api/v1/diagnoses", tags=["diagnoses-review"])


# --- Retraining exports ----------------------------------------------------
#
# Trimmed to just what a retraining pipeline needs: the input text
# (pathology_reports.source_diagnosis — the same "Clinical Diagnoses" cell
# text mirrored into GCS, but reading it from the DB avoids a per-row GCS
# fetch across tens of thousands of rows) paired with the finalized label
# (cancer_type + icd_o_code). Scoped to real data with a settled
# confirmed/corrected label — 'pending' isn't finalized yet and 'rejected'
# means a human said this isn't a valid diagnosis, so neither is a usable
# training pair.

_EXPORT_CSV_COLUMNS = ["case_id", "diagnosis_index", "clinical_diagnosis", "cancer_type", "icd_o_code"]

# Characters spreadsheet apps (Excel, Google Sheets) interpret as formula
# starters — prefix with a tab to defuse CSV formula injection. Mirrors
# app.services.export_service._safe_csv_value.
_FORMULA_CHARS = frozenset("=+-@\t")


def _safe_csv_value(value: str) -> str:
    if value and value.lstrip()[0:1] in _FORMULA_CHARS:
        return "\t" + value
    return value


async def _generate_diagnoses_export_csv(db: AsyncSession, audited_only: bool) -> str:
    stmt = (
        select(
            Patient.anon_id,
            CaseDiagnosis.diagnosis_index,
            PathologyReport.source_diagnosis,
            CancerType.name.label("cancer_type"),
            CaseDiagnosis.icd_o_code,
        )
        .select_from(CaseDiagnosis)
        .join(Patient, Patient.id == CaseDiagnosis.patient_id)
        .join(PathologyReport, PathologyReport.id == CaseDiagnosis.pathology_report_id)
        .join(CancerType, CancerType.id == CaseDiagnosis.cancer_type_id)
        .where(Patient.data_source == "petbert")
        .where(CaseDiagnosis.review_status.in_(["confirmed", "corrected"]))
        .where(PathologyReport.source_diagnosis.isnot(None))
        .where(PathologyReport.source_diagnosis != "")
        .order_by(Patient.anon_id, CaseDiagnosis.diagnosis_index)
    )
    if audited_only:
        stmt = stmt.where(CaseDiagnosis.reviewed_by_email.isnot(None))

    rows = (await db.execute(stmt)).all()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_EXPORT_CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "case_id": _safe_csv_value(row.anon_id or ""),
            "diagnosis_index": row.diagnosis_index if row.diagnosis_index is not None else "",
            "clinical_diagnosis": _safe_csv_value(row.source_diagnosis or ""),
            "cancer_type": _safe_csv_value(row.cancer_type or ""),
            "icd_o_code": _safe_csv_value(row.icd_o_code or ""),
        })
    return output.getvalue()


@router.get("/export/audited.csv")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def export_audited_diagnoses_csv(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """Manually audited diagnoses only (a human confirmed/corrected it). Admin-only."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin role required")
    csv_text = await _generate_diagnoses_export_csv(db, audited_only=True)
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audited_diagnoses.csv"},
    )


@router.get("/export/all.csv")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def export_all_diagnoses_csv(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """All finalized diagnoses (confirmed/corrected) — includes the manually audited subset. Admin-only."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin role required")
    csv_text = await _generate_diagnoses_export_csv(db, audited_only=False)
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=all_diagnoses.csv"},
    )
