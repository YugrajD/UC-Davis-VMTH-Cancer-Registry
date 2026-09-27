"""The dashboard review worklist (audit list) and gold export.

Pairs with database/migrations/033_gold_review.sql and
ml/documentation/audit-list-change-request.md. Admin-only for now (import);
worklist/review-screen/export endpoints land in later commits.

Endpoints:
  POST /api/v1/audit-review/lists/import   - import audit_list_<id>.txt + its
                                              .manifest.json sidecar, replacing
                                              the active worklist
"""

import hashlib
import json

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import CurrentUser, require_admin
from app.config import settings
from app.database import get_db
from app.models.models import AuditList, AuditListCase, Patient
from app.rate_limit import limiter
from app.services.ingestion_service import normalize_anon_id

router = APIRouter(prefix="/api/v1/audit-review", tags=["audit-review"])

# Matches ml/handoff/contracts.py's AUDIT_LIST_EXPORT_KIND / SCHEMA_VERSION —
# duplicated here since production has no /ml to import them from. Bump the
# version only in lockstep with a matching change on ML's side.
_AUDIT_LIST_KIND = "audit_list"
_AUDIT_LIST_SCHEMA_VERSION = 1
_FILENAME_PREFIX = "audit_list_"
_FILENAME_SUFFIX = ".txt"


class AuditListImportSummary(BaseModel):
    list_id: str
    case_count: int
    replaced_list_id: str | None
    not_found: list[str]


def _parse_list_id(filename: str) -> str:
    name = (filename or "").strip()
    if not (name.startswith(_FILENAME_PREFIX) and name.endswith(_FILENAME_SUFFIX)):
        raise HTTPException(
            status_code=400,
            detail=f"File name must look like {_FILENAME_PREFIX}<list_id>{_FILENAME_SUFFIX}, got {filename!r}",
        )
    list_id = name[len(_FILENAME_PREFIX):-len(_FILENAME_SUFFIX)]
    if not list_id:
        raise HTTPException(status_code=400, detail="File name is missing a list_id")
    return list_id


def _parse_case_ids(list_bytes: bytes) -> list[str]:
    try:
        text = list_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Audit list must be UTF-8 encoded")

    case_ids = [line.strip() for line in text.split("\n") if line.strip()]
    if not case_ids:
        raise HTTPException(status_code=400, detail="Audit list has no case IDs")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"Audit list repeats case_id(s): {', '.join(sorted(duplicates)[:20])}",
        )
    return case_ids


@router.post("/lists/import")
@limiter.limit(settings.RATE_LIMIT_WRITE)
async def import_audit_list(
    request: Request,
    list_file: UploadFile = File(...),
    manifest_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_admin),
) -> AuditListImportSummary:
    """Admin-only. Loads audit_list_<list_id>.txt as the specialist's worklist,
    replacing whichever list is currently active. Every list is the complete
    current worklist, not an increment — a case that drops off a newer list
    simply stops appearing, though its audit_list_cases row (and any review)
    is kept: a case is still safe to export as long as it was ever on some
    list, since ML refuses the whole gold file for a case_id that never was."""
    list_id = _parse_list_id(list_file.filename or "")

    list_bytes = await list_file.read()
    if not list_bytes:
        raise HTTPException(status_code=400, detail="Audit list file is empty")

    manifest_bytes = await manifest_file.read()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Manifest sidecar is not valid JSON")

    if manifest.get("kind") != _AUDIT_LIST_KIND:
        raise HTTPException(
            status_code=400,
            detail=f"Manifest kind {manifest.get('kind')!r} != {_AUDIT_LIST_KIND!r}",
        )
    if manifest.get("schema_version") != _AUDIT_LIST_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Manifest schema_version {manifest.get('schema_version')!r} != "
                f"{_AUDIT_LIST_SCHEMA_VERSION!r} (this backend doesn't know that shape)"
            ),
        )
    expected_sha256 = manifest.get("sha256")
    if not expected_sha256:
        raise HTTPException(status_code=400, detail="Manifest is missing sha256")
    actual_sha256 = hashlib.sha256(list_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HTTPException(
            status_code=400,
            detail=f"sha256 {actual_sha256} does not match the manifest's recorded {expected_sha256}",
        )

    case_ids = _parse_case_ids(list_bytes)

    existing = (
        await db.execute(select(AuditList.id).where(AuditList.list_id == list_id))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"list_id {list_id!r} was already imported")

    previous_active = (
        await db.execute(select(AuditList.list_id).where(AuditList.is_active.is_(True)))
    ).scalar_one_or_none()

    await db.execute(update(AuditList).where(AuditList.is_active.is_(True)).values(is_active=False))

    new_list = AuditList(
        list_id=list_id,
        imported_by_email=user.email,
        sha256=actual_sha256,
        case_count=len(case_ids),
        is_active=True,
    )
    db.add(new_list)
    await db.flush()

    db.add_all(
        AuditListCase(audit_list_id=new_list.id, case_id=case_id, position=position)
        for position, case_id in enumerate(case_ids)
    )

    # Informational only — an unmatched case still gets imported (shown in the
    # worklist flagged "no record found"), never dropped from the list.
    normalized_ids = {normalize_anon_id(case_id) for case_id in case_ids}
    normalized_ids.discard("")
    found = set(
        (await db.execute(select(Patient.anon_id).where(Patient.anon_id.in_(normalized_ids)))).scalars()
    )
    not_found = [case_id for case_id in case_ids if normalize_anon_id(case_id) not in found]

    await db.commit()

    return AuditListImportSummary(
        list_id=list_id,
        case_count=len(case_ids),
        replaced_list_id=previous_active,
        not_found=not_found,
    )
