"""Schemas + versions for every file that crosses between ML and the registry app.

Per ml-rewrite-plan.md's Artefacts ("Handoff") and icd-mapping-strategy.md
("Data crosses manually, through exports and imports the backend developer
builds"): every exchanged file gets a schema version stamped in a sidecar
``<file>.manifest.json`` ({"kind", "schema_version", "sha256", "written_at"}),
so a schema change on either side is visible rather than silently
misinterpreted. Column lists here are read from their owning module's own
constants where one already exists (``diagnosis_mapping.silver``,
``coding.adopt``, ``coding.queue``) rather than duplicated, per CLAUDE.md
("Don't reach through modules... call the public interface").

Bump a ``*_SCHEMA_VERSION`` constant whenever that file's columns change in a
way the other side must know about (a column added, removed or renamed).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from coding.adopt import ADOPTED_CODES_COLUMNS
from coding.queue import REVIEW_QUEUE_COLUMNS
from diagnosis_mapping.silver import ANNOTATION_COLUMNS, DIAG_NUM_COL, ID_COL, TEXT_COL
from generations.manifest import sha256_file

SIDECAR_SUFFIX = ".manifest.json"

# ---------------------------------------------------------------------------
# Inbox: pending diagnoses (cloud -> ML). Case text (the ``diagnosis`` column),
# never committed — lands under config.HANDOFF_INBOX_DIR only.
# ---------------------------------------------------------------------------
PENDING_DIAGNOSES_KIND = "pending_diagnoses"
PENDING_DIAGNOSES_SCHEMA_VERSION = 1
PENDING_DIAGNOSES_REQUIRED_COLUMNS = [ID_COL, TEXT_COL]
PENDING_DIAGNOSES_COLUMNS = [ID_COL, DIAG_NUM_COL, TEXT_COL]

# ---------------------------------------------------------------------------
# Inbox: gold (cloud -> ML). ``origin`` is mandatory; storage itself is
# manual_audit.gold's job (see handoff/imports.py) — this only names the
# minimum shape an import must have before it can even try that ingest.
# ---------------------------------------------------------------------------
GOLD_IMPORT_KIND = "gold"
GOLD_IMPORT_SCHEMA_VERSION = 1
GOLD_IMPORT_REQUIRED_COLUMNS = ["case_id", "origin"]

# ---------------------------------------------------------------------------
# Outbox: silver codes (ML -> cloud). No ``diagnosis`` text column — the cloud
# already holds the diagnosis text it sent as pending_diagnoses; re-sending it
# would be an unnecessary export of case text.
# ---------------------------------------------------------------------------
SILVER_EXPORT_KIND = "silver_codes"
SILVER_EXPORT_SCHEMA_VERSION = 1
SILVER_EXPORT_COLUMNS = [c for c in ANNOTATION_COLUMNS if c != TEXT_COL] + ["silver_generation"]

# ---------------------------------------------------------------------------
# Outbox: adopted codes + review queue (ML -> cloud). Both are already
# text-free (coding.adopt / coding.queue never carry report or diagnosis text).
# ---------------------------------------------------------------------------
ADOPTED_CODES_EXPORT_KIND = "adopted_codes"
ADOPTED_CODES_EXPORT_SCHEMA_VERSION = 1
ADOPTED_CODES_EXPORT_COLUMNS = ADOPTED_CODES_COLUMNS

REVIEW_QUEUE_EXPORT_KIND = "review_queue"
REVIEW_QUEUE_EXPORT_SCHEMA_VERSION = 1
REVIEW_QUEUE_EXPORT_COLUMNS = REVIEW_QUEUE_COLUMNS

# ---------------------------------------------------------------------------
# Outbox: report-mapping generation bundle (ML -> cloud, for ml-worker). The
# bundle's own internal shape is report_mapping.model.generation's layout
# (see handoff/worker_format.py); this is just the tarball's own schema version.
# ---------------------------------------------------------------------------
BUNDLE_KIND = "report_mapping_bundle"
BUNDLE_SCHEMA_VERSION = 1


def write_sidecar(path: str | Path, *, kind: str, schema_version: int) -> Path:
    """Write ``<path><SIDECAR_SUFFIX>`` recording {kind, schema_version, sha256,
    written_at (ISO UTC)}. The one sidecar shape every handoff file gets."""
    path = Path(path)
    sidecar = {
        "kind": kind,
        "schema_version": schema_version,
        "sha256": sha256_file(path),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar_path = path.with_name(path.name + SIDECAR_SUFFIX)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


class SidecarError(Exception):
    """A handoff file's sidecar is missing, or the file no longer matches it."""


def read_sidecar(path: str | Path) -> dict:
    path = Path(path)
    sidecar_path = path.with_name(path.name + SIDECAR_SUFFIX)
    if not sidecar_path.is_file():
        raise SidecarError(f"{sidecar_path} is missing")
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def verify_sidecar(path: str | Path, *, expected_kind: str | None = None) -> dict:
    """Verify ``path`` still matches its sidecar's recorded sha256 (and, if
    given, ``expected_kind``). Returns the sidecar on success."""
    path = Path(path)
    sidecar = read_sidecar(path)
    if expected_kind is not None and sidecar.get("kind") != expected_kind:
        raise SidecarError(f"{path}: sidecar kind {sidecar.get('kind')!r} != expected {expected_kind!r}")
    actual = sha256_file(path)
    if actual != sidecar.get("sha256"):
        raise SidecarError(f"{path}: sha256 {actual} does not match its sidecar ({sidecar.get('sha256')})")
    return sidecar
