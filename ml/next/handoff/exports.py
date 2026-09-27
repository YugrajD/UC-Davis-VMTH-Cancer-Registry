"""Exports: versioned files the backend developer's cloud imports consume.

Three outbox kinds (ml-rewrite-plan.md Artefacts, "Handoff"):

- ``export_silver`` — ``silver_codes_<silver_id>.csv``: the diagnosis-mapping
  cascade's codes for a silver generation, so the backend can apply the coding
  rule. No diagnosis text (the cloud already has it — it sent it as
  pending_diagnoses).
- ``export_coding`` — ``adopted_codes_<run>.csv`` + ``review_queue_<run>.csv``:
  the already-computed ``coding.adopt`` / ``coding.queue`` outputs, re-stamped
  with a schema version and copied to the outbox. Both are already free of
  report/diagnosis text.
- ``export_bundle`` — a tarball of a report-mapping generation directory (what
  ml-worker needs to serve predictions), plus a sha256 sidecar file. The
  source generation is verified (manifest file-hashes + embedding fingerprint,
  via ``handoff.worker_format.verify_worker_bundle``) *before* bundling, so a
  stale or tampered generation is refused rather than shipped;
  ``verify_bundle`` re-runs the same check against the tarball's own contents
  after extraction, plus the sha256 sidecar, for the receiving side (and for
  this module's own tests) to confirm the tarball travelled intact.

Every non-bundle export gets a ``contracts.write_sidecar`` (schema_version,
sha256) next to it, per the module docstring's inbox/outbox convention.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from generations.manifest import sha256_file
from handoff import contracts, worker_format
from report_mapping.model.generation import resolve_generation_dir


class HandoffExportError(Exception):
    """An export was refused, or a bundle failed verification."""


def _check_required_columns(path: Path, actual, expected: list[str]) -> None:
    missing = set(expected) - set(actual)
    if missing:
        raise HandoffExportError(f"{path}: missing required column(s) {sorted(missing)}")


def export_silver(silver_id: str, *, out_dir: str | Path | None = None) -> Path:
    """Write ``silver_codes_<silver_id>.csv`` (codes only, no diagnosis text)."""
    df = load_silver(silver_id)
    df = df[contracts.SILVER_EXPORT_COLUMNS]

    out_dir = Path(out_dir) if out_dir is not None else config.HANDOFF_OUTBOX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"silver_codes_{silver_id}.csv"
    io_utils.write_csv(df, out_path)
    contracts.write_sidecar(out_path, kind=contracts.SILVER_EXPORT_KIND, schema_version=contracts.SILVER_EXPORT_SCHEMA_VERSION)
    return out_path


def export_coding(
    run_id: str,
    *,
    adopted_csv: str | Path | None = None,
    queue_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Write ``adopted_codes_<run_id>.csv`` and ``review_queue_<run_id>.csv``
    from the already-built ``coding.adopt`` / ``coding.queue`` outputs."""
    adopted_src = Path(adopted_csv) if adopted_csv is not None else config.ADOPTED_CODES_CSV
    queue_src = Path(queue_csv) if queue_csv is not None else config.REVIEW_QUEUE_CSV
    out_dir = Path(out_dir) if out_dir is not None else config.HANDOFF_OUTBOX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    adopted = io_utils.read_csv(adopted_src, encoding="utf-8", dtype=str, keep_default_na=False)
    _check_required_columns(adopted_src, adopted.columns, contracts.ADOPTED_CODES_EXPORT_COLUMNS)
    adopted_out = out_dir / f"adopted_codes_{run_id}.csv"
    io_utils.write_csv(adopted, adopted_out)
    contracts.write_sidecar(
        adopted_out, kind=contracts.ADOPTED_CODES_EXPORT_KIND, schema_version=contracts.ADOPTED_CODES_EXPORT_SCHEMA_VERSION,
    )

    queue = io_utils.read_csv(queue_src, encoding="utf-8", dtype=str, keep_default_na=False)
    _check_required_columns(queue_src, queue.columns, contracts.REVIEW_QUEUE_EXPORT_COLUMNS)
    queue_out = out_dir / f"review_queue_{run_id}.csv"
    io_utils.write_csv(queue, queue_out)
    contracts.write_sidecar(
        queue_out, kind=contracts.REVIEW_QUEUE_EXPORT_KIND, schema_version=contracts.REVIEW_QUEUE_EXPORT_SCHEMA_VERSION,
    )

    return {
        "adopted_codes_path": adopted_out, "review_queue_path": queue_out,
        "adopted_rows": len(adopted), "review_queue_rows": len(queue),
    }


def export_bundle(generation_dir: str | Path | None = None, *, out_dir: str | Path | None = None) -> dict:
    """Tar ``generation_dir`` (default ``current``) and write a sha256 sidecar.

    Refuses (via ``worker_format.verify_worker_bundle``) before bundling if the
    source generation's manifest or embedding fingerprint doesn't check out.
    """
    directory = resolve_generation_dir(generation_dir) if generation_dir is not None else config.REPORT_MAPPING_CURRENT_DIR
    manifest = worker_format.verify_worker_bundle(directory)
    generation_id = manifest.get("generation_id", directory.name)

    out_dir = Path(out_dir) if out_dir is not None else config.HANDOFF_BUNDLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = out_dir / f"bundle_{generation_id}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(directory, arcname=directory.name)

    checksum = sha256_file(tarball_path)
    sha256_path = tarball_path.with_name(tarball_path.name + ".sha256")
    sha256_path.write_text(f"{checksum}  {tarball_path.name}\n", encoding="utf-8", newline="\n")

    return {
        "tarball_path": tarball_path, "sha256_path": sha256_path,
        "sha256": checksum, "generation_id": generation_id,
    }


def verify_bundle(tarball_path: str | Path, *, sha256_path: str | Path | None = None) -> dict:
    """Verify a bundle's sha256 sidecar, then extract it to a temp dir and
    re-run ``worker_format.verify_worker_bundle`` against its contents — the
    receiving side's (and this module's own tests') confirmation that the
    tarball travelled intact and is still a loadable generation."""
    tarball_path = Path(tarball_path)
    sha256_path = Path(sha256_path) if sha256_path is not None else tarball_path.with_name(tarball_path.name + ".sha256")
    if not sha256_path.is_file():
        raise HandoffExportError(f"{sha256_path} is missing")
    recorded = sha256_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(tarball_path)
    if actual != recorded:
        raise HandoffExportError(f"{tarball_path}: sha256 {actual} does not match {sha256_path} ({recorded})")

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        extracted = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(extracted) != 1:
            raise HandoffExportError(f"{tarball_path}: expected exactly one top-level directory, found {len(extracted)}")
        manifest = worker_format.verify_worker_bundle(extracted[0])

    return {"sha256_verified": True, "generation_id": manifest.get("generation_id")}
