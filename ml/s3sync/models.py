"""publish / pull the report-mapping generation ``current/`` through S3.

Remote layout under the prefix: ``generations/<generation_id>/<relpath>`` (a readable, immutable copy of the
generation directory, ``manifest.json`` written last so its presence means "complete") and
``generations/CURRENT.json`` (the one mutable pointer, fast-forwarded like a set HEAD). Only ``current/`` is ever
published, so unpromoted candidates and archived generations never reach S3. Dry runs make no writes.

``generations.promote`` (torch, pandas) is imported lazily so plain set operations stay light; its errors
surface here as ``S3SyncError``.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import date

import config
from generations.manifest import ManifestError, read_manifest, sha256_file, verify_manifest
from s3sync import state
from s3sync.files import PARTIAL_SUFFIX, check_rel, download_verified
from s3sync.remote import ConflictError, Remote, S3SyncError

_MANIFEST = "manifest.json"


@contextmanager
def _generation_checks():
    """Yield ``generations.promote``; its checks refuse with PromotionError / ManifestError / GenerationError."""
    from generations import promote
    from report_mapping.model.generation import GenerationError

    try:
        yield promote
    except (promote.PromotionError, ManifestError, GenerationError) as error:
        raise S3SyncError(str(error)) from error


def _pointer(remote: Remote) -> tuple[str, str] | None:
    """(generation_id, etag) of the remote CURRENT pointer, or None if nothing was ever published."""
    current = remote.get_json(remote.key("generations", "CURRENT.json"))
    if current is None:
        return None
    generation_id = current[0].get("generation_id") if isinstance(current[0], dict) else None
    if not isinstance(generation_id, str):
        raise S3SyncError("malformed CURRENT.json (no generation_id)")
    check_rel(generation_id)
    if "/" in generation_id:
        raise S3SyncError(f"malformed CURRENT.json (generation_id {generation_id!r})")
    return generation_id, current[1]


def _moved(pointer, synced) -> bool:
    return pointer is not None and (synced is None or pointer[1] != synced["etag"])


def _local_id() -> str | None:
    current = config.REPORT_MAPPING_CURRENT_DIR
    if not current.exists():
        return None
    try:
        return read_manifest(current)["generation_id"]
    except ManifestError as error:
        raise S3SyncError(str(error)) from error


def status(remote: Remote) -> dict:
    pointer, synced = _pointer(remote), state.synced(remote, state.MODELS)
    return {"remote_generation": pointer[0] if pointer else None, "local_generation": _local_id(),
            "never_synced": synced is None, "remote_moved": _moved(pointer, synced)}


def publish(remote: Remote, apply: bool = False) -> dict:
    current = config.REPORT_MAPPING_CURRENT_DIR
    with _generation_checks() as promote:
        manifest = promote.check_candidate(current)  # manifest, embedding fingerprint and calibration
    generation_id, files = manifest["generation_id"], manifest["files"]
    pointer, synced = _pointer(remote), state.synced(remote, state.MODELS)
    result = {"generation_id": generation_id, "applied": False, "nothing_to_do": False,
              "files": len(files), "bytes": sum((current / rel).stat().st_size for rel in files),
              "already_uploaded": False}

    if pointer is not None and pointer[0] == generation_id:
        result["nothing_to_do"] = True
        if apply:
            state.save(remote, state.MODELS, {"generation_id": generation_id, "etag": pointer[1]})
        return result
    if _moved(pointer, synced):
        raise ConflictError("remote model moved, pull-model first")
    manifest_key = remote.key("generations", generation_id, _MANIFEST)
    result["already_uploaded"] = remote.head_etag(manifest_key) is not None
    if not apply:
        return result

    if not result["already_uploaded"]:
        for rel, sha in files.items():
            remote.put_file_if_absent(remote.key("generations", generation_id, rel), current / rel, sha)
    # Also when already uploaded: an existing manifest.json with a different checksum must refuse, not be trusted.
    remote.put_file_if_absent(manifest_key, current / _MANIFEST, sha256_file(current / _MANIFEST))
    try:
        etag = remote.move_pointer(remote.key("generations", "CURRENT.json"), {"generation_id": generation_id},
                                   pointer[1] if pointer else None)
    except ConflictError as error:
        raise ConflictError("remote model moved, pull-model first") from error
    state.save(remote, state.MODELS, {"generation_id": generation_id, "etag": etag})
    result["applied"] = True
    return result


def pull_model(remote: Remote, apply: bool = False) -> dict:
    candidate, current = config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR
    pointer, synced, local_id = _pointer(remote), state.synced(remote, state.MODELS), _local_id()
    result = {"action": "nothing", "applied": False, "remote_generation": pointer[0] if pointer else None,
              "local_generation": local_id, "files": 0, "archive": None, "warning": None}
    if pointer is None:
        return result
    generation_id, etag = pointer
    if local_id == generation_id:
        if apply:
            state.save(remote, state.MODELS, {"generation_id": generation_id, "etag": etag})
        return result
    if local_id is not None and synced is not None and synced["etag"] == etag:
        result["action"] = f"keep local: current {local_id} is unpublished; run publish-model"
        return result
    if candidate.exists():
        raise S3SyncError("candidate/ exists; promote or delete it first")

    today = date.today()
    if local_id is not None:  # adopt will archive it: refuse now if it cannot, before downloading anything
        with _generation_checks() as promote:
            verify_manifest(current)
            result["archive"] = promote.archive_path(local_id, today=today)
        if result["archive"].exists():
            raise S3SyncError(f"{result['archive']} already exists; move it aside before pulling")
        if synced is None or synced["generation_id"] != local_id:
            result["warning"] = f"local current {local_id} is UNPUBLISHED and will be archived to {result['archive']}"

    manifest = remote.get_json(remote.key("generations", generation_id, _MANIFEST))
    if manifest is None:
        raise S3SyncError(f"generation {generation_id!r} has no manifest.json; its publish never completed")
    files = manifest[0].get("files") if isinstance(manifest[0], dict) else None
    if not isinstance(files, dict):
        raise S3SyncError(f"generation {generation_id!r} has a malformed manifest.json")
    for rel in files:
        check_rel(rel)
    result.update(action=f"fetch {generation_id} into candidate/ and adopt it", files=len(files))
    if not apply:
        return result

    try:
        candidate.mkdir(parents=True)
    except FileExistsError as error:  # created since the check above; not ours, so never cleaned up
        raise S3SyncError("candidate/ exists; promote or delete it first") from error
    try:  # from here on candidate/ is ours: any failure removes it whole
        for rel, sha in files.items():
            dest = candidate / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + PARTIAL_SUFFIX)
            download_verified(remote, remote.key("generations", generation_id, rel), temp, sha)
            os.replace(temp, dest)
        remote.download(remote.key("generations", generation_id, _MANIFEST), candidate / _MANIFEST)
        with _generation_checks() as promote:
            if promote.check_candidate(candidate)["generation_id"] != generation_id:
                raise S3SyncError(f"downloaded manifest is not generation {generation_id!r}")
            result["adopted"] = promote.adopt(today=today)
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)  # no-op once adopt has renamed it into current/
        raise
    if result["warning"]:
        result["warning"] = result["warning"].replace("will be archived", "was archived")
    state.save(remote, state.MODELS, {"generation_id": generation_id, "etag": etag})
    result["applied"] = True
    return result
