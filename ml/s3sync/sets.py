"""status / push / pull for mutable file sets (see ``config.S3_SYNC_SETS``).

Remote layout under the prefix: ``blobs/<sha256>`` (immutable contents), ``sets/<set>/manifests/<id>.json``
(immutable) and ``sets/<set>/HEAD.json`` (the one mutable pointer). Push and pull are dry runs unless ``apply``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

import config
from generations.manifest import sha256_file
from s3sync import state
from s3sync.files import PARTIAL_SUFFIX, check_rel, download_verified
from s3sync.remote import ConflictError, Remote, S3SyncError

_JUNK_NAMES = {".DS_Store"}


def set_names(name: str) -> list[str]:
    if any(configured.startswith("_") for configured in config.S3_SYNC_SETS):
        raise S3SyncError("set names starting with '_' are reserved for sync state keys")
    if name == "all":
        return list(config.S3_SYNC_SETS)
    if name not in config.S3_SYNC_SETS:
        raise S3SyncError(f"unknown set {name!r}; known: {', '.join(config.S3_SYNC_SETS)} or all")
    return [name]


def _is_excluded(root: Path, rel: str) -> bool:
    """Junk files and anything under ``config.S3_SYNC_EXCLUDED_DIRS`` are never part of a set."""
    parts = Path(rel).parts
    if parts[-1] in _JUNK_NAMES or parts[-1].endswith(PARTIAL_SUFFIX) or "__pycache__" in parts:
        return True
    return any((root / rel).is_relative_to(Path(d)) for d in config.S3_SYNC_EXCLUDED_DIRS)


def local_files(set_name: str) -> dict[str, dict]:
    """{posix relpath: {sha256, size}} of the set's directory, minus junk and excluded dirs."""
    root = Path(config.S3_SYNC_SETS[set_name])
    files = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and not _is_excluded(root, rel):
            files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return files


def _shas(files: dict[str, dict]) -> dict[str, str]:
    return {rel: info["sha256"] for rel, info in files.items()}


def _diff(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(new.keys() - old.keys()),
        "changed": sorted(rel for rel in new.keys() & old.keys() if new[rel] != old[rel]),
        "removed": sorted(old.keys() - new.keys()),
    }


def _remote_head(remote: Remote, set_name: str):
    """(manifest id, manifest files {rel: {sha256, size}}, HEAD etag), or None if the set was never pushed."""
    head = remote.get_json(remote.key("sets", set_name, "HEAD.json"))
    if head is None:
        return None
    manifest_id = head[0].get("manifest")
    if not isinstance(manifest_id, str):
        raise S3SyncError(f"{set_name}: malformed HEAD.json (no manifest id)")
    manifest = remote.get_json(remote.key("sets", set_name, "manifests", f"{manifest_id}.json"))
    if manifest is None:
        raise S3SyncError(f"{set_name}: HEAD points to missing manifest {manifest_id!r}")
    return manifest_id, manifest[0]["files"], head[1]


def _remote_moved(head, synced) -> bool:
    return head is not None and (synced is None or head[2] != synced["etag"])


def status(remote: Remote, set_name: str) -> dict:
    synced = state.synced(remote, set_name)
    head = _remote_head(remote, set_name)
    return {
        "set": set_name,
        "never_synced": synced is None,
        "remote_manifest": head[0] if head else None,
        "remote_moved": _remote_moved(head, synced),
        "local": _diff(synced["files"] if synced else {}, _shas(local_files(set_name))),
    }


def push(remote: Remote, set_name: str, apply: bool = False) -> dict:
    synced = state.synced(remote, set_name)
    head = _remote_head(remote, set_name)
    if _remote_moved(head, synced):
        raise ConflictError(f"{set_name}: remote moved, pull first")
    root = Path(config.S3_SYNC_SETS[set_name])
    local = local_files(set_name)
    # An absent or empty directory would publish "delete everything".
    if head is not None and head[1] and (not root.is_dir() or not local):
        raise S3SyncError(f"{set_name}: local directory {root} is missing or empty but the remote set has files")
    change = _diff(_shas(head[1]) if head else {}, _shas(local))
    result = {"set": set_name, "applied": False, "change": change, "nothing_to_do": not any(change.values())}
    if result["nothing_to_do"] or not apply:
        return result

    for sha, rel in {info["sha256"]: rel for rel, info in local.items()}.items():
        remote.put_file_if_absent(remote.key("blobs", sha), root / rel, sha)
    created = datetime.now(timezone.utc)
    manifest = {"set": set_name, "created_at": created.isoformat(), "created_by": socket.gethostname(),
                "parent": head[0] if head else None, "files": local}
    digest = hashlib.sha256(json.dumps(local, sort_keys=True).encode()).hexdigest()[:8]
    manifest_id = f"{created:%Y%m%dT%H%M%SZ}-{digest}"
    remote.put_json_if_absent(remote.key("sets", set_name, "manifests", f"{manifest_id}.json"), manifest)
    try:
        etag = remote.move_pointer(remote.key("sets", set_name, "HEAD.json"), {"manifest": manifest_id},
                                   head[2] if head else None)
    except ConflictError as error:  # the orphan manifest and blobs are harmless
        raise ConflictError(f"{set_name}: remote moved, pull first") from error
    state.save(remote, set_name, {"manifest": manifest_id, "etag": etag, "files": _shas(local)})
    result["applied"] = True
    return result


def _backup(source: Path, set_name: str, rel: str, stamp: str, move: bool) -> str:
    target = Path(config.S3_SYNC_BACKUP_DIR) / stamp / set_name / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(source, target)
    return rel


def _plan_pull(set_name: str, wanted: dict[str, str], base: dict[str, str]) -> dict:
    """3-way per file: base = last-synced sha, mine = local sha, theirs = remote sha."""
    local = _shas(local_files(set_name))
    plan = {"fetch": [], "conflicts": [], "kept_modified": [], "remove": []}
    for rel, theirs in sorted(wanted.items()):
        mine, old = local.get(rel), base.get(rel)
        if mine == theirs:
            continue
        if theirs == old:  # remote did not change it, so what differs is our edit (or deletion)
            plan["kept_modified"].append(rel)
        elif mine is None or mine == old:  # only the remote changed it (or nothing local to lose)
            plan["fetch"].append(rel)
        else:  # both changed it differently: remote wins, our copy is backed up
            plan["conflicts"].append(rel)
    for rel in sorted(base.keys() - wanted.keys()):  # deleted remotely
        if rel in local:
            plan["remove" if local[rel] == base[rel] else "kept_modified"].append(rel)
    plan["kept_modified"].sort()
    return plan


def pull(remote: Remote, set_name: str, apply: bool = False) -> dict:
    synced = state.synced(remote, set_name)
    head = _remote_head(remote, set_name)
    result = {"set": set_name, "applied": False, "fetch": [], "conflicts": [], "kept_modified": [],
              "remove": [], "skipped_excluded": [], "backed_up": []}
    if head is None:
        return result
    manifest_id, remote_files, etag = head
    root = Path(config.S3_SYNC_SETS[set_name])

    wanted = {}
    for rel, info in remote_files.items():
        check_rel(rel)
        if _is_excluded(root, rel):
            result["skipped_excluded"].append(rel)
        else:
            wanted[rel] = info["sha256"]
    result["skipped_excluded"].sort()
    result.update(_plan_pull(set_name, wanted, synced["files"] if synced else {}))
    if not apply:
        return result

    # Download and verify everything before touching any local file.
    temps = {}
    try:
        for rel in result["fetch"] + result["conflicts"]:
            info = remote_files[rel]
            temp = (root / rel).with_name((root / rel).name + PARTIAL_SUFFIX)
            temp.parent.mkdir(parents=True, exist_ok=True)
            temps[rel] = temp
            download_verified(remote, remote.key("blobs", info["sha256"]), temp, info["sha256"], info["size"])
    except Exception:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        raise

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        for rel in list(temps):
            if (root / rel).exists():
                result["backed_up"].append(_backup(root / rel, set_name, rel, stamp, move=False))
            os.replace(temps[rel], root / rel)
            del temps[rel]
        for rel in result["remove"]:
            result["backed_up"].append(_backup(root / rel, set_name, rel, stamp, move=True))
    except OSError as error:
        raise S3SyncError("pull partially applied; close the locked file and re-run pull") from error
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
    state.save(remote, set_name, {"manifest": manifest_id, "etag": etag, "files": wanted})
    result["applied"] = True
    return result
