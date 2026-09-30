"""Checks shared by everything that writes remote data into local directories."""

from __future__ import annotations

from pathlib import Path

from generations.manifest import sha256_file
from s3sync.remote import GuardError, Remote, S3SyncError

PARTIAL_SUFFIX = ".s3sync-partial"  # a download that has not yet been verified
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
                   *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}


class VerificationError(S3SyncError):
    """A downloaded file does not match its manifest."""


def check_rel(rel: str) -> None:
    """Refuse a manifest path that could write outside its directory (or to a Windows device); manifests are remote data."""
    segments = rel.split("/")
    unsafe = (
        "\\" in rel or ":" in rel or rel.startswith("/")
        or any(s in ("", ".", "..") or s.endswith((".", " ")) or s.split(".")[0].upper() in _RESERVED_NAMES
               for s in segments)
    )
    if unsafe:
        raise GuardError(f"manifest holds an unsafe path {rel!r}")


def download_verified(remote: Remote, key: str, temp: Path, sha256: str, size: int | None = None) -> None:
    """Download ``key`` to ``temp`` and refuse unless its sha256 (and size, when known) match."""
    remote.download(key, temp)
    if (size is not None and temp.stat().st_size != size) or sha256_file(temp) != sha256:
        raise VerificationError(f"{key}: downloaded content does not match the manifest")
