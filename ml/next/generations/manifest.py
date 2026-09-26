"""manifest.json for every generation directory (splits, silver, report-mapping, ...).

A manifest is the caller's ``fields`` plus ``created_at`` (ISO UTC), ``git_sha``
(the code's HEAD, or ``"unknown"`` outside git) and ``files`` — a sha256 for
every file under the directory, keyed by its path relative to the directory.
``verify_manifest`` refuses a directory whose listed files are missing or changed,
or that holds a file the manifest does not list.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import config

MANIFEST_NAME = "manifest.json"


class ManifestError(Exception):
    """manifest.json is missing, a file it lists is missing or changed, or an unlisted file was added."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    # HEAD of the repo holding this code (not of the output directory), so the
    # manifest records which code version wrote the generation.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PACKAGE_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _listed_paths(directory: Path) -> list[str]:
    """Every file under ``directory`` (relative, posix), except manifest.json."""
    return [
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.relative_to(directory).as_posix() != MANIFEST_NAME
    ]


def write_manifest(directory: str | Path, fields: dict) -> dict:
    directory = Path(directory)
    files = {rel_path: sha256_file(directory / rel_path) for rel_path in _listed_paths(directory)}
    manifest = {
        **fields,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "files": files,
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(directory: str | Path) -> dict:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(directory: str | Path) -> None:
    directory = Path(directory)
    listed = read_manifest(directory)["files"]
    added = sorted(set(_listed_paths(directory)) - set(listed))
    if added:
        raise ManifestError(f"{directory} holds file(s) not in its manifest: {', '.join(added)}")
    for rel_path, expected in listed.items():
        path = directory / rel_path
        if not path.is_file():
            raise ManifestError(f"{path} is listed in the manifest but missing")
        if sha256_file(path) != expected:
            raise ManifestError(f"{path} does not match its manifest sha256")
