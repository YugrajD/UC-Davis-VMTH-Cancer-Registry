"""Guarded S3 access. Every key comes from ``Remote.key``; there is no delete call anywhere.

Objects are content-addressed and immutable (written with ``IfNoneMatch='*'``); the only mutable object is a
set's HEAD pointer, moved with ``IfMatch``. Rollback therefore never needs S3 object versions.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from botocore.exceptions import ClientError

import config

_MISSING_CODES = {"NoSuchKey", "404", "NotFound"}
_CONFLICT_CODES = {"PreconditionFailed", "ConditionalRequestConflict"}
_CHUNK = 1 << 20


class S3SyncError(Exception):
    """Base class: the sync refused to proceed."""


class GuardError(S3SyncError):
    """A key or prefix would leave our area of the bucket."""


class ConflictError(S3SyncError):
    """A conditional write lost the race (HTTP 412)."""


def _check_segments(path: str, what: str) -> None:
    if not path or "\\" in path or path.startswith("/"):
        raise GuardError(f"bad {what}: {path!r}")
    if any(segment in ("", ".", "..") for segment in path.rstrip("/").split("/")):
        raise GuardError(f"bad {what}: {path!r}")


class Remote:
    def __init__(self, client, prefix: str | None = None):
        self.client = client
        self.bucket = config.S3_BUCKET
        prefix = (prefix or config.S3_PREFIX).rstrip("/") + "/"
        _check_segments(prefix, "prefix")
        self.prefix = prefix
        self.key()  # validates the prefix against S3_PREFIX and the protected prefixes

    def key(self, *parts: str) -> str:
        for part in parts:
            _check_segments(part, "key part")
        key = self.prefix + "/".join(parts)
        if not key.startswith(config.S3_PREFIX):
            raise GuardError(f"key {key!r} is outside {config.S3_PREFIX!r}")
        if any(key.startswith(protected) for protected in config.S3_PROTECTED_PREFIXES):
            raise GuardError(f"key {key!r} is under a protected prefix")
        return key

    def _get(self, key: str):
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in _MISSING_CODES:
                return None
            raise

    def head_etag(self, key: str) -> str | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)["ETag"]
        except ClientError as error:
            if error.response["Error"]["Code"] in _MISSING_CODES:
                return None
            raise

    def get_json(self, key: str) -> tuple[dict, str] | None:
        """(document, etag), or None when the key does not exist."""
        response = self._get(key)
        if response is None:
            return None
        return json.loads(response["Body"].read()), response["ETag"]

    def download(self, key: str, dest: Path) -> None:
        response = self._get(key)
        if response is None:
            raise S3SyncError(f"missing remote object {key}")
        with open(dest, "wb") as file:
            for chunk in iter(lambda: response["Body"].read(_CHUNK), b""):
                file.write(chunk)

    def _put(self, key: str, body, **condition) -> str:
        try:
            return self.client.put_object(Bucket=self.bucket, Key=key, Body=body,
                                          ServerSideEncryption="AES256", **condition)["ETag"]
        except ClientError as error:
            code = error.response["Error"]["Code"]
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            # IfMatch on a key that is gone means it is no longer the object we saw.
            if code in _CONFLICT_CODES or status == 412 or ("IfMatch" in condition and code in _MISSING_CODES):
                raise ConflictError(key) from error
            raise

    def put_json_if_absent(self, key: str, document: dict) -> bool:
        """True if written; False when the key already existed (first writer wins, content is not compared)."""
        try:
            self._put(key, json.dumps(document, indent=1, sort_keys=True).encode(), IfNoneMatch="*")
        except ConflictError:
            return False
        return True

    def put_file_if_absent(self, key: str, path: Path, sha256: str) -> bool:
        """Upload ``path`` as the blob ``sha256``; S3 rejects the write if the bytes read now do not hash to it."""
        if self.head_etag(key) is not None:
            return False
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
        try:
            with open(path, "rb") as file:
                self._put(key, file, IfNoneMatch="*", ChecksumSHA256=checksum)
        except ConflictError:
            return False
        except ClientError as error:
            if error.response["Error"]["Code"] == "BadDigest":
                raise S3SyncError(f"{path} changed during push; re-run push") from error
            raise
        return True

    def move_pointer(self, key: str, document: dict, etag: str | None) -> str:
        """Write the mutable pointer: only if absent (etag None) or unchanged since ``etag``. Returns the new etag."""
        condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
        return self._put(key, json.dumps(document).encode(), **condition)
