"""In-memory stand-in for the boto3 S3 client, supporting exactly the calls s3sync makes, plus machine setup."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from botocore.exceptions import ClientError

import config


def _error(code: str, status: int, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, operation)


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.checksums: dict[str, str] = {}  # base64 sha256, for objects put with ChecksumSHA256
        self.calls: list[tuple[str, dict]] = []  # (method name, kwargs) for every call made

    def _etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", {"Key": Key}))
        if Key not in self.objects:
            raise _error("NoSuchKey", 404, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": self._etag(Key)}

    def head_object(self, Bucket, Key, ChecksumMode=None):
        self.calls.append(("head_object", {"Key": Key}))
        if Key not in self.objects:
            raise _error("404", 404, "HeadObject")
        response = {"ETag": self._etag(Key)}
        if ChecksumMode == "ENABLED" and Key in self.checksums:
            response["ChecksumSHA256"] = self.checksums[Key]
        return response

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.calls.append(("put_object", {"Key": Key, **kwargs}))
        body = Body if isinstance(Body, bytes) else Body.read()
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise _error("PreconditionFailed", 412, "PutObject")
        if "IfMatch" in kwargs:
            if Key not in self.objects:
                raise _error("NoSuchKey", 404, "PutObject")  # real S3 answers 404, not 412, here
            if self._etag(Key) != kwargs["IfMatch"]:
                raise _error("PreconditionFailed", 412, "PutObject")
        if "ChecksumSHA256" in kwargs and kwargs["ChecksumSHA256"] != base64.b64encode(hashlib.sha256(body).digest()).decode():
            raise _error("BadDigest", 400, "PutObject")
        self.objects[Key] = body
        if "ChecksumSHA256" in kwargs:
            self.checksums[Key] = kwargs["ChecksumSHA256"]
        else:
            self.checksums.pop(Key, None)
        return {"ETag": self._etag(Key)}

    def writes(self) -> list[dict]:
        return [kwargs for name, kwargs in self.calls if name == "put_object"]


def use_machine(monkeypatch, root: Path) -> dict[str, Path]:
    """Point config at a per-machine directory tree: sets ``data`` and ``coding`` (with an excluded bundles dir)."""
    dirs = {"data": root / "data", "coding": root / "coding"}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "S3_SYNC_SETS", dirs)
    monkeypatch.setattr(config, "S3_SYNC_EXCLUDED_DIRS", (dirs["coding"] / "bundles",))
    monkeypatch.setattr(config, "S3_SYNC_STATE_JSON", root / "state" / "s3sync_state.json")
    monkeypatch.setattr(config, "S3_SYNC_BACKUP_DIR", root / "backup")
    return dirs
