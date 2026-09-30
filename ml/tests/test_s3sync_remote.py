"""The key guard, conditional writes and the no-delete / always-encrypt invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from s3sync import sets
from s3sync.remote import ConflictError, GuardError, Remote

from .s3fake import FakeS3, use_machine


def test_guard_refuses_malformed_key_parts():
    remote = Remote(FakeS3())
    for parts in (("..", "x"), ("a", ".."), ("a\\b",), ("/abs",), ("",), ("a", "", "b")):
        with pytest.raises(GuardError):
            remote.key(*parts)


def test_guard_refuses_prefixes_outside_ours():
    prefixes = ("database/x/", "ml/x/", "other/", "ml-Revised-ICD-Mapping/../ml/",
                "ml-Revised-ICD-Mappingx/", "/ml-Revised-ICD-Mapping/", "ml-Revised-ICD-Mapping\\x/")
    for prefix in prefixes:
        with pytest.raises(GuardError):
            Remote(FakeS3(), prefix)


def test_guard_refuses_protected_prefix_even_if_configured(monkeypatch):
    monkeypatch.setattr(config, "S3_PREFIX", "ml/")
    with pytest.raises(GuardError):
        Remote(FakeS3())


def test_guard_accepts_scratch_subprefix():
    remote = Remote(FakeS3(), config.S3_PREFIX + "_scratch/run1/")
    assert remote.key("blobs", "abc") == config.S3_PREFIX + "_scratch/run1/blobs/abc"
    assert Remote(FakeS3()).key("sets", "data", "HEAD.json") == config.S3_PREFIX + "sets/data/HEAD.json"


def test_conditional_writes():
    remote = Remote(FakeS3())
    key = remote.key("sets", "x", "HEAD.json")
    assert remote.put_json_if_absent(key, {"a": 1}) is True
    assert remote.put_json_if_absent(key, {"a": 2}) is False
    assert remote.get_json(key)[0] == {"a": 1}
    assert remote.get_json(remote.key("nope")) is None and remote.head_etag(remote.key("nope")) is None
    etag = remote.get_json(key)[1]
    new_etag = remote.move_pointer(key, {"a": 3}, etag)
    with pytest.raises(ConflictError):
        remote.move_pointer(key, {"a": 4}, etag)
    with pytest.raises(ConflictError):
        remote.move_pointer(key, {"a": 4}, None)
    assert remote.get_json(key) == ({"a": 3}, new_etag)
    with pytest.raises(ConflictError):  # IfMatch on a key that does not exist (S3 answers 404)
        remote.move_pointer(remote.key("sets", "y", "HEAD.json"), {"a": 1}, '"stale"')


def test_full_cycle_never_deletes_and_always_encrypts(monkeypatch, tmp_path):
    fake = FakeS3()
    remote = Remote(fake)
    a = use_machine(monkeypatch, tmp_path / "a")
    (a["data"] / "f.csv").write_text("one")
    sets.push(remote, "data", apply=True)
    b = use_machine(monkeypatch, tmp_path / "b")
    sets.pull(remote, "data", apply=True)
    (b["data"] / "f.csv").write_text("two")
    (b["data"] / "g.csv").write_text("three")
    sets.push(remote, "data", apply=True)
    use_machine(monkeypatch, tmp_path / "a")
    sets.pull(remote, "data", apply=True)
    (a["data"] / "g.csv").unlink()
    sets.push(remote, "data", apply=True)

    assert not [name for name, _ in fake.calls if "delete" in name.lower()]
    writes = fake.writes()
    assert writes and all(kwargs["ServerSideEncryption"] == "AES256" for kwargs in writes)
    assert all(kwargs["Key"].startswith(config.S3_PREFIX) for _, kwargs in fake.calls)


def test_no_delete_api_referenced_in_module_source():
    for path in (Path(config.ML_ROOT) / "s3sync").glob("*.py"):
        assert "delete_object" not in path.read_text(), path
