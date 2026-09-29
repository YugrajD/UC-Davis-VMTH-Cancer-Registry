"""push / pull / status of file sets against the in-memory bucket."""

from __future__ import annotations

import json
import os

import pytest

import config
from generations.manifest import sha256_file
from s3sync import sets
from s3sync.remote import ConflictError, GuardError, Remote, S3SyncError

from .s3fake import FakeS3, use_machine


@pytest.fixture
def fake():
    return FakeS3()


@pytest.fixture
def remote(fake):
    return Remote(fake)


def _put(directory, rel, text):
    path = directory / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_first_push_creates_blobs_manifest_head_then_second_is_noop(monkeypatch, tmp_path, fake, remote):
    dirs = use_machine(monkeypatch, tmp_path)
    _put(dirs["data"], "a.csv", "aaa")
    _put(dirs["data"], "sub/b.csv", "bbb")
    assert sets.push(remote, "data", apply=True)["applied"]
    keys = sorted(fake.objects)
    assert sum("/blobs/" in k for k in keys) == 2
    assert sum("/manifests/" in k for k in keys) == 1
    assert any(k.endswith("sets/data/HEAD.json") for k in keys)
    before = len(fake.writes())
    result = sets.push(remote, "data", apply=True)
    assert result["nothing_to_do"] and not result["applied"] and len(fake.writes()) == before


def test_two_machines_conflict_and_recovery(monkeypatch, tmp_path, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "v1")
    _put(a["data"], "g.csv", "g1")
    sets.push(remote, "data", apply=True)

    b = use_machine(monkeypatch, tmp_path / "b")
    with pytest.raises(ConflictError, match="pull first"):  # never pulled
        sets.push(remote, "data", apply=True)
    sets.pull(remote, "data", apply=True)
    assert (b["data"] / "f.csv").read_text() == "v1"
    (b["data"] / "f.csv").write_text("v2-from-b")
    sets.push(remote, "data", apply=True)

    use_machine(monkeypatch, tmp_path / "a")
    (a["data"] / "f.csv").write_text("v2-from-a")
    (a["data"] / "g.csv").write_text("g2-from-a")  # only A edits g
    assert sets.status(remote, "data")["remote_moved"]
    with pytest.raises(ConflictError, match="pull first"):
        sets.push(remote, "data", apply=True)
    result = sets.pull(remote, "data", apply=True)
    assert result["conflicts"] == ["f.csv"] and result["kept_modified"] == ["g.csv"]
    assert (a["data"] / "f.csv").read_text() == "v2-from-b"
    assert (a["data"] / "g.csv").read_text() == "g2-from-a"  # unpushed edit survives
    assert [p.read_text() for p in (tmp_path / "a" / "backup").rglob("f.csv")] == ["v2-from-a"]
    (a["data"] / "h.csv").write_text("new")
    assert sets.push(remote, "data", apply=True)["applied"]

    use_machine(monkeypatch, tmp_path / "b")
    sets.pull(remote, "data", apply=True)
    assert (b["data"] / "g.csv").read_text() == "g2-from-a"


def test_pull_keeps_local_edit_and_fetches_remote_only_change(monkeypatch, tmp_path, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "f1")
    _put(a["data"], "g.csv", "g1")
    sets.push(remote, "data", apply=True)
    b = use_machine(monkeypatch, tmp_path / "b")
    sets.pull(remote, "data", apply=True)
    (b["data"] / "f.csv").write_text("f-edit-b")

    use_machine(monkeypatch, tmp_path / "a")
    (a["data"] / "g.csv").write_text("g2")
    sets.push(remote, "data", apply=True)

    use_machine(monkeypatch, tmp_path / "b")
    dry = sets.pull(remote, "data")
    assert dry["fetch"] == ["g.csv"] and dry["kept_modified"] == ["f.csv"] and dry["conflicts"] == []
    sets.pull(remote, "data", apply=True)
    assert (b["data"] / "f.csv").read_text() == "f-edit-b" and (b["data"] / "g.csv").read_text() == "g2"
    assert [q.name for q in (tmp_path / "b" / "backup").rglob("*.csv")] == ["g.csv"]  # f.csv was never overwritten
    assert sets.push(remote, "data", apply=True)["change"]["changed"] == ["f.csv"]


def test_push_loses_head_race_and_refuses(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "v1")
    sets.push(remote, "data", apply=True)
    _put(a["data"], "f.csv", "v2")
    real_put = fake.put_object

    def racing_put(Bucket, Key, Body, **kwargs):
        if Key.endswith("HEAD.json"):  # someone else moves HEAD between our check and our write
            fake.objects[Key] = b'{"manifest": "elsewhere"}'
        return real_put(Bucket=Bucket, Key=Key, Body=Body, **kwargs)

    fake.put_object = racing_put
    with pytest.raises(ConflictError, match="pull first"):
        sets.push(remote, "data", apply=True)


def test_pull_rejects_corrupt_blob_and_leaves_local_file(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "good")
    sets.push(remote, "data", apply=True)
    fake.objects[remote.key("blobs", sha256_file(a["data"] / "f.csv"))] = b"tampered"

    b = use_machine(monkeypatch, tmp_path / "b")
    _put(b["data"], "f.csv", "mine")
    with pytest.raises(sets.VerificationError):
        sets.pull(remote, "data", apply=True)
    assert (b["data"] / "f.csv").read_text() == "mine"
    assert not list(b["data"].glob("*.s3sync-partial"))


def test_pull_backs_up_overwrites_removes_remote_deletions_keeps_local_only(monkeypatch, tmp_path, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "v1")
    _put(a["data"], "gone.csv", "bye")
    _put(a["data"], "edited.csv", "orig")
    sets.push(remote, "data", apply=True)

    b = use_machine(monkeypatch, tmp_path / "b")
    sets.pull(remote, "data", apply=True)
    _put(b["data"], "local_only.csv", "keep me")
    _put(b["data"], "edited.csv", "edit by b")
    _put(b["data"], "f.csv", "b clobbers")

    use_machine(monkeypatch, tmp_path / "a")
    (a["data"] / "f.csv").write_text("v2")
    (a["data"] / "gone.csv").unlink()
    (a["data"] / "edited.csv").unlink()
    sets.push(remote, "data", apply=True)

    use_machine(monkeypatch, tmp_path / "b")
    result = sets.pull(remote, "data", apply=True)
    assert (b["data"] / "f.csv").read_text() == "v2"
    assert not (b["data"] / "gone.csv").exists()
    assert (b["data"] / "local_only.csv").read_text() == "keep me"
    assert (b["data"] / "edited.csv").read_text() == "edit by b"  # remotely deleted but locally modified
    assert result["kept_modified"] == ["edited.csv"] and result["conflicts"] == ["f.csv"]
    assert result["remove"] == ["gone.csv"]
    backups = list((tmp_path / "b" / "backup").rglob("*.csv"))
    assert {p.name: p.read_text() for p in backups} == {"f.csv": "b clobbers", "gone.csv": "bye"}


def test_dry_runs_make_no_writes_and_change_nothing(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "v1")
    result = sets.push(remote, "data")
    assert not result["applied"] and result["change"]["added"] == ["f.csv"]
    assert fake.writes() == [] and not (tmp_path / "a" / "state").exists()

    sets.push(remote, "data", apply=True)
    b = use_machine(monkeypatch, tmp_path / "b")
    _put(b["data"], "f.csv", "other")
    writes = len(fake.writes())
    result = sets.pull(remote, "data")
    assert result["conflicts"] == ["f.csv"] and not result["applied"]
    assert len(fake.writes()) == writes
    assert (b["data"] / "f.csv").read_text() == "other"
    assert not (tmp_path / "b" / "state").exists() and not (tmp_path / "b" / "backup").exists()


def test_excluded_dirs_and_junk_are_not_in_the_manifest(monkeypatch, tmp_path, fake, remote):
    dirs = use_machine(monkeypatch, tmp_path)
    _put(dirs["coding"], "keep.csv", "x")
    _put(dirs["coding"], "bundles/big.tar", "bundle")
    _put(dirs["coding"], ".DS_Store", "junk")
    _put(dirs["coding"], "partial.csv.s3sync-partial", "junk")
    _put(dirs["coding"], "__pycache__/m.pyc", "junk")
    _put(dirs["coding"], "real.tmp", "a legitimate file")
    assert list(sets.local_files("coding")) == ["keep.csv", "real.tmp"]
    sets.push(remote, "coding", apply=True)
    manifest = next(v for k, v in fake.objects.items() if "/manifests/" in k)
    assert list(json.loads(manifest)["files"]) == ["keep.csv", "real.tmp"]


@pytest.mark.parametrize("bad", [
    "../evil.csv", "a/../b.csv", "C:/x.csv", "a:b.csv", "a\\b.csv", "/abs.csv", "a//b.csv", "./x.csv",
    "NUL", "con.txt", "sub/COM1.csv", "lpt9", "trailing.", "sub/space ",
])
def test_pull_refuses_unsafe_manifest_path(monkeypatch, tmp_path, fake, remote, bad):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "x")
    sets.push(remote, "data", apply=True)
    key = next(k for k in fake.objects if "/manifests/" in k)
    fake.objects[key] = fake.objects[key].replace(b'"f.csv"', json.dumps(bad).encode())
    b = use_machine(monkeypatch, tmp_path / "b")
    with pytest.raises(GuardError, match="unsafe"):
        sets.pull(remote, "data", apply=True)
    assert not (tmp_path / "evil.csv").exists() and not list(b["data"].iterdir())


def test_pull_skips_excluded_manifest_paths(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["coding"], "keep.csv", "x")
    sets.push(remote, "coding", apply=True)
    key = next(k for k in fake.objects if "/manifests/" in k)
    manifest = json.loads(fake.objects[key])
    for rel in ("bundles/big.tar", ".DS_Store", "x.csv.s3sync-partial"):
        manifest["files"][rel] = {"sha256": "0" * 64, "size": 1}  # blob does not exist: must never be fetched
    fake.objects[key] = json.dumps(manifest).encode()

    b = use_machine(monkeypatch, tmp_path / "b")
    result = sets.pull(remote, "coding", apply=True)
    assert result["skipped_excluded"] == [".DS_Store", "bundles/big.tar", "x.csv.s3sync-partial"]
    assert result["fetch"] == ["keep.csv"] and result["applied"]
    assert not (b["coding"] / "bundles").exists()
    assert sets.pull(remote, "coding")["fetch"] == []  # not re-fetched forever


def test_scratch_prefix_state_is_isolated(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path)
    _put(a["data"], "f.csv", "x")
    scratch = Remote(fake, config.S3_PREFIX + "_scratch/run1/")
    sets.push(scratch, "data", apply=True)
    assert sets.status(remote, "data")["never_synced"]
    assert sets.status(scratch, "data")["never_synced"] is False
    sets.push(remote, "data", apply=True)
    assert sorted(json.loads(config.S3_SYNC_STATE_JSON.read_text())) == sorted([remote.prefix, scratch.prefix])


def test_mid_replace_failure_cleans_partials_and_reports(monkeypatch, tmp_path, remote):
    a = use_machine(monkeypatch, tmp_path / "a")
    _put(a["data"], "f.csv", "1")
    _put(a["data"], "g.csv", "1")
    sets.push(remote, "data", apply=True)
    b = use_machine(monkeypatch, tmp_path / "b")
    real_replace = os.replace

    def locked(src, dst):
        if str(dst).endswith("g.csv"):
            raise PermissionError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(sets.os, "replace", locked)
    with pytest.raises(S3SyncError, match="partially applied"):
        sets.pull(remote, "data", apply=True)
    assert not list(b["data"].glob("*.s3sync-partial"))
    monkeypatch.setattr(sets.os, "replace", real_replace)
    sets.pull(remote, "data", apply=True)  # re-run finishes the job
    assert (b["data"] / "g.csv").read_text() == "1" and (b["data"] / "f.csv").read_text() == "1"


def test_head_pointing_to_missing_manifest_is_refused(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path)
    _put(a["data"], "f.csv", "x")
    sets.push(remote, "data", apply=True)
    for key in [k for k in fake.objects if "/manifests/" in k]:
        del fake.objects[key]
    for operation in (sets.pull, sets.push, sets.status):
        with pytest.raises(S3SyncError, match="missing manifest"):
            operation(remote, "data")


def test_push_refuses_missing_or_emptied_local_set(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path)
    _put(a["data"], "f.csv", "x")
    sets.push(remote, "data", apply=True)
    (a["data"] / "f.csv").unlink()
    with pytest.raises(S3SyncError, match="missing or empty"):
        sets.push(remote, "data", apply=True)
    a["data"].rmdir()
    with pytest.raises(S3SyncError, match="missing or empty"):
        sets.push(remote, "data", apply=True)
    assert sets.push(remote, "coding")["nothing_to_do"]  # empty dir, empty remote: fine
    a["coding"].rmdir()
    assert sets.push(remote, "coding", apply=True)["nothing_to_do"]  # never-created dir, empty remote: fine


def test_blob_changed_between_hash_and_upload_is_rejected(monkeypatch, tmp_path, fake, remote):
    a = use_machine(monkeypatch, tmp_path)
    path = _put(a["data"], "f.csv", "before")
    real_head = remote.head_etag

    def mutate_then_head(key):
        if "/blobs/" in key:  # runs after local_files() hashed the file, before the upload reads it
            path.write_text("after!")
        return real_head(key)

    monkeypatch.setattr(remote, "head_etag", mutate_then_head)
    with pytest.raises(S3SyncError, match="changed during push"):
        sets.push(remote, "data", apply=True)
    assert not any("/blobs/" in k for k in fake.objects)
    assert not any(k.endswith("HEAD.json") for k in fake.objects)


def test_status_reports_local_changes(monkeypatch, tmp_path, remote):
    a = use_machine(monkeypatch, tmp_path)
    _put(a["data"], "f.csv", "v1")
    _put(a["data"], "g.csv", "v1")
    sets.push(remote, "data", apply=True)
    (a["data"] / "f.csv").write_text("v2")
    (a["data"] / "g.csv").unlink()
    _put(a["data"], "h.csv", "new")
    status = sets.status(remote, "data")
    assert status["local"] == {"added": ["h.csv"], "changed": ["f.csv"], "removed": ["g.csv"]}
    assert not status["remote_moved"] and not status["never_synced"]


def test_set_names(monkeypatch, tmp_path):
    use_machine(monkeypatch, tmp_path)
    assert sets.set_names("all") == ["data", "coding"]
    assert sets.set_names("data") == ["data"]
    with pytest.raises(S3SyncError, match="unknown set"):
        sets.set_names("nope")
