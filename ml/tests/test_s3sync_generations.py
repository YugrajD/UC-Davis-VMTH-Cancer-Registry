"""publish-model / pull-model against the in-memory bucket, on tiny synthetic generations."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from datetime import date

import pytest

import config
from generations.manifest import read_manifest, verify_manifest, write_manifest
from s3sync import models
from s3sync.files import VerificationError
from s3sync.remote import ConflictError, GuardError, Remote, S3SyncError

from . import fixtures as fx
from .s3fake import FakeS3


@pytest.fixture
def fake():
    return FakeS3()


@pytest.fixture
def remote(fake):
    return Remote(fake)


@pytest.fixture
def machine(monkeypatch, tmp_path, tiny_bert_dir):
    """machine("a") points config at tmp_path/a; machine("a", "gen-1") also gives it a current/ generation."""
    def switch(name: str, generation_id: str | None = None, status: str = "calibrated") -> None:
        fx.point_promotion_config_at(monkeypatch, tmp_path / name)
        monkeypatch.setattr(config, "S3_SYNC_STATE_JSON", tmp_path / name / "s3sync_state.json")
        if generation_id:
            _generation(config.REPORT_MAPPING_CURRENT_DIR, tiny_bert_dir, generation_id, status)
    switch.tiny_bert_dir = tiny_bert_dir
    return switch


def _generation(directory, tiny_bert_dir, generation_id, status="calibrated"):
    if directory.exists():
        shutil.rmtree(directory)
    fx.build_report_mapping_bundle(directory, tiny_bert_dir)
    fields = {k: v for k, v in read_manifest(directory).items() if k not in ("files", "created_at", "git_sha")}
    write_manifest(directory, {**fields, "generation_id": generation_id, "calibration": {"status": status}})


def _sha_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode()


def _key(remote, generation_id, rel):
    return remote.key("generations", generation_id, rel)


def test_publish_uploads_files_then_manifest_then_pointer(machine, fake, remote):
    machine("a", "gen-1")
    files = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["files"]
    result = models.publish(remote, apply=True)
    assert result["applied"] and result["files"] == len(files) and result["bytes"] > 0

    puts = [kwargs for kwargs in fake.writes()]
    assert [p["Key"] for p in puts[-2:]] == [_key(remote, "gen-1", "manifest.json"),
                                             remote.key("generations", "CURRENT.json")]
    assert {p["Key"] for p in puts[:-2]} == {_key(remote, "gen-1", rel) for rel in files}  # readable names
    assert all(p["ServerSideEncryption"] == "AES256" and p["IfNoneMatch"] == "*" for p in puts)
    assert all("ChecksumSHA256" in p for p in puts[:-1])
    assert json.loads(fake.objects[remote.key("generations", "CURRENT.json")]) == {"generation_id": "gen-1"}


def test_republish_is_a_noop(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    writes = len(fake.writes())
    result = models.publish(remote, apply=True)
    assert result["nothing_to_do"] and not result["applied"] and len(fake.writes()) == writes


def test_retried_partial_publish_completes(machine, fake, remote):
    machine("a", "gen-1")
    current = config.REPORT_MAPPING_CURRENT_DIR
    files = read_manifest(current)["files"]
    for rel in sorted(files)[:2]:  # an earlier attempt got this far and died before the manifest
        remote.put_file_if_absent(_key(remote, "gen-1", rel), current / rel, files[rel])
    assert _key(remote, "gen-1", "manifest.json") not in fake.objects
    assert models.publish(remote, apply=True)["applied"]
    assert all(_key(remote, "gen-1", rel) in fake.objects for rel in [*files, "manifest.json"])


def test_present_file_with_a_different_checksum_is_refused(machine, fake, remote):
    machine("a", "gen-1")
    rel = sorted(read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["files"])[0]
    other = b"someone else's bytes"
    fake.put_object(Bucket="", Key=_key(remote, "gen-1", rel), Body=other, ChecksumSHA256=_sha_b64(other))
    with pytest.raises(S3SyncError, match="different checksum"):
        models.publish(remote, apply=True)
    assert _key(remote, "gen-1", "manifest.json") not in fake.objects
    assert remote.key("generations", "CURRENT.json") not in fake.objects


def test_publish_refuses_uncalibrated_or_manifest_broken_current(machine, remote):
    machine("a", "gen-1", status="pending")
    with pytest.raises(S3SyncError, match="calibration.status"):
        models.publish(remote, apply=True)
    machine("b", "gen-1")
    victim = next(config.REPORT_MAPPING_CURRENT_DIR.glob("checkpoints/*.json"))
    victim.write_text("tampered")
    with pytest.raises(S3SyncError, match="manifest"):
        models.publish(remote, apply=True)


def test_pull_onto_a_fresh_machine_adopts_the_generation(machine, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b")
    assert not config.REPORT_MAPPING_CURRENT_DIR.exists()
    result = models.pull_model(remote, apply=True)
    assert result["applied"] and result["adopted"]["archive"] is None
    current = config.REPORT_MAPPING_CURRENT_DIR
    assert read_manifest(current)["generation_id"] == "gen-1" and read_manifest(current)["status"] == "current"
    verify_manifest(current)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()
    assert models.pull_model(remote)["action"] == "nothing"


def test_pull_onto_an_older_current_archives_it(machine, remote):
    machine("a", "gen-2")
    models.publish(remote, apply=True)
    machine("b", "gen-1")
    result = models.pull_model(remote, apply=True)
    archive = result["adopted"]["archive"]
    assert archive.parent == config.ARCHIVE_ROOT
    assert read_manifest(archive)["generation_id"] == "gen-1" and read_manifest(archive)["status"] == "archived"
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-2"


def test_pull_refuses_when_candidate_exists(machine, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b")
    config.REPORT_MAPPING_CANDIDATE_DIR.mkdir(parents=True)
    with pytest.raises(S3SyncError, match="candidate/ exists"):
        models.pull_model(remote, apply=True)
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists()  # not ours: left alone


def test_tampered_remote_file_is_refused_and_candidate_removed(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    rel = sorted(read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["files"])[0]
    fake.objects[_key(remote, "gen-1", rel)] = b"tampered"
    machine("b", "gen-0")
    with pytest.raises(VerificationError):
        models.pull_model(remote, apply=True)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.S3_SYNC_STATE_JSON.exists()
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-0"


def test_tampered_fingerprint_is_refused_and_candidate_removed(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    key = _key(remote, "gen-1", "manifest.json")
    manifest = json.loads(fake.objects[key])
    manifest["embedding_fingerprint"] = "0" * 64
    fake.objects[key] = json.dumps(manifest).encode()
    machine("b", "gen-0")
    with pytest.raises(S3SyncError, match="fingerprint mismatch"):
        models.pull_model(remote, apply=True)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.S3_SYNC_STATE_JSON.exists()
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-0"


def test_unsafe_relpath_in_remote_manifest_is_refused(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    key = _key(remote, "gen-1", "manifest.json")
    manifest = json.loads(fake.objects[key])
    manifest["files"]["../../evil.txt"] = "0" * 64
    fake.objects[key] = json.dumps(manifest).encode()
    machine("b")
    with pytest.raises(GuardError, match="unsafe"):
        models.pull_model(remote, apply=True)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_stale_publish_is_refused(machine, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b")
    models.pull_model(remote, apply=True)
    machine("a", "gen-2")  # a moves the model on
    models.publish(remote, apply=True)
    machine("b", "gen-3")  # b, unaware, trains its own
    assert models.status(remote)["remote_moved"]
    with pytest.raises(ConflictError, match="pull-model first"):
        models.publish(remote, apply=True)


def test_unpublished_local_current_is_kept_on_pull(machine, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b")
    models.pull_model(remote, apply=True)
    machine("b", "gen-local")  # replaced by something unpublished
    result = models.pull_model(remote, apply=True)
    assert result["action"].startswith("keep local") and "publish-model" in result["action"]
    assert not result["applied"]
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-local"
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_dry_runs_make_no_writes(machine, fake, remote):
    machine("a", "gen-1")
    plan = models.publish(remote)
    assert not plan["applied"] and plan["files"] > 0 and plan["bytes"] > 0 and fake.writes() == []
    models.publish(remote, apply=True)
    writes = len(fake.writes())
    machine("b")
    plan = models.pull_model(remote)
    assert plan["files"] > 0 and not plan["applied"] and "gen-1" in plan["action"]
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.REPORT_MAPPING_CURRENT_DIR.exists()
    assert not config.S3_SYNC_STATE_JSON.exists() and len(fake.writes()) == writes


def test_every_key_is_under_the_prefix_and_nothing_is_deleted(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b", "gen-0")
    models.pull_model(remote, apply=True)
    assert fake.calls and all(kwargs["Key"].startswith(config.S3_PREFIX + "generations/") for _, kwargs in fake.calls)
    assert not [name for name, _ in fake.calls if "delete" in name.lower()]
    assert all(kwargs["ServerSideEncryption"] == "AES256" for kwargs in fake.writes())


def test_publish_ignores_a_candidate_directory(machine, fake, remote, tiny_bert_dir):
    machine("a", "gen-1")
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_bert_dir, "gen-cand")
    models.publish(remote, apply=True)
    assert fake.objects and not any("gen-cand" in key for key in fake.objects)


def test_pointer_race_on_publish_is_refused_and_leaves_state_unwritten(machine, fake, remote):
    machine("a", "gen-1")
    real_put = fake.put_object

    def racing_put(Bucket, Key, Body, **kwargs):
        if Key.endswith("CURRENT.json"):  # someone else publishes between our read and our write
            fake.objects[Key] = b'{"generation_id": "elsewhere"}'
        return real_put(Bucket=Bucket, Key=Key, Body=Body, **kwargs)

    fake.put_object = racing_put
    with pytest.raises(ConflictError, match="pull-model first"):
        models.publish(remote, apply=True)
    assert not config.S3_SYNC_STATE_JSON.exists()


def test_republish_of_an_uploaded_generation_still_checks_its_manifest(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    fake.objects.pop(remote.key("generations", "CURRENT.json"))  # the pointer alone is gone; the generation is there
    manifest_key = _key(remote, "gen-1", "manifest.json")
    fake.objects[manifest_key] = b"{}"  # a different manifest.json under the same id
    fake.checksums[manifest_key] = _sha_b64(b"{}")
    with pytest.raises(S3SyncError, match="different checksum"):
        models.publish(remote, apply=True)
    assert remote.key("generations", "CURRENT.json") not in fake.objects


def test_existing_object_without_a_checksum_is_refused(machine, fake, remote):
    machine("a", "gen-1")
    rel = sorted(read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["files"])[0]
    fake.put_object(Bucket="", Key=_key(remote, "gen-1", rel), Body=b"no checksum given")  # fail closed
    with pytest.raises(S3SyncError, match="different checksum"):
        models.publish(remote, apply=True)


@pytest.mark.parametrize("pointer", [b'["gen-1"]', b'{"nope": 1}', b'{"generation_id": "C:x"}',
                                     b'{"generation_id": "a/b"}', b'{"generation_id": 3}'])
def test_malformed_pointer_is_refused(machine, fake, remote, pointer):
    machine("a", "gen-1")
    fake.objects[remote.key("generations", "CURRENT.json")] = pointer
    for operation in (models.publish, models.pull_model, models.status):
        with pytest.raises(S3SyncError):
            operation(remote)


def test_pointer_to_a_generation_without_a_manifest_is_refused(machine, fake, remote):
    machine("a")
    fake.objects[remote.key("generations", "CURRENT.json")] = b'{"generation_id": "ghost"}'
    with pytest.raises(S3SyncError, match="no manifest.json"):
        models.pull_model(remote, apply=True)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.S3_SYNC_STATE_JSON.exists()


def test_candidate_created_during_the_pull_survives(machine, remote, monkeypatch):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b")
    real_get_json = remote.get_json

    def get_json_then_race(key):
        result = real_get_json(key)
        if key.endswith("gen-1/manifest.json"):  # between the exists() check and mkdir
            config.REPORT_MAPPING_CANDIDATE_DIR.mkdir(parents=True)
            (config.REPORT_MAPPING_CANDIDATE_DIR / "precious.pt").write_text("keep")
        return result

    monkeypatch.setattr(remote, "get_json", get_json_then_race)
    with pytest.raises(S3SyncError, match="candidate/ exists"):
        models.pull_model(remote, apply=True)
    assert (config.REPORT_MAPPING_CANDIDATE_DIR / "precious.pt").read_text() == "keep"


def test_archive_destination_taken_is_refused_before_downloading(machine, fake, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b", "gen-0")
    taken = config.ARCHIVE_ROOT / f"{date.today().isoformat()}_gen-0"
    taken.mkdir(parents=True)
    gets = len([c for c in fake.calls if c[0] == "get_object"])
    for apply in (False, True):
        with pytest.raises(S3SyncError, match="already exists"):
            models.pull_model(remote, apply=apply)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()
    assert len([c for c in fake.calls if c[0] == "get_object"]) == gets + 2  # only the pointer reads


def test_failed_adopt_removes_the_candidate_and_keeps_current(machine, remote, monkeypatch):
    from generations import promote

    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b", "gen-0")

    def refuse(*args, **kwargs):
        raise promote.PromotionError("swap refused")

    monkeypatch.setattr(promote, "adopt", refuse)
    with pytest.raises(S3SyncError, match="swap refused"):
        models.pull_model(remote, apply=True)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.S3_SYNC_STATE_JSON.exists()
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-0"


def test_unpublished_local_current_is_flagged_and_archived_on_pull(machine, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b", "gen-local")  # never synced: unpublished work
    plan = models.pull_model(remote)
    archive = config.ARCHIVE_ROOT / f"{date.today().isoformat()}_gen-local"
    assert plan["archive"] == archive
    assert "UNPUBLISHED" in plan["warning"] and str(archive) in plan["warning"]
    done = models.pull_model(remote, apply=True)
    assert "UNPUBLISHED" in done["warning"] and read_manifest(archive)["generation_id"] == "gen-local"


def test_published_local_current_is_archived_without_the_unpublished_warning(machine, remote):
    machine("a", "gen-1")
    models.publish(remote, apply=True)
    machine("b")
    models.pull_model(remote, apply=True)  # b now has gen-1, recorded as synced
    machine("a", "gen-2")
    models.publish(remote, apply=True)
    machine("b")
    plan = models.pull_model(remote)
    assert plan["archive"] is not None and plan["warning"] is None
