"""generations/manifest.py: round-trip and tamper detection."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from io_utils import write_csv
from generations.manifest import ManifestError, read_manifest, sha256_file, verify_manifest, write_manifest


@pytest.fixture
def generation_dir(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01\x02")
    return tmp_path


def test_sha256_file_matches_hashlib(generation_dir):
    assert sha256_file(generation_dir / "a.txt") == hashlib.sha256(b"alpha\n").hexdigest()


def test_round_trip(generation_dir):
    written = write_manifest(generation_dir, {"split_id": "x", "counts": {"train": 1}})
    assert read_manifest(generation_dir) == written
    assert json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8")) == written
    assert written["split_id"] == "x" and written["counts"] == {"train": 1}
    assert set(written["files"]) == {"a.txt", "sub/b.bin"}  # manifest.json itself not listed
    assert written["files"]["sub/b.bin"] == sha256_file(generation_dir / "sub" / "b.bin")
    assert written["created_at"].endswith("+00:00")
    assert isinstance(written["git_sha"], str) and written["git_sha"]
    verify_manifest(generation_dir)


def test_written_files_use_lf_on_every_os(generation_dir):
    # Windows and the Mac must write byte-identical artefacts, or their sha256s disagree.
    write_manifest(generation_dir, {"split_id": "x"})
    write_csv(pd.DataFrame({"case_id": ["C1", "C2"]}), generation_dir / "t.csv")
    for name in ("manifest.json", "t.csv"):
        assert b"\r" not in (generation_dir / name).read_bytes()


def test_rewrite_does_not_list_old_manifest(generation_dir):
    write_manifest(generation_dir, {})
    assert "manifest.json" not in write_manifest(generation_dir, {})["files"]


def test_tampered_file_detected(generation_dir):
    write_manifest(generation_dir, {})
    (generation_dir / "a.txt").write_text("alpha\nCASE-9999\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="sha256"):
        verify_manifest(generation_dir)


def test_missing_listed_file_detected(generation_dir):
    write_manifest(generation_dir, {})
    (generation_dir / "sub" / "b.bin").unlink()
    with pytest.raises(ManifestError, match="missing"):
        verify_manifest(generation_dir)


def test_added_file_detected(generation_dir):
    write_manifest(generation_dir, {})
    (generation_dir / "sub" / "calibration_cases.txt").write_text("CASE-0001\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="sub/calibration_cases.txt"):
        verify_manifest(generation_dir)


def test_missing_manifest_detected(generation_dir):
    with pytest.raises(ManifestError, match="missing"):
        verify_manifest(generation_dir)
