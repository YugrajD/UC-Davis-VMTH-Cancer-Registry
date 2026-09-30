"""scripts/sync.py turns AWS failures (not just S3SyncError) into `REFUSED:` and exit status 1."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

import config

from .s3fake import use_fake_s3

_SYNC_PY = Path(__file__).resolve().parents[1] / "scripts" / "sync.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_sync", _SYNC_PY)
sync_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, sync_script)
_spec.loader.exec_module(sync_script)


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["sync.py", *argv])
    return sync_script.main()


@pytest.mark.parametrize("error", [ProfileNotFound(profile="nope"), NoCredentialsError()])
def test_client_construction_failure_is_refused_not_a_traceback(monkeypatch, tmp_path, capsys, error):
    use_fake_s3(monkeypatch, tmp_path, {"data": tmp_path / "data"})

    def broken():
        raise error

    monkeypatch.setattr("s3sync.client.make_client", broken)
    # sync.py imported make_client by name, so patch its own reference too.
    monkeypatch.setattr(sync_script, "make_client", broken)
    for command in ("status", "push", "pull-model"):
        assert _run(monkeypatch, command) == 1
        assert capsys.readouterr().err.startswith("REFUSED: ")


def test_access_denied_from_s3_is_refused(monkeypatch, tmp_path, capsys):
    (tmp_path / "data").mkdir()
    fake = use_fake_s3(monkeypatch, tmp_path, {"data": tmp_path / "data"})
    monkeypatch.setattr(sync_script, "make_client", lambda: fake)

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "nope"},
                           "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject")

    fake.get_object = denied
    assert _run(monkeypatch, "status", "data") == 1
    err = capsys.readouterr().err
    assert err.startswith("REFUSED: ") and "AccessDenied" in err


def test_guard_refusal_still_reports_and_a_clean_run_exits_0(monkeypatch, tmp_path, capsys):
    (tmp_path / "data").mkdir()
    fake = use_fake_s3(monkeypatch, tmp_path, {"data": tmp_path / "data"})
    monkeypatch.setattr(sync_script, "make_client", lambda: fake)
    assert _run(monkeypatch, "--prefix", "database/x/", "status", "data") == 1
    assert "REFUSED" in capsys.readouterr().err
    assert _run(monkeypatch, "--prefix", config.S3_PREFIX + "_scratch/t/", "status", "data") == 0
