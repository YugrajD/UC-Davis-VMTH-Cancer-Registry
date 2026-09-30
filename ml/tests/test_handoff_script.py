"""scripts/handoff.py: thin CLI wiring over handoff.imports/exports.

Minimal coverage, mirroring test_code_cases.py: one happy-path invocation per
subcommand, checking the CLI wires args through correctly — the underlying
behavior is already covered by test_handoff_imports.py / test_handoff_exports.py.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

import config
import io_utils
from coding.combine import COMBINED_PREDICTIONS_COLUMNS
from coding.queue import REVIEW_QUEUE_COLUMNS

from . import fixtures as fx

_HANDOFF_PY = Path(__file__).resolve().parents[1] / "scripts" / "handoff.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_handoff", _HANDOFF_PY)
handoff_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, handoff_script)
_spec.loader.exec_module(handoff_script)


@pytest.fixture
def handoff_root(monkeypatch, tmp_path):
    fx.point_handoff_config_at(monkeypatch, tmp_path)
    return tmp_path


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["handoff.py"] + argv)
    return handoff_script.main()


def test_import_pending_cli(monkeypatch, handoff_root):
    csv_path = handoff_root / "pending.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "diagnosis_number", "diagnosis"])
        writer.writerow(["CASE-A", 1, "MAST CELL TUMOR"])

    assert _run(monkeypatch, ["import-pending", "--csv", str(csv_path), "--export-id", "e1"]) == 0
    assert config.HANDOFF_PENDING_DIAGNOSES_CSV.is_file()


def _make_stamped_silver_generation(silver_id: str, rows: list[tuple]) -> str:
    """See test_handoff_exports.py's helper of the same name and docstring."""
    df = pd.DataFrame(rows, columns=fx.ANNOTATION_COLUMNS)
    df["silver_generation"] = silver_id
    directory = config.SILVER_DIR / silver_id
    directory.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(df, directory / "annotation.csv")
    from generations.manifest import write_manifest
    write_manifest(directory, {"silver_id": silver_id, "llm_enabled": True})
    return silver_id


def test_export_silver_cli(monkeypatch, handoff_root):
    _make_stamped_silver_generation("s1", [
        ("CASE-A", 1, "TEXT", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3", "kw", "Exact", 1.0, "tier1_exact"),
    ])
    assert _run(monkeypatch, ["export-silver", "--silver-id", "s1"]) == 0
    assert (config.HANDOFF_OUTBOX_DIR / "silver_codes_s1.csv").is_file()


def test_export_coding_cli(monkeypatch, handoff_root):
    config.COMBINED_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in COMBINED_PREDICTIONS_COLUMNS}]), config.COMBINED_PREDICTIONS_CSV)
    config.REVIEW_QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in REVIEW_QUEUE_COLUMNS}]), config.REVIEW_QUEUE_CSV)

    assert _run(monkeypatch, ["export-coding", "--run-id", "run1"]) == 0
    assert (config.HANDOFF_OUTBOX_DIR / "combined_predictions_run1.csv").is_file()
    assert (config.HANDOFF_OUTBOX_DIR / "review_queue_run1.csv").is_file()


def test_export_bundle_cli(monkeypatch, handoff_root, tiny_bert_dir):
    fx.build_report_mapping_bundle(config.REPORT_MAPPING_CURRENT_DIR, tiny_bert_dir)
    assert _run(monkeypatch, ["export-bundle", "--generation", "current"]) == 0
    assert any(config.HANDOFF_BUNDLES_DIR.glob("bundle_*.tar.gz"))


# ---- --push (opt-in S3 hook) ----

from .s3fake import forbid_s3, pushed_sets, use_fake_s3  # noqa: E402


def _s3(monkeypatch, handoff_root):
    return use_fake_s3(monkeypatch, handoff_root, {"handoff": config.HANDOFF_DIR, "manual_audit": config.MANUAL_AUDIT_DIR})


def _pending_csv(path):
    path.write_text("case_id,diagnosis_number,diagnosis\nCASE-A,1,MAST CELL TUMOR\n", encoding="utf-8")
    return path


def _gold_csv(monkeypatch, handoff_root, labels_csv):
    monkeypatch.setattr(config, "LABELS_CSV", labels_csv)
    ledger = handoff_root / "ledger.csv"
    io_utils.write_csv(pd.DataFrame({"case_id": ["CASE-A"]}), ledger)
    monkeypatch.setattr(config, "EVAL_BATCH_LEDGER_CSV", ledger)
    path = handoff_root / "gold.csv"
    io_utils.write_csv(pd.DataFrame([{"case_id": "CASE-A", "term": "Mast cell tumor, malignant", "origin": "eval_batch"}]), path)
    return path


def test_push_after_import_pending_pushes_only_the_handoff_set(monkeypatch, handoff_root):
    fake = _s3(monkeypatch, handoff_root)
    csv_path = _pending_csv(handoff_root / "pending.csv")
    assert _run(monkeypatch, ["import-pending", "--csv", str(csv_path), "--export-id", "e1", "--push"]) == 0
    assert pushed_sets(fake) == {"handoff"}
    assert all(kwargs["ServerSideEncryption"] == "AES256" for kwargs in fake.writes())


def test_push_after_import_gold_pushes_manual_audit_and_handoff(monkeypatch, handoff_root, labels_csv):
    fake = _s3(monkeypatch, handoff_root)
    csv_path = _gold_csv(monkeypatch, handoff_root, labels_csv)
    assert _run(monkeypatch, ["import-gold", "--csv", str(csv_path), "--export-id", "g1", "--reviewer", "Dr. T", "--push"]) == 0
    assert pushed_sets(fake) == {"manual_audit", "handoff"}


def test_push_after_export_silver_pushes_the_handoff_set(monkeypatch, handoff_root):
    fake = _s3(monkeypatch, handoff_root)
    _make_stamped_silver_generation("s1", [
        ("CASE-A", 1, "TEXT", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3", "kw", "Exact", 1.0, "tier1_exact"),
    ])
    assert _run(monkeypatch, ["export-silver", "--silver-id", "s1", "--push"]) == 0
    assert pushed_sets(fake) == {"handoff"}


def test_push_after_export_audit_list_pushes_manual_audit_and_handoff(monkeypatch, handoff_root):
    fake = _s3(monkeypatch, handoff_root)
    assert _run(monkeypatch, ["export-audit-list", "--list-id", "L1", "--no-review-queue", "--push"]) == 0
    assert config.AUDIT_LIST_LEDGER_CSV.is_file()
    assert pushed_sets(fake) == {"manual_audit", "handoff"}


def test_push_refusal_reports_local_work_done_and_exits_1(monkeypatch, handoff_root, capsys):
    fake = _s3(monkeypatch, handoff_root)
    assert _run(monkeypatch, ["import-pending", "--csv", str(_pending_csv(handoff_root / "p1.csv")),
                              "--export-id", "e1", "--push"]) == 0
    config.S3_SYNC_STATE_JSON.unlink()  # this "machine" never pulled what is now on the remote
    capsys.readouterr()
    assert _run(monkeypatch, ["import-pending", "--csv", str(_pending_csv(handoff_root / "p2.csv")),
                              "--export-id", "e2", "--push"]) == 1
    err = capsys.readouterr().err
    assert "import-pending done locally; pushed: -; FAILED: handoff (" in err and "remote moved, pull first" in err
    assert "not attempted: -" in err
    assert "run ml/scripts/sync.py pull handoff --apply, then ml/scripts/sync.py push handoff --apply" in err
    assert (config.HANDOFF_INBOX_DIR / "pending_diagnoses_e2.csv").is_file()  # local output intact


def test_partial_multi_set_push_reports_pushed_failed_and_not_attempted(monkeypatch, handoff_root, labels_csv, capsys):
    _s3(monkeypatch, handoff_root)
    assert _run(monkeypatch, ["import-pending", "--csv", str(_pending_csv(handoff_root / "p1.csv")),
                              "--export-id", "e1", "--push"]) == 0
    config.S3_SYNC_STATE_JSON.unlink()  # handoff is now behind the remote; manual_audit has no remote yet
    capsys.readouterr()
    csv_path = _gold_csv(monkeypatch, handoff_root, labels_csv)
    assert _run(monkeypatch, ["import-gold", "--csv", str(csv_path), "--export-id", "g1", "--reviewer", "Dr. T",
                              "--push"]) == 1
    captured = capsys.readouterr()
    assert "pushed manual_audit" in captured.out
    assert captured.err.startswith("import-gold done locally; pushed: manual_audit; FAILED: handoff (")
    assert "not attempted: - (stopped at the first failure)" in captured.err

    # The failing set first: the other one is reported as not attempted.
    monkeypatch.setitem(handoff_script.PUSH_SETS, "import-gold", ("handoff", "manual_audit"))
    assert _run(monkeypatch, ["import-gold", "--csv", str(csv_path), "--export-id", "g2", "--reviewer", "Dr. T",
                              "--push"]) == 1
    err = capsys.readouterr().err
    assert "pushed: -; FAILED: handoff (" in err and "not attempted: manual_audit (stopped" in err


def test_push_transient_failure_says_to_rerun_the_push(monkeypatch, handoff_root, capsys):
    from botocore.exceptions import ClientError

    fake = _s3(monkeypatch, handoff_root)

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "PutObject")

    fake.put_object = denied
    assert _run(monkeypatch, ["import-pending", "--csv", str(_pending_csv(handoff_root / "p.csv")),
                              "--export-id", "e1", "--push"]) == 1
    err = capsys.readouterr().err
    assert "FAILED: handoff" in err and "re-run ml/scripts/sync.py push handoff --apply" in err
    assert (config.HANDOFF_INBOX_DIR / "pending_diagnoses_e1.csv").is_file()


def test_push_with_a_bad_aws_profile_is_a_clean_failure(monkeypatch, handoff_root, capsys):
    from botocore.exceptions import ProfileNotFound

    def no_profile():
        raise ProfileNotFound(profile="nope")

    _s3(monkeypatch, handoff_root)
    monkeypatch.setattr("s3sync.client.make_client", no_profile)
    assert _run(monkeypatch, ["import-pending", "--csv", str(_pending_csv(handoff_root / "p.csv")),
                              "--export-id", "e1", "--push"]) == 1
    assert "import-pending done locally; pushed: -; FAILED: handoff" in capsys.readouterr().err


def _seed_silver():
    _make_stamped_silver_generation("s1", [
        ("CASE-A", 1, "TEXT", "Mast cell tumor, malignant", "Mast Cell Tumors", "1001/3", "kw", "Exact", 1.0, "tier1_exact"),
    ])


def _seed_coding():
    config.COMBINED_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in COMBINED_PREDICTIONS_COLUMNS}]), config.COMBINED_PREDICTIONS_CSV)
    io_utils.write_csv(pd.DataFrame([{c: "" for c in REVIEW_QUEUE_COLUMNS}]), config.REVIEW_QUEUE_CSV)


def test_push_after_export_coding_pushes_the_handoff_set(monkeypatch, handoff_root):
    fake = _s3(monkeypatch, handoff_root)
    _seed_coding()
    assert _run(monkeypatch, ["export-coding", "--run-id", "run1", "--push"]) == 0
    assert pushed_sets(fake) == {"handoff"}


def _snapshot(root):
    import hashlib

    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("command", sorted(handoff_script.PUSH_SETS))
def test_push_sets_cover_every_file_the_command_writes(monkeypatch, handoff_root, labels_csv, command):
    """PUSH_SETS is hand-kept: a file a command writes outside its sets would silently never sync."""
    _s3(monkeypatch, handoff_root)
    argv = {
        "import-pending": lambda: ["import-pending", "--csv", str(_pending_csv(handoff_root / "in.csv")), "--export-id", "e1"],
        "import-gold": lambda: ["import-gold", "--csv", str(_gold_csv(monkeypatch, handoff_root, labels_csv)),
                                "--export-id", "g1", "--reviewer", "Dr. T"],
        "export-silver": lambda: (_seed_silver(), ["export-silver", "--silver-id", "s1"])[1],
        "export-coding": lambda: (_seed_coding(), ["export-coding", "--run-id", "run1"])[1],
        "export-audit-list": lambda: ["export-audit-list", "--list-id", "L1", "--no-review-queue"],
    }[command]()
    before = _snapshot(handoff_root)
    assert _run(monkeypatch, argv) == 0
    after = _snapshot(handoff_root)
    written = [p for p in after if before.get(p) != after[p]]
    assert written
    allowed = [Path(config.S3_SYNC_SETS[name]) for name in handoff_script.PUSH_SETS[command]]
    allowed += [Path(d) for d in config.S3_SYNC_EXCLUDED_DIRS]
    stray = [p for p in written if not any(p.is_relative_to(d) for d in allowed)]
    assert not stray, f"{command} wrote outside its pushed sets: {stray}"


def test_export_bundle_has_no_push_flag(monkeypatch, handoff_root):
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["export-bundle", "--push"])


def test_nothing_touches_s3_without_push(monkeypatch, handoff_root):
    forbid_s3(monkeypatch)
    assert _run(monkeypatch, ["import-pending", "--csv", str(_pending_csv(handoff_root / "p.csv")),
                              "--export-id", "e1"]) == 0
    assert _run(monkeypatch, ["export-audit-list", "--list-id", "L1", "--no-review-queue"]) == 0


def test_push_sets_are_configured_sets():
    assert set().union(*handoff_script.PUSH_SETS.values()) <= set(config.S3_SYNC_SETS)
    assert "export-bundle" not in handoff_script.PUSH_SETS


@pytest.mark.parametrize("script", ["handoff.py", "promote.py"])
def test_help_does_not_import_boto3(script):
    import subprocess

    result = subprocess.run([sys.executable, "-X", "importtime", str(_HANDOFF_PY.with_name(script)), "--help"],
                            capture_output=True, text=True, check=True)
    assert not any(name in result.stderr for name in ("boto3", "botocore", "s3sync"))
