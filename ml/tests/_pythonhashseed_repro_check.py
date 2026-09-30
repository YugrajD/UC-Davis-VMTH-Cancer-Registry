"""Standalone helper for test_eval_batch.py's PYTHONHASHSEED reproducibility check.

Not a test module itself (no ``test_`` prefix, so pytest never collects it) —
run as a subprocess: ``python _pythonhashseed_repro_check.py <tmp_dir>``.
Builds an isolated split + silver generation under ``<tmp_dir>``, draws one
eval batch, and prints the resulting ledger CSV to stdout. The calling test
runs this twice with different ``PYTHONHASHSEED`` values and asserts the two
ledgers are byte-identical — set/dict iteration order (which depends on
PYTHONHASHSEED) must never affect the draw.

Config is patched by direct attribute assignment rather than
``monkeypatch.setattr``, since this runs as a plain script, not under pytest.
"""

import sys
from pathlib import Path

ML_NEXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_NEXT))

import config  # noqa: E402
from manual_audit import eval_batch  # noqa: E402
from tests import fixtures as fx  # noqa: E402


def main(root: Path) -> None:
    splits_dir = root / "output" / "splits"
    manual_audit_dir = root / "output" / "manual_audit"
    config.ML_ROOT = root
    config.SPLITS_DIR = splits_dir
    config.MANUAL_AUDIT_DIR = manual_audit_dir
    config.AUDIT_STORE_CSV = manual_audit_dir / "audit_store.csv"
    config.GOLD_STORE_CSV = manual_audit_dir / "gold_store.csv"
    config.EVAL_BATCH_LEDGER_CSV = manual_audit_dir / "eval_batch_ledger.csv"
    config.CAUSE_STORE_CSV = manual_audit_dir / "cause_store.csv"
    config.EVAL_BATCH_DIR = manual_audit_dir / "eval_batch"
    config.SILVER_DIR = root / "output" / "silver"
    config.DIAGNOSIS_MAPPING_AUDIT_BATCH1_TXT = manual_audit_dir / "diagnosis_mapping_audit_batch1.txt"
    config.CORRECTED_ANNOTATIONS_CSV = root / "output" / "coding" / "corrected_annotations.csv"

    train_ids = [f"TRAIN-{i:04d}" for i in range(1, 6)]
    test_ids = fx.EVAL_BATCH_CASE_IDS
    fx.make_two_way_split_generation("legacy-80-20", train_ids=train_ids, test_ids=test_ids)

    silver_id = fx.make_eval_batch_silver_generation()
    no_dm_audit = root / "dm_audit_batch1_cases.txt"
    no_dm_audit.write_text("", encoding="utf-8")

    eval_batch.generate_batch(
        "eval-batch-1", silver_id=silver_id, fraction=0.5, split_id="legacy-80-20",
        seed=42, dm_audit_batch1_cases=no_dm_audit,
    )
    sys.stdout.write(config.EVAL_BATCH_LEDGER_CSV.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
