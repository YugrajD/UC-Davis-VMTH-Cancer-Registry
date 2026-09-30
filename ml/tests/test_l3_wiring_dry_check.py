"""Dry-check of the full L3 cycle's wiring on tiny fixtures — no real
training, just confirming the pipeline chain holds together and, critically,
that the shared embedding cache is only ever written once across the whole
3-seed cycle.

train.py --stage heads --out candidate (no --model) must resolve the
backbone to the *current* generation's petbert/ and, since heads-only
training embeds through the same content-hash cache predict.py reads
(report_mapping.training.embeddings.get_or_build ->
report_mapping.inference.predict.build_fresh_cache), it must hit the SAME
cache key predict.py would use for "current" — so re-running heads for a
second and third seed, then calibrating and predicting each seed's candidate,
never re-embeds. This is exactly what lets Windows run the L3 3-seed cycle
without touching the (machine-local) imported legacy cache a second time.

Chain per seed: train.py --stage heads -> calibrate.py -> predict.py. Reads
scripts/train.py and report_mapping/training/calibrate.py
as fixed, already-tested dependencies (not edited here); only
report_mapping/inference/predict.py's cache-dir wiring is this round's
change under test.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

from report_mapping.inference.predict import run_predict
from report_mapping.model.generation import generation_paths, load_generation
from report_mapping.training import calibrate as cal

from . import fixtures as fx

_TRAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_train_l3check", _TRAIN_PY)
train_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, train_script)
_spec.loader.exec_module(train_script)

SPLIT_ID = "l3-dry-check-split"
EXPECTED_COLUMNS = [
    "case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
    "case_presence_prob", "confidence", "group_prob", "method", "generation_id",
]


def test_l3_three_seed_cycle_hits_the_shared_cache_once(tmp_path, monkeypatch, tiny_bert_dir):
    fx.point_training_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    ann_path = tmp_path / "annotation.csv"
    fx.make_training_annotation_csv(ann_path)

    cancer_ids, noncancer_ids = fx.TRAINING_CASE_IDS[:15], fx.TRAINING_CASE_IDS[15:]
    train_ids = cancer_ids[:10] + noncancer_ids[:10]
    calibration_ids = cancer_ids[10:13] + noncancer_ids[10:13]
    test_ids = cancer_ids[13:15] + noncancer_ids[13:15]
    fx.make_three_way_split_generation(SPLIT_ID, train_ids, calibration_ids, test_ids)

    current_petbert = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).petbert_dir
    current_petbert.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_bert_dir, current_petbert)
    current_thresholds = generation_paths(config.REPORT_MAPPING_CURRENT_DIR).thresholds_json
    current_thresholds.parent.mkdir(parents=True, exist_ok=True)
    current_thresholds.write_text('{"case_presence_gate": 0.5}', encoding="utf-8")

    candidate_dir = config.REPORT_MAPPING_CANDIDATE_DIR
    predictions_dir = tmp_path / "l3_predictions"
    predictions_dir.mkdir()

    for seed in (1, 2, 3):
        rc = train_script.main([
            "--stage", "heads", "--labels", str(ann_path), "--split", SPLIT_ID,
            "--seed", str(seed), "--device", "cpu", "--out", "candidate", "--local-only",
        ])
        assert rc == 0
        # The backbone was resolved to current's petbert/ (copied in, byte-identical) on
        # seed 1 and reused as-is on seeds 2/3 -- never re-copied, never a second backbone.
        assert generation_paths(candidate_dir).petbert_dir.is_dir()

        cal.calibrate(candidate_dir, labels=str(ann_path), split_id=SPLIT_ID, device="cpu")
        gen = load_generation(candidate_dir)
        assert gen.manifest["calibration"]["status"] == "calibrated"

        # Exactly one cache entry throughout: heads-only training's own
        # get_or_build call and calibrate's load_cached_embeddings both
        # resolve to the SAME content-hash key as predict.py would for
        # "current" (same report.csv, same labels.csv, byte-identical
        # backbone copy) -- so nothing here ever re-embeds.
        assert len(list(config.EMBEDDING_CACHE_DIR.glob("*.npz"))) == 1

        out_path = predictions_dir / f"seed{seed}.csv"
        result = run_predict(generation_dir=candidate_dir, local_only=True, device_arg="cpu", out_path=out_path)
        assert result == out_path and out_path.is_file()
        assert len(list(config.EMBEDDING_CACHE_DIR.glob("*.npz"))) == 1  # predict didn't re-embed either

    seed_csvs = sorted(predictions_dir.glob("seed*.csv"))
    assert len(seed_csvs) == 3
    for path in seed_csvs:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(table.columns) == EXPECTED_COLUMNS
        assert len(table) > 0
