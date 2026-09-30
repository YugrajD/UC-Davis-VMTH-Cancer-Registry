"""WP15's gold path on mock gold: gold-train rows reach the corrected annotations, a candidate trained on
them records that gold in its manifest, and the gold-train-growth trigger reads it back.

Synthetic cases and a synthetic gold store only; real gold does not exist yet.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

import io_utils
from coding.corrected import write_corrected_annotations
from generations import triggers
from generations.manifest import read_manifest
from manual_audit import gold

from . import fixtures as fx

_TRAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("ml_scripts_train_mock_gold", _TRAIN_PY)
train_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, train_script)
_spec.loader.exec_module(train_script)

SPLIT = "all-train"
MOCK_GOLD = [(case_id, "1001/3", "Mast cell tumor, malignant", "Mast Cell Tumors")
             for case_id in fx.TRAINING_CASE_IDS[15:18]]  # three silver no-cancer cases, "corrected" by review


def test_mock_gold_train_reaches_the_manifest_and_fires_the_growth_trigger(tmp_path, monkeypatch, tiny_bert_dir):
    fx.point_promotion_config_at(monkeypatch, tmp_path)
    import config

    config.REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fx.make_training_reports_csv(config.REPORT_CSV)
    fx.make_labels_csv(config.LABELS_CSV)
    fx.make_two_way_split_generation(SPLIT, train_ids=fx.TRAINING_CASE_IDS, test_ids=[])
    with open(fx.make_training_annotation_csv(tmp_path / "annotation.csv"), newline="", encoding="utf-8") as file:
        fx.make_silver_generation("silver-M", list(csv.reader(file))[1:])
    shutil.copytree(tiny_bert_dir, config.REPORT_MAPPING_CURRENT_DIR / "petbert")  # heads train on current's backbone

    rows = [{field: "" for field in gold.GOLD_STORE_FIELDS}
            | {"case_id": c, "code": code, "term": term, "group": group, "origin": "review_queue"}
            for c, code, term, group in MOCK_GOLD]
    config.GOLD_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(pd.DataFrame(rows, columns=gold.GOLD_STORE_FIELDS), config.GOLD_STORE_CSV)

    corrected = write_corrected_annotations("silver-M", SPLIT)
    table = io_utils.read_csv(corrected, encoding="utf-8", dtype=str, keep_default_na=False)
    assert set(table.loc[table["label_source"] == "gold", "case_id"]) == {c for c, *_ in MOCK_GOLD}

    assert train_script.main(["--stage", "heads", "--labels", str(corrected), "--split", SPLIT,
                              "--seed", "42", "--device", "cpu", "--out", "candidate", "--local-only"]) == 0
    parents = read_manifest(config.REPORT_MAPPING_CANDIDATE_DIR)["parents"]
    assert parents["silver_id"] == "silver-M" and parents["gold_train_codes"] == 3
    assert parents["gold_train_snapshot"] == gold.gold_snapshot_hash(SPLIT)

    # Against an incumbent trained on no gold the three codes are new; against the candidate itself, none are.
    assert triggers.gold_train_growth({"parents": {}}, SPLIT, threshold=3).met
    assert not triggers.gold_train_growth({"parents": parents}, SPLIT, threshold=1).met
