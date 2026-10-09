"""scripts/train.py, calibrate.py and predict.py: the thin entry points, end to end on a tiny offline ModernBERT.

train.py is one step (labels + split -> candidate with calibration pending); calibrate.py fits the thresholds;
predict.py refuses an uncalibrated generation and stamps its predictions with the generation id.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

import config
from generations.manifest import read_manifest, verify_manifest
from report_mapping import generation
from report_mapping.inference import decide
from report_mapping.model import ENCODER_ID

from . import fixtures as fx

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ml_next_scripts_{name}_tests", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


train_script, calibrate_script, predict_script = _load("train"), _load("calibrate"), _load("predict")


@pytest.fixture
def world(tmp_path, monkeypatch):
    labels, split_id = fx.make_training_world(tmp_path, monkeypatch)
    return labels, split_id


def _train(world, tiny_modernbert_dir, *extra):
    labels, split_id = world
    return train_script.main(["--labels", labels, "--split", split_id, "--device", "cpu",
                              "--model", str(tiny_modernbert_dir), "--max-steps", "2", *extra])


def test_train_produces_a_pending_candidate_that_verifies(world, tiny_modernbert_dir):
    assert _train(world, tiny_modernbert_dir) == 0

    candidate = config.REPORT_MAPPING_CANDIDATE_DIR
    verify_manifest(candidate)
    manifest = generation.verify_spec(candidate, require_calibrated=False)
    assert manifest["calibration"] == {"status": "pending"} and manifest["status"] == "candidate"
    assert re.fullmatch(r"gen-\d{8}T\d{6}Z", manifest["generation_id"])
    assert manifest["parents"]["split_id"] == world[1] and manifest["training"]["seed"] == 42
    assert manifest["model"]["encoder_id"] == str(tiny_modernbert_dir)
    assert not (candidate / "thresholds.json").exists()
    # A freshly trained candidate is uncalibrated, so predict.py refuses it.
    with pytest.raises(generation.GenerationError, match="calibrate"):
        generation.verify_spec(candidate)


def test_train_calibrate_predict_end_to_end(world, tiny_modernbert_dir, tmp_path, capsys):
    labels, split_id = world
    assert _train(world, tiny_modernbert_dir, "--seed", "7") == 0
    candidate = config.REPORT_MAPPING_CANDIDATE_DIR
    assert read_manifest(candidate)["training"]["seed"] == 7

    assert predict_script.main(["--generation", "candidate", "--device", "cpu"]) == 1   # not calibrated yet
    assert "REFUSED" in capsys.readouterr().err

    # (--max-fp-share 1.0: with random heads the default cap would leave no accepted case to fit the review cut on)
    assert calibrate_script.main(["--labels", labels, "--split", split_id, "--device", "cpu",
                                  "--max-fp-share", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "calibrated on calibration of" in out
    manifest = generation.verify_spec(candidate)                    # now whole, calibrated and bound to its weights
    assert manifest["calibration"]["max_fp_share"] == 1.0
    assert manifest["calibration"]["review_target_share"] == config.REVIEW_TARGET_SHARE
    assert set(generation.read_thresholds(candidate)) == set(generation.THRESHOLD_KEYS)

    assert calibrate_script.main(["--labels", labels, "--split", split_id, "--device", "cpu",
                                  "--max-fp-share", "1.0"]) == 1   # calibrated once
    assert "recalibrate" in capsys.readouterr().err

    predictions = tmp_path / "out" / "candidate_predictions.csv"
    assert predict_script.main(["--generation", "candidate", "--device", "cpu", "--out", str(predictions)]) == 0
    frame = pd.read_csv(predictions, dtype=str, keep_default_na=False)
    assert list(frame.columns) == decide.COLUMNS
    assert set(frame["generation_id"]) == {manifest["generation_id"]} and len(frame) > 0


def test_predict_defaults_to_current_and_writes_to_the_predictions_dir(world, tiny_generation_dir, monkeypatch):
    import shutil

    shutil.copytree(tiny_generation_dir, config.REPORT_MAPPING_CURRENT_DIR)
    assert predict_script.main(["--device", "cpu"]) == 0
    expected = config.PREDICTIONS_DIR / f"{read_manifest(config.REPORT_MAPPING_CURRENT_DIR)['generation_id']}_predictions.csv"
    assert expected.is_file()


def test_train_refuses_a_corrupt_split_before_writing_anything(world, tiny_modernbert_dir, capsys):
    labels, _ = world
    ids = fx.TRAINING_CASE_IDS
    fx.make_three_way_split_generation("corrupt", ids[:10], ids[:2], ids[13:15])    # train and calibration overlap
    assert train_script.main(["--labels", labels, "--split", "corrupt", "--device", "cpu",
                              "--model", str(tiny_modernbert_dir), "--max-steps", "1"]) == 1
    assert "REFUSED" in capsys.readouterr().err
    assert not (config.REPORT_MAPPING_CANDIDATE_DIR / "manifest.json").exists()


def test_train_refuses_a_non_empty_out_directory(world, tiny_modernbert_dir, capsys):
    config.REPORT_MAPPING_CANDIDATE_DIR.mkdir(parents=True)
    (config.REPORT_MAPPING_CANDIDATE_DIR / "old.txt").write_text("x", encoding="utf-8")
    assert _train(world, tiny_modernbert_dir) == 1
    assert "not empty" in capsys.readouterr().err
    assert (config.REPORT_MAPPING_CANDIDATE_DIR / "old.txt").is_file()


def test_train_script_defaults_and_flags_reach_the_trainer(world, monkeypatch):
    labels, split_id = world
    seen = []

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return {"generation_id": "gen-x", "model": {"n_terms": 0}}

    monkeypatch.setattr(train_script, "train", spy)
    assert train_script.main(["--labels", labels, "--device", "cpu"]) == 0
    (args, kwargs), = seen
    assert args[:2] == (labels, config.DEFAULT_SPLIT_ID) and args[2] == config.REPORT_MAPPING_CANDIDATE_DIR
    assert args[3].seed == 42 and args[4] == "cpu"
    assert kwargs == {"model_id": ENCODER_ID, "limit": 0, "max_steps": 0}

    seen.clear()
    train_script.main(["--labels", labels, "--split", split_id, "--seed", "3", "--device", "cpu",
                       "--out", "elsewhere", "--model", "some/checkpoint", "--limit", "9", "--max-steps", "4"])
    (args, kwargs), = seen
    assert args[0] == labels and args[1] == split_id and args[2] == Path("elsewhere") and args[3].seed == 3
    assert kwargs == {"model_id": "some/checkpoint", "limit": 9, "max_steps": 4}


def test_calibrate_script_defaults_to_the_candidate_and_the_config_cap(world, monkeypatch):
    labels, _ = world
    seen = []

    def spy(*args):
        seen.append(args)
        return {"partition": "calibration", "split_id": "s", "n_cases": 1, "n_cases_with_text": 1, "n_accepted": 1,
                "values": {"gate_t": 0.5}, "partition_gs_share": 0.0, "partition_good_share": 0.0,
                "review_share_below_cut": 0.0}

    monkeypatch.setattr(calibrate_script, "calibrate", spy)
    assert calibrate_script.main(["--labels", labels, "--device", "cpu"]) == 0
    assert seen[-1] == (config.REPORT_MAPPING_CANDIDATE_DIR, labels, config.DEFAULT_SPLIT_ID, config.FP_CAP,
                        config.REVIEW_TARGET_SHARE, "cpu")
    calibrate_script.main(["--labels", labels, "--generation", "current", "--split", "s2", "--max-fp-share", "0.5",
                           "--device", "cpu"])
    assert seen[-1] == (config.REPORT_MAPPING_CURRENT_DIR, labels, "s2", 0.5, config.REVIEW_TARGET_SHARE, "cpu")


@pytest.mark.parametrize("script, argv", [
    (predict_script, ["--model", "m"]), (predict_script, ["--embed-only"]), (predict_script, ["--cache-dir", "d"]),
    (predict_script, ["--local-only"]),
    (train_script, ["--labels", "x", "--stage", "heads"]), (train_script, ["--labels", "x", "--local-only"]),
])
def test_flags_of_the_old_pipeline_are_gone(script, argv):
    with pytest.raises(SystemExit) as exit_info:
        script.main(argv)
    assert exit_info.value.code == 2


def test_resolve_device_auto_prefers_cuda_and_passes_explicit_names_through(monkeypatch):
    import torch

    from report_mapping.model import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda" and resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cuda") == "cuda"
