"""scripts/oof.py: the thin cross-fit entry point, end to end on a tiny offline ModernBERT (k=2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

import config
from generations.splits import load_split
from report_mapping.model import ENCODER_ID
from report_mapping.training import crossfit as crossfit_module

from . import fixtures as fx

_OOF_PY = Path(__file__).resolve().parents[1] / "scripts" / "oof.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_oof_tests", _OOF_PY)
oof_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, oof_script)
_spec.loader.exec_module(oof_script)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return fx.make_training_world(tmp_path, monkeypatch)


def _run(world, tiny_modernbert_dir, *extra):
    labels, split_id = world
    return oof_script.main(["--labels", labels, "--split", split_id, "--k", "2", "--device", "cpu",
                            "--model", str(tiny_modernbert_dir), "--max-steps", "2", *extra])


def test_crossfit_csv_is_numbers_only_and_covers_the_train_partition(world, tiny_modernbert_dir, tmp_path):
    out = tmp_path / "scratch" / "oof.csv"
    assert _run(world, tiny_modernbert_dir, "--out", str(out)) == 0
    frame = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(frame.columns) == crossfit_module.COLUMNS     # ids, fold, targets, probabilities, term names: no text
    train = load_split(world[1]).train
    assert set(frame["case_id"]) == train and set(frame["fold"]) == {"0", "1"}
    assert frame.groupby("case_id")["fold"].nunique().eq(1).all()      # a case is held out in exactly one fold
    cancer = frame[frame["gate_target"] == "1"]
    assert (cancer["term"] != "").all() and (frame[frame["gate_target"] == "0"]["term"] == "").all()
    assert frame["gate_prob"].astype(float).between(0, 1).all()


def test_limit_restricts_the_cases(world, tiny_modernbert_dir, tmp_path):
    out = tmp_path / "limited.csv"
    assert _run(world, tiny_modernbert_dir, "--out", str(out), "--limit", "12") == 0
    frame = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert 0 < frame["case_id"].nunique() <= 12


def test_default_out_is_named_from_labels_split_and_k_under_the_oof_dir(world, tiny_modernbert_dir):
    assert _run(world, tiny_modernbert_dir) == 0
    expected = config.OOF_DIR / f"{Path(world[0]).stem}_{world[1]}_k2.csv"
    assert expected.is_file()


def test_refuses_to_overwrite_unless_forced(world, tiny_modernbert_dir, tmp_path, capsys):
    out = tmp_path / "oof.csv"
    out.write_text("keep me", encoding="utf-8")
    assert _run(world, tiny_modernbert_dir, "--out", str(out)) == 1
    assert "REFUSED" in capsys.readouterr().err and out.read_text(encoding="utf-8") == "keep me"
    assert _run(world, tiny_modernbert_dir, "--out", str(out), "--force") == 0
    assert out.read_text(encoding="utf-8").startswith("case_id,fold,")


def test_refuses_a_corrupt_split_before_writing_anything(world, tiny_modernbert_dir, tmp_path, capsys):
    labels, _ = world
    ids = fx.TRAINING_CASE_IDS
    fx.make_three_way_split_generation("corrupt", ids[:10], ids[:2], ids[13:15])    # train and calibration overlap
    out = tmp_path / "oof.csv"
    assert oof_script.main(["--labels", labels, "--split", "corrupt", "--k", "2", "--device", "cpu",
                            "--model", str(tiny_modernbert_dir), "--max-steps", "1", "--out", str(out)]) == 1
    assert "REFUSED" in capsys.readouterr().err and not out.exists()


def test_defaults_and_flags_reach_crossfit(world, monkeypatch):
    labels, split_id = world
    seen = []

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return args[5]

    monkeypatch.setattr(oof_script, "crossfit", spy)
    assert oof_script.main(["--labels", labels, "--device", "cpu"]) == 0
    (args, kwargs), = seen
    assert args[:3] == (labels, config.DEFAULT_SPLIT_ID, 5) and args[3].seed == 42 and args[4] == "cpu"
    assert args[5] == config.OOF_DIR / f"{Path(labels).stem}_{config.DEFAULT_SPLIT_ID}_k5.csv"
    assert kwargs == {"limit": 0, "max_steps": 0, "model_id": ENCODER_ID}

    seen.clear()
    oof_script.main(["--labels", "silver-x", "--split", "s2", "--k", "3", "--seed", "9", "--device", "cpu",
                     "--out", "somewhere.csv", "--model", "some/checkpoint", "--limit", "7", "--max-steps", "4"])
    (args, kwargs), = seen
    assert args[:3] == ("silver-x", "s2", 3) and args[3].seed == 9 and args[5] == Path("somewhere.csv")
    assert kwargs == {"limit": 7, "max_steps": 4, "model_id": "some/checkpoint"}
