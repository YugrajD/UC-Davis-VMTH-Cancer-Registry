"""report_mapping.training.calibrate on synthetic scores and a synthetic generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generations.guards import GuardViolation
from generations.manifest import read_manifest, verify_manifest, write_manifest
from report_mapping.inference.predict import run_predict
from report_mapping.model.generation import generation_paths, load_generation
from report_mapping.training import calibrate as cal

from . import fixtures as fx

SPLIT_ID = "calib-test"
CALIBRATION_IDS = fx.CASE_IDS[:6]


@pytest.fixture
def setup(tmp_path, monkeypatch, report_mapping_bundle: Path):
    """A synthetic generation with a populated embedding cache, a three-way split and a labels table."""
    import config

    fx.point_generations_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "REPORT_CSV", fx.make_reports_csv(tmp_path / "report.csv"))
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", tmp_path / "embedding_cache")
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    run_predict(generation_dir=report_mapping_bundle, local_only=True, device_arg="cpu", embed_only=True)
    fx.make_three_way_split_generation(SPLIT_ID, ["CASE-0101", "CASE-0102"], CALIBRATION_IDS, fx.CASE_IDS[6:])
    return report_mapping_bundle, str(fx.make_calibration_labels_csv(tmp_path / "labels.csv"))


def test_fit_lp_threshold_recovers_known_optimum():
    # Perfect separation for any t in (0.55, 0.62]: the only grid point there is 0.60 (lowest F1-max wins).
    probs = np.array([0.62, 0.71, 0.93, 0.10, 0.50, 0.55])
    targets = np.array([1, 1, 1, 0, 0, 0])
    assert cal.fit_lp_threshold(probs, targets) == 0.60
    # Every point ties at F1 0 without positives -> the first grid point.
    assert cal.fit_lp_threshold(probs, np.zeros(6)) == cal.LP_GRID[0]


def test_fit_stage_thresholds_recovers_known_optimum_and_breaks_ties_by_good_then_grid_order():
    target = {"case_presence_gate": 0.75, "group": 0.60, "tail_max_predictions": 3, "tail_max_group_prob_gap": 0.10}

    def score(point):
        distance = sum(abs(point[k] - v) for k, v in target.items())
        return -distance, 0.0

    best, table = cal.fit_stage_thresholds(score)
    assert best == target
    assert len(table) == len(cal.GATE_GRID) * len(cal.GROUP_GRID) * len(cal.TAIL_GRID)

    best, _ = cal.fit_stage_thresholds(lambda p: (0.5, 1.0 if p["group"] == 0.90 else 0.0))
    assert best["group"] == 0.90 and best["case_presence_gate"] == cal.GATE_GRID[0]  # good share, then grid order
    best, _ = cal.fit_stage_thresholds(lambda p: (0.5, 0.5))
    assert best == {"case_presence_gate": cal.GATE_GRID[0], "group": cal.GROUP_GRID[0],
                    "tail_max_predictions": cal.TAIL_GRID[0][0], "tail_max_group_prob_gap": cal.TAIL_GRID[0][1]}


def test_refuses_non_calibration_partition(setup):
    generation_dir, labels = setup
    with pytest.raises(ValueError, match="calibration"):
        cal.calibrate(generation_dir, labels=labels, split_id=SPLIT_ID, partition="test")


def test_refuses_labels_with_no_row_on_the_calibration_partition(setup, tmp_path):
    # A corrected-annotations table covers train only; fitting on it would silently pick the first grid point.
    generation_dir, labels = setup
    train_only = tmp_path / "train_only.csv"
    pd.DataFrame({"case_id": ["NOT-IN-CALIBRATION"], "matched_term": [""], "matched_group": [""],
                  "matched_code": [""]}).to_csv(train_only, index=False)
    with pytest.raises(ValueError, match="no rows on the 'calibration' partition"):
        cal.calibrate(generation_dir, labels=str(train_only), split_id=SPLIT_ID)


def test_refuses_calibration_cases_the_generation_trained_on(setup):
    generation_dir, labels = setup
    fx.make_three_way_split_generation("trained-on", ["CASE-0001"], [], fx.CASE_IDS[1:])
    fields = {k: v for k, v in read_manifest(generation_dir).items() if k not in ("files", "created_at", "git_sha")}
    write_manifest(generation_dir, {**fields, "parents": {"split_id": "trained-on"}})
    with pytest.raises(GuardViolation, match="CASE-0001"):
        cal.calibrate(generation_dir, labels=labels, split_id=SPLIT_ID)


def test_refuses_to_embed_on_a_cache_miss(setup, tmp_path, monkeypatch):
    import config

    generation_dir, labels = setup
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", tmp_path / "empty_cache")
    with pytest.raises(FileNotFoundError, match="no embedding cache"):
        cal.calibrate(generation_dir, labels=labels, split_id=SPLIT_ID)
    assert not (tmp_path / "empty_cache").exists()


def test_writes_thresholds_and_a_valid_manifest(setup):
    generation_dir, labels = setup
    result = cal.calibrate(generation_dir, labels=labels, split_id=SPLIT_ID)

    verify_manifest(generation_dir)
    gen = load_generation(generation_dir)  # manifest + fingerprint still verify
    block = gen.manifest["calibration"]
    assert block["status"] == "calibrated"
    assert block["partition"] == "calibration" and block["split_id"] == SPLIT_ID
    assert block["n_cases"] == len(CALIBRATION_IDS)
    assert gen.thresholds["case_presence_gate"] in cal.GATE_GRID
    assert gen.thresholds["group"] in cal.GROUP_GRID
    assert (gen.thresholds["tail_max_predictions"], gen.thresholds["tail_max_group_prob_gap"]) in cal.TAIL_GRID
    assert set(gen.lp_thresholds) <= set(gen.label_presence_heads)
    assert all(t in cal.LP_GRID for t in gen.lp_thresholds.values())
    assert block["values"]["label_presence"] == gen.lp_thresholds

    diagnostics_path = generation_paths(generation_dir).thresholds_json.parent / cal.DIAGNOSTICS_NAME
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["gate"]["n_cases"] == len(CALIBRATION_IDS)
    assert diagnostics["group"]["n_cancer_cases"] == 4  # CASE-0001, -0003, -0004, -0005
    assert set(diagnostics["label_presence"]) == set(gen.lp_thresholds)
    assert result["diagnostics"]["partition_gs_share"] == block["partition_gs_share"]
    best = max(diagnostics["grid_scores"], key=lambda r: (r["gs_share"], r["good_share"]))
    assert best["gs_share"] == pytest.approx(block["partition_gs_share"])


def _set_thresholds(generation_dir: Path, thresholds: dict) -> None:
    generation_paths(generation_dir).thresholds_json.write_text(json.dumps(thresholds), encoding="utf-8")
    fields = {k: v for k, v in read_manifest(generation_dir).items() if k not in ("files", "created_at", "git_sha")}
    write_manifest(generation_dir, fields)


@pytest.mark.parametrize("gate, group, tail", [
    (cal.GATE_GRID[0], cal.GROUP_GRID[0], cal.TAIL_GRID[0]),
    (cal.GATE_GRID[0], cal.GROUP_GRID[1], cal.TAIL_GRID[-1]),
    (cal.GATE_GRID[3], cal.GROUP_GRID[-1], cal.TAIL_GRID[3]),
    (cal.GATE_GRID[-1], cal.GROUP_GRID[5], cal.TAIL_GRID[5]),
    ("median", cal.GROUP_GRID[0], cal.TAIL_GRID[-1]),  # a gate that rejects some cases and passes others
])
def test_gate_last_shortcut_matches_run_predict(setup, tmp_path, gate, group, tail):
    """At each point, the shortcut (every case through the stages, gate applied last) gives exactly the rows
    run_predict writes with that point's thresholds in the generation (gate applied first)."""
    generation_dir, _ = setup
    inp = cal._load_partition_inputs(load_generation(generation_dir), frozenset(fx.CASE_IDS))
    if gate == "median":
        gate = float(np.median(inp.case_presence_probs))
        passed = inp.case_presence_probs >= gate
        assert passed.any() and not passed.all()
    k, gap = tail
    _set_thresholds(generation_dir, {"case_presence_gate": gate, "group": group, "tail_max_predictions": k,
                                     "tail_max_group_prob_gap": gap, "label_presence_fallback": 0.5})
    gen = load_generation(generation_dir)
    out = run_predict(generation_dir=generation_dir, local_only=True, device_arg="cpu", out_path=tmp_path / "p.csv")
    expected = pd.read_csv(out, dtype=str, keep_default_na=False)[["case_id", "predicted_term", "predicted_group"]]

    passed_rows = cal._categorize(gen, inp, group, k, gap, gen.lp_thresholds)
    rows = cal._prediction_rows(inp, passed_rows, gate)
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected.reset_index(drop=True))


def test_gate_and_group_diagnostics_count_a_term_without_a_group_as_cancer(setup):
    """Legacy's gate/group evaluators: cancer = any non-empty matched_term. LP pairs still need the group."""
    generation_dir, labels = setup
    with open(labels, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(["CASE-0002", "Mast cell tumor, malignant", "", ""])
    diagnostics = cal.calibrate(generation_dir, labels=labels, split_id=SPLIT_ID)["diagnostics"]

    assert diagnostics["group"]["n_cancer_cases"] == 5  # CASE-0001, -0002, -0003, -0004, -0005
    assert diagnostics["gate"]["tp"] + diagnostics["gate"]["fn"] == 5
    assert diagnostics["label_presence"]["Mast Cell Tumors"]["n_cases"] == 2  # CASE-0001, -0003 only
