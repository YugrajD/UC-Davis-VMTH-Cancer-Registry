"""bronze_rr.phase2: the one-shot test-set scorer. Synthetic labels / predictions / generation in tmp paths; nothing real is read."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bronze_rr import phase2, ruler
from evaluation import silver_eval, verdicts
from generations.manifest import ManifestError, write_manifest
from generations.promote import MARGIN

UNCOMMON = frozenset({"G3"})
GID = "rr-test"
COLUMNS = ["case_id", "predicted_term", "predicted_group", "generation_id"]
CASES = frozenset({"ZZ-1", "ZZ-2", "ZZ-3", "ZZ-4", "ZZ-5", "ZZ-6", "ZZ-7", "ZZ-8"})

# ZZ-1..5 are cancer cases, ZZ-6/7 have empty-term label rows, ZZ-8 has no label rows
LABELS = pd.DataFrame([("ZZ-1", "t1", "G1"), ("ZZ-2", "t1", "G1"), ("ZZ-3", "t3", "G2"), ("ZZ-4", "t4", "G3"),
                       ("ZZ-5", "t2", "G1"), ("ZZ-6", "", ""), ("ZZ-7", "", "")],
                      columns=["case_id", "matched_term", "matched_group"])
NC = ("Non-Cancer", "Non-Cancer")


def _predictions(rows: dict[str, tuple[str, str]], generation_id: str = GID) -> pd.DataFrame:
    return pd.DataFrame([(c, t, g, generation_id) for c, (t, g) in rows.items()], columns=COLUMNS)


# new: good ZZ-1, ZZ-3, ZZ-4; FN ZZ-2; slightly_off ZZ-5; FP ZZ-7        -> 6 rows: 3 good, 1 slight, 1 fp, 1 fn
NEW = _predictions({"ZZ-1": ("t1", "G1"), "ZZ-2": NC, "ZZ-3": ("t3", "G2"), "ZZ-4": ("t4", "G3"),
                    "ZZ-5": ("t1", "G1"), "ZZ-6": NC, "ZZ-7": ("t1", "G1"), "ZZ-8": NC})
# base: good ZZ-1, ZZ-2, ZZ-5; ZZ-3 completely_off + FN row; FN ZZ-4; FP ZZ-6 -> 7 rows: 3 good, 1 co, 1 fp, 2 fn
BASE = _predictions({"ZZ-1": ("t1", "G1"), "ZZ-2": ("t1", "G1"), "ZZ-3": ("t1", "G1"), "ZZ-4": NC,
                     "ZZ-5": ("t2", "G1"), "ZZ-6": ("t1", "G1"), "ZZ-7": NC, "ZZ-8": NC}, "gen-base")


def _score(new=NEW, base=BASE, n_boot=200):
    return phase2.score_systems(new, base, LABELS, CASES, UNCOMMON, n_boot=n_boot)


# --- scoring, bootstrap, discordant -------------------------------------------------------------------------------

def test_both_systems_are_scored_on_the_same_universe_and_the_differences_are_hand_checkable():
    r = _score()
    assert [r["new"][v] for v in verdicts.VERDICTS] == [3, 1, 0, 1, 1] and r["new"]["total"] == 6
    assert [r["baseline"][v] for v in verdicts.VERDICTS] == [3, 0, 1, 1, 2] and r["baseline"]["total"] == 7
    assert r["differences"]["good"]["diff"] == pytest.approx(3 / 6 - 3 / 7)
    assert r["differences"]["good_plus_slight"]["diff"] == pytest.approx(4 / 6 - 3 / 7)
    assert r["differences"]["false_negative"]["diff"] == pytest.approx(1 / 6 - 2 / 7)
    assert r["test_cases"] == 8 and r["cases_with_prediction_rows"] == {"new": 8, "baseline": 8}
    assert r["wp14_sanity"] == "MISMATCH"                  # 7 synthetic rows are not WP14's 4,299


def test_identical_systems_give_zero_difference_and_a_zero_interval():
    r = _score(NEW, NEW.copy())
    for d in r["differences"].values():
        assert d["diff"] == d["lo"] == d["hi"] == 0
    assert r["all_pass"] and r["kill_rule_gaps"]["good_gap_pp"] == 0


def test_the_interval_brackets_the_point_estimate_in_a_non_degenerate_case():
    for d in _score(n_boot=300)["differences"].values():
        assert d["lo"] <= d["hi"]
    assert _score(n_boot=300)["differences"] == _score(n_boot=300)["differences"]      # seed 0 every time


def test_discordant_cases_are_counted_at_case_level():
    d = _score()["discordant"]
    assert d == {"new_good_baseline_none": 2,                           # ZZ-3, ZZ-4
                 "baseline_good_new_none": 2,                           # ZZ-2, ZZ-5
                 "cancer_cases": 5,
                 "cancer_new_non_cancer_baseline_cancer_term": 1,       # ZZ-2
                 "cancer_baseline_non_cancer_new_cancer_term": 1}       # ZZ-4


def test_a_case_with_two_good_rows_counts_once():
    new = pd.concat([NEW, _predictions({"ZZ-1": ("t1", "G1")}).assign(predicted_term="t1")], ignore_index=True)
    assert _score(new, BASE)["discordant"]["new_good_baseline_none"] == 2


def test_wp14_sanity_matches_when_the_baseline_row_equals_the_recorded_numbers(monkeypatch):
    base = silver_eval.evaluate(BASE, LABELS, CASES, UNCOMMON)["summary"]
    monkeypatch.setattr(ruler, "WP14_TEST", {"total": base["total"],
                                             **{v: round(100 * base[f"{v}_share"], 2) for v in verdicts.VERDICTS}})
    r = _score()
    assert r["wp14_sanity"] == "MATCH"
    r["provenance"] = {"generation_id": GID, "split_id": "s", "n_boot": 200, "bootstrap_seed": 0}
    assert "WARNING" not in phase2.render_summary(r)
    monkeypatch.setattr(ruler, "WP14_TEST", {**ruler.WP14_TEST, "total": base["total"] + 1})
    mismatch = _score()
    mismatch["provenance"] = r["provenance"]
    assert mismatch["wp14_sanity"] == "MISMATCH" and "WARNING" in phase2.render_summary(mismatch)


# --- bars ---------------------------------------------------------------------------------------------------------

def _summary(good=0.4, gs=0.6, fp=0.0235):
    return {"good_share": good, "good_plus_slight_share": gs, "false_positive_share": fp}


def _diffs(good_lo):
    return {"good": {"diff": 0.0, "lo": good_lo, "hi": 0.05}}


def _bars(good_lo=0.0, new=None, base=None):
    return phase2.evaluate_bars(new or _summary(), base or _summary(), _diffs(good_lo))


def test_margin_matches_the_promotion_rule():
    assert phase2.GOOD_MARGIN == MARGIN


def test_good_bar_edge_is_inclusive_at_minus_two_pp():
    assert _bars(-0.02)["bars"]["good_lower_bound"]["pass"]
    assert not _bars(-0.0201)["bars"]["good_lower_bound"]["pass"]
    assert _bars(0.01)["bars"]["good_lower_bound"]["pass"]


def test_gs_bar_needs_new_at_least_equal_to_baseline():
    assert _bars(new=_summary(gs=0.6), base=_summary(gs=0.6))["bars"]["gs_not_below_baseline"]["pass"]
    assert not _bars(new=_summary(gs=0.5999), base=_summary(gs=0.6))["bars"]["gs_not_below_baseline"]["pass"]


def test_fp_bar_edge_is_inclusive_at_baseline_plus_one_pp():
    base = _summary(fp=0.0235)
    assert _bars(new=_summary(fp=0.0235 + 0.01), base=base)["bars"]["fp_within_1pp"]["pass"]
    assert _bars(new=_summary(fp=0.0335), base=base)["bars"]["fp_within_1pp"]["pass"]
    assert not _bars(new=_summary(fp=0.0336), base=base)["bars"]["fp_within_1pp"]["pass"]


def test_overall_needs_all_three_bars_and_nan_fails_the_good_bar():
    assert _bars()["all_pass"]
    assert not _bars(good_lo=-0.03)["all_pass"]
    assert not _bars(new=_summary(gs=0.5))["all_pass"]
    assert not _bars(new=_summary(fp=0.05))["all_pass"]
    assert not _bars(good_lo=float("nan"))["bars"]["good_lower_bound"]["pass"]


def test_kill_rule_gaps_are_reported_in_pp_and_flag_more_than_five():
    g = _bars(new=_summary(good=0.35, gs=0.5), base=_summary(good=0.41, gs=0.55))["kill_rule_gaps"]
    assert g["good_gap_pp"] == pytest.approx(6.0) and g["good_behind_more_than_5pp"]
    assert g["gs_gap_pp"] == pytest.approx(5.0) and not g["gs_behind_more_than_5pp"]
    ahead = _bars(new=_summary(good=0.5), base=_summary(good=0.4))["kill_rule_gaps"]
    assert ahead["good_gap_pp"] == pytest.approx(-10.0) and not ahead["good_behind_more_than_5pp"]


# --- run: guard, write-once, generation checks ---------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(silver_eval, "load_labels", lambda _id: LABELS)
    monkeypatch.setattr(silver_eval, "partition_case_ids", lambda split_id, partition, half=None: CASES)
    monkeypatch.setattr(phase2, "load_uncommon_groups", lambda: UNCOMMON)
    generation_dir = tmp_path / "gen" / GID
    generation_dir.mkdir(parents=True)
    (generation_dir / "thresholds.json").write_text(json.dumps({"gate_t": 0.7, "tau": 0.1, "rho": 0.25}), encoding="utf-8")
    (generation_dir / "uncommon_groups.txt").write_text("G3\n", encoding="utf-8")
    write_manifest(generation_dir, {"generation_id": GID, "retriever": {"seed": 42}, "gate": {"kind": "joint_head"},
                                    "calibration": {"status": "calibrated", "values": {"gate_t": 0.7, "tau": 0.1, "rho": 0.25}}})
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    NEW.to_csv(predictions_dir / f"{GID}_predictions.csv", index=False)
    baseline = predictions_dir / "base_predictions.csv"
    BASE.to_csv(baseline, index=False)
    return {"generation_dir": generation_dir, "baseline": baseline, "predictions_dir": predictions_dir,
            "out_root": tmp_path / "phase2"}


def _run(world, allow_test=True):
    return phase2.run(world["generation_dir"], world["baseline"], allow_test=allow_test,
                      out_root=world["out_root"], predictions_dir=world["predictions_dir"])


def test_the_guard_flag_is_required_and_nothing_is_written_without_it(world):
    with pytest.raises(SystemExit, match="--i-understand-this-spends-the-test-set"):
        _run(world, allow_test=False)
    assert not world["out_root"].exists()


def test_run_writes_aggregates_only_and_is_write_once(world):
    result = _run(world)
    out = world["out_root"] / GID
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert scores["new"]["total"] == 6 and scores["wp14_sanity"] == "MISMATCH" and "WARNING" in summary
    assert scores["provenance"]["generation_id"] == GID and len(scores["provenance"]["manifest_sha256"]) == 64
    assert scores["provenance"]["n_boot"] == 1000 and scores["provenance"]["bootstrap_seed"] == 0
    assert scores["provenance"]["generation_uncommon_list_matches"] is True
    assert "ZZ-" not in summary + (out / "scores.json").read_text(encoding="utf-8")      # no case ids
    assert result["output_dir"] == str(out)

    before = (out / "scores.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="write-once"):
        _run(world)
    assert (out / "scores.json").read_text(encoding="utf-8") == before


def test_predictions_with_another_generation_id_are_refused(world):
    _predictions({"ZZ-1": ("t1", "G1")}, "rr-other").to_csv(world["predictions_dir"] / f"{GID}_predictions.csv", index=False)
    with pytest.raises(SystemExit, match="generation_id"):
        _run(world)
    assert not world["out_root"].exists()


def test_a_generation_that_is_not_calibrated_is_refused(world):
    directory = world["generation_dir"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["calibration"] = {"status": "uncalibrated"}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="calibrated"):
        _run(world)


def test_a_generation_that_no_longer_matches_its_manifest_is_refused(world):
    (world["generation_dir"] / "thresholds.json").write_text(json.dumps({"gate_t": 0.1, "tau": 0.1, "rho": 0.25}), encoding="utf-8")
    with pytest.raises(ManifestError):
        _run(world)
    assert not world["out_root"].exists()
