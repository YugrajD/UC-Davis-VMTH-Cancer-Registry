"""generations/promote.py + triggers.py + scripts/{promote,retrain_cycle}.py on synthetic generations, gold and
predictions.

No model runs: predictions CSVs are written directly; generations are copies of the session's tiny generation
with rewritten manifest fields.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import config
import io_utils
from evaluation.silver_eval import SilverEvalError
from generations import promote, triggers
from generations.manifest import read_manifest, update_manifest, verify_manifest
from manual_audit import eval_batch, gold
from report_mapping.generation import ARCHITECTURE, GenerationError

from . import fixtures as fx

_PROMOTE_PY = Path(__file__).resolve().parents[1] / "scripts" / "promote.py"
_spec = importlib.util.spec_from_file_location("ml_next_scripts_promote", _PROMOTE_PY)
promote_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, promote_script)
_spec.loader.exec_module(promote_script)

SPLIT = "promo-split"
TRAIN = [f"PTRAIN-{i:03d}" for i in range(1, 11)]
TEST = [f"PTEST-{i:03d}" for i in range(1, 41)]
TERMS = [("Term A", "Group A", "8000/3"), ("Term B", "Group B", "8001/3")]
REASON = "documentation/history/decisions/0009-test-adopt.md"


def _generation(directory: Path, template: Path, generation_id: str, silver_id: str, pending: bool = False,
                architecture: dict | None = None) -> None:
    """A copy of the tiny generation with this id, silver lineage and (optionally) an other architecture."""
    shutil.copytree(template, directory)
    fields = {"generation_id": generation_id, "parents": {"silver_id": silver_id}}
    if pending:
        fields["calibration"] = {"status": "pending"}
    if architecture:
        fields["architecture"] = architecture
    update_manifest(directory, fields)


def _gold_store(rows: list[dict]) -> None:
    frame = pd.DataFrame([{field: "" for field in gold.GOLD_STORE_FIELDS} | row for row in rows],
                         columns=gold.GOLD_STORE_FIELDS)
    config.GOLD_STORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(frame, config.GOLD_STORE_CSV)


def _eval_batch_gold() -> None:
    rows, ledger = [], []
    for i, case_id in enumerate(TEST):
        term, group, code = TERMS[i % 2]
        rows.append({"case_id": case_id, "code": code, "term": term, "group": group, "origin": "eval_batch"})
        ledger.append({"case_id": case_id, "batch_id": "b1", "split_id": SPLIT, "stratum": group,
                       "N_h": "400", "review_mode": eval_batch.REVIEW_MODE})
    _gold_store(rows)
    io_utils.write_csv(pd.DataFrame(ledger).reindex(columns=eval_batch.EVAL_BATCH_LEDGER_FIELDS, fill_value=""),
                       config.EVAL_BATCH_LEDGER_CSV)


def _predictions(path: Path, generation_id: str, wrong: int = 0) -> Path:
    rows = []
    for i, case_id in enumerate(TEST):
        term, group, code = TERMS[(i + (1 if i < wrong else 0)) % 2]
        rows.append({"case_id": case_id, "diagnosis_index": "1", "predicted_term": term, "predicted_group": group,
                     "predicted_code": code, "confidence": "0.9", "generation_id": generation_id})
    io_utils.write_csv(pd.DataFrame(rows), path)
    return path


def _reason_file(name: str = "0009-test-adopt.md") -> Path:
    path = config.HISTORY_DIR / "decisions" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# ADR\n", encoding="utf-8")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch, tiny_generation_dir):
    fx.point_promotion_config_at(monkeypatch, tmp_path)
    config.UNCOMMON_GROUPS_TXT.write_text("Group Z\n", encoding="utf-8")
    fx.make_two_way_split_generation(SPLIT, TRAIN, TEST)
    _generation(config.REPORT_MAPPING_CURRENT_DIR, tiny_generation_dir, "gen-A", "silver-A")
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_generation_dir, "gen-B", "silver-B")
    return tmp_path


def _recommend(env: Path, wrong: int = 0) -> dict:
    return promote.recommend(_predictions(env / "b.csv", "gen-B", wrong), _predictions(env / "a.csv", "gen-A"),
                             split_id=SPLIT, n_boot=200)


# ---- retraining: the gold rule ----

def test_no_gold_eval_is_refused(env):
    with pytest.raises(promote.PromotionError, match="no gold-eval"):
        _recommend(env)


def test_uncalibrated_candidate_is_refused(env, tiny_generation_dir):
    _eval_batch_gold()
    shutil.rmtree(config.REPORT_MAPPING_CANDIDATE_DIR)
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_generation_dir, "gen-B", "silver-B", pending=True)
    with pytest.raises(GenerationError, match="calibration.status"):
        _recommend(env)


def test_a_candidate_with_an_other_architecture_than_current_is_refused_and_pointed_to_adopt(env):
    _eval_batch_gold()
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {**ARCHITECTURE, "name": "petbert-four-stage"}})
    with pytest.raises(promote.PromotionError, match=r"architectural change.*promote\.py --adopt --reason"):
        _recommend(env)


def test_the_architecture_check_comes_before_the_gold_check(env):
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {"name": "petbert-four-stage"}})
    with pytest.raises(promote.PromotionError, match="architectural change"):  # no gold-eval rows exist here
        _recommend(env)


def test_a_case_without_prediction_rows_scores_as_predicting_nothing(env):
    # predict.py writes no row for a case with no report text; that case is a miss, and the count is reported.
    _eval_batch_gold()
    partial = _predictions(env / "b.csv", "gen-B")
    io_utils.write_csv(io_utils.read_csv(partial, encoding="utf-8", dtype=str).iloc[1:], partial)
    c = promote.recommend(partial, _predictions(env / "a.csv", "gen-A"), split_id=SPLIT, n_boot=50)["comparison"]
    assert (c["challenger_unpredicted"], c["incumbent_unpredicted"]) == (1, 0)
    assert c["challenger_good"] == pytest.approx(39 / 40) and c["incumbent_good"] == 1.0


def test_predictions_of_another_generation_are_refused(env):
    _eval_batch_gold()
    with pytest.raises(SilverEvalError, match="generation_id"):
        promote.recommend(_predictions(env / "b.csv", "gen-B"), _predictions(env / "a.csv", "gen-Z"),
                          split_id=SPLIT, n_boot=50)


def test_equal_challenger_with_new_silver_is_promoted_and_applied(env):
    _eval_batch_gold()
    result = _recommend(env)
    assert result["comparison"]["good_diff"] == 0 and result["comparison"]["good_lo"] == 0
    assert [t.met for t in result["triggers"]] == [True, False, False]
    assert result["promote"]

    outcome = promote.apply(result, "test", today=date(2026, 9, 26))
    archive = config.ARCHIVE_ROOT / "2026-09-26_test"
    assert outcome == {"action": "promoted", "generation_id": "gen-B", "archive": archive}
    assert read_manifest(archive)["generation_id"] == "gen-A"
    assert read_manifest(archive)["status"] == "archived"
    assert not (archive / "embedding_cache").exists()      # a generation carries no embedding cache any more
    current = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    assert current["generation_id"] == "gen-B" and current["status"] == "current"
    verify_manifest(config.REPORT_MAPPING_CURRENT_DIR)  # a status change leaves the file hashes valid
    verify_manifest(archive)
    assert "adoption" not in current
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_trigger_status_runs_before_any_candidate_and_without_gold_eval(env):
    fired = promote.trigger_status("silver-B", env / "never-read.csv", split_id=SPLIT)
    assert [(t.name, t.met) for t in fired] == [
        ("new_silver_lineage", True), ("random_slice_drop", False), ("gold_train_growth", False)]
    assert not any(t.met for t in promote.trigger_status("silver-A", env / "never-read.csv", split_id=SPLIT))


def test_no_trigger_means_no_promotion(env):
    _eval_batch_gold()
    update_manifest(config.REPORT_MAPPING_CANDIDATE_DIR, {"parents": {"silver_id": "silver-A"}})
    result = _recommend(env)
    assert not any(t.met for t in result["triggers"]) and not result["promote"]


def test_worse_challenger_is_not_promoted_and_apply_deletes_it(env):
    _eval_batch_gold()
    result = _recommend(env, wrong=8)  # 8 wrong: 8 completely_off + 8 FN rows -> good 32/48
    assert result["comparison"]["good_diff"] == pytest.approx(32 / 48 - 1)
    assert result["comparison"]["good_lo"] < promote.MARGIN and not result["promote"]
    assert promote.apply(result)["action"] == "discarded"
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"


def test_failed_swap_restores_current(env, monkeypatch):
    _eval_batch_gold()
    result = _recommend(env)
    real_rename, calls = os.rename, []

    def flaky_rename(src, dst):
        calls.append((src, dst))
        if len(calls) == 2:
            raise OSError("disk full")
        real_rename(src, dst)

    monkeypatch.setattr(promote.os, "rename", flaky_rename)
    with pytest.raises(OSError, match="disk full"):
        promote.apply(result, "test")
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"
    assert read_manifest(config.REPORT_MAPPING_CANDIDATE_DIR)["generation_id"] == "gen-B"
    assert not any(config.ARCHIVE_ROOT.iterdir())


def test_same_generation_id_is_refused(env):
    _eval_batch_gold()
    update_manifest(config.REPORT_MAPPING_CANDIDATE_DIR, {"generation_id": "gen-A"})
    with pytest.raises(promote.PromotionError, match="share generation_id"):
        _recommend(env)


def test_apply_refuses_when_the_candidate_changed_since_the_recommendation(env):
    _eval_batch_gold()
    result = _recommend(env)
    update_manifest(config.REPORT_MAPPING_CANDIDATE_DIR, {"generation_id": "gen-other"})
    with pytest.raises(promote.PromotionError, match="candidate/ changed"):
        promote.apply(result)


def test_gold_train_growth_counts_codes_since_incumbent(env):
    rows = [{"case_id": c, "code": "8000/3", "term": "Term A", "group": "Group A", "origin": "review_queue"}
            for c in TRAIN[:6]]
    rows.append({"case_id": TRAIN[6], "code": gold.NO_CANCER, "origin": "review_queue"})
    _gold_store(rows)
    met = triggers.gold_train_growth({"parents": {"gold_train_codes": 1}}, SPLIT, threshold=5)
    assert met.met and met.numbers["new_codes"] == 5 and met.numbers["gold_train_codes"] == 6
    assert not triggers.gold_train_growth({"parents": {"gold_train_codes": 2}}, SPLIT, threshold=5).met


def test_new_silver_lineage_unknown_is_not_met():
    assert not triggers.new_silver_lineage({"parents": {}}, "silver-B").met
    assert triggers.new_silver_lineage({"parents": {"silver_id": "silver-A"}}, "silver-B").met


def test_random_slice_drop_needs_non_overlapping_intervals():
    def case(i, origin, period=""):
        return {"case_id": f"C{i:03d}", "origin": origin, "upload_period": period}

    gold_rows = pd.DataFrame([case(i, "eval_batch") for i in range(60)]
                             + [case(i, "random_slice", "2026-10") for i in range(60, 100)])
    cases = pd.DataFrame({"weight": 1.0, "stratum": "s"}, index=gold_rows["case_id"])

    def table(slice_good: int) -> pd.DataFrame:
        verdict = ["good"] * 60 + ["good"] * slice_good + ["completely_off"] * (40 - slice_good)
        return pd.DataFrame({"case_id": gold_rows["case_id"], "verdict": verdict, "weight": 1.0})

    assert triggers.random_slice_drop(table(10), cases, gold_rows, n_boot=200).met
    assert not triggers.random_slice_drop(table(40), cases, gold_rows, n_boot=200).met
    no_slice = triggers.random_slice_drop(table(40), cases, gold_rows[gold_rows["origin"] == "eval_batch"])
    assert not no_slice.met and no_slice.numbers == {"slice_cases": 0}


# ---- promote.py: recommend / --apply ----

def _argv(env, *extra, wrong=0):
    return ["promote.py", "--candidate-predictions", str(_predictions(env / "b.csv", "gen-B", wrong)),
            "--incumbent-predictions", str(_predictions(env / "a.csv", "gen-A")), "--split", SPLIT, "--n-boot", "50",
            *extra]


def test_script_refuses_cleanly_without_gold_eval(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _argv(env))
    assert promote_script.main() == 1
    assert "no gold-eval" in capsys.readouterr().err


def test_script_prints_recommendation(env, monkeypatch, capsys):
    _eval_batch_gold()
    monkeypatch.setattr(sys, "argv", _argv(env))
    assert promote_script.main() == 0
    out = capsys.readouterr().out
    assert "Recommendation: PROMOTE" in out
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists()  # recommend-only: nothing moved


def test_script_refuses_an_architectural_change_and_names_adopt(env, monkeypatch, capsys):
    _eval_batch_gold()
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {"name": "petbert-four-stage"}})
    monkeypatch.setattr(sys, "argv", _argv(env, "--apply"))
    assert promote_script.main() == 1
    assert "--adopt --reason" in capsys.readouterr().err
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_script_flag_combinations(env, monkeypatch, capsys):
    for argv, message in ((["--adopt"], "--adopt requires --reason"),
                          (["--adopt", "--apply", "--reason", "x.md"], "mutually exclusive"),
                          (["--reason", "x.md"], "only for --adopt"),
                          (["--publish"], "--publish requires --apply or --adopt")):
        monkeypatch.setattr(sys, "argv", ["promote.py", *argv])
        with pytest.raises(SystemExit) as exit_info:
            promote_script.main()
        assert exit_info.value.code == 2 and message in capsys.readouterr().err
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.ARCHIVE_ROOT.exists()


# ---- architectural adopt ----

def test_adopt_architecture_needs_no_gold_and_records_the_reason(env):
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {"name": "petbert-four-stage"}})
    reason = _reason_file()
    outcome = promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR,
                                         reason, "joint-modernbert", today=date(2026, 10, 9))
    archive = config.ARCHIVE_ROOT / "2026-10-09_joint-modernbert"
    assert outcome == {"action": "adopted", "generation_id": "gen-B", "archive": archive}
    assert read_manifest(archive)["generation_id"] == "gen-A" and read_manifest(archive)["status"] == "archived"
    assert read_manifest(archive)["architecture"] == {"name": "petbert-four-stage"}   # the old current, whole
    current = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    assert current["generation_id"] == "gen-B" and current["status"] == "current"
    assert current["adoption"]["reason"] == REASON      # repo-relative, posix
    assert pd.Timestamp(current["adoption"]["adopted_at"]).tzinfo is not None
    verify_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    verify_manifest(archive)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


@pytest.mark.parametrize("reason", ["missing", "outside", "directory"])
def test_adopt_architecture_refuses_a_reason_that_is_not_an_existing_history_file(env, reason):
    path = {"missing": config.HISTORY_DIR / "decisions" / "nope.md", "outside": env / "elsewhere.md",
            "directory": config.HISTORY_DIR}[reason]
    if reason == "outside":
        path.write_text("# not history\n", encoding="utf-8")
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with pytest.raises(promote.PromotionError, match="existing file under"):
        promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR, path)
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.ARCHIVE_ROOT.exists()


def test_adopt_architecture_refuses_an_uncalibrated_candidate(env, tiny_generation_dir):
    shutil.rmtree(config.REPORT_MAPPING_CANDIDATE_DIR)
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_generation_dir, "gen-B", "silver-B", pending=True)
    with pytest.raises(GenerationError, match="calibration.status"):
        promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR,
                                   _reason_file())
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"


def test_adopt_architecture_on_a_machine_without_current_just_renames_the_candidate(env):
    shutil.rmtree(config.REPORT_MAPPING_CURRENT_DIR)
    outcome = promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR,
                                         _reason_file())
    assert outcome["archive"] is None
    current = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    assert current["generation_id"] == "gen-B" and current["adoption"]["reason"] == REASON
    assert not config.ARCHIVE_ROOT.exists() and not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_adopt_architecture_restores_current_when_the_swap_fails(env, monkeypatch):
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {"name": "petbert-four-stage"}})
    real_rename, calls = os.rename, []

    def flaky_rename(src, dst):
        calls.append((src, dst))
        if len(calls) == 2:
            raise OSError("disk full")
        real_rename(src, dst)

    monkeypatch.setattr(promote.os, "rename", flaky_rename)
    with pytest.raises(OSError, match="disk full"):
        promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR,
                                   _reason_file())
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"
    assert "adoption" not in read_manifest(config.REPORT_MAPPING_CANDIDATE_DIR)


def test_adopt_script_end_to_end(env, monkeypatch, capsys):
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {"name": "petbert-four-stage"}})
    reason = _reason_file()
    monkeypatch.setattr(sys, "argv", ["promote.py", "--adopt", "--reason", str(reason), "--description", "adopted"])
    assert promote_script.main() == 0
    assert "Adopted gen-B" in capsys.readouterr().out
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["adoption"]["reason"] == REASON
    assert (config.ARCHIVE_ROOT / f"{date.today().isoformat()}_adopted").is_dir()


def test_adopt_script_refuses_a_reason_outside_history(env, monkeypatch, capsys):
    outside = env / "notes.md"
    outside.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["promote.py", "--adopt", "--reason", str(outside)])
    assert promote_script.main() == 1
    assert "REFUSED" in capsys.readouterr().err
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"


# ---- adopt_published (a generation promoted elsewhere, pulled by s3sync) ----

def test_adopt_published_swaps_a_verified_candidate_in_and_archives_the_incumbent_without_the_rule(env):
    # No gold-eval, no predictions: the rule is not consulted.
    outcome = promote.adopt_published("adopted", today=date(2026, 9, 26))
    archive = config.ARCHIVE_ROOT / "2026-09-26_adopted"
    assert outcome == {"action": "adopted", "generation_id": "gen-B", "archive": archive}
    assert read_manifest(archive)["generation_id"] == "gen-A" and read_manifest(archive)["status"] == "archived"
    current = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    assert current["status"] == "current" and "adoption" not in current
    verify_manifest(archive)
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_adopt_published_on_a_fresh_machine_just_renames_the_candidate(env):
    shutil.rmtree(config.REPORT_MAPPING_CURRENT_DIR)
    outcome = promote.adopt_published()
    assert outcome["archive"] is None
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["status"] == "current"
    assert not config.ARCHIVE_ROOT.exists() and not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_adopt_published_refuses_an_uncalibrated_candidate(env, tiny_generation_dir):
    shutil.rmtree(config.REPORT_MAPPING_CANDIDATE_DIR)
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_generation_dir, "gen-B", "silver-B", pending=True)
    with pytest.raises(GenerationError, match="calibration.status"):
        promote.adopt_published()
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"


def test_check_candidate_is_the_spec_check(env):
    assert promote.check_candidate(config.REPORT_MAPPING_CANDIDATE_DIR)["generation_id"] == "gen-B"
    with pytest.raises(promote.PromotionError, match="no candidate"):
        promote.check_candidate(env / "nowhere")
    (config.REPORT_MAPPING_CANDIDATE_DIR / "head.pt").write_bytes(b"tampered")
    with pytest.raises(GenerationError, match="does not match"):
        promote.check_candidate(config.REPORT_MAPPING_CANDIDATE_DIR)


# ---- retrain_cycle.py ----

def _load_cycle_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "retrain_cycle.py"
    spec = importlib.util.spec_from_file_location("ml_next_scripts_retrain_cycle", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def cycle(env, monkeypatch, tiny_generation_dir):
    """retrain_cycle.py with every step stubbed: records each script it runs; train.py leaves a
    candidate (gen-C) behind, as the real one would."""
    script = _load_cycle_script()
    shutil.rmtree(config.REPORT_MAPPING_CANDIDATE_DIR)
    config.PREDICTIONS_DIR.mkdir(parents=True)
    _predictions(config.PREDICTIONS_DIR / "gen-A_predictions.csv", "gen-A")
    ran, real_run = [], subprocess.run

    def fake_run(command, **kwargs):
        if command[0] != sys.executable:  # e.g. the manifest writer's git rev-parse
            return real_run(command, **kwargs)
        ran.append([Path(command[1]).name, *command[2:]])
        if ran[-1][0] == "train.py" and not config.REPORT_MAPPING_CANDIDATE_DIR.exists():
            _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_generation_dir, "gen-C", "silver-B")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["retrain_cycle.py", "--split", SPLIT, *argv])
        return script.main()

    return main, ran, script


def test_cycle_stops_without_gold_eval(cycle, capsys):
    main, ran, _ = cycle
    assert main("--silver", "silver-B") == 0 and ran == []
    assert "no gold-eval" in capsys.readouterr().out


def test_cycle_stops_when_no_trigger_is_met(cycle, capsys):
    main, ran, _ = cycle
    _eval_batch_gold()
    assert main("--silver", "silver-A") == 0 and ran == []
    assert "STOP: no retraining trigger met" in capsys.readouterr().out


def test_cycle_trains_calibrates_predicts_and_recommends_in_order(cycle):
    main, ran, _ = cycle
    _eval_batch_gold()
    assert main("--silver", "silver-B", "--device", "cuda") == 0
    assert [step[0] for step in ran] == ["code_cases.py", "train.py", "calibrate.py", "predict.py", "promote.py"]
    # One training step from the base checkpoint: no stage, backbone or heads flags.
    assert "--stage" not in ran[1] and "--local-only" not in ran[1]
    assert ran[1][ran[1].index("--device") + 1] == "cuda" and ran[1][ran[1].index("--out") + 1] == "candidate"
    assert "--apply" not in ran[-1]      # recommend-only by default
    # Training uses the corrected table (train only); calibration is labelled by silver.
    assert ran[1][ran[1].index("--labels") + 1] == str(config.CORRECTED_ANNOTATIONS_CSV)
    assert ran[2][ran[2].index("--labels") + 1] == "silver-B"
    assert str(config.PREDICTIONS_DIR / "gen-C_predictions.csv") in ran[3]


def test_cycle_refuses_when_a_candidate_exists_and_has_no_backbone_flag(cycle, capsys):
    main, ran, _ = cycle
    _eval_batch_gold()
    assert main("--silver", "silver-A", "--force") == 0
    assert [step[0] for step in ran].count("train.py") == 1
    assert main("--silver", "silver-B") == 1  # candidate/ now exists
    assert "REFUSED" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main("--silver", "silver-A", "--backbone")


@pytest.mark.parametrize("rule_passes", [True, False])
def test_cycle_apply_promotes_only_when_the_gold_rule_passes(cycle, monkeypatch, rule_passes):
    main, ran, script = cycle
    _eval_batch_gold()
    seen = {}

    def recommend(candidate_csv, incumbent_csv, *, split_id, n_boot):
        seen.update(candidate=Path(candidate_csv).name, incumbent=Path(incumbent_csv).name, split=split_id)
        return {"promote": rule_passes}

    monkeypatch.setattr(script, "recommend", recommend)
    assert main("--silver", "silver-B", "--apply") == 0
    assert seen == {"candidate": "gen-C_predictions.csv", "incumbent": "gen-A_predictions.csv", "split": SPLIT}
    assert ("--apply" in ran[-1]) is rule_passes and ran[-1][0] == "promote.py"


def test_cycle_apply_stops_when_the_rule_cannot_be_evaluated(cycle, monkeypatch, capsys):
    main, ran, script = cycle
    _eval_batch_gold()
    monkeypatch.setattr(script, "recommend", lambda *a, **k: (_ for _ in ()).throw(ValueError("no good")))
    assert main("--silver", "silver-B", "--apply") == 1
    assert "REFUSED: no good" in capsys.readouterr().err
    assert ran[-1][0] == "predict.py"


# ---- --publish (opt-in S3 hook) ----

from .s3fake import forbid_s3, use_fake_s3  # noqa: E402


def _run_promote(monkeypatch, env, *extra, wrong=0):
    monkeypatch.setattr(sys, "argv", _argv(env, *extra, wrong=wrong))
    return promote_script.main()


def test_publish_without_apply_is_rejected(env, monkeypatch, capsys):
    forbid_s3(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        _run_promote(monkeypatch, env, "--publish")
    assert exit_info.value.code == 2 and "--publish requires --apply" in capsys.readouterr().err
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists()  # nothing was promoted


def test_promote_publish_lands_the_generation_in_s3(env, monkeypatch, capsys):
    _eval_batch_gold()
    fake = use_fake_s3(monkeypatch, env)
    assert _run_promote(monkeypatch, env, "--apply", "--publish") == 0
    assert "Published gen-B to S3" in capsys.readouterr().out
    prefix = config.S3_PREFIX + "generations/"
    assert json.loads(fake.objects[prefix + "CURRENT.json"]) == {"generation_id": "gen-B"}
    assert prefix + "gen-B/manifest.json" in fake.objects
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"


def test_adopt_publish_lands_the_generation_in_s3(env, monkeypatch, capsys):
    update_manifest(config.REPORT_MAPPING_CURRENT_DIR, {"architecture": {"name": "petbert-four-stage"}})
    fake = use_fake_s3(monkeypatch, env)
    monkeypatch.setattr(sys, "argv", ["promote.py", "--adopt", "--reason", str(_reason_file()), "--publish"])
    assert promote_script.main() == 0
    assert "Published gen-B to S3" in capsys.readouterr().out
    assert json.loads(fake.objects[config.S3_PREFIX + "generations/CURRENT.json"]) == {"generation_id": "gen-B"}


def test_discarded_candidate_is_not_published(env, monkeypatch, capsys):
    _eval_batch_gold()
    forbid_s3(monkeypatch)
    assert _run_promote(monkeypatch, env, "--apply", "--publish", wrong=8) == 0
    out = capsys.readouterr().out
    assert "Deleted losing candidate" in out and "(--publish ignored: nothing promoted)" in out
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-A"


def test_publish_failure_keeps_the_local_promotion_and_exits_1(env, monkeypatch, capsys):
    _eval_batch_gold()
    fake = use_fake_s3(monkeypatch, env)
    fake.objects[config.S3_PREFIX + "generations/CURRENT.json"] = b'{"generation_id": "elsewhere"}'
    assert _run_promote(monkeypatch, env, "--apply", "--publish") == 1
    err = capsys.readouterr().err
    assert "PROMOTED locally, but the remote model moved" in err
    assert "ml/scripts/sync.py pull-model --apply (archives this promotion locally)" in err
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_publish_transient_failure_says_to_rerun_publish_model(env, monkeypatch, capsys):
    from botocore.exceptions import ClientError

    _eval_batch_gold()
    fake = use_fake_s3(monkeypatch, env)

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "PutObject")

    fake.put_object = denied
    assert _run_promote(monkeypatch, env, "--apply", "--publish") == 1
    err = capsys.readouterr().err
    assert "PROMOTED locally, but publish failed" in err and "ml/scripts/sync.py publish-model --apply" in err
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"


def test_publish_with_a_bad_aws_profile_is_a_clean_failure(env, monkeypatch, capsys):
    from botocore.exceptions import ProfileNotFound

    _eval_batch_gold()

    def no_profile():
        raise ProfileNotFound(profile="nope")

    monkeypatch.setattr("s3sync.client.make_client", no_profile)
    assert _run_promote(monkeypatch, env, "--apply", "--publish") == 1
    assert "PROMOTED locally, but publish failed" in capsys.readouterr().err
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"


def test_apply_without_publish_never_touches_s3(env, monkeypatch):
    _eval_batch_gold()
    forbid_s3(monkeypatch)
    assert _run_promote(monkeypatch, env, "--apply") == 0
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"


def test_adopt_architecture_refuses_a_same_architecture_candidate(env):
    # A retrain (same code, new weights) must go through the gold rule, never around it.
    with pytest.raises(promote.PromotionError, match="same architecture"):
        promote.adopt_architecture(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR,
                                   _reason_file(), "retrain")
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists() and not config.ARCHIVE_ROOT.exists()
