"""generations/promote.py + triggers.py + scripts/promote.py on synthetic generations, gold and predictions.

No model runs: predictions CSVs are written directly; generations are the tiny
fixture bundle with rewritten manifests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import config
import io_utils
from generations import promote, triggers
from generations.manifest import read_manifest, write_manifest
from manual_audit import eval_batch, gold

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


def _generation(directory: Path, tiny_bert_dir: Path, generation_id: str, silver_id: str,
                status: str = "calibrated") -> None:
    fx.build_report_mapping_bundle(directory, tiny_bert_dir)
    fields = {k: v for k, v in read_manifest(directory).items() if k not in ("files", "created_at", "git_sha")}
    write_manifest(directory, {**fields, "generation_id": generation_id, "parents": {"silver_id": silver_id},
                               "calibration": {"status": status}})


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


@pytest.fixture
def env(tmp_path, monkeypatch, tiny_bert_dir):
    fx.point_promotion_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation(SPLIT, TRAIN, TEST)
    _generation(config.REPORT_MAPPING_CURRENT_DIR, tiny_bert_dir, "gen-A", "silver-A")
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR, tiny_bert_dir, "gen-B", "silver-B")
    return tmp_path


def _recommend(env: Path, wrong: int = 0) -> dict:
    return promote.recommend(_predictions(env / "b.csv", "gen-B", wrong), _predictions(env / "a.csv", "gen-A"),
                             split_id=SPLIT, n_boot=200)


def test_no_gold_eval_is_refused(env):
    with pytest.raises(promote.PromotionError, match="no gold-eval"):
        _recommend(env)


def test_uncalibrated_candidate_is_refused(env, tiny_bert_dir):
    _eval_batch_gold()
    _generation(config.REPORT_MAPPING_CANDIDATE_DIR.with_name("pending"), tiny_bert_dir, "gen-B", "silver-B",
                status="pending")
    os.rename(config.REPORT_MAPPING_CANDIDATE_DIR, env / "old-candidate")
    os.rename(config.REPORT_MAPPING_CANDIDATE_DIR.with_name("pending"), config.REPORT_MAPPING_CANDIDATE_DIR)
    with pytest.raises(promote.PromotionError, match="calibration.status"):
        _recommend(env)


def test_predictions_must_cover_gold_eval(env):
    _eval_batch_gold()
    partial = _predictions(env / "b.csv", "gen-B")
    io_utils.write_csv(io_utils.read_csv(partial, encoding="utf-8", dtype=str).iloc[1:], partial)
    with pytest.raises(promote.PromotionError, match="miss 1 gold-eval case"):
        promote.recommend(partial, _predictions(env / "a.csv", "gen-A"), split_id=SPLIT, n_boot=50)


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
    assert read_manifest(config.REPORT_MAPPING_CURRENT_DIR)["generation_id"] == "gen-B"
    assert not config.REPORT_MAPPING_CANDIDATE_DIR.exists()


def test_no_trigger_means_no_promotion(env, tiny_bert_dir):
    _eval_batch_gold()
    manifest_path = config.REPORT_MAPPING_CANDIDATE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parents"]["silver_id"] = "silver-A"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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
    manifest_path = config.REPORT_MAPPING_CANDIDATE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_id"] = "gen-A"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(promote.PromotionError, match="share generation_id"):
        _recommend(env)


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


def test_script_refuses_cleanly_without_gold_eval(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["promote.py", "--candidate-predictions", str(_predictions(env / "b.csv", "gen-B")),
                                      "--incumbent-predictions", str(_predictions(env / "a.csv", "gen-A")),
                                      "--split", SPLIT])
    assert promote_script.main() == 1
    assert "no gold-eval" in capsys.readouterr().err


def test_script_prints_recommendation(env, monkeypatch, capsys):
    _eval_batch_gold()
    monkeypatch.setattr(sys, "argv", ["promote.py", "--candidate-predictions", str(_predictions(env / "b.csv", "gen-B")),
                                      "--incumbent-predictions", str(_predictions(env / "a.csv", "gen-A")),
                                      "--split", SPLIT, "--n-boot", "50"])
    assert promote_script.main() == 0
    out = capsys.readouterr().out
    assert "Recommendation: PROMOTE" in out
    assert config.REPORT_MAPPING_CANDIDATE_DIR.exists()  # recommend-only: nothing moved
