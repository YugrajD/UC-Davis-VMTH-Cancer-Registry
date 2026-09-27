"""manual_audit/eval_batch.py: stratification, code-targeted allocation, ledger, sheet ingest."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import config
import io_utils
from diagnosis_mapping.silver import load_silver
from manual_audit import eval_batch, sheets, tier3_audit

from . import fixtures as fx


@pytest.fixture
def eval_batch_env(tmp_path, monkeypatch):
    """A split whose test partition is exactly EVAL_BATCH_CASE_IDS, + a silver generation."""
    fx.point_manual_audit_config_at(monkeypatch, tmp_path)
    fx.make_two_way_split_generation(
        "legacy-80-20", train_ids=[f"TRAIN-{i:04d}" for i in range(1, 6)], test_ids=fx.EVAL_BATCH_CASE_IDS,
    )
    silver_id = fx.make_eval_batch_silver_generation()
    return silver_id


@pytest.fixture
def no_tier3_exclusions(tmp_path):
    """An empty Tier-3 batch-1 ledger, passed explicitly so tests don't need the
    default-path exclusion file unless they're specifically testing it."""
    path = tmp_path / "tier3_batch1_cases.txt"
    path.write_text("", encoding="utf-8")
    return path


def test_multi_group_case_stratum_rule(eval_batch_env):
    silver_df = load_silver(eval_batch_env)
    strata = eval_batch._silver_strata(silver_df, set(fx.EVAL_BATCH_CASE_IDS))
    # CASE-0029: empty group at diagnosis 1, "Mast Cell Tumors" at 2 -> its only non-empty group.
    assert strata["CASE-0029"] == "Mast Cell Tumors"
    # CASE-0030: "Rare Sarcomas" at diagnosis 1, "Rare Carcinomas" at 2 -> the
    # lowest diagnosis_number with a non-empty group wins.
    assert strata["CASE-0030"] == "Rare Sarcomas"
    assert strata["CASE-0001"] == "Mast Cell Tumors"
    assert strata["CASE-0021"] == "no_cancer"


def test_plan_targets_code_targeted_big_and_rare_and_no_cancer():
    populations = {"Big Group": 200, "Rare Group": 3}
    own_code_counts = {"Big Group": 300, "Rare Group": 3}  # Big Group: 1.5 codes/case; Rare: 1.0
    targets = eval_batch._plan_targets(
        populations, own_code_counts, no_cancer_population=5000,
        target_codes_per_group=30, big_group_share=0.01, rare_stratum_n=2, no_cancer_target=100,
    )
    # Big Group: share = 300/303 >> 1% -> ceil(30 / 1.5) = 20, well under its population of 200.
    assert targets["Big Group"] == 20
    # Rare Group: share = 3/303 < 1% -> falls back to the fixed rare floor,
    # min(N_h, rare_n) = min(3, 2) = 2.
    assert targets["Rare Group"] == 2
    assert targets["no_cancer"] == 100  # fixed target, population (5000) isn't the constraint


def test_plan_targets_rare_share_falls_back_to_fixed_n():
    populations = {"Common": 500, "Obscure": 50}
    own_code_counts = {"Common": 5000, "Obscure": 5}  # Obscure: 5/5005 << 1% share
    targets = eval_batch._plan_targets(
        populations, own_code_counts, no_cancer_population=10,
        target_codes_per_group=30, big_group_share=0.01, rare_stratum_n=2, no_cancer_target=100,
    )
    assert targets["Obscure"] == 2
    assert targets["no_cancer"] == 10  # capped by the (tiny) population here


def test_sheet_has_no_prediction_or_silver_column(eval_batch_env, no_tier3_exclusions):
    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    rows = sheets.read_csv(result["sheet_path"])
    header = set(rows[0].keys())
    for forbidden in ("predicted", "prediction", "silver", "bronze", "confidence", "matched_term", "matched_group",
                      "matched_code", "decision_stage"):
        assert not any(forbidden in col.lower() for col in header)
    expected = {"case_id", "record_pointer"} | {f"term_{i}" for i in range(1, 6)} | {"no_cancer", "reviewer", "notes"}
    assert header == expected
    for row in rows:
        assert row["record_pointer"] == row["case_id"]
        for col in expected - {"case_id", "record_pointer"}:
            assert row[col] == ""


def test_ledger_records_silver_id_seed_and_review_mode(eval_batch_env, no_tier3_exclusions):
    eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20", seed=7,
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(ledger.columns) == eval_batch.EVAL_BATCH_LEDGER_FIELDS
    assert (ledger["silver_id"] == eval_batch_env).all()
    assert (ledger["seed"] == "7").all()
    assert (ledger["review_mode"] == "app_non_blind").all()


def test_weights_sum_to_stratum_populations_within_a_batch(eval_batch_env, no_tier3_exclusions):
    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    for stratum, group in ledger.groupby("stratum"):
        n_h = group["n_h"].astype(int).unique()
        N_h = group["N_h"].astype(int).unique()
        weight = group["sample_weight"].astype(float).unique()
        assert len(n_h) == len(N_h) == len(weight) == 1
        assert len(group) == n_h[0]
        assert n_h[0] * weight[0] == pytest.approx(N_h[0], rel=1e-4)
        assert N_h[0] == result["stratum_populations"][stratum]


def test_fraction_draws_a_share_of_the_remaining_target(eval_batch_env, no_tier3_exclusions):
    first = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    for stratum, target in first["targets"].items():
        assert first["stratum_counts"][stratum] == math.ceil(0.5 * target)

    second = eval_batch.generate_batch(
        "eval-batch-2", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    # fraction=1.0 on whatever remains mops up the rest of every target exactly.
    for stratum, target in second["targets"].items():
        drawn_total = first["stratum_counts"][stratum] + second["stratum_counts"][stratum]
        assert drawn_total == target


def test_refuses_to_redraw_same_batch_id(eval_batch_env, no_tier3_exclusions):
    eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
                              tier3_batch1_ledger=no_tier3_exclusions)
    with pytest.raises(eval_batch.EvalBatchError, match="eval-batch-1"):
        eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
                                  tier3_batch1_ledger=no_tier3_exclusions)


def test_refuses_invalid_fraction(eval_batch_env, no_tier3_exclusions):
    with pytest.raises(eval_batch.EvalBatchError, match="fraction"):
        eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0, split_id="legacy-80-20",
                                  tier3_batch1_ledger=no_tier3_exclusions)
    with pytest.raises(eval_batch.EvalBatchError, match="fraction"):
        eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=1.5, split_id="legacy-80-20",
                                  tier3_batch1_ledger=no_tier3_exclusions)


def test_tier3_missing_file_refuses(eval_batch_env, tmp_path):
    with pytest.raises(eval_batch.EvalBatchError, match="Tier-3 batch-1"):
        eval_batch.generate_batch(
            "eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
            tier3_batch1_ledger=tmp_path / "does_not_exist.txt",
        )


def test_tier3_cases_excluded_and_counted_within_partition(eval_batch_env, tmp_path):
    tier3_cases = fx.EVAL_BATCH_CASE_IDS[:3]
    tier3_path = tmp_path / "tier3_batch1_cases.txt"
    tier3_path.write_text("\n".join(tier3_cases) + "\n", encoding="utf-8")

    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=tier3_path,
    )
    ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    assert not (set(tier3_cases) & set(ledger["case_id"]))
    assert result["excluded_count"] == 3
    assert "tier3_audit_batch1_cases.txt" in result["excluded_ledgers"]


def test_tier3_default_path_used_when_present(eval_batch_env):
    new_path = config.TIER3_AUDIT_BATCH1_EXCLUSION_TXT
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(fx.EVAL_BATCH_CASE_IDS[0] + "\n", encoding="utf-8")
    # Its own path, distinct from tier3_audit's own batch-1 case ledger — the
    # two must never collide (see item 5 of the WP7 fixes).
    assert new_path != tier3_audit.batch_ledger_path(1)

    result = eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20")
    assert fx.EVAL_BATCH_CASE_IDS[0] not in set(
        io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)["case_id"]
    )
    assert "tier3_audit_batch1_cases.txt" in result["excluded_ledgers"]


def test_tier3_default_path_missing_refuses(eval_batch_env):
    assert not config.TIER3_AUDIT_BATCH1_EXCLUSION_TXT.exists()
    with pytest.raises(eval_batch.EvalBatchError, match="Tier-3 batch-1"):
        eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20")


def test_excludes_prior_ledger_cases(eval_batch_env, no_tier3_exclusions):
    first = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
        tier3_batch1_ledger=no_tier3_exclusions,
    )
    first_ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    claimed = set(first_ledger["case_id"])

    second = eval_batch.generate_batch(
        "eval-batch-2", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        tier3_batch1_ledger=no_tier3_exclusions,
    )
    second_ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    second_rows = second_ledger[second_ledger["batch_id"] == "eval-batch-2"]
    assert not (claimed & set(second_rows["case_id"]))
    assert "eval_batch_ledger.csv" in second["excluded_ledgers"]


def test_ingest_sheet_round_trip(eval_batch_env, no_tier3_exclusions, labels_csv):
    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    rows = sheets.read_csv(result["sheet_path"])
    ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    stratum_of = dict(zip(ledger["case_id"], ledger["stratum"]))

    filled = []
    for row in rows:
        row = dict(row)
        if stratum_of[row["case_id"]] == "no_cancer":
            row["no_cancer"] = "x"
        else:
            row["term_1"] = "Mast cell tumor, malignant"
        filled.append(row)
    sheets.write_csv(result["sheet_path"], list(rows[0].keys()), filled)

    ingest_result = eval_batch.ingest_sheet(result["sheet_path"], "eval-batch-1", "Dr. Test",
                                            labels_csv=labels_csv)
    assert ingest_result["total_cases"] == len(rows)
    from manual_audit import gold
    store = gold.load_gold()
    assert (store["origin"] == "eval_batch").all()
    assert (store["batch_or_export_id"] == "eval-batch-1").all()


def test_ingest_sheet_refuses_case_not_in_batch(eval_batch_env, no_tier3_exclusions, tmp_path):
    eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    bad_sheet = tmp_path / "bad_sheet.csv"
    sheets.write_csv(bad_sheet, ["case_id", "record_pointer", "term_1", "term_2", "term_3", "term_4", "term_5",
                                "no_cancer", "reviewer", "notes"],
                     [{"case_id": "NOT-IN-BATCH", "record_pointer": "NOT-IN-BATCH", "term_1": "Mast cell tumor, malignant",
                       "term_2": "", "term_3": "", "term_4": "", "term_5": "", "no_cancer": "", "reviewer": "", "notes": ""}])
    with pytest.raises(eval_batch.EvalBatchError, match="not one of batch"):
        eval_batch.ingest_sheet(bad_sheet, "eval-batch-1", "Dr. Test")


def test_ingest_sheet_refuses_blank_and_both_filled_rows(eval_batch_env, no_tier3_exclusions):
    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    rows = sheets.read_csv(result["sheet_path"])
    header = list(rows[0].keys())

    blank_rows = [dict(r) for r in rows]  # every row left blank
    sheets.write_csv(result["sheet_path"], header, blank_rows)
    with pytest.raises(eval_batch.EvalBatchError, match="neither no_cancer nor any term"):
        eval_batch.ingest_sheet(result["sheet_path"], "eval-batch-1", "Dr. Test")

    both_rows = [dict(r) for r in rows]
    both_rows[0]["term_1"] = "Mast cell tumor, malignant"
    both_rows[0]["no_cancer"] = "x"
    sheets.write_csv(result["sheet_path"], header, both_rows)
    with pytest.raises(eval_batch.EvalBatchError, match="both no_cancer and term"):
        eval_batch.ingest_sheet(result["sheet_path"], "eval-batch-1", "Dr. Test")


def test_ingest_sheet_refuses_repeated_case_id(eval_batch_env, no_tier3_exclusions, labels_csv):
    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
    )
    rows = sheets.read_csv(result["sheet_path"])
    header = list(rows[0].keys())
    duplicated = list(rows) + [dict(rows[0])]  # the first case_id now appears twice
    sheets.write_csv(result["sheet_path"], header, duplicated)
    with pytest.raises(eval_batch.EvalBatchError, match="appears more than once"):
        eval_batch.ingest_sheet(result["sheet_path"], "eval-batch-1", "Dr. Test", labels_csv=labels_csv)


def test_series_consistency_refuses_changed_silver_split_or_target_params(eval_batch_env, no_tier3_exclusions):
    eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
                              rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions)

    other_silver_id = fx.make_eval_batch_silver_generation(silver_id="a-different-silver")
    with pytest.raises(eval_batch.EvalBatchError, match="silver_id"):
        eval_batch.generate_batch("eval-batch-2", silver_id=other_silver_id, fraction=0.5, split_id="legacy-80-20",
                                  rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions)

    with pytest.raises(eval_batch.EvalBatchError, match="no_cancer_target"):
        eval_batch.generate_batch("eval-batch-2", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
                                  rare_stratum_n=2, no_cancer_target=999, tier3_batch1_ledger=no_tier3_exclusions)

    with pytest.raises(eval_batch.EvalBatchError, match="rare_stratum_n"):
        eval_batch.generate_batch("eval-batch-2", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
                                  rare_stratum_n=3, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions)

    # Same parameters as batch 1: allowed.
    eval_batch.generate_batch("eval-batch-2", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
                              rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions)


def test_generate_batch_excludes_cases_with_existing_gold(eval_batch_env, no_tier3_exclusions, tmp_path):
    gold_csv = tmp_path / "gold_store.csv"
    already_gold_case = fx.EVAL_BATCH_CASE_IDS[0]  # a Mast Cell Tumors case
    io_utils.write_csv(pd.DataFrame([{"case_id": already_gold_case, "origin": "review_queue"}]), gold_csv)

    result = eval_batch.generate_batch(
        "eval-batch-1", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
        rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions, gold_csv=gold_csv,
    )
    ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    assert already_gold_case not in set(ledger["case_id"])
    assert "gold_store.csv" in result["excluded_ledgers"]
    assert result["excluded_count"] >= 1  # already_gold_case is in the split's test partition


def test_gold_ingest_between_batches_does_not_change_series_targets(tmp_path, monkeypatch, no_tier3_exclusions, labels_csv):
    """WP7 fix 8: N_h/targets must come from split.test - tier3 only, never
    from split.test - tier3 - gold. Ingesting batch 1's own gold before
    drawing batch 2 must not shrink any stratum's population or target."""

    def _run_series(root, *, ingest_after_batch1: bool) -> dict:
        fx.point_manual_audit_config_at(monkeypatch, root)
        fx.make_two_way_split_generation(
            "legacy-80-20", train_ids=[f"TRAIN-{i:04d}" for i in range(1, 6)], test_ids=fx.EVAL_BATCH_CASE_IDS,
        )
        silver_id = fx.make_eval_batch_silver_generation()

        first = eval_batch.generate_batch(
            "eval-batch-1", silver_id=silver_id, fraction=0.5, split_id="legacy-80-20",
            rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
        )
        if ingest_after_batch1:
            ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
            stratum_of = dict(zip(ledger["case_id"], ledger["stratum"]))
            sheet_rows = sheets.read_csv(first["sheet_path"])
            filled = []
            for row in sheet_rows:
                row = dict(row)
                if stratum_of[row["case_id"]] == "no_cancer":
                    row["no_cancer"] = "x"
                else:
                    row["term_1"] = "Mast cell tumor, malignant"
                filled.append(row)
            sheets.write_csv(first["sheet_path"], list(sheet_rows[0].keys()), filled)
            eval_batch.ingest_sheet(first["sheet_path"], "eval-batch-1", "Dr. Test", labels_csv=labels_csv)

        return eval_batch.generate_batch(
            "eval-batch-2", silver_id=silver_id, fraction=1.0, split_id="legacy-80-20",
            rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions,
        )

    without_ingest = _run_series(tmp_path / "no_ingest", ingest_after_batch1=False)
    with_ingest = _run_series(tmp_path / "with_ingest", ingest_after_batch1=True)

    assert with_ingest["targets"] == without_ingest["targets"]
    assert with_ingest["stratum_populations"] == without_ingest["stratum_populations"]


def test_pooled_weights_over_a_multi_batch_series(eval_batch_env, no_tier3_exclusions):
    eval_batch.generate_batch("eval-batch-1", silver_id=eval_batch_env, fraction=0.5, split_id="legacy-80-20",
                              rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions)
    eval_batch.generate_batch("eval-batch-2", silver_id=eval_batch_env, fraction=1.0, split_id="legacy-80-20",
                              rare_stratum_n=2, no_cancer_target=5, tier3_batch1_ledger=no_tier3_exclusions)
    ledger = io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)

    weights = eval_batch.pooled_weights(ledger)
    for stratum, group in ledger.groupby("stratum"):
        N_h = int(group["N_h"].iloc[0])
        total_n_h = len(group)  # every row is one case; Σ n_h across the series
        assert weights[stratum] == pytest.approx(N_h / total_n_h)

    # Also accepts a path directly.
    weights_from_path = eval_batch.pooled_weights(config.EVAL_BATCH_LEDGER_CSV)
    pd.testing.assert_series_equal(weights.sort_index(), weights_from_path.sort_index())


def test_draw_is_reproducible_across_pythonhashseeds(tmp_path):
    """The regression test for the PYTHONHASHSEED-dependent shuffle-order bug:
    two fresh processes with different hash seeds must draw byte-identical
    ledgers from the same seed and inputs."""
    script = Path(__file__).with_name("_pythonhashseed_repro_check.py")
    outputs = {}
    for hash_seed in ("1", "2"):
        run_dir = tmp_path / f"seed_{hash_seed}"
        run_dir.mkdir()
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, str(script), str(run_dir)],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout, "expected the ledger CSV on stdout"
        outputs[hash_seed] = result.stdout
    assert outputs["1"] == outputs["2"]
