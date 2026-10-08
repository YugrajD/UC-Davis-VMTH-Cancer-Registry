"""Phase 2: the ONE-SHOT head-to-head of a frozen rr generation against the WP14 baseline on the test partition.

Spends the test set: needs ``--i-understand-this-spends-the-test-set`` and is write-once (the output directory
``<RR_PHASE1_DIR>/phase2/<generation_id>/`` must not exist; there is no overwrite). Reads two existing predictions
CSVs, so no model runs. Both systems go through ``silver_eval.evaluate`` on the SAME test universe (the split's test
cases) with the same silver labels and WP14's uncommon-groups list.

The paired bootstrap is ``intervals.paired_bootstrap`` exactly as ``generations.promote.compare`` calls it (stratified
case-cluster resampling, 1000 resamples, seed 0, 95% percentile interval, both systems on the same resampled cases).
promote.compare takes its strata from the gold-eval sampling design (ledger stratum / random-slice period); silver
test has no sampling design (the test partition is the split's random draw of all cases), so it is ONE stratum holding
every test case, true negatives and cases without rows included, as promote's population does. The metrics are
unweighted row shares (promote's carry gold-eval weights; silver rows have none).

Pre-registered bars (RR_HANDOFF.md): (1) Good: lower bound of the 95% interval of (new - WP14) >= -2.0 pp, the same
margin as promote.MARGIN; (2) G+S: new point estimate >= WP14's; (3) FP: new FP share <= WP14's + 1.0 pp. All three
must pass. Kill rule context: kill if still more than 5 pp behind after one variant; the gaps are reported.
Aggregates only: nothing in the output carries report text or case ids.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pandas as pd

import config
from bronze_rr import generation, ruler, scoring
from bronze_rr.data import SILVER_ID, load_uncommon_groups
from evaluation import intervals, silver_eval, verdicts
from generations.manifest import MANIFEST_NAME, sha256_file

N_BOOT = 1000
SEED = 0
GOOD_MARGIN = -0.02      # same as generations.promote.MARGIN
FP_MARGIN = 0.01
KILL_GAP = 0.05
EPS = 1e-9               # float guard so a bar met exactly at its edge is a pass
BASELINE_CSV = config.PREDICTIONS_DIR / "gen-20260927T025939Z_predictions.csv"
OUT_ROOT = config.RR_PHASE1_DIR / "phase2"

# difference rows: name -> verdicts counted
DIFFERENCES = {**{v: (v,) for v in verdicts.VERDICTS}, "good_plus_slight": verdicts.GOOD_PLUS_SLIGHT}


def case_strata(case_ids) -> pd.Series:
    """One stratum holding every test case."""
    return pd.Series("test", index=sorted(case_ids))


def paired_differences(new_table: pd.DataFrame, base_table: pd.DataFrame, strata: pd.Series,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """metric -> {diff, lo, hi} for share(new) - share(WP14), fractions of the verdict rows."""
    out = {}
    for name, hits in DIFFERENCES.items():
        diff, low, high = intervals.paired_bootstrap(new_table, base_table, partial(verdicts.share, verdicts=hits),
                                                     strata, n_boot, seed)
        out[name] = {"diff": diff, "lo": low, "hi": high}
    return out


def evaluate_bars(new: dict, base: dict, diffs: dict) -> dict:
    """The three pre-registered bars and the kill-rule gaps, from two ``verdicts.summarize`` dicts and the differences."""
    good_lo, fp_new, fp_cap = diffs["good"]["lo"], new["false_positive_share"], base["false_positive_share"] + FP_MARGIN
    bars = {
        "good_lower_bound": {"value_pp": 100 * good_lo, "threshold_pp": 100 * GOOD_MARGIN,
                             "pass": bool(good_lo >= GOOD_MARGIN - EPS)},
        "gs_not_below_baseline": {"new_pp": 100 * new["good_plus_slight_share"],
                                  "baseline_pp": 100 * base["good_plus_slight_share"],
                                  "pass": bool(new["good_plus_slight_share"] >= base["good_plus_slight_share"] - EPS)},
        "fp_within_1pp": {"new_pp": 100 * fp_new, "limit_pp": 100 * fp_cap, "pass": bool(fp_new <= fp_cap + EPS)},
    }
    gaps = {"good_gap_pp": 100 * (base["good_share"] - new["good_share"]),
            "gs_gap_pp": 100 * (base["good_plus_slight_share"] - new["good_plus_slight_share"])}
    gaps["good_behind_more_than_5pp"] = bool(base["good_share"] - new["good_share"] > KILL_GAP + EPS)
    gaps["gs_behind_more_than_5pp"] = bool(base["good_plus_slight_share"] - new["good_plus_slight_share"] > KILL_GAP + EPS)
    return {"bars": bars, "all_pass": all(b["pass"] for b in bars.values()), "kill_rule_gaps": gaps}


def discordant_counts(new_table: pd.DataFrame, base_table: pd.DataFrame, new_pred: pd.DataFrame,
                      base_pred: pd.DataFrame, labels: pd.DataFrame, case_ids) -> dict:
    """Case-level counts where the two systems disagree. Cancer cases are the test cases with a silver term."""
    def good(table):
        return set(table.loc[table["verdict"] == "good", "case_id"])

    cancer = set(labels.loc[labels["case_id"].isin(case_ids) & (labels["matched_term"].astype(str).str.strip() != ""),
                            "case_id"])

    def stance(pred):
        rows = pred[pred["case_id"].isin(cancer)]
        abstained = rows["predicted_term"].isin(verdicts.NON_CANCER_PRED_TERMS)
        return set(rows.loc[abstained, "case_id"]), set(rows.loc[~abstained, "case_id"])

    (new_nc, new_ca), (base_nc, base_ca) = stance(new_pred), stance(base_pred)
    return {"new_good_baseline_none": len(good(new_table) - good(base_table)),
            "baseline_good_new_none": len(good(base_table) - good(new_table)),
            "cancer_cases": len(cancer),
            "cancer_new_non_cancer_baseline_cancer_term": len(new_nc & base_ca),
            "cancer_baseline_non_cancer_new_cancer_term": len(base_nc & new_ca)}


def score_systems(new_pred: pd.DataFrame, base_pred: pd.DataFrame, labels: pd.DataFrame, case_ids, uncommon,
                  n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Everything the report needs from the two predictions tables; pure (no IO)."""
    new, base = (silver_eval.evaluate(p, labels, case_ids, uncommon) for p in (new_pred, base_pred))
    diffs = paired_differences(new["table"], base["table"], case_strata(case_ids), n_boot, seed)
    in_test = lambda p: p[p["case_id"].isin(case_ids)]  # noqa: E731
    return {"new": new["summary"], "baseline": base["summary"],
            "wp14_sanity": "MATCH" if ruler.matches_wp14_test(base["summary"]) else "MISMATCH",
            "differences": diffs, **evaluate_bars(new["summary"], base["summary"], diffs),
            "discordant": discordant_counts(new["table"], base["table"], in_test(new_pred), in_test(base_pred),
                                            labels, case_ids),
            "test_cases": len(case_ids),
            "cases_with_prediction_rows": {"new": int(in_test(new_pred)["case_id"].nunique()),
                                           "baseline": int(in_test(base_pred)["case_id"].nunique())}}


def render_summary(result: dict) -> str:
    new, base, prov = result["new"], result["baseline"], result["provenance"]
    lines = [f"# Phase 2: {prov['generation_id']} vs WP14 on {prov['split_id']} test", ""]
    if result["wp14_sanity"] != "MATCH":
        lines += ["**WARNING: THE WP14 ROW DOES NOT MATCH THE RECORDED BASELINE "
                  f"({ruler.WP14_TEST}); the ruler or the baseline file is wrong. Do not trust this comparison.**", ""]
    lines += [f"**Overall: {'PASS' if result['all_pass'] else 'FAIL'}** (all three bars must pass)", "",
              "## Verdicts (share of scored rows, %; counts good/slight/co/fp/fn)", "", ruler.SUMMARY_HEADER,
              ruler.summary_row(prov["generation_id"], new), ruler.summary_row("WP14 baseline", base), "",
              f"WP14 sanity (4,299 rows, Good 46.27 / Slight 15.86 / CO 12.84 / FP 2.35 / FN 22.68): "
              f"**{result['wp14_sanity']}**", "",
              f"Test cases {result['test_cases']}; cases with prediction rows: new "
              f"{result['cases_with_prediction_rows']['new']}, WP14 {result['cases_with_prediction_rows']['baseline']}.",
              "", f"## Paired bootstrap, new - WP14 (pp, 95% interval; {prov['n_boot']} resamples, seed {prov['bootstrap_seed']})",
              "", "| metric | difference | 95% low | 95% high |", "|---|---|---|---|"]
    for name, d in result["differences"].items():
        lines.append(f"| {name} | {100 * d['diff']:+.2f} | {100 * d['lo']:+.2f} | {100 * d['hi']:+.2f} |")
    b = result["bars"]
    lines += ["", "## Pre-registered bars", "", "| bar | value | limit | result |", "|---|---|---|---|",
              f"| Good: lower bound of new - WP14 >= -2.0 pp | {b['good_lower_bound']['value_pp']:+.2f} pp | "
              f"{b['good_lower_bound']['threshold_pp']:+.1f} pp | {'PASS' if b['good_lower_bound']['pass'] else 'FAIL'} |",
              f"| G+S: new >= WP14 | {b['gs_not_below_baseline']['new_pp']:.2f}% | "
              f"{b['gs_not_below_baseline']['baseline_pp']:.2f}% | {'PASS' if b['gs_not_below_baseline']['pass'] else 'FAIL'} |",
              f"| FP: new <= WP14 + 1.0 pp | {b['fp_within_1pp']['new_pp']:.2f}% | "
              f"{b['fp_within_1pp']['limit_pp']:.2f}% | {'PASS' if b['fp_within_1pp']['pass'] else 'FAIL'} |"]
    g = result["kill_rule_gaps"]
    lines += ["", "Kill-rule context (kill if still more than 5 pp behind after one variant; positive = new is behind WP14): "
              f"Good gap {g['good_gap_pp']:+.2f} pp, G+S gap {g['gs_gap_pp']:+.2f} pp.", "",
              "## Discordant cases (case-level counts)", ""]
    d = result["discordant"]
    lines += [f"- new has >= 1 Good row, WP14 none: {d['new_good_baseline_none']}",
              f"- WP14 has >= 1 Good row, new none: {d['baseline_good_new_none']}",
              f"- of {d['cancer_cases']} cancer cases, new Non-Cancer and WP14 a cancer term: "
              f"{d['cancer_new_non_cancer_baseline_cancer_term']}",
              f"- of {d['cancer_cases']} cancer cases, WP14 Non-Cancer and new a cancer term: "
              f"{d['cancer_baseline_non_cancer_new_cancer_term']}", "", "## Provenance", ""]
    lines += [f"- {key}: {value}" for key, value in prov.items()]
    return "\n".join(lines) + "\n"


def run(generation_dir: Path, baseline_csv: Path = BASELINE_CSV, *, allow_test: bool,
        split_id: str = config.DEFAULT_SPLIT_ID, labels_id: str = SILVER_ID, out_root: Path = OUT_ROOT,
        predictions_dir: Path = config.PREDICTIONS_DIR) -> dict:
    """Score once, write ``scores.json`` and ``summary.md`` to ``out_root/<generation_id>/``, return the result."""
    scoring.check_partition_allowed("test", allow_test)
    manifest, _ = generation.load_generation_info(generation_dir)  # also verifies every file against the manifest
    generation_id = manifest["generation_id"]
    calibration = manifest.get("calibration") or {}
    if calibration.get("status") != "calibrated" or not calibration.get("values"):
        raise SystemExit(f"{generation_dir}: calibration status is {calibration.get('status')!r} with "
                         f"thresholds {calibration.get('values')!r}; Phase 2 needs a calibrated generation")
    directory = out_root / generation_id
    if directory.exists():
        raise SystemExit(f"{directory} already exists: Phase 2 is write-once and the test set is spent")

    predictions_csv = predictions_dir / f"{generation_id}_predictions.csv"
    new_pred = silver_eval.read_predictions(predictions_csv)
    found = sorted(set(new_pred["generation_id"]))
    if found != [generation_id]:
        raise SystemExit(f"{predictions_csv.name} carries generation_id {found}, expected [{generation_id!r}]")
    base_pred = silver_eval.read_predictions(baseline_csv)

    case_ids = silver_eval.partition_case_ids(split_id, "test")
    uncommon = load_uncommon_groups()
    labels = silver_eval.load_labels(labels_id)
    result = score_systems(new_pred, base_pred, labels, case_ids, uncommon)

    own_list = generation_dir / "uncommon_groups.txt"
    result["provenance"] = {
        "generation_id": generation_id, "manifest_sha256": sha256_file(generation_dir / MANIFEST_NAME),
        "retriever_seed": (manifest.get("retriever") or {}).get("seed"), "gate": (manifest.get("gate") or {}).get("kind"),
        "thresholds": calibration["values"],
        "predictions_csv": predictions_csv.name, "predictions_sha256": sha256_file(predictions_csv),
        "baseline_csv": baseline_csv.name, "baseline_sha256": sha256_file(baseline_csv),
        "baseline_generation_ids": sorted(set(base_pred["generation_id"])) if "generation_id" in base_pred.columns else [],
        "split_id": split_id, "labels": labels_id, "partition": "test",
        "uncommon_groups_n": len(uncommon),
        "uncommon_groups_sha256": hashlib.sha256("\n".join(sorted(uncommon)).encode("utf-8")).hexdigest(),
        "generation_uncommon_list_matches": (frozenset(l.strip() for l in own_list.read_text(encoding="utf-8").splitlines()
                                                       if l.strip()) == uncommon) if own_list.is_file() else None,
        "bootstrap": "intervals.paired_bootstrap as promote.compare; one stratum of all test cases; unweighted row shares",
        "n_boot": N_BOOT, "bootstrap_seed": SEED,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        directory.mkdir(parents=True)  # claim the directory; exist_ok is deliberately not allowed
    except FileExistsError:
        raise SystemExit(f"{directory} already exists: Phase 2 is write-once and the test set is spent") from None
    (directory / "scores.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (directory / "summary.md").write_text(render_summary(result), encoding="utf-8")
    result["output_dir"] = str(directory)
    return result
