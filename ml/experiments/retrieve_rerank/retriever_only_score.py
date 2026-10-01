"""Diagnostic: skip the reranker. Predict the retriever's own top-1 term behind a cancer / no-cancer gate.

  ml/.venv/Scripts/python.exe ml/experiments/retrieve_rerank/retriever_only_score.py

Never calls a model: reads the retriever's calibration_candidates.csv (rank, term, score), the OpenJev pilot raw
file (Q1 probability only) and the WP14 baseline predictions, and writes scores.json and summary.md to
<RR_PHASE0_DIR>/retriever_only/. Only aggregates are written.

The candidates CSV "score" is the retriever's raw LOGIT (eval_recall.py saves the head output, no sigmoid). The BCE
retriever's per-term probability is sigmoid(score); thresholds are fitted on that probability (0.01..0.99 grid).

Prediction per case: the rank-1 term (group / code from the taxonomy), plus the rank-2 term if its probability is
>= a second-term threshold; a gate-rejected case gets one Non-Cancer row. Method string "retriever_top1".

Gates (each a variant):
  openjev_q1          pass iff OpenJev's Q1 neoplasm probability >= t (pilot cases only)
  wp14_case_presence  pass iff the WP14 baseline has a row for the case whose group is not Non-Cancer (no fitting)
  retriever_score     pass iff the rank-1 probability >= t (diagnostic: the retriever saw cancer cases only)

Scoring sets (pilot_score's md5 fit / eval halves; every threshold is fitted on the fit half, every number is on the
eval half). Thresholds as in pilot_score: a gate threshold maximises the F1 of detecting cancer; the second-term
threshold maximises the F1 of "rank-2 term is gold" over the fit-half cases that pass the gate.
  pilot        the OpenJev pilot cases (50/50 cancer mix); also reports verdict shares re-weighted to the natural mix
  calibration  every calibration case (already the natural mix, so unweighted only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: F401  must load before scipy/sklearn on Windows (libomp DLL clash, WinError 1114)
import numpy as np
import pandas as pd

import config
import io_utils
from evaluation import silver_eval, verdicts
from experiments.retrieve_rerank.data import SILVER_ID, load_phase0_data
from experiments.retrieve_rerank.pilot_score import (NON_CANCER, PREDICTION_COLUMNS, best_f1_threshold, in_fit_half,
                                                     natural_mix_weights, q1_p_yes, verdict_summary)

GENERATION_ID = "rr-phase0-retriever-only"
METHOD = "retriever_top1"
WP14 = "wp14_case_presence"
GATE_SCORES = {"openjev_q1": lambda v: v["p_yes"], "retriever_score": lambda v: float(v["prob"][0])}


def retriever_views(candidates: pd.DataFrame, gold: dict) -> dict[str, dict]:
    """case_id -> {terms best first, prob = sigmoid(logit), gold flags, is_cancer} from the candidates CSV."""
    views = {}
    ordered = candidates.assign(rank=candidates["rank"].astype(int)).sort_values(["case_id", "rank"])
    for case_id, rows in ordered.groupby("case_id", sort=False):
        terms = rows["term"].tolist()
        views[case_id] = {"case_id": case_id, "is_cancer": case_id in gold, "terms": terms,
                          "prob": 1 / (1 + np.exp(-rows["score"].astype(float).to_numpy())),
                          "gold": np.array([t in gold.get(case_id, ()) for t in terms])}
    return views


def wp14_passes(baseline: pd.DataFrame) -> set[str]:
    """Cases the production gate let through: >= 1 baseline row whose group is not Non-Cancer. Absent = rejected."""
    return set(baseline.loc[baseline["predicted_group"] != NON_CANCER, "case_id"])


def fit_gate(name: str, fit: list[dict], wp14_pass: set[str]):
    """(view -> pass?, fitted threshold or None) for one gate; the threshold is fitted on ``fit`` only."""
    if name == WP14:
        return (lambda v: v["case_id"] in wp14_pass), None
    score = GATE_SCORES[name]
    threshold, _ = best_f1_threshold(np.array([score(v) for v in fit]), np.array([bool(v["is_cancer"]) for v in fit]))
    return (lambda v: score(v) >= threshold), threshold


def fit_second(fit: list[dict], passes) -> float:
    passed = [v for v in fit if passes(v)]
    second_gold = np.array([bool(v["gold"][1]) for v in passed])
    if not second_gold.any():
        return 1.01
    return best_f1_threshold(np.array([float(v["prob"][1]) for v in passed]), second_gold)[0]


def make_predictions(views: list[dict], passes, second: float, data) -> pd.DataFrame:
    """Predictions-CSV rows: one Non-Cancer row for a rejected case, else the rank-1 term (+ rank 2 if its
    probability clears ``second``)."""
    rows = []
    for v in views:
        base = {"case_id": v["case_id"], "case_presence_prob": "", "group_prob": "", "generation_id": GENERATION_ID}
        if not passes(v):
            rows.append({**base, "diagnosis_index": 1, "predicted_term": NON_CANCER, "predicted_group": NON_CANCER,
                         "predicted_code": "", "confidence": "0.00", "method": "rejected_by_gate"})
            continue
        for rank in range(2):
            if rank == 1 and v["prob"][1] < second:
                break
            term = v["terms"][rank]
            rows.append({**base, "diagnosis_index": rank + 1, "predicted_term": term,
                         "predicted_group": data.term_group[term], "predicted_code": data.term_code[term],
                         "confidence": f"{v['prob'][rank]:.2f}", "method": METHOD})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).astype(str)


def score_set(views: list[dict], gates: list[str], baseline: pd.DataFrame, data, labels, weights: dict) -> dict:
    fit = [v for v in views if in_fit_half(v["case_id"])]
    evaluation = [v for v in views if not in_fit_half(v["case_id"])]
    eval_ids = [v["case_id"] for v in evaluation]
    cancer = [v for v in evaluation if v["is_cancer"]]
    wp14_pass = wp14_passes(baseline)

    systems = {}
    for name in gates:
        passes, gate_t = fit_gate(name, fit, wp14_pass)
        second = fit_second(fit, passes)
        predictions = make_predictions(evaluation, passes, second, data)
        systems[name] = {
            "gate_threshold": gate_t, "second_term_threshold": second,
            "gate_eval": {"recall": float(np.mean([passes(v) for v in cancer])),
                          "false_alarm_rate": float(np.mean([passes(v) for v in evaluation if not v["is_cancer"]]))},
            "verdicts": verdict_summary(predictions, labels, eval_ids, data.uncommon_groups, weights)}
    return {"n_fit": len(fit), "n_eval": len(evaluation), "n_eval_cancer": len(cancer),
            "retriever_top1_accuracy_eval_cancer": float(np.mean([bool(v["gold"][0]) for v in cancer])),
            "systems": systems,
            "wp14_baseline": verdict_summary(baseline, labels, eval_ids, data.uncommon_groups, weights)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", type=Path, default=config.RR_PHASE0_DIR / "bce" / "calibration_candidates.csv")
    parser.add_argument("--raw", type=Path, default=config.RR_PHASE0_DIR / "pilot" / "raw_bce_n30.jsonl")
    parser.add_argument("--pilot-scores", type=Path, default=config.RR_PHASE0_DIR / "pilot" / "scores_bce_n30.json",
                        help="pilot_score output, only used to check the WP14 row reproduces")
    parser.add_argument("--baseline-predictions", type=Path,
                        default=config.PREDICTIONS_DIR / "gen-20260927T025939Z_predictions.csv")
    parser.add_argument("--out-dir", type=Path, default=config.RR_PHASE0_DIR / "retriever_only")
    args = parser.parse_args()

    data = load_phase0_data()
    labels = silver_eval.load_labels(SILVER_ID)
    baseline = silver_eval.read_predictions(args.baseline_predictions)
    candidates = io_utils.read_csv(args.candidates_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    by_case = retriever_views(candidates, data.gold)
    records = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines()]

    pilot_views = [{**by_case[r["case_id"]], "p_yes": q1_p_yes(r)} for r in records]
    pilot_weights = natural_mix_weights({r["case_id"]: r["is_cancer"] for r in records}, data)
    calibration_views = [by_case[c] for c in data.calibration_ids]

    result = {"pilot": score_set(pilot_views, ["openjev_q1", WP14, "retriever_score"], baseline, data, labels, pilot_weights),
              "calibration": score_set(calibration_views, [WP14, "retriever_score"], baseline, data, labels,
                                       {c: 1.0 for c in data.calibration_ids})}
    result["pilot"]["natural_mix"] = True
    result["calibration"]["natural_mix"] = False

    # The WP14 row must be the one pilot_score reported for the same eval cases.
    if args.pilot_scores.is_file():
        earlier = json.loads(args.pilot_scores.read_text(encoding="utf-8"))["verdicts_wp14_baseline"]
        mine = result["pilot"]["wp14_baseline"]
        same = all(earlier[k] == mine[k] for k in ("total", *verdicts.VERDICTS))
        result["wp14_row_matches_pilot_score"] = same
        print(f"check: WP14 pilot row rows={mine['total']} (pilot_score {earlier['total']}), "
              f"Good {100 * mine['good_share']:.1f}% (pilot_score {100 * earlier['good_share']:.1f}%): "
              f"{'OK' if same else 'MISMATCH'}")
    else:
        result["wp14_row_matches_pilot_score"] = None
        print(f"check skipped: {args.pilot_scores} not found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scores.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    markdown = render_markdown(result)
    (args.out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def render_markdown(r: dict) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    def row(name: str, s: dict, natural: bool) -> str:
        cells = [pct(s[f"{v}_share"]) for v in verdicts.VERDICTS]
        counts = "/".join(str(s[v]) for v in verdicts.VERDICTS)
        line = f"| {name} | {s['total']} | " + " | ".join(cells) + f" | {pct(s['good_plus_slight_share'])} | {counts} |"
        return line + (" " + " / ".join(pct(s["natural_mix"][v]) for v in verdicts.VERDICTS) + " |" if natural else "")

    lines = ["# Retriever-only top-1 diagnostic", ""]
    for key, title in (("pilot", "Pilot cases"), ("calibration", "All calibration cases")):
        s, natural = r[key], r[key]["natural_mix"]
        lines += [f"## {title} (eval half: {s['n_eval']} cases, {s['n_eval_cancer']} cancer; fit half {s['n_fit']})", "",
                  f"Retriever top-1 accuracy on eval-half cancer cases: {pct(s['retriever_top1_accuracy_eval_cancer'])}%", "",
                  "Verdict shares in % of scored rows; counts are good/slightly_off/completely_off/false_positive/false_negative"
                  + ("; last column = natural cancer mix, same order" if natural else " (unweighted: already the natural mix)"), "",
                  "| system | rows | Good | Slight | CO | FP | FN | G+S | counts |" + (" natural mix |" if natural else ""),
                  "|---|---|---|---|---|---|---|---|---|" + ("---|" if natural else "")]
        lines += [row(f"retriever_top1 + {name}", sys_["verdicts"], natural) for name, sys_ in s["systems"].items()]
        lines += [row("WP14 baseline", s["wp14_baseline"], natural), "",
                  "| gate | gate threshold | second-term threshold | cancer passed | non-cancer passed |", "|---|---|---|---|---|"]
        for name, sys_ in s["systems"].items():
            t = sys_["gate_threshold"]
            lines.append(f"| {name} | {'n/a' if t is None else f'{t:.2f}'} | {sys_['second_term_threshold']:.2f} | "
                         f"{pct(sys_['gate_eval']['recall'])}% | {pct(sys_['gate_eval']['false_alarm_rate'])}% |")
        lines.append("")
    check = r["wp14_row_matches_pilot_score"]
    lines.append("WP14 pilot row vs pilot_score: " + ("not checked" if check is None else "OK" if check else "MISMATCH"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
