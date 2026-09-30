"""Score the OpenJev pilot from its raw probabilities (cheap to re-run; never calls a model).

  ml/.venv/Scripts/python.exe ml/experiments/retrieve_rerank/pilot_score.py --loss bce --n 30

Reads <pilot-dir>/raw_<loss>_<tag>.jsonl, writes scores_<loss>_<tag>.json and summary_<loss>_<tag>.md beside it. Two variants
per run: "retriever" (candidates = the retriever's top-n) and "oracle" (gold terms forced in).

Fit half / eval half: the repo's md5 half rule (generations.splits.in_sweep_half) is constant over the
calibration partition (calibration IS the sweep half), so it cannot split the pilot. This script uses the
next md5 bit instead: fit half iff (int(md5(case_id), 16) >> 1) % 2 == 0. Thresholds and the temperature
are fitted on the fit half only; every reported metric is on the eval half (the F1-vs-threshold scan is a
diagnostic on the eval half and is not used to fit anything).

Candidate scores: P(entailment) of each candidate hypothesis. Confidence = those scores normalised over
the candidates (the card's decide() rule); the temperature T is applied to log P(entailment) before that
normalisation. Q1 probability = P_ent(yes) / (P_ent(yes) + P_ent(no)).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: F401  must load before scipy/sklearn on Windows (libomp DLL clash, WinError 1114)
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

import config
from evaluation import silver_eval, verdicts
from experiments.retrieve_rerank.data import SILVER_ID, load_phase0_data

ECE_BINS = 15
ENT = 1  # index of "entailment" in [contradiction, entailment, neutral]
NON_CANCER = "Non-Cancer"
GENERATION_ID = "rr-phase0-pilot"
PREDICTION_COLUMNS = ["case_id", "diagnosis_index", "predicted_term", "predicted_group", "predicted_code",
                      "case_presence_prob", "confidence", "group_prob", "method", "generation_id"]
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 3)


def in_fit_half(case_id: str) -> bool:
    return (int(hashlib.md5(case_id.encode()).hexdigest(), 16) >> 1) % 2 == 0


def case_view(record: dict, variant: str, n: int) -> dict | None:
    """One case's candidates sorted by P(entailment), for the retriever or oracle candidate set."""
    cands = [c for c in record["candidates"]
             if (c["rank"] is not None and c["rank"] <= n) or (variant == "oracle" and c["gold"])]
    if not cands:
        return None
    ent = np.array([c["probs"][ENT] for c in cands])
    order = np.argsort(-ent, kind="stable")
    yes, no = record["q1_yes"][ENT], record["q1_no"][ENT]
    return {"case_id": record["case_id"], "is_cancer": record["is_cancer"],
            "terms": [cands[i]["term"] for i in order], "ent": ent[order],
            "gold": np.array([cands[i]["gold"] for i in order]), "p_yes": yes / max(yes + no, 1e-9)}


def probs(view: dict, temperature: float) -> np.ndarray:
    z = np.log(np.clip(view["ent"], 1e-9, None)) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_temperature(views: list[dict]) -> float:
    """Minimise the NLL of the probability mass on gold, over cancer cases whose gold is in the candidates."""
    usable = [v for v in views if v["gold"].any()]
    def nll(log_t: float) -> float:
        return -float(np.mean([np.log(max(probs(v, np.exp(log_t))[v["gold"]].sum(), 1e-12)) for v in usable]))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(conf: np.ndarray, correct: np.ndarray) -> float:
    bins = np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).sum()
                     for b in range(ECE_BINS) if (bins == b).any()) / len(conf))


def best_f1_threshold(score: np.ndarray, positive: np.ndarray, accepted_correct: np.ndarray | None = None) -> tuple[float, float]:
    """(threshold, F1) maximising F1 of 'accept if score >= t'. ``accepted_correct`` (defaults to
    ``positive``) marks which accepted items count as true positives; ``positive`` sets the recall base."""
    hit = positive if accepted_correct is None else accepted_correct
    best = (1.01, 0.0)
    for t in THRESHOLD_GRID:
        accept = score >= t
        tp = float((accept & hit).sum())
        f1 = 2 * tp / max(accept.sum() + positive.sum(), 1)
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def f1_at(score: np.ndarray, positive: np.ndarray, hit: np.ndarray, t: float) -> float:
    accept = score >= t
    return float(2 * (accept & hit).sum() / max(accept.sum() + positive.sum(), 1))


def accept_arrays(views: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per case: (normalised top-1 confidence, raw top-1 entailment, top-1 is gold, is cancer)."""
    return (np.array([probs(v, 1.0)[0] for v in views]), np.array([v["ent"][0] for v in views]),
            np.array([bool(v["is_cancer"]) and bool(v["gold"][0]) for v in views]),
            np.array([bool(v["is_cancer"]) for v in views]))


def make_predictions(views: list[dict], temperature: float, gate: float, second: float, data) -> pd.DataFrame:
    """Rows in the predictions-CSV format: Non-Cancer below the gate, else the top-1 term (plus a second
    term if its probability clears ``second``)."""
    rows = []
    for v in views:
        base = {"case_id": v["case_id"], "case_presence_prob": f"{v['p_yes']:.4f}", "group_prob": "",
                "generation_id": GENERATION_ID}
        if v["p_yes"] < gate:
            rows.append({**base, "diagnosis_index": 1, "predicted_term": NON_CANCER, "predicted_group": NON_CANCER,
                         "predicted_code": "", "confidence": "0.00", "method": "rejected_by_gate"})
            continue
        p = probs(v, temperature)
        for rank in range(min(2, len(p))):
            if rank == 1 and p[1] < second:
                break
            term = v["terms"][rank]
            rows.append({**base, "diagnosis_index": rank + 1, "predicted_term": term,
                         "predicted_group": data.term_group[term], "predicted_code": data.term_code[term],
                         "confidence": f"{p[rank]:.2f}", "method": "openjev_rerank"})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).astype(str)


def verdict_summary(predictions: pd.DataFrame, labels: pd.DataFrame, case_ids, uncommon, weights: dict) -> dict:
    table = silver_eval.evaluate(predictions, labels, case_ids, uncommon)["table"].copy()
    table["w"] = table["case_id"].map(weights).astype(float)
    summary = verdicts.summarize(table)
    summary["natural_mix"] = {v: verdicts.share(table, (v,), "w") for v in verdicts.VERDICTS}
    summary["natural_mix"]["good_plus_slight"] = verdicts.share(table, verdicts.GOOD_PLUS_SLIGHT, "w")
    return summary


def score_variant(views: list[dict], data, labels, weights: dict) -> dict:
    fit = [v for v in views if in_fit_half(v["case_id"])]
    evaluation = [v for v in views if not in_fit_half(v["case_id"])]
    fit_cancer = [v for v in fit if v["is_cancer"]]
    temperature = fit_temperature(fit_cancer)

    # Fit on the fit half: neoplasm gate (F1 of detecting cancer) and the second-term threshold.
    p_yes_fit = np.array([v["p_yes"] for v in fit])
    cancer_fit = np.array([v["is_cancer"] for v in fit], dtype=bool)
    gate, gate_f1 = best_f1_threshold(p_yes_fit, cancer_fit)
    passed = [v for v in fit if v["p_yes"] >= gate and len(v["terms"]) >= 2]
    second_p = np.array([probs(v, temperature)[1] for v in passed])
    second_gold = np.array([bool(v["gold"][1]) for v in passed])
    second, second_f1 = best_f1_threshold(second_p, second_gold) if second_gold.any() else (1.01, 0.0)

    # Eval-half metrics.
    cancer = [v for v in evaluation if v["is_cancer"]]
    with_gold = [v for v in cancer if v["gold"].any()]
    top1_hit = lambda vs: np.array([bool(v["gold"][0]) for v in vs])
    raw_conf = np.array([probs(v, 1.0)[0] for v in with_gold])
    cal_conf = np.array([probs(v, temperature)[0] for v in with_gold])
    hit = top1_hit(with_gold)

    all_raw, all_ent, all_hit, positives = accept_arrays(evaluation)
    fit_raw, fit_ent, fit_hit, fit_positives = accept_arrays(fit)
    t_norm, f1_norm = best_f1_threshold(all_raw, positives, all_hit)
    t_ent, f1_ent = best_f1_threshold(all_ent, positives, all_hit)
    t_norm_fit, _ = best_f1_threshold(fit_raw, fit_positives, fit_hit)
    t_ent_fit, _ = best_f1_threshold(fit_ent, fit_positives, fit_hit)
    cliff_grid = (0.3, 0.5, 0.7, 0.9, 0.95, 0.99)

    p_yes = np.array([v["p_yes"] for v in evaluation])
    eval_ids = [v["case_id"] for v in evaluation]
    predictions = make_predictions(evaluation, temperature, gate, second, data)
    return {
        "n_fit": len(fit), "n_eval": len(evaluation), "n_eval_cancer": len(cancer),
        "temperature": temperature, "gate_threshold": gate, "gate_f1_fit": gate_f1,
        "second_term_threshold": second, "second_term_f1_fit": second_f1,
        "q1_auc_eval": float(roc_auc_score(positives, p_yes)),
        "q1_gate_eval": {"recall": float((p_yes[positives] >= gate).mean()),
                         "false_alarm_rate": float((p_yes[~positives] >= gate).mean())},
        "retrieval_hit_rate_eval_cancer": len(with_gold) / max(len(cancer), 1),
        "top1_acc_gold_in_candidates": float(hit.mean()),
        "top1_acc_all_cancer_cases": float(hit.sum() / max(len(cancer), 1)),
        "ece_raw": ece(raw_conf, hit), "ece_temperature": ece(cal_conf, hit),
        "mean_conf_raw": float(raw_conf.mean()), "mean_conf_temperature": float(cal_conf.mean()),
        "accept_top1_f1": {
            "normalised_conf": {"best_threshold": t_norm, "best_f1": f1_norm, "best_threshold_fit_half": t_norm_fit,
                                "f1_eval_at_fit_threshold": f1_at(all_raw, positives, all_hit, t_norm_fit),
                                "f1_at": {str(t): f1_at(all_raw, positives, all_hit, t) for t in cliff_grid}},
            "raw_entailment": {"best_threshold": t_ent, "best_f1": f1_ent, "best_threshold_fit_half": t_ent_fit,
                               "f1_eval_at_fit_threshold": f1_at(all_ent, positives, all_hit, t_ent_fit),
                               "f1_at": {str(t): f1_at(all_ent, positives, all_hit, t) for t in cliff_grid}},
        },
        "verdicts_openjev": verdict_summary(predictions, labels, eval_ids, data.uncommon_groups, weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", required=True, help="retriever loss dir name the pilot was run with (bce / softmax)")
    parser.add_argument("--n", type=int, default=30, help="top-n retriever candidates to keep (<= the run's n)")
    parser.add_argument("--tag", default=None, help="raw file suffix after the loss; default n<N>")
    parser.add_argument("--pilot-dir", type=Path, default=config.RR_PHASE0_DIR / "pilot")
    parser.add_argument("--baseline-predictions", type=Path,
                        default=config.PREDICTIONS_DIR / "gen-20260927T025939Z_predictions.csv")
    args = parser.parse_args()
    tag = f"{args.loss}_{args.tag or f'n{args.n}'}"

    raw = args.pilot_dir / f"raw_{tag}.jsonl"
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    data = load_phase0_data()
    labels = silver_eval.load_labels(SILVER_ID)

    # Natural-mix weights: the pilot is 50/50 cancer / non-cancer, calibration is not.
    n_cancer_pop = len(data.calibration_cancer_ids())
    n_non_pop = len(data.calibration_ids) - n_cancer_pop
    n_cancer_s = sum(r["is_cancer"] for r in records)
    weights = {r["case_id"]: (n_cancer_pop / n_cancer_s if r["is_cancer"] else n_non_pop / (len(records) - n_cancer_s))
               for r in records}

    seconds = np.array([r["seconds"] for r in records])
    result = {"n": args.n, "tag": tag, "n_cases": len(records), "n_cancer": n_cancer_s,
              "calibration_population": {"cancer": n_cancer_pop, "non_cancer": n_non_pop},
              "truncated_cases": int(sum(r["truncated"] for r in records)),
              "mean_premise_tokens": float(np.mean([r["n_premise_tokens"] for r in records])),
              "latency_seconds_per_case": {"mean": float(seconds.mean()), "median": float(np.median(seconds)),
                                           "p95": float(np.percentile(seconds, 95))}}
    for variant in ("retriever", "oracle"):
        views = [v for v in (case_view(r, variant, args.n) for r in records) if v is not None]
        result[variant] = score_variant(views, data, labels, weights)

    eval_ids = [r["case_id"] for r in records if not in_fit_half(r["case_id"])]
    baseline = silver_eval.read_predictions(args.baseline_predictions)
    result["verdicts_wp14_baseline"] = verdict_summary(baseline, labels, eval_ids, data.uncommon_groups, weights)
    base_eval = baseline[baseline["case_id"].isin(eval_ids)]
    unidentified = (base_eval.get("method", "") == "unidentified_cancer") | (base_eval["predicted_group"] == "Unidentified Group")
    result["wp14_baseline_unidentified_rows_eval"] = {"unidentified": int(unidentified.sum()), "rows": len(base_eval)}

    (args.pilot_dir / f"scores_{tag}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (args.pilot_dir / f"summary_{tag}.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


def render_markdown(r: dict) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    def verdict_row(name: str, s: dict) -> str:
        cells = [pct(s[f"{v}_share"]) for v in verdicts.VERDICTS]
        natural = [pct(s["natural_mix"][v]) for v in verdicts.VERDICTS]
        return (f"| {name} | {s['total']} | " + " | ".join(cells) + " | " + pct(s["good_plus_slight_share"]) + " | "
                + " / ".join(natural) + " |")

    lines = [f"# OpenJev pilot scores (n={r['n']}, {r['n_cases']} cases, {r['n_cancer']} cancer)", "",
             f"Truncated cases: {r['truncated_cases']}; mean premise tokens {r['mean_premise_tokens']:.0f}; "
             f"latency per case mean {r['latency_seconds_per_case']['mean']:.2f}s, "
             f"median {r['latency_seconds_per_case']['median']:.2f}s, p95 {r['latency_seconds_per_case']['p95']:.2f}s.", ""]
    for variant in ("retriever", "oracle"):
        s = r[variant]
        f1n, f1e = s["accept_top1_f1"]["normalised_conf"], s["accept_top1_f1"]["raw_entailment"]
        lines += [f"## {variant} (eval half: {s['n_eval']} cases, {s['n_eval_cancer']} cancer)", "",
                  f"- Q1 AUC {s['q1_auc_eval']:.3f}; gate t={s['gate_threshold']:.2f} recall {pct(s['q1_gate_eval']['recall'])}%, "
                  f"false-alarm {pct(s['q1_gate_eval']['false_alarm_rate'])}%",
                  f"- Retrieval hit rate (cancer, gold in candidates) {pct(s['retrieval_hit_rate_eval_cancer'])}%; "
                  f"top-1 accuracy | gold in candidates {pct(s['top1_acc_gold_in_candidates'])}%; over all cancer cases {pct(s['top1_acc_all_cancer_cases'])}%",
                  f"- ECE(15) raw {s['ece_raw']:.3f} -> after T={s['temperature']:.2f}: {s['ece_temperature']:.3f}",
                  f"- Accept-top-1 F1 (normalised conf): eval-half best {f1n['best_f1']:.3f} at t={f1n['best_threshold']:.2f} "
                  f"(diagnostic); fit-half best t={f1n['best_threshold_fit_half']:.2f} -> eval F1 {f1n['f1_eval_at_fit_threshold']:.3f}; "
                  f"F1 at t={f1n['f1_at']}",
                  f"- Accept-top-1 F1 (raw entailment): eval-half best {f1e['best_f1']:.3f} at t={f1e['best_threshold']:.2f} "
                  f"(diagnostic); fit-half best t={f1e['best_threshold_fit_half']:.2f} -> eval F1 {f1e['f1_eval_at_fit_threshold']:.3f}",
                  f"- Second-term threshold {s['second_term_threshold']:.2f}", ""]
    lines += ["## Verdicts on the eval-half pilot cases (%; last column = natural cancer mix, Good / Slight / CO / FP / FN)", "",
              "| system | rows | Good | Slight | CO | FP | FN | G+S | natural mix |", "|---|---|---|---|---|---|---|---|---|",
              verdict_row("OpenJev, retriever candidates", r["retriever"]["verdicts_openjev"]),
              verdict_row("OpenJev, oracle candidates", r["oracle"]["verdicts_openjev"]),
              verdict_row("WP14 baseline", r["verdicts_wp14_baseline"]), "",
              f"WP14 baseline rows with an unidentified cancer (method unidentified_cancer / group Unidentified Group) among "
              f"the eval cases: {r['wp14_baseline_unidentified_rows_eval']['unidentified']} of "
              f"{r['wp14_baseline_unidentified_rows_eval']['rows']} prediction rows.",
              "Note: the baseline's thresholds were calibrated on the whole calibration partition (which includes this "
              "eval half), so this comparison is mildly baseline-optimistic.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
