"""Gate bake-off on the CALIBRATION partition only: fit half -> eval half.

For each gate source (old = WP14 PetBERT gate probabilities from its predictions CSV; new = the ModernBERT gate)
and each available retriever seed (and, for a "shared" gate, a head on one retriever's pooled vector, only with
the retriever it was trained on): fit {gate_t, tau, rho} on the fit half, score the eval half through
``silver_eval.evaluate`` with WP14's uncommon list, and report gate AUC, retriever top-1 accuracy, the objective
along gate_t (the "no cliff" check) and the WP14 baseline on the same eval-half cases.

Fit / eval half: the repo's md5 half rule is constant over the calibration partition (calibration IS the sweep
half), so, as in the Phase 0 pilot, the next md5 bit splits it.
WP14's own thresholds were fitted on the whole calibration partition, so the baseline row has seen this eval
half; the comparison is mildly baseline-optimistic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import config
from bronze_rr import calibrate, decide, gate as gate_mod, scoring, shared_gate
from bronze_rr.data import BronzeData, load_uncommon_groups
from bronze_rr.decide import Thresholds
from bronze_rr.ruler import SUMMARY_HEADER, summary_row
from evaluation import silver_eval

OLD_GATE_CSV = config.PREDICTIONS_DIR / "gen-20260927T025939Z_predictions.csv"


def in_fit_half(case_id: str) -> bool:
    return (int(hashlib.md5(case_id.encode()).hexdigest(), 16) >> 1) % 2 == 0


def available_retrievers(root: Path) -> dict[str, Path]:
    """name -> dir: the Phase 0 BCE retriever (seed 42) and every trained <root>/retriever_seed<S>."""
    found = {"seed42": config.RR_PHASE0_DIR / "bce"}
    for directory in sorted(root.glob("retriever_seed*")):
        found[directory.name.removeprefix("retriever_")] = directory
    return {name: d for name, d in found.items() if (d / "train_log.json").is_file()}


def _top1_accuracy(scores: scoring.RetrieverScores, gold: dict) -> float:
    cancer = [i for i, c in enumerate(scores.case_ids) if c in gold]
    return float(np.mean([scores.terms[scores.term_idx[i, 0]] in gold[scores.case_ids[i]] for i in cancer]))


def _half(data: BronzeData, fit: bool, limit: int) -> tuple[list[str], frozenset[str]]:
    """(scored calibration cases in the half, the half of the split's calibration partition). ``limit`` > 0 is
    a smoke: only the first N report cases, and the half of the partition shrinks to them."""
    in_half = [c for c in data.partition_ids["calibration"][:limit or None] if in_fit_half(c) == fit]
    if limit:
        return in_half, frozenset(in_half)
    return in_half, frozenset(c for c in data.split_cases["calibration"] if in_fit_half(c) == fit)


def run_system(gate_name: str, gate_by_case: dict, retriever_name: str, scores: scoring.RetrieverScores,
               data: BronzeData, labels, uncommon, limit: int) -> dict:
    fit_cases, fit_universe = _half(data, True, limit)
    eval_cases, eval_universe = _half(data, False, limit)
    fit_scores, eval_scores = scores.subset(fit_cases), scores.subset(eval_cases)
    pc_fit = calibrate.build_pattern_counts(fit_scores, gate_mod.probs_for(fit_cases, gate_by_case), data.has_text,
                                            labels, fit_universe, uncommon, data.term_info)
    fitted = calibrate.fit_thresholds(pc_fit)
    th: Thresholds = fitted["thresholds"]

    pc_eval = calibrate.build_pattern_counts(eval_scores, gate_mod.probs_for(eval_cases, gate_by_case), data.has_text,
                                             labels, eval_universe, uncommon, data.term_info)
    eval_curve = []
    for t in calibrate.GATE_GRID:
        gs, good = pc_eval.objective(Thresholds(t, th.tau, th.rho))
        eval_curve.append({"gate_t": t, "gs": gs, "good": good})
    predictions = decide.decide_all(eval_scores, gate_by_case, data.has_text, th, data.term_info, f"bakeoff-{gate_name}")
    summary = silver_eval.evaluate(predictions, labels, eval_universe, uncommon)["summary"]

    gate_eval = gate_mod.probs_for(eval_cases, gate_by_case)
    cancer = np.array([c in data.gold for c in eval_cases])
    return {
        "gate": gate_name, "retriever": retriever_name, "thresholds": th.as_dict(),
        "fit_objective": {"gs": fitted["gs_share"], "good": fitted["good_share"]},
        "n_fit_cases": len(fit_cases), "n_eval_cases": len(eval_cases), "n_eval_cancer": int(cancer.sum()),
        "gate_auc_eval": gate_mod.roc_auc(gate_eval, cancer),
        "gate_eval_at_threshold": {"recall": float((gate_eval[cancer] >= th.gate_t).mean()),
                                   "false_alarm_rate": float((gate_eval[~cancer] >= th.gate_t).mean())},
        "retriever_top1_accuracy_eval_cancer": _top1_accuracy(eval_scores, data.gold),
        "verdicts": summary, "curve_fit": fitted["curve"], "curve_eval_at_chosen_tau_rho": eval_curve,
    }


def run_bakeoff(data: BronzeData, labels, *, root: Path = config.RR_PHASE1_DIR, limit: int = 0) -> dict:
    """Models, score caches and outputs live under ``root`` (the smoke passes a scratch root with ``limit``)."""
    uncommon = load_uncommon_groups()
    scores_dir, new_gate_dir, out_dir = root / "scores", root / "gate", root / "bakeoff"
    gates = {"old": gate_mod.load_old_gate_probs(OLD_GATE_CSV)}
    if (new_gate_dir / "train_log.json").is_file():
        gates["new"] = scoring.gate_scores(new_gate_dir, data, "calibration", out_dir=scores_dir, limit=limit)
    else:
        print(f"no trained gate in {new_gate_dir}: only the old gate is compared")

    systems = []
    retrievers = available_retrievers(root)
    for retriever_name, directory in retrievers.items():
        scores = scoring.retriever_scores(directory, data, "calibration", out_dir=scores_dir, limit=limit)
        for gate_name, gate_by_case in gates.items():
            systems.append(run_system(gate_name, gate_by_case, retriever_name, scores, data, labels, uncommon, limit))

    for gate_dir in shared_gate.available(root):
        wanted = shared_gate.read_source(gate_dir)["retriever_tag"]
        paired = [(n, d) for n, d in retrievers.items() if scoring.model_tag(d) == wanted]
        if not paired:
            print(f"{gate_dir.name}: its retriever ({wanted}) is not available; skipped")
            continue
        retriever_name, directory = paired[0]
        gate_by_case = shared_gate.gate_scores(gate_dir, directory, data, "calibration", out_dir=scores_dir, limit=limit)
        scores = scoring.retriever_scores(directory, data, "calibration", out_dir=scores_dir, limit=limit)
        systems.append(run_system("shared", gate_by_case, retriever_name, scores, data, labels, uncommon, limit))

    _, eval_universe = _half(data, False, limit)
    baseline = silver_eval.evaluate(silver_eval.read_predictions(OLD_GATE_CSV), labels, eval_universe, uncommon)["summary"]
    result = {"split_id": data.split_id, "partition": "calibration", "eval_half": "md5 bit 1 == 1",
              "systems": systems, "wp14_baseline_eval_half": baseline}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict) -> str:
    lines = ["# Phase 1 gate bake-off (calibration partition, fit half -> eval half)", "",
             "Shares are % of scored rows; counts are good/slight/co/fp/fn. Row totals differ between systems.",
             "Caveat: WP14's thresholds were fitted on the whole calibration partition, so its row has seen this eval half.",
             "", SUMMARY_HEADER]
    lines += [summary_row(f"{s['retriever']} + {s['gate']} gate", s["verdicts"]) for s in result["systems"]]
    lines += [summary_row("WP14 baseline", result["wp14_baseline_eval_half"]), "",
              "| system | gate_t | tau | rho | fit G+S | fit Good | gate AUC (eval) | cancer passed | non-cancer passed | top-1 acc (eval cancer) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in result["systems"]:
        t, g = s["thresholds"], s["gate_eval_at_threshold"]
        lines.append(f"| {s['retriever']} + {s['gate']} | {t['gate_t']:.2f} | {t['tau']:.2f} | {t['rho']:.2f} | "
                     f"{100 * s['fit_objective']['gs']:.2f} | {100 * s['fit_objective']['good']:.2f} | "
                     f"{s['gate_auc_eval']:.4f} | {100 * g['recall']:.1f}% | {100 * g['false_alarm_rate']:.1f}% | "
                     f"{100 * s['retriever_top1_accuracy_eval_cancer']:.1f}% |")
    lines += ["", "## Objective along gate_t (no-cliff check)", "",
              "Per system: fit-half G+S at the best (tau, rho) for that gate_t; fit-half and eval-half G+S at the chosen (tau, rho). All %.", ""]
    for s in result["systems"]:
        lines += [f"### {s['retriever']} + {s['gate']} gate (chosen gate_t {s['thresholds']['gate_t']:.2f})", "",
                  "| gate_t | fit best G+S | fit G+S @chosen | eval G+S @chosen | eval Good @chosen |", "|---|---|---|---|---|"]
        for fit_row, eval_row in zip(s["curve_fit"], s["curve_eval_at_chosen_tau_rho"]):
            lines.append(f"| {fit_row['gate_t']:.2f} | {100 * fit_row['best_gs']:.2f} | {100 * fit_row['at_chosen_gs']:.2f} | "
                         f"{100 * eval_row['gs']:.2f} | {100 * eval_row['good']:.2f} |")
        lines.append("")
    return "\n".join(lines) + "\n"
