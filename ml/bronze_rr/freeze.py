"""Freeze a Phase 1 system: fit the thresholds on the FULL calibration partition, write the generation directory,
then predict every case that has report text (train + calibration + test) to a predictions CSV.

Predicting test cases is not scoring them: nothing here evaluates against test labels.
"""

from __future__ import annotations

from pathlib import Path

import io_utils
import config
from bronze_rr import calibrate, decide, gate as gate_mod, generation, joint, joint_scoring, scoring, shared_gate
from bronze_rr.bakeoff import OLD_GATE_CSV
from bronze_rr.data import BronzeData, load_uncommon_groups


def run_freeze(data: BronzeData, labels, *, gate: str, retriever_dir: Path, root: Path = config.RR_PHASE1_DIR,
               limit: int = 0, max_fp_share: float | None = None) -> dict:
    """``gate`` is "old" (WP14 predictions CSV), "new" (<root>/gate), "shared" (the head trained on this retriever,
    <root>/gate_shared_<retriever tag>; one retriever pass per report gives both the gate and the term
    probabilities) or "joint" (``retriever_dir`` is a jointly trained model, <root>/joint, holding both heads: one
    encoder pass gives both too). Models and caches live under ``root``; the predictions CSV goes to
    config.PREDICTIONS_DIR, or to <root>/predictions in a smoke (``limit`` > 0). ``max_fp_share``: the threshold fit
    only considers points whose false-positive share on the calibration partition is <= it (None: no cap)."""
    scores_dir, gate_dir = root / "scores", (root / "gate" if gate == "new" else None)
    shared_dir = shared_gate.gate_dir_for(root, retriever_dir) if gate == "shared" else None
    uncommon = load_uncommon_groups()

    if gate == "new":
        gate_by_case = scoring.gate_scores(gate_dir, data, "all", out_dir=scores_dir, limit=limit)
    elif gate == "joint":  # first, so the same pass fills the retriever score caches read below
        if not (retriever_dir / joint.GATE_HEAD_FILE).is_file():
            raise SystemExit(f"{retriever_dir} is not a jointly trained model (no {joint.GATE_HEAD_FILE})")
        _, gate_by_case = joint_scoring.score(retriever_dir, data, "all", out_dir=scores_dir, limit=limit)
    elif gate == "shared":  # first, so the same pass fills the retriever score caches read below
        gate_by_case = shared_gate.gate_scores(shared_dir, retriever_dir, data, "all", out_dir=scores_dir, limit=limit)
    else:
        gate_by_case = gate_mod.load_old_gate_probs(OLD_GATE_CSV)
    calibration_scores = scoring.retriever_scores(retriever_dir, data, "calibration", out_dir=scores_dir, limit=limit)
    all_scores = scoring.retriever_scores(retriever_dir, data, "all", out_dir=scores_dir, limit=limit)

    universe = frozenset(calibration_scores.case_ids) if limit else data.split_cases["calibration"]
    counts = calibrate.build_pattern_counts(calibration_scores, gate_mod.probs_for(calibration_scores.case_ids, gate_by_case),
                                            data.has_text, labels, universe, uncommon, data.term_info)
    fitted = calibrate.fit_thresholds(counts, max_fp_share)
    thresholds = fitted["thresholds"]

    generation_id = generation.new_generation_id()
    calibration = {"status": "calibrated", "partition": "calibration", "split_id": data.split_id,
                   "labels": data.silver_id, "n_cases": len(universe), "objective": calibrate.OBJECTIVE,
                   "grid": {"gate_t": list(calibrate.GATE_GRID), "tau": list(calibrate.TAU_GRID),
                            "rho": list(calibrate.RHO_GRID)},
                   "partition_gs_share": fitted["gs_share"], "partition_good_share": fitted["good_share"],
                   "max_fp_share": max_fp_share, "chosen_verdicts_in_sample": fitted["verdicts"],
                   "curve_along_gate_t": fitted["curve"]}
    directory = generation.write_generation(
        root / "generation", generation_id=generation_id, retriever_dir=retriever_dir, gate_dir=gate_dir, shared_gate_dir=shared_dir,
        joint_gate=gate == "joint",
        thresholds_extra={"max_fp_share": max_fp_share, "calibration_verdicts_in_sample": fitted["verdicts"]},
        old_gate_csv=OLD_GATE_CSV if gate == "old" else None, thresholds=thresholds, calibration=calibration,
        split_id=data.split_id, silver_id=data.silver_id, uncommon_groups_txt=config.RR_PHASE1_UNCOMMON_GROUPS_TXT,
        max_length=scoring.trained_max_length(retriever_dir))

    predictions = decide.decide_all(all_scores, gate_by_case, data.has_text, thresholds, data.term_info, generation_id)
    predictions_dir = root / "predictions" if limit else config.PREDICTIONS_DIR
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = predictions_dir / f"{generation_id}_predictions.csv"
    io_utils.write_csv(predictions, predictions_csv)
    by_partition = {name: {"rows": int(predictions["case_id"].isin(cases).sum()),
                           "cases": predictions.loc[predictions["case_id"].isin(cases), "case_id"].nunique()}
                    for name, cases in data.split_cases.items()}
    return {"generation_id": generation_id, "generation_dir": str(directory), "predictions_csv": str(predictions_csv),
            "prediction_rows": len(predictions), "cases_predicted": predictions["case_id"].nunique(),
            "predictions_by_partition": by_partition,
            "thresholds": thresholds.as_dict(), "max_fp_share": max_fp_share,
            "calibration_gs_share": fitted["gs_share"], "calibration_good_share": fitted["good_share"],
            "calibration_verdicts_in_sample": fitted["verdicts"], "curve_along_gate_t": fitted["curve"]}


def curve_table(result: dict) -> str:
    """The gate_t curve of a freeze result as text: best eligible (tau, rho) per gate_t, shares in percent."""
    lines = ["gate_t   tau   rho    G+S   Good     FP  (calibration, in-sample)"]
    for row in result["curve_along_gate_t"]:
        if row["eligible"]:
            lines.append(f"{row['gate_t']:6.2f}  {row['best_tau']:4.2f}  {row['best_rho']:4.2f}  {100 * row['best_gs']:5.2f}  "
                         f"{100 * row['best_good']:5.2f}  {100 * row['best_fp_share']:5.2f}")
        else:
            lines.append(f"{row['gate_t']:6.2f}  ineligible: every (tau, rho) is over the FP cap")
    return "\n".join(lines)
