"""Fit {gate_t, tau, rho} by grid search and the review cut on the calibration partition.

Objective: G+S share of the evaluation.verdicts table of the predictions against the labels on the partition,
among the grid points whose false-positive share is within the FP cap; ties go to the higher Good share, then
to the first grid point.

Speed: a case's verdicts depend only on which of three row patterns it gets - rejected (one Non-Cancer row),
top-1 only, or top-1 + top-2 - and not on any other case. The three patterns are scored ONCE through
``verdicts.score`` on the rows ``decide`` writes; a threshold point then just picks one pattern per case and sums
per-case verdict counts. The numbers equal ``verdicts.summarize`` of the real predictions (see the tests).

The review cut is not part of the grid: after the thresholds are fitted it is set so that ``review_target_share``
of the accepted calibration cases fall below it (``fit_review_cut``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
from evaluation import verdicts
from generations import guards
from generations.splits import load_split
from report_mapping import generation, text
from report_mapping.inference import decide, scoring
from report_mapping.inference.decide import NEVER, Thresholds
from report_mapping.inference.scoring import RankedScores
from report_mapping.model import load_joint
from report_mapping.training import labels as labels_mod

GATE_GRID = tuple(sorted({round(0.05 * k, 2) for k in range(1, 20)} | {0.80}))
TAU_GRID = tuple(round(0.1 * k, 1) for k in range(1, 10)) + (NEVER,)
RHO_GRID = (0.0, 0.25, 0.5, 0.75)
OBJECTIVE = ("G+S share of the evaluation.verdicts table vs the labels on the calibration cases; "
             "ties -> higher Good share, then first grid point (gate_t, tau, rho order)")

FP_INDEX = verdicts.VERDICTS.index("false_positive")
REJECT, TOP1, TOP2 = 0, 1, 2
_PATTERN_THRESHOLDS = {REJECT: Thresholds(2.0, NEVER, 0.0), TOP1: Thresholds(0.0, NEVER, 0.0), TOP2: Thresholds(0.0, 0.0, 0.0)}


@dataclass(frozen=True)
class PatternCounts:
    gate_probs: np.ndarray   # (N,) float
    p1: np.ndarray           # (N,) rank-1 probability
    p2: np.ndarray           # (N,) rank-2 probability
    counts: np.ndarray       # (3 patterns, N, 5 verdicts) int
    fixed: np.ndarray        # (5,) verdict counts of labelled cases that were not scored (same in every pattern)

    def pattern(self, th: Thresholds) -> np.ndarray:
        passed = self.gate_probs >= th.gate_t
        ratio = self.p2 / np.where(self.p1 > 0, self.p1, 1.0)  # same arithmetic as decide.wants_second
        second = passed & (self.p1 > 0) & (self.p2 >= th.tau) & (ratio >= th.rho)
        return np.where(~passed, REJECT, np.where(second, TOP2, TOP1))

    def totals(self, th: Thresholds) -> np.ndarray:
        chosen = self.counts[self.pattern(th), np.arange(len(self.gate_probs))]
        return chosen.sum(axis=0) + self.fixed

    def objective(self, th: Thresholds) -> tuple[float, float]:
        """(G+S share, Good share) of the verdict rows at ``th``."""
        return _objective(self.totals(th))


def _objective(totals: np.ndarray) -> tuple[float, float]:
    n = int(totals.sum())
    return ((int(totals[0] + totals[1]) / n, int(totals[0]) / n) if n else (0.0, 0.0))


def _fp_share(totals: np.ndarray) -> float:
    n = int(totals.sum())
    return int(totals[FP_INDEX]) / n if n else 0.0


def verdict_summary(totals: np.ndarray) -> dict:
    """Counts and shares of the verdict rows, named as ``verdicts.summarize`` names them."""
    n = int(totals.sum())
    summary: dict = {"total": n}
    for verdict, count in zip(verdicts.VERDICTS, totals):
        summary[verdict] = int(count)
        summary[f"{verdict}_share"] = int(count) / n if n else 0.0
    summary["good_plus_slight"] = int(totals[0] + totals[1])
    summary["good_plus_slight_share"] = _objective(totals)[0]
    return summary


def build_pattern_counts(scores: RankedScores, labels: pd.DataFrame, universe_ids, uncommon_groups,
                         term_info) -> PatternCounts:
    """``labels``: (case_id, matched_term, matched_group). ``universe_ids``: every case the objective counts (the
    partition); labelled ones that were not scored (no report text) contribute FN rows in every pattern."""
    expectations = labels[labels["case_id"].isin(universe_ids)]
    index = {c: i for i, c in enumerate(scores.case_ids)}
    counts = np.zeros((3, len(index), len(verdicts.VERDICTS)), dtype=np.int64)
    fixed = np.zeros(len(verdicts.VERDICTS), dtype=np.int64)
    for pattern, th in _PATTERN_THRESHOLDS.items():
        predictions = decide.decide_all(scores, th, term_info, "calibration")
        table = verdicts.score(expectations, predictions, uncommon_groups)
        per_case = pd.crosstab(table["case_id"], table["verdict"]).reindex(columns=list(verdicts.VERDICTS), fill_value=0)
        scored = np.fromiter((c in index for c in per_case.index), dtype=bool, count=len(per_case))
        rows = [index[c] for c in per_case.index[scored]]
        counts[pattern, rows] = per_case.to_numpy()[scored]
        if pattern == REJECT:
            fixed = per_case.to_numpy()[~scored].sum(axis=0).astype(np.int64)
    return PatternCounts(np.asarray(scores.gate_probs, dtype=float), scores.probs[:, 0], scores.probs[:, 1], counts, fixed)


def fit_thresholds(pc: PatternCounts, max_fp_share: float | None = None) -> dict:
    """Grid search. Returns {"thresholds": Thresholds, "gs_share", "good_share", "verdicts", "curve"}: ``verdicts``
    is the chosen point's verdict counts and shares (``verdict_summary``); ``curve`` has, per gate_t, the best
    (tau, rho) there and its objective and FP share ("best_*") and the objective and FP share at the chosen
    (tau, rho) ("at_chosen_*"), for the "no cliff along gate_t" check.

    ``max_fp_share``: only grid points whose false-positive share of the verdict rows is <= it are eligible
    (objective and tie-break unchanged among them). A gate_t with no eligible point has "eligible": False and
    best_* None. Raises ValueError if no grid point is eligible; the cap is never relaxed."""
    best, best_key, best_totals = None, None, None
    per_gate: dict[float, tuple] = {}
    lowest_fp = None
    for gate_t in GATE_GRID:
        for tau in TAU_GRID:
            for rho in RHO_GRID:
                th = Thresholds(gate_t, tau, rho)
                totals = pc.totals(th)
                fp = _fp_share(totals)
                lowest_fp = fp if lowest_fp is None else min(lowest_fp, fp)
                if max_fp_share is not None and fp > max_fp_share:
                    continue
                key = _objective(totals)
                if best_key is None or key > best_key:
                    best, best_key, best_totals = th, key, totals
                if gate_t not in per_gate or key > per_gate[gate_t][0]:
                    per_gate[gate_t] = (key, th, fp)
    if best is None:
        raise ValueError(f"no grid point has a false-positive share <= {max_fp_share} "
                         f"(the lowest on the grid is {lowest_fp:.6f}); not relaxing the cap")
    curve = []
    for gate_t in GATE_GRID:
        at_totals = pc.totals(Thresholds(gate_t, best.tau, best.rho))
        at_gs, at_good = _objective(at_totals)
        at_fp = _fp_share(at_totals)
        if gate_t in per_gate:
            (gs, good), th, fp = per_gate[gate_t]
            row = {"gate_t": gate_t, "eligible": True, "best_tau": th.tau, "best_rho": th.rho, "best_gs": gs,
                   "best_good": good, "best_fp_share": fp}
        else:
            row = {"gate_t": gate_t, "eligible": False, "best_tau": None, "best_rho": None, "best_gs": None,
                   "best_good": None, "best_fp_share": None}
        curve.append({**row, "at_chosen_gs": at_gs, "at_chosen_good": at_good, "at_chosen_fp_share": at_fp})
    return {"thresholds": best, "gs_share": best_key[0], "good_share": best_key[1],
            "verdicts": verdict_summary(best_totals), "curve": curve}


def fit_review_cut(rank1_probs: np.ndarray, target_share: float) -> float:
    """The cut such that the share of the accepted calibration cases (``rank1_probs``: their rank-1 confidences)
    strictly below it is closest to ``target_share``; exact when the confidences are distinct, the nearest
    achievable share when they tie. The review queue holds the accepted cases below the cut."""
    if len(rank1_probs) == 0:
        raise ValueError("no accepted calibration case to fit the review cut on")
    ordered = np.sort(rank1_probs)
    candidates = np.unique(ordered)
    share_below = np.searchsorted(ordered, candidates, side="left") / len(ordered)
    return float(candidates[np.argmin(np.abs(share_below - target_share))])


def _check_not_trained_on(manifest: dict, used_ids: set[str]) -> None:
    """Calibration cases must not be in the train partition the generation was fitted on."""
    parent_split_id = (manifest.get("parents") or {}).get("split_id")
    leaked = sorted(used_ids & load_split(parent_split_id).train) if parent_split_id else []
    if leaked:
        raise guards.GuardViolation(f"{len(leaked)} case(s) are calibration inputs but in the train partition of "
                                    f"the generation's split {parent_split_id!r}: {', '.join(leaked[:5])}")


def calibrate(generation_dir: Path, labels: str, split_id: str, max_fp_share: float | None,
              review_target_share: float, device) -> dict:
    """Fit the thresholds and the review cut on ``split_id``'s calibration partition and write them into the
    generation (thresholds.json, manifest calibration block bound to its weights). ``labels``: a silver_id or
    labels CSV path. ``max_fp_share``: the FP cap on the grid (None: no cap). Returns the calibration block.

    A generation is calibrated once; new thresholds for the same weights are a new generation (recalibrate)."""
    directory = Path(generation_dir)
    manifest = generation.verify_spec(directory, require_calibrated=False)
    if manifest["calibration"]["status"] != "pending":
        raise generation.GenerationError(f"{directory} is already calibrated; recalibrate it into a new generation")

    partition_ids = load_split(split_id).calibration
    if not partition_ids:
        raise ValueError(f"split {split_id!r} has no calibration partition")
    table = labels_mod.load_labels_table(labels)
    expectations = table.assign(case_id=table["case_id"].astype(str).str.strip())
    expectations = expectations[expectations["case_id"].isin(partition_ids)]
    if expectations.empty:
        # e.g. a corrected-annotations table, which covers the train partition only: every grid point
        # would score 0 and the first one would be written as if fitted.
        raise ValueError(f"labels {labels!r} have no rows on the calibration partition of {split_id!r}; "
                         "calibrate against a silver generation")
    reports = text.read_reports()
    reports = reports[reports["case_id"].isin(partition_ids)]
    texts = text.build_report_texts(reports)
    scored = [i for i, t in enumerate(texts) if t]
    case_ids = [reports["case_id"].iloc[i] for i in scored]

    used_ids = set(reports["case_id"]) | set(expectations["case_id"])
    guards.check_all(split_id, calibration_case_ids=used_ids)
    _check_not_trained_on(manifest, used_ids)

    model, tokenizer = load_joint(directory)
    term_logits, gate_logits, over = scoring.logits_for(model.to(device), tokenizer, [texts[i] for i in scored],
                                                        manifest["model"]["max_length"], device)
    scores = scoring.rank(case_ids, model.terms, term_logits, gate_logits)
    uncommon_groups = frozenset(line.strip() for line in config.UNCOMMON_GROUPS_TXT.read_text(encoding="utf-8")
                                .splitlines() if line.strip())
    counts = build_pattern_counts(scores, expectations, partition_ids, uncommon_groups,
                                  generation.read_term_info(directory))
    fitted = fit_thresholds(counts, max_fp_share)
    thresholds = fitted["thresholds"]

    # The cut is fitted on the 2-decimal confidences the predictions CSV carries, so the queue rule on that CSV
    # reproduces the fitted share.
    accepted = scores.gate_probs >= thresholds.gate_t
    confidence = np.array([float(f"{p:.2f}") for p in scores.probs[accepted, 0]])
    review_cut = fit_review_cut(confidence, review_target_share)

    values = {**thresholds.as_dict(), "review_cut": review_cut}
    block = {
        "partition": "calibration", "split_id": split_id, "labels": labels, "n_cases": len(partition_ids),
        "n_cases_with_text": len(case_ids), "n_reports_over_max_length": over, "objective": OBJECTIVE,
        "grid": {"gate_t": list(GATE_GRID), "tau": list(TAU_GRID), "rho": list(RHO_GRID)},
        "values": values, "max_fp_share": max_fp_share, "review_target_share": review_target_share,
        "n_accepted": int(accepted.sum()), "review_share_below_cut": float((confidence < review_cut).mean()),
        "partition_gs_share": fitted["gs_share"], "partition_good_share": fitted["good_share"],
        "chosen_verdicts_in_sample": fitted["verdicts"], "curve_along_gate_t": fitted["curve"],
    }
    generation.write_calibration(directory, {**values, "max_fp_share": max_fp_share,
                                             "review_target_share": review_target_share}, block)
    return block
