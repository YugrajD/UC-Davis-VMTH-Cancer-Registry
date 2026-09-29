"""Fit every inference threshold on one split partition (calibration only), from cached embeddings.

Replaces ``ml/scripts/sweep_lp_thresholds.py`` (per-LP thresholds),
``ml/scripts/sweep_tail_gate.py`` (tail gate) and the per-stage evaluators
``ml/evaluation/evaluate_{case_presence,groups,label_presence}.py`` (now the
small diagnostics block below).

1. **Per-LP thresholds**, carried from ``sweep_lp_thresholds.py``: for each
   label-presence head, every (case, label) pair of the in-scope annotated
   partition cases (``evaluate_label_presence.py:155-212``); F1 at every grid
   threshold; the lowest threshold with the strictly highest F1 wins
   (``sweep_lp_thresholds.py:41-61, 134-140`` with ``--beta 1``). Probabilities are
   rounded to 4 dp first, because legacy wrote them to CSV as ``:.4f``
   (``evaluate_label_presence.py:208``) and swept the re-read values.
2. **Gate, group and tail**, jointly over the small fixed grid below, with the
   LP thresholds from (1). Objective: per-code G+S share of the
   ``evaluation.verdicts`` table against the labels table on the partition;
   ties go to the higher ``good`` share, then to the first point in grid order.

Every case read is checked by ``guards.check_all`` (which runs
``check_calibration_inputs``), and against the generation's own training split.

The whole manifest ``calibration`` block is replaced, whatever placeholder was
there. Only ``calibration.status`` becomes "calibrated"; the top-level
``status`` (candidate / current) is intentionally left alone (promotion owns it).

Inference uses the public ``report_mapping.inference`` pieces on the cached
embeddings and never re-embeds. The gate is applied after the other stages: a
gate-rejected case always yields exactly one "Non-Cancer" row (its group
probabilities are zeroed, so no group clears a threshold > 0), so each (group,
tail) point is categorized once and every gate value is derived from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import config
import io_utils
from evaluation import verdicts
from generations import guards
from generations.manifest import read_manifest, write_manifest
from generations.splits import load_split
from report_mapping import sections
from report_mapping.inference import embedding_cache, predict, stages
from report_mapping.model import generation as generation_mod
from report_mapping.training.labels import load_labels_table

CALIBRATION_PARTITION = "calibration"
DIAGNOSTICS_NAME = "calibration_diagnostics.json"  # written beside thresholds.json


def _legacy_grid(start: float, stop: float, step: float) -> list[float]:
    # Built exactly as sweep_lp_thresholds.py:91-96 builds it (float accumulation, round to 4 dp).
    thresholds, t = [], start
    while t <= stop + 1e-9:
        thresholds.append(round(t, 4))
        t += step
    return thresholds


# The grid that actually produced the legacy lp_thresholds.json: "0.05,0.95,0.05". The script's --grid
# default and legacy-tree/training-guide.md Step 8 say 0.01, but every legacy value is a multiple of 0.05, and on the
# three-way-v1 calibration partition the 0.05 grid reproduces all 25 legacy values while 0.01 reproduces 12
# (WP5b real-data check). history/experiments/2026-05-10-per-lp-thresholds.md also records 0.05 steps.
LP_GRID = _legacy_grid(0.05, 0.95, 0.05)
GATE_GRID = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
GROUP_GRID = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)  # all > 0: the gate-after-stages shortcut needs it
assert all(t > 0 for t in GROUP_GRID)  # a gate-rejected case (group probs zeroed) must clear no group
# (tail_max_predictions, tail_max_group_prob_gap): sweep_tail_gate.py CONFIGS, verbatim.
TAIL_GRID = ((1, 1.00), (2, 0.02), (2, 0.05), (2, 0.08), (2, 0.10), (3, 0.10), (5, 1.00))

# Applied when no per-LP threshold covers a label (legacy production default).
LABEL_PRESENCE_FALLBACK = 0.5

LP_OBJECTIVE = ("per-LP F1 over the (case, label) pairs of partition cases annotated in the head's group(s), "
                "probabilities rounded to 4 dp; lowest grid threshold with the highest F1 "
                "(ml/scripts/sweep_lp_thresholds.py, beta=1)")
STAGE_OBJECTIVE = ("per-code G+S share of the evaluation.verdicts table vs the labels table on the partition; "
                   "ties -> higher good share, then first in grid order (group, tail, gate)")


# ---------------------------------------------------------------------------
# Objectives (pure; tested on synthetic scores)
# ---------------------------------------------------------------------------

def fit_lp_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """sweep_lp_thresholds.py's per-LP pick on (4-dp-rounded) probs and 0/1 targets."""
    targets = targets.astype(bool)
    best_t, best_score = None, -1.0
    for t in LP_GRID:
        pred = probs >= t
        tp = int((pred & targets).sum())
        fp = int((pred & ~targets).sum())
        fn = int((~pred & targets).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = (1 + 1.0) * p * r / (1.0 * p + r) if (1.0 * p + r) > 0 else 0.0  # f_beta, beta=1
        if score > best_score:
            best_score, best_t = score, t
    return best_t


def fit_stage_thresholds(score: Callable[[dict], tuple[float, float]]) -> tuple[dict, list[dict]]:
    """Joint grid search. ``score(thresholds) -> (gs_share, good_share)``.

    Returns (best thresholds, every grid point with its scores). Iterates gate
    innermost so a scorer can reuse one categorization per (group, tail) point.
    """
    best, best_key, table = None, None, []
    for group_t in GROUP_GRID:
        for k, gap in TAIL_GRID:
            for gate_t in GATE_GRID:
                point = {"case_presence_gate": gate_t, "group": group_t,
                         "tail_max_predictions": k, "tail_max_group_prob_gap": gap}
                gs, good = score(point)
                table.append({**point, "gs_share": gs, "good_share": good})
                if best_key is None or (gs, good) > best_key:
                    best, best_key = point, (gs, good)
    return best, table


# ---------------------------------------------------------------------------
# Partition inputs and inference (report_mapping.inference public functions only)
# ---------------------------------------------------------------------------

@dataclass
class _PartitionInputs:
    ids: list[str]                   # partition cases in report.csv order
    texts: list[str]                 # merged per-case text (keyword correction; never printed)
    concat_3: np.ndarray             # (N, 3*d) gate + LP view
    case_presence_probs: np.ndarray  # (N,) float32
    group_probs: np.ndarray          # (N, G), not gate-zeroed
    label_embeddings: np.ndarray
    cache: embedding_cache.EmbeddingCache


def _load_partition_inputs(gen, partition_ids: frozenset[str]) -> _PartitionInputs:
    reports = io_utils.read_csv(config.REPORT_CSV, encoding="latin-1")
    reports = sections.build_section_frame(reports)
    all_ids = reports["case_id"].map(sections.clean_text)
    reports = reports.loc[all_ids.isin(partition_ids).to_numpy()]
    ids = reports["case_id"].map(sections.clean_text).tolist()

    cache = predict.load_cached_embeddings(gen)  # raises on a miss: calibration never embeds
    cache_index = {cid: i for i, cid in enumerate(cache.case_ids)}
    missing = [cid for cid in ids if cid not in cache_index]
    if missing:
        raise ValueError(f"{len(missing)} partition case(s) missing from the embedding cache, e.g. {missing[:5]}")
    sel = [cache_index[cid] for cid in ids]

    concat_3 = cache.col_embeddings[sections.CONCAT_3_KEY][sel].astype(np.float32)
    _, cp_probs = stages.run_case_presence(gen.case_presence, concat_3, 0.0)
    group_probs = stages.run_group(gen.group_head, predict.group_classifier_input(cache, sel),
                                   np.ones(len(ids), bool))
    return _PartitionInputs(ids=ids, texts=sections.merged_texts(reports), concat_3=concat_3,
                            case_presence_probs=cp_probs, group_probs=group_probs,
                            label_embeddings=cache.label_embeddings, cache=cache)


def _categorize(gen, inp: _PartitionInputs, group_t: float, k: int, gap: float, lp_thresholds: dict) -> pd.DataFrame:
    """Every stage after the gate, with every case let through: predict.py's rows, plus each row's case position."""
    labels = [tl.term for tl in gen.taxonomy_labels]
    result = stages.categorize_cases(
        texts=inp.texts, lp_embeddings=inp.concat_3, label_embeddings=inp.label_embeddings,
        taxonomy_labels=gen.taxonomy_labels, labels=labels,
        group_probs=inp.group_probs, group_names=gen.group_names, group_threshold=group_t,
        tail_max_predictions=k, tail_max_group_prob_gap=gap,
        presence_mask=np.ones(len(inp.ids), dtype=bool), uncommon_groups=gen.uncommon_groups,
        label_presence_heads=gen.label_presence_heads,
        label_presence_fallback=LABEL_PRESENCE_FALLBACK,
        lp_thresholds=lp_thresholds,
    )
    rows = predict.prediction_rows(inp.ids, result, gen.taxonomy_labels, labels, inp.case_presence_probs,
                                   gen.generation_id)
    table = pd.DataFrame(rows, columns=["case_id", "predicted_term", "predicted_group"])
    position = {cid: i for i, cid in enumerate(inp.ids)}
    return table.assign(position=table["case_id"].map(position))


def _prediction_rows(inp: _PartitionInputs, passed_rows: pd.DataFrame, gate_t: float) -> pd.DataFrame:
    """Apply the gate last: a rejected case with text gets exactly one "Non-Cancer" row (as predict.py
    writes it), an empty-text case none, whatever the gate says."""
    passed = inp.case_presence_probs >= gate_t  # same float32 comparison as stages.run_case_presence
    rejected = [i for i in range(len(inp.ids)) if not passed[i] and inp.texts[i]]
    rejected_rows = pd.DataFrame({"case_id": [inp.ids[i] for i in rejected], "predicted_term": "Non-Cancer",
                                  "predicted_group": "Non-Cancer", "position": rejected})
    kept = passed_rows[passed[passed_rows["position"].to_numpy(dtype=int)]]
    table = pd.concat([kept, rejected_rows], ignore_index=True).sort_values("position", kind="stable")
    return table.drop(columns="position").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-LP rows and diagnostics
# ---------------------------------------------------------------------------

def _terms_by_case_group(expectations: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """case -> group -> annotated terms (evaluate_label_presence.py:51-67)."""
    out: dict[str, dict[str, set[str]]] = {}
    for cid, term, group in zip(expectations["case_id"], expectations["matched_term"], expectations["matched_group"]):
        term, group = sections.clean_text(term), sections.clean_text(group)
        if term and group:
            out.setdefault(cid, {}).setdefault(group, set()).add(term)
    return out


def _lp_pairs(gen, inp: _PartitionInputs, expectations: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """head -> (4-dp probs, 0/1 targets, n_cases) over its in-scope partition cases (evaluate_label_presence.py)."""
    labels = [tl.term for tl in gen.taxonomy_labels]
    group_to_labels: dict[str, list[int]] = {}
    for j, tl in enumerate(gen.taxonomy_labels):
        group_to_labels.setdefault(tl.group, []).append(j)
    cache_index = {cid: i for i, cid in enumerate(inp.cache.case_ids)}
    by_case = _terms_by_case_group(expectations)
    concat_all = inp.cache.col_embeddings[sections.CONCAT_3_KEY]

    pairs = {}
    for head_group, model in sorted(gen.label_presence_heads.items()):
        scope = gen.uncommon_groups if head_group == "Uncommon" else frozenset({head_group})
        label_idxs = [j for g in sorted(scope) for j in group_to_labels.get(g, [])]
        cases = [c for c, groups in by_case.items() if c in cache_index and any(g in groups for g in scope)]
        if not label_idxs or not cases:
            continue
        probs = model.score_matrix(
            torch.from_numpy(concat_all[[cache_index[c] for c in cases]].astype(np.float32)),
            torch.from_numpy(inp.label_embeddings[label_idxs]),
        ).numpy()
        positives = [set().union(*(by_case[c].get(g, set()) for g in scope)) for c in cases]
        targets = np.array([[labels[j] in positives[ci] for j in label_idxs] for ci in range(len(cases))])
        rounded = np.array([float(f"{float(p):.4f}") for p in probs.ravel()])
        pairs[head_group] = (rounded, targets.ravel(), len(cases))
    return pairs


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _diagnostics(gen, inp: _PartitionInputs, expectations: pd.DataFrame, gate_t: float,
                 lp_pairs: dict, lp_thresholds: dict) -> dict:
    """Gate P/R/F1, group top-k accuracy, per-LP P/R — the old per-stage evaluators, on the partition."""
    # Cancer = any row with a non-empty matched_term, group or not (evaluate_case_presence.py:50-56,
    # evaluate_groups.py:49-73); only the LP pairs also need the group.
    groups_by_case: dict[str, set[str]] = {}
    for cid, term, group in zip(expectations["case_id"], expectations["matched_term"], expectations["matched_group"]):
        if sections.clean_text(term):
            groups_by_case.setdefault(cid, set()).update({sections.clean_text(group)} - {""})
    cancer = np.array([cid in groups_by_case for cid in inp.ids])
    passed = inp.case_presence_probs >= gate_t
    gate = _prf(int((passed & cancer).sum()), int((passed & ~cancer).sum()), int((~passed & cancer).sum()))

    # evaluate_groups.py: cancer cases only, not gated; expected groups outside the head's vocabulary -> "Uncommon".
    vocab = set(gen.group_names)
    hits = {1: 0, 3: 0, 5: 0}
    for i in np.flatnonzero(cancer):
        expected = {g if g in vocab else "Uncommon" for g in groups_by_case[inp.ids[i]]}
        order = np.argsort(-inp.group_probs[i])
        for k in hits:
            hits[k] += bool({gen.group_names[g] for g in order[:k]} & expected)
    n_cancer = int(cancer.sum())
    group = {"n_cancer_cases": n_cancer, **{f"top{k}_acc": (h / n_cancer if n_cancer else 0.0) for k, h in hits.items()}}

    label_presence = {}
    for head_group, (probs, targets, n_cases) in lp_pairs.items():
        t = lp_thresholds[head_group]
        pred, targets = probs >= t, targets.astype(bool)
        label_presence[head_group] = {"threshold": t, "n_cases": n_cases, "support": int(targets.sum()),
                                      **_prf(int((pred & targets).sum()), int((pred & ~targets).sum()),
                                             int((~pred & targets).sum()))}
    return {"gate": {"threshold": gate_t, "n_cases": len(inp.ids), **gate}, "group": group,
            "label_presence": label_presence}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _check_not_trained_on(manifest: dict, used_ids: set[str]) -> None:
    """Calibration cases must not be in the train partition the generation was fitted on."""
    parent_split_id = (manifest.get("parents") or {}).get("split_id")
    leaked = sorted(used_ids & load_split(parent_split_id).train) if parent_split_id else []
    if leaked:
        raise guards.GuardViolation(f"{len(leaked)} case(s) are calibration inputs but in the train partition of "
                                    f"the generation's split {parent_split_id!r}: {', '.join(leaked[:5])}")


def calibrate(generation_dir, *, labels: str, split_id: str, partition: str = CALIBRATION_PARTITION,
              device: str = "cpu") -> dict:
    """Fit thresholds on ``partition`` of ``split_id``, write them into the generation, rewrite its manifest.

    ``labels``: a silver_id or labels-table CSV path. Returns the manifest's
    calibration block plus ``diagnostics``.
    """
    if partition != CALIBRATION_PARTITION:
        raise ValueError(f"thresholds are fitted on the {CALIBRATION_PARTITION!r} partition only (got {partition!r})")
    directory = Path(generation_dir)
    gen = generation_mod.load_generation(directory, allow_uncalibrated=True)  # verifies manifest + embedding fingerprint
    for head in (gen.case_presence, gen.group_head, *gen.label_presence_heads.values()):
        head.to(device)

    partition_ids = getattr(load_split(split_id), partition)
    if not partition_ids:
        raise ValueError(f"split {split_id!r} has no {partition!r} partition")
    table = load_labels_table(labels)
    expectations = table.assign(case_id=table["case_id"].astype(str).str.strip())
    expectations = expectations[expectations["case_id"].isin(partition_ids)]
    if expectations.empty:
        # e.g. a corrected-annotations table, which covers the train partition only: every grid point
        # would score 0 and the first one would be written as if fitted.
        raise ValueError(f"labels {labels!r} have no rows on the {partition!r} partition of {split_id!r}; "
                         "calibrate against a silver generation")
    inp = _load_partition_inputs(gen, partition_ids)

    used_ids = set(inp.ids) | set(expectations["case_id"])
    guards.check_all(split_id, calibration_case_ids=used_ids)
    _check_not_trained_on(gen.manifest, used_ids)

    lp_pairs = _lp_pairs(gen, inp, expectations)
    lp_thresholds = {g: fit_lp_threshold(probs, targets) for g, (probs, targets, _) in lp_pairs.items()}

    @lru_cache(maxsize=1)
    def categorized(group_t: float, k: int, gap: float) -> pd.DataFrame:
        return _categorize(gen, inp, group_t, k, gap, lp_thresholds)

    def score(point: dict) -> tuple[float, float]:
        passed_rows = categorized(point["group"], point["tail_max_predictions"], point["tail_max_group_prob_gap"])
        verdict_table = verdicts.score(expectations, _prediction_rows(inp, passed_rows, point["case_presence_gate"]),
                                       gen.uncommon_groups)
        return verdicts.share(verdict_table, verdicts.GOOD_PLUS_SLIGHT), verdicts.share(verdict_table, verdicts.GOOD)

    stage_values, grid_scores = fit_stage_thresholds(score)
    thresholds = {**stage_values, "label_presence_fallback": LABEL_PRESENCE_FALLBACK}
    gs, good = score(stage_values)

    diagnostics = {
        **_diagnostics(gen, inp, expectations, thresholds["case_presence_gate"], lp_pairs, lp_thresholds),
        "partition_gs_share": gs, "partition_good_share": good, "grid_scores": grid_scores,
    }
    block = {
        "status": "calibrated",
        "partition": partition,
        "split_id": split_id,
        "labels": str(labels),
        "n_cases": len(used_ids),
        "objective": {"label_presence": LP_OBJECTIVE, "stages": STAGE_OBJECTIVE},
        "grid": {"label_presence": LP_GRID, "case_presence_gate": list(GATE_GRID), "group": list(GROUP_GRID),
                 "tail": [list(p) for p in TAIL_GRID]},
        "values": {**thresholds, "label_presence": lp_thresholds},
        "partition_gs_share": gs,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    paths = generation_mod.generation_paths(directory)
    paths.thresholds_json.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8", newline="\n")
    paths.lp_thresholds_json.write_text(json.dumps(lp_thresholds, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8", newline="\n")
    (paths.thresholds_json.parent / DIAGNOSTICS_NAME).write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    # Manifest last: until it is rewritten, the new threshold files fail verification and nothing loads them.
    fields = {k: v for k, v in read_manifest(directory).items() if k not in ("files", "created_at", "git_sha")}
    write_manifest(directory, {**fields, "calibration": block})
    return {**block, "diagnostics": diagnostics}
