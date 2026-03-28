"""Per-label score calibration for the PresenceClassifier.

After mean-centering the score matrix, different labels still have different score
variances. A label the model is uncertain about (low variance, scores clustered near 0)
loses argmax to higher-variance labels even when it is the correct answer.

Calibration adds a learned scalar offset b_l per label, optimized on the evaluation
set to maximize Good+Slight accuracy: the offset is chosen so that at least one label
in label l's ICD group wins argmax for cases whose ground truth is label l.

    calibrated_score_l = (score_l - mean_l) + b_l

Labels with fewer than min_cases labeled examples keep b_l = 0 to avoid overfitting.

Output: ml/output/calibration/label_offsets.json  {label_term: offset_float}
        (missing labels default to 0 at inference time)

Run via:
  python ml/scripts/run_training.py --mode calibrate [--device xpu] [--local-only]
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import config
from model.presence_classifier import PresenceClassifier
from production.petbert_pipeline import device_from_arg, load_cache
from ICD_labels import label_catalog_for_config


_OFFSET_GRID = np.linspace(-0.3, 0.3, 61, dtype=np.float32)
_MIN_CASES = 10


def _load_ground_truth(annotation_csv: str) -> dict[str, set[str]]:
    """Return {case_id: set of matched_term} from annotation CSV."""
    case_terms: dict[str, set[str]] = defaultdict(set)
    with open(annotation_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row["matched_term"].strip()
            if term:
                case_terms[row["case_id"]].add(term)
    return dict(case_terms)


def calibrate(
    *,
    annotation_csv: str = config.KEYWORD_ANNOTATION_CSV,
    model_path: str | None = None,
    cache_path: str = config.EMBEDDING_CACHE_NPZ,
    out_path: str = config.CALIBRATION_OFFSETS_JSON,
    device_arg: str = "auto",
    min_cases: int = _MIN_CASES,
) -> None:
    """Compute per-label score offsets and save to out_path.

    Args:
        annotation_csv: Verified annotation CSV (ground truth).
        model_path:     Path to PresenceClassifier .pt file. None = auto-detect best.
        cache_path:     Path to embedding cache npz.
        out_path:       Where to write label_offsets.json.
        device_arg:     Compute device ("auto", "cpu", "cuda", "mps", "xpu").
        min_cases:      Labels with fewer GT cases keep offset = 0 (overfitting guard).
    """
    device = device_from_arg(device_arg)

    # ── Resolve best classifier if not specified ──────────────────────────────
    if model_path is None:
        model_path, subdir = config.best_checkpoint_info()
        if model_path is None:
            raise RuntimeError("No classifier checkpoint found. Train one first.")
    else:
        subdir = "contrastive" if "contrastive" in model_path else "binary"

    model_name = (
        config.CHECKPOINT_CONTRASTIVE_DIR if subdir == "contrastive"
        else "SAVSNET/PetBERT"
    )

    # ── Load embedding cache ───────────────────────────────────────────────────
    cache = load_cache(
        cache_path,
        model_name=model_name,
        report_csv_path=config.REPORTS_CSV,
        labels_csv_path=config.LABELS_CSV,
    )
    if cache is None:
        raise RuntimeError(
            f"Embedding cache not found or stale: {cache_path}\n"
            "Run one training cycle first to build it."
        )

    case_ids: list[str] = cache["case_ids"]
    col_names: list[str] = cache["col_names"]
    col_embeddings: dict[str, np.ndarray] = cache["col_embeddings"]
    col_has_content: dict[str, np.ndarray] = cache["col_has_content"]
    # Use enriched label embeddings when available — matches pipeline.py behaviour.
    label_embeddings: np.ndarray = (
        cache["enriched_label_embeddings"]
        if cache.get("enriched_label_embeddings") is not None
        else cache["label_embeddings"]
    )  # (M, 768)

    # ── Build per-column concat embedding — mirrors pipeline.py exactly ───────
    col_emb_concat = np.concatenate(
        [np.where(col_has_content[col][:, None], col_embeddings[col], 0.0)
         for col in col_names],
        axis=1,
    ).astype(np.float32)  # (N, n_cols * 768)

    # ── Load taxonomy labels (same order as cache) ────────────────────────────
    label_catalog = label_catalog_for_config(config.LABELS_CSV)
    labels: list[str] = label_catalog.labels  # (M,)
    label_index: dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

    if len(labels) != label_embeddings.shape[0]:
        raise RuntimeError(
            f"Label count mismatch: catalog has {len(labels)} labels but "
            f"cache has {label_embeddings.shape[0]}. Delete the cache and rebuild."
        )

    # ── Compute score matrix (N, M) ───────────────────────────────────────────
    print(f"Loading classifier from {model_path}...")
    classifier = PresenceClassifier.load(model_path)
    classifier.to(device)
    print(f"Computing score matrix ({len(case_ids)} cases × {len(labels)} labels)...")
    score_matrix = classifier.score_matrix(
        torch.from_numpy(col_emb_concat),
        torch.from_numpy(label_embeddings),
    ).numpy()  # (N, M) float32
    classifier.cpu()
    del classifier

    # ── Mean-center — mirrors run_categorization() exactly ────────────────────
    finite_mask = np.isfinite(score_matrix)
    label_means = (
        np.where(finite_mask, score_matrix, 0.0).sum(axis=0)
        / np.maximum(finite_mask.sum(axis=0), 1)
    )
    centered = score_matrix - label_means[np.newaxis, :]  # (N, M)
    del score_matrix

    # ── Build group/term maps ─────────────────────────────────────────────────
    label_to_group: list[str] = [tl.group for tl in label_catalog.taxonomy_labels]
    term_to_group: dict[str, str] = {tl.term: tl.group for tl in label_catalog.taxonomy_labels}

    # ── Build case index and ground truth mappings ────────────────────────────
    case_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(case_ids)}
    print(f"Loading ground truth from {annotation_csv}...")
    case_terms = _load_ground_truth(annotation_csv)

    # Per-annotated-row GT groups and label_to_gt_rows index.
    gt_groups_by_row: dict[int, set[str]] = {}
    label_to_gt_rows: dict[int, list[int]] = defaultdict(list)
    for cid, terms in case_terms.items():
        row_idx = case_id_to_idx.get(cid)
        if row_idx is None:
            continue
        gt_groups_by_row[row_idx] = {term_to_group[t] for t in terms if t in term_to_group}
        for term in terms:
            l_idx = label_index.get(term)
            if l_idx is not None:
                label_to_gt_rows[l_idx].append(row_idx)

    annotated_rows_arr = np.array(sorted(gt_groups_by_row), dtype=np.int32)  # (n_ann,)

    # ── Precompute global baseline state ──────────────────────────────────────
    row_max = centered.max(axis=1)            # (N,)
    current_argmax = centered.argmax(axis=1)  # (N,)

    # Is the current winner's group the GT group for each annotated row?
    currently_correct_arr = np.array(
        [label_to_group[current_argmax[r]] in gt_groups_by_row[r]
         for r in annotated_rows_arr],
        dtype=bool,
    )

    # ── Grid-search per-label offsets — net Good+Slight gain across ALL cases ─
    # For each candidate offset b applied to label L:
    #   - Only rows where 0 < margin < b are affected (L steals the argmax win).
    #   - gain: affected row was wrong-group, L's group is the GT group.
    #   - loss: affected row was right-group, L's group is not the GT group.
    # Keep offset b only if (gains - losses) > 0, ensuring the label is net helpful.
    print(
        f"Optimizing offsets for {len(label_to_gt_rows)} labels "
        f"(min {min_cases} cases each)..."
    )
    offsets: dict[str, float] = {}
    n_calibrated = 0
    n_skipped_rare = 0

    for l_idx, gt_rows in label_to_gt_rows.items():
        if len(gt_rows) < min_cases:
            n_skipped_rare += 1
            continue

        l_group = label_to_group[l_idx]

        # Is L's group the GT group for each annotated row?
        l_correct_arr = np.array(
            [l_group in gt_groups_by_row[r] for r in annotated_rows_arr],
            dtype=bool,
        )

        # margin[i] = how far L's score is below the current winner (0 if L already wins).
        margin_ann = row_max[annotated_rows_arr] - centered[annotated_rows_arr, l_idx]
        not_already_winner = margin_ann > 0

        # Gain: adding b converts a wrong-group prediction to L's (correct) group.
        # Loss: adding b converts a right-group prediction to L's (wrong) group.
        gain_arr = not_already_winner & (~currently_correct_arr) & l_correct_arr
        loss_arr = not_already_winner & currently_correct_arr & (~l_correct_arr)

        best_b = 0.0
        best_net = 0  # offset only applied if strictly net-positive
        for b in _OFFSET_GRID:
            if b <= 0.0:
                continue
            wins = not_already_winner & (margin_ann < b)
            net = int(gain_arr[wins].sum()) - int(loss_arr[wins].sum())
            if net > best_net:
                best_net = net
                best_b = float(b)

        if best_b != 0.0:
            offsets[labels[l_idx]] = best_b
            n_calibrated += 1

    # ── Project Good+Slight accuracy change on annotation set ─────────────────
    offset_arr = np.zeros(len(labels), dtype=np.float32)
    for term, b in offsets.items():
        offset_arr[label_index[term]] = b
    calibrated = centered + offset_arr[np.newaxis, :]

    baseline_argmax = centered.argmax(axis=1)
    calibrated_argmax = calibrated.argmax(axis=1)

    baseline_correct = 0
    calibrated_correct = 0
    n_gt_cases = len(gt_groups_by_row)
    for row_idx, gt_groups in gt_groups_by_row.items():
        base_label = labels[baseline_argmax[row_idx]]
        if term_to_group.get(base_label, "") in gt_groups:
            baseline_correct += 1
        cal_label = labels[calibrated_argmax[row_idx]]
        if term_to_group.get(cal_label, "") in gt_groups:
            calibrated_correct += 1

    print(f"\n=== Calibration Results ===")
    print(f"  Labels calibrated : {n_calibrated}")
    print(f"  Labels skipped    : {n_skipped_rare} (fewer than {min_cases} GT cases)")
    if n_gt_cases > 0:
        print(
            f"  Baseline  Good+Slight: {baseline_correct}/{n_gt_cases} "
            f"({100 * baseline_correct / n_gt_cases:.1f}%)"
        )
        print(
            f"  Calibrated Good+Slight: {calibrated_correct}/{n_gt_cases} "
            f"({100 * calibrated_correct / n_gt_cases:.1f}%)"
        )
        delta = calibrated_correct - baseline_correct
        print(f"  Net change           : {delta:+d} cases ({100 * delta / n_gt_cases:+.2f}pp)")

    if offsets:
        vals = list(offsets.values())
        print(
            f"\n  Offset distribution: "
            f"min={min(vals):.3f}  max={max(vals):.3f}  "
            f"mean={sum(vals)/len(vals):.3f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(offsets, f, indent=2, sort_keys=True)
    print(f"\nSaved {len(offsets)} non-zero label offsets -> {out_path}")
    print("Apply at inference time with --calibration-offsets in run_production.py")
