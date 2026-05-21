"""Stage 3 evaluation — per-group LabelPresenceClassifier (within-group label scoring).

For each LP checkpoint in --classifier-dir, scores its in-scope test cases:
  • For LP 'Group X':  cases whose annotation has matched_group == X.
  • For LP 'Uncommon': cases whose annotation has matched_group ∈ uncommon_groups.

For each in-scope case, all labels in the group form (case, label) pairs:
  • Positive: annotation has matched_term == label_term.
  • Negative: otherwise.

This matches the training distribution from training/label_presence/build_training_pairs.py
(except every label in the group is evaluated, not just sampled negatives).

Output files (written to --out-dir):
  label_presence_evaluation.csv         — per (case, label) verdict
  label_presence_evaluation_summary.csv — per-LP + macro/micro
  label_presence_evaluation_history.csv — appended one row per call
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from ICD_labels import label_catalog_for_config
from evaluation.common import load_filter_ids, load_uncommon_groups, prf, safe_div
from model.label_presence_classifier import LabelPresenceClassifier
from utils.encoding import safe_filename


def _load_cache_minimal(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read case_ids, concat-3 report embeddings, and label_embeddings."""
    data = np.load(path, allow_pickle=True)
    if "col_concat_3" not in data.files:
        raise ValueError(
            f"Embedding cache {path} is missing the 'col_concat_3' key. "
            "Rebuild the cache by deleting it and re-running run_production.py."
        )
    return list(data["case_ids"]), data["col_concat_3"], data["label_embeddings"]


def _load_annotation(annotation_csv: Path) -> tuple[
    dict[str, dict[str, set[str]]], set[str]
]:
    """Return (case_id -> matched_group -> set of matched_terms, set of cancer case_ids)."""
    per_case_per_group: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    cancer: set[str] = set()
    with open(annotation_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row.get("matched_term", "").strip()
            grp = row.get("matched_group", "").strip()
            if not term:
                continue
            cid = row["case_id"]
            cancer.add(cid)
            if grp:
                per_case_per_group[cid][grp].add(term)
    return per_case_per_group, cancer


def _resolve_lp_files(
    classifier_dir: Path, all_groups: set[str], uncommon_groups: frozenset[str]
) -> dict[str, Path]:
    """Match each .pt file to a group name.

    Real groups: safe_filename(group_name).pt → group_name.
    Special: 'uncommon.pt' → 'Uncommon'.
    """
    candidates = {safe_filename(g): g for g in all_groups}
    if uncommon_groups:
        candidates["uncommon"] = "Uncommon"

    matches: dict[str, Path] = {}
    for pt_path in sorted(classifier_dir.glob("*.pt")):
        stem = pt_path.stem
        if stem in candidates:
            matches[candidates[stem]] = pt_path
        else:
            print(f"  Warning: LP checkpoint {pt_path.name} did not match any group; skipping.")
    return matches


def evaluate_label_presence(
    classifier_dir: Path,
    embedding_cache_path: Path,
    annotation_csv: Path,
    labels_csv: str,
    out_dir: Path,
    cases_txt: str = "",
    uncommon_groups_path: str = config.UNCOMMON_GROUPS_TXT,
    threshold: float = 0.5,
    history_label: str = "",
) -> dict:
    """Score every per-group LabelPresenceClassifier in classifier_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids, mean_embeddings, label_embeddings = _load_cache_minimal(embedding_cache_path)
    print(f"  Loaded embedding cache: {len(case_ids)} cases, {label_embeddings.shape[0]} labels.")

    catalog = label_catalog_for_config(labels_csv)
    if catalog.taxonomy_labels is None:
        raise ValueError("LabelCatalog has no taxonomy_labels — cannot evaluate.")
    labels = catalog.labels

    group_to_label_indices: dict[str, list[int]] = defaultdict(list)
    for j, tl in enumerate(catalog.taxonomy_labels):
        group_to_label_indices[tl.group].append(j)

    uncommon_groups = load_uncommon_groups(uncommon_groups_path)
    if uncommon_groups:
        uncommon_label_indices = [
            j for g in uncommon_groups for j in group_to_label_indices.get(g, [])
        ]
    else:
        uncommon_label_indices = []

    annotation_by_case, cancer_case_ids = _load_annotation(annotation_csv)
    print(f"  Annotation: {len(cancer_case_ids)} cancer cases.")

    filter_ids = load_filter_ids(cases_txt)
    case_id_to_idx = {cid: i for i, cid in enumerate(case_ids)}

    lp_files = _resolve_lp_files(classifier_dir, set(group_to_label_indices.keys()), uncommon_groups)
    print(f"  Found {len(lp_files)} LP checkpoints in {classifier_dir}.")

    eval_rows: list[dict] = []
    per_lp_summary: list[dict] = []
    micro_tp = micro_fp = micro_fn = 0
    macro_p_sum = macro_r_sum = macro_f1_sum = 0.0
    macro_n = 0
    n_lps_with_data = 0

    for lp_group, pt_path in lp_files.items():
        # Determine candidate label indices for this LP.
        if lp_group == "Uncommon":
            label_idxs = uncommon_label_indices
            in_scope_groups = uncommon_groups
        else:
            label_idxs = group_to_label_indices.get(lp_group, [])
            in_scope_groups = frozenset({lp_group})

        if not label_idxs:
            print(f"  Skipping {lp_group!r}: no candidate labels.")
            continue

        # In-scope cases: cancer test cases with any annotation in in_scope_groups.
        in_scope_cases = [
            cid for cid, grp_to_terms in annotation_by_case.items()
            if (filter_ids is None or cid in filter_ids)
            and cid in case_id_to_idx
            and any(g in grp_to_terms for g in in_scope_groups)
        ]
        if not in_scope_cases:
            print(f"  Skipping {lp_group!r}: no in-scope cases.")
            continue

        # Gather (case, label) ground truth.
        sel_idx = [case_id_to_idx[cid] for cid in in_scope_cases]
        case_embs = mean_embeddings[sel_idx]
        group_label_embs = label_embeddings[label_idxs]

        # Score.
        lp_model = LabelPresenceClassifier.load(pt_path)
        lp_model.eval()
        probs = lp_model.score_matrix(
            torch.from_numpy(case_embs), torch.from_numpy(group_label_embs)
        ).numpy()  # (n_cases, n_labels_in_group)
        del lp_model

        tp = fp = fn = 0
        n_pos_pairs = 0
        for ci, cid in enumerate(in_scope_cases):
            grp_to_terms = annotation_by_case[cid]
            # All terms annotated for this case in any in-scope group are positives.
            positive_terms: set[str] = set()
            for g in in_scope_groups:
                positive_terms |= grp_to_terms.get(g, set())
            for li, label_idx in enumerate(label_idxs):
                term = labels[label_idx]
                true_pos = term in positive_terms
                pred_pos = bool(probs[ci, li] >= threshold)
                if true_pos and pred_pos:
                    v = "TP"
                    tp += 1
                elif pred_pos:
                    v = "FP"
                    fp += 1
                elif true_pos:
                    v = "FN"
                    fn += 1
                else:
                    v = "TN"
                if true_pos:
                    n_pos_pairs += 1
                eval_rows.append({
                    "case_id": cid,
                    "group": lp_group,
                    "label": term,
                    "prob": f"{float(probs[ci, li]):.4f}",
                    "true_label": "1" if true_pos else "0",
                    "pred_label": "1" if pred_pos else "0",
                    "verdict": v,
                })

        p, r, f = prf(tp, fp, fn)
        per_lp_summary.append({
            "scope": lp_group,
            "n_cases": len(in_scope_cases),
            "n_pairs": len(in_scope_cases) * len(label_idxs),
            "support": n_pos_pairs,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        })
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        if n_pos_pairs > 0:
            macro_p_sum += p
            macro_r_sum += r
            macro_f1_sum += f
            macro_n += 1
        n_lps_with_data += 1

    macro_p = safe_div(macro_p_sum, macro_n)
    macro_r = safe_div(macro_r_sum, macro_n)
    macro_f1 = safe_div(macro_f1_sum, macro_n)
    micro_p, micro_r, micro_f1 = prf(micro_tp, micro_fp, micro_fn)

    # Write per-(case, label) eval CSV
    eval_path = out_dir / "label_presence_evaluation.csv"
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "group", "label", "prob",
                           "true_label", "pred_label", "verdict"],
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    summary_path = out_dir / "label_presence_evaluation_summary.csv"
    summary_fields = ["scope", "n_cases", "n_pairs", "support",
                      "tp", "fp", "fn", "precision", "recall", "f1"]
    aggregate_rows = [
        {"scope": "OVERALL_macro", "n_cases": "", "n_pairs": "",
         "support": sum(r["support"] for r in per_lp_summary),
         "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
         "precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        {"scope": "OVERALL_micro", "n_cases": "", "n_pairs": "",
         "support": sum(r["support"] for r in per_lp_summary),
         "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
         "precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows + sorted(per_lp_summary, key=lambda r: -r["support"]))

    history_path = out_dir / "label_presence_evaluation_history.csv"
    history_fields = [
        "timestamp", "label", "threshold", "n_lps_evaluated",
        "macro_p", "macro_r", "macro_f1",
        "micro_p", "micro_r", "micro_f1",
    ]
    history_row = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "label": history_label,
        "threshold": round(threshold, 4),
        "n_lps_evaluated": n_lps_with_data,
        "macro_p": round(macro_p, 4),
        "macro_r": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_p": round(micro_p, 4),
        "micro_r": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
    }
    new_file = not history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        if new_file:
            writer.writeheader()
        writer.writerow(history_row)

    print(
        f"\n=== Stage 3 — LabelPresenceClassifier (threshold={threshold:.2f}) ===\n"
        f"  LPs evaluated: {n_lps_with_data}/{len(lp_files)}\n"
        f"  Macro: P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}\n"
        f"  Micro: P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}\n"
        f"\nWrote:\n  {eval_path}\n  {summary_path}\n  {history_path}"
    )
    return history_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-dir", default=config.CHECKPOINT_LABEL_PRESENCE_DIR)
    parser.add_argument("--embedding-cache", default=config.EMBEDDING_CACHE_NPZ)
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument("--labels-csv", default=config.LABELS_CSV)
    parser.add_argument("--out-dir",
                        default=f"{config.OUTPUT_EVALUATION_DIR}/{config.BEST_PREDICTIONS_SUBDIR}")
    parser.add_argument("--test-cases", default="")
    parser.add_argument("--uncommon-groups", default=config.UNCOMMON_GROUPS_TXT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    evaluate_label_presence(
        classifier_dir=Path(args.classifier_dir),
        embedding_cache_path=Path(args.embedding_cache),
        annotation_csv=Path(args.annotation_csv),
        labels_csv=args.labels_csv,
        out_dir=Path(args.out_dir),
        cases_txt=args.test_cases,
        uncommon_groups_path=args.uncommon_groups,
        threshold=args.threshold,
        history_label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
