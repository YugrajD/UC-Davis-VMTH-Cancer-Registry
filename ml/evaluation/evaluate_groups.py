"""Stage 2 evaluation — GroupClassifier (multi-label group prediction).

Scores the GroupClassifier on cancer test cases only (cases with >=1 annotation
row containing a non-empty matched_term). Cases that the CasePresenceClassifier
might gate out are still evaluated here — this isolates Stage 2 performance from
Stage 1 errors.

Per case, the expected group set is derived from the annotation CSV. Any
expected matched_group that is in the uncommon-groups list is replaced by the
literal token "Uncommon" to align with the GroupClassifier output vocabulary.

Output files (written to --out-dir):
  groups_evaluation.csv         — per (case, group) verdict (TP/FP/FN)
  groups_evaluation_summary.csv — per-group + overall macro/micro + top-k accuracy
  groups_evaluation_history.csv — appended one row per call
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
from evaluation.common import load_filter_ids, load_uncommon_groups, prf, safe_div
from model.group_classifier import GroupClassifier
from utils.encoding import npz_col_key


def _load_cache_minimal(path: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Read case_ids, mean_embeddings, per-column embeddings + content masks."""
    data = np.load(path, allow_pickle=True)
    col_names = list(data["col_names"])
    col_embeddings = {col: data[f"col_{npz_col_key(col)}"] for col in col_names}
    col_has_content = {col: data[f"has_{npz_col_key(col)}"] for col in col_names}
    return list(data["case_ids"]), data["mean_embeddings"], col_embeddings, col_has_content, col_names


def _load_annotation_groups(
    annotation_csv: Path, uncommon_groups: frozenset[str]
) -> tuple[dict[str, set[str]], set[str]]:
    """Return (case_id -> expected groups, set of cancer case_ids).

    Uncommon matched_groups are folded into the literal token "Uncommon" to
    align with the GroupClassifier output vocabulary.
    """
    expected: dict[str, set[str]] = defaultdict(set)
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
                if grp in uncommon_groups:
                    expected[cid].add("Uncommon")
                else:
                    expected[cid].add(grp)
    return expected, cancer


def evaluate_groups(
    classifier_path: Path,
    embedding_cache_path: Path,
    annotation_csv: Path,
    out_dir: Path,
    cases_txt: str = "",
    uncommon_groups_path: str = config.UNCOMMON_GROUPS_TXT,
    threshold: float = 0.85,
    history_label: str = "",
) -> dict:
    """Score GroupClassifier on cancer test cases."""
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids, mean_embeddings, col_embeddings, col_has_content, col_names = \
        _load_cache_minimal(embedding_cache_path)
    print(f"  Loaded embedding cache: {len(case_ids)} cases, columns={col_names}.")

    uncommon_groups = load_uncommon_groups(uncommon_groups_path)
    if uncommon_groups:
        print(f"  Uncommon groups loaded: {len(uncommon_groups)} groups.")

    expected_per_case, cancer_case_ids = _load_annotation_groups(annotation_csv, uncommon_groups)
    print(f"  Annotation: {len(cancer_case_ids)} cancer cases (across all splits).")

    filter_ids = load_filter_ids(cases_txt)

    # Build col_emb_concat in the same order as production (pipeline.py:200-204).
    col_emb_concat = np.concatenate(
        [np.where(col_has_content[col][:, None], col_embeddings[col], 0.0) for col in col_names],
        axis=1,
    ).astype(np.float32)

    # Filter to (test split if requested) ∩ cancer cases.
    keep_idx = [
        i for i, cid in enumerate(case_ids)
        if cid in cancer_case_ids and (filter_ids is None or cid in filter_ids)
    ]
    case_ids = [case_ids[i] for i in keep_idx]
    col_emb_concat = col_emb_concat[keep_idx]
    mean_embeddings = mean_embeddings[keep_idx]
    print(f"  Evaluating {len(case_ids)} cancer cases (intersection with filter).")

    print(f"  Loading classifier from {classifier_path}...")
    model, group_names = GroupClassifier.load(classifier_path)
    # Use mean_embeddings if model expects 768-dim input; else col_emb_concat.
    if model.emb_dim == mean_embeddings.shape[1]:
        feats = mean_embeddings
    elif model.emb_dim == col_emb_concat.shape[1]:
        feats = col_emb_concat
    else:
        raise ValueError(
            f"GroupClassifier emb_dim={model.emb_dim} matches neither "
            f"mean_embeddings dim={mean_embeddings.shape[1]} nor "
            f"col_emb_concat dim={col_emb_concat.shape[1]}"
        )
    group_probs = model.predict_proba(torch.from_numpy(feats)).numpy()
    del model

    n = len(case_ids)
    G = len(group_names)
    group_idx = {g: i for i, g in enumerate(group_names)}

    # Per-group counters
    per_group_tp = [0] * G
    per_group_fp = [0] * G
    per_group_fn = [0] * G

    # Top-k counters
    topk_hits = {1: 0, 3: 0, 5: 0}
    exact_match_hits = 0

    eval_rows: list[dict] = []
    for i, cid in enumerate(case_ids):
        probs = group_probs[i]
        predicted_set = {group_names[g] for g in range(G) if probs[g] >= threshold}
        expected_set = expected_per_case.get(cid, set())

        # TP: in both, FP: predicted only, FN: expected only
        for g_name in predicted_set | expected_set:
            in_pred = g_name in predicted_set
            in_exp = g_name in expected_set
            if in_pred and in_exp:
                v = "TP"
                if g_name in group_idx:
                    per_group_tp[group_idx[g_name]] += 1
            elif in_pred:
                v = "FP"
                if g_name in group_idx:
                    per_group_fp[group_idx[g_name]] += 1
            else:
                v = "FN"
                if g_name in group_idx:
                    per_group_fn[group_idx[g_name]] += 1
            prob_val = float(probs[group_idx[g_name]]) if g_name in group_idx else 0.0
            eval_rows.append({
                "case_id": cid,
                "group": g_name,
                "prob": f"{prob_val:.4f}",
                "true_label": "1" if in_exp else "0",
                "pred_label": "1" if in_pred else "0",
                "verdict": v,
            })

        # Top-k accuracy: top k by prob ranking, intersect with expected.
        order = np.argsort(-probs)
        for k in (1, 3, 5):
            topk_groups = {group_names[g] for g in order[:k].tolist()}
            if topk_groups & expected_set:
                topk_hits[k] += 1

        if predicted_set == expected_set:
            exact_match_hits += 1

    # Per-group metrics
    group_summary: list[dict] = []
    macro_p_sum = macro_r_sum = macro_f1_sum = 0.0
    macro_n = 0
    for g_i, g_name in enumerate(group_names):
        support = per_group_tp[g_i] + per_group_fn[g_i]
        p, r, f = prf(per_group_tp[g_i], per_group_fp[g_i], per_group_fn[g_i])
        group_summary.append({
            "scope": g_name,
            "support": support,
            "tp": per_group_tp[g_i],
            "fp": per_group_fp[g_i],
            "fn": per_group_fn[g_i],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        })
        if support > 0:
            macro_p_sum += p
            macro_r_sum += r
            macro_f1_sum += f
            macro_n += 1

    macro_p = safe_div(macro_p_sum, macro_n)
    macro_r = safe_div(macro_r_sum, macro_n)
    macro_f1 = safe_div(macro_f1_sum, macro_n)
    micro_tp = sum(per_group_tp)
    micro_fp = sum(per_group_fp)
    micro_fn = sum(per_group_fn)
    micro_p, micro_r, micro_f1 = prf(micro_tp, micro_fp, micro_fn)

    top1_acc = safe_div(topk_hits[1], n)
    top3_acc = safe_div(topk_hits[3], n)
    top5_acc = safe_div(topk_hits[5], n)
    exact_match_acc = safe_div(exact_match_hits, n)

    # Write per-(case, group) CSV
    eval_path = out_dir / "groups_evaluation.csv"
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "group", "prob", "true_label", "pred_label", "verdict"],
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    # Write summary CSV
    summary_path = out_dir / "groups_evaluation_summary.csv"
    summary_fields = ["scope", "support", "tp", "fp", "fn", "precision", "recall", "f1"]
    aggregate_rows = [
        {"scope": "OVERALL_macro", "support": sum(per_group_tp) + sum(per_group_fn),
         "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
         "precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        {"scope": "OVERALL_micro", "support": sum(per_group_tp) + sum(per_group_fn),
         "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
         "precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        {"scope": "top1_acc", "support": n, "tp": topk_hits[1], "fp": 0, "fn": 0,
         "precision": "", "recall": "", "f1": round(top1_acc, 4)},
        {"scope": "top3_acc", "support": n, "tp": topk_hits[3], "fp": 0, "fn": 0,
         "precision": "", "recall": "", "f1": round(top3_acc, 4)},
        {"scope": "top5_acc", "support": n, "tp": topk_hits[5], "fp": 0, "fn": 0,
         "precision": "", "recall": "", "f1": round(top5_acc, 4)},
        {"scope": "exact_match_acc", "support": n, "tp": exact_match_hits, "fp": 0, "fn": 0,
         "precision": "", "recall": "", "f1": round(exact_match_acc, 4)},
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows + sorted(group_summary, key=lambda r: -r["support"]))

    # History row
    history_path = out_dir / "groups_evaluation_history.csv"
    history_fields = [
        "timestamp", "label", "threshold", "n_cancer_cases",
        "macro_p", "macro_r", "macro_f1",
        "micro_p", "micro_r", "micro_f1",
        "top1_acc", "top3_acc", "top5_acc", "exact_match_acc",
    ]
    history_row = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "label": history_label,
        "threshold": round(threshold, 4),
        "n_cancer_cases": n,
        "macro_p": round(macro_p, 4),
        "macro_r": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_p": round(micro_p, 4),
        "micro_r": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "top1_acc": round(top1_acc, 4),
        "top3_acc": round(top3_acc, 4),
        "top5_acc": round(top5_acc, 4),
        "exact_match_acc": round(exact_match_acc, 4),
    }
    new_file = not history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        if new_file:
            writer.writeheader()
        writer.writerow(history_row)

    print(
        f"\n=== Stage 2 — GroupClassifier (threshold={threshold:.2f}) ===\n"
        f"  N cancer cases: {n}\n"
        f"  Macro: P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}  "
        f"(over {macro_n} groups with support)\n"
        f"  Micro: P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}\n"
        f"  Top-1={top1_acc:.4f}  Top-3={top3_acc:.4f}  Top-5={top5_acc:.4f}  "
        f"Exact-match={exact_match_acc:.4f}\n"
        f"\nWrote:\n  {eval_path}\n  {summary_path}\n  {history_path}"
    )
    return history_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier",
                        default=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_best.pt")
    parser.add_argument("--embedding-cache", default=config.EMBEDDING_CACHE_NPZ)
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument("--out-dir",
                        default=f"{config.OUTPUT_EVALUATION_DIR}/{config.BEST_PREDICTIONS_SUBDIR}")
    parser.add_argument("--test-cases", default="")
    parser.add_argument("--uncommon-groups", default=config.UNCOMMON_GROUPS_TXT)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    evaluate_groups(
        classifier_path=Path(args.classifier),
        embedding_cache_path=Path(args.embedding_cache),
        annotation_csv=Path(args.annotation_csv),
        out_dir=Path(args.out_dir),
        cases_txt=args.test_cases,
        uncommon_groups_path=args.uncommon_groups,
        threshold=args.threshold,
        history_label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
