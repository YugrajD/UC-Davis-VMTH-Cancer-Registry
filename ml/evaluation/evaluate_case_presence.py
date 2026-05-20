"""Stage 1 evaluation — CasePresenceClassifier (cancer vs non-cancer).

Scores the binary case-level cancer presence classifier against ground truth
derived from the annotation CSV. A case is positive iff it has >=1 annotation
row with a non-empty matched_term (same definition used by
training/binary/build_case_presence_dataset.py).

Output files (written to --out-dir):
  case_presence_evaluation.csv         — per-case predictions + verdict (TP/TN/FP/FN)
  case_presence_evaluation_summary.csv — single overall row of metrics
  case_presence_evaluation_history.csv — appended one row per call
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.common import load_filter_ids, safe_div
from model.case_presence_classifier import CasePresenceClassifier
from utils.encoding import npz_col_key


def _load_cache_minimal(
    path: Path,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Read case_ids, mean_embeddings, and per-column embeddings + content masks."""
    data = np.load(path, allow_pickle=True)
    col_names = list(data["col_names"])
    col_embeddings = {col: data[f"col_{npz_col_key(col)}"] for col in col_names}
    col_has_content = {col: data[f"has_{npz_col_key(col)}"] for col in col_names}
    return (
        list(data["case_ids"]),
        data["mean_embeddings"],
        col_embeddings,
        col_has_content,
        col_names,
    )


def _load_cancer_case_ids(annotation_csv: Path) -> set[str]:
    """A case is cancer iff it has >=1 annotation row with non-empty matched_term."""
    cancer: set[str] = set()
    with open(annotation_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("matched_term", "").strip():
                cancer.add(row["case_id"])
    return cancer


def _verdict(true_pos: bool, pred_pos: bool) -> str:
    if true_pos and pred_pos:
        return "TP"
    if not true_pos and not pred_pos:
        return "TN"
    if pred_pos and not true_pos:
        return "FP"
    return "FN"


def evaluate_case_presence(
    classifier_path: Path,
    embedding_cache_path: Path,
    annotation_csv: Path,
    out_dir: Path,
    cases_txt: str = "",
    threshold: float = 0.5,
    history_label: str = "",
) -> dict:
    """Score CasePresenceClassifier against annotation-derived cancer labels.

    Returns the summary metrics dict (also written to summary CSV).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids, mean_embeddings, col_embeddings, col_has_content, col_names = \
        _load_cache_minimal(embedding_cache_path)
    print(f"  Loaded embedding cache: {len(case_ids)} cases, columns={col_names}.")

    # Match production: the gate feeds the 2304-dim concat-3 per-row view.
    # Exclude the "concat_3" alias so we re-concat the per-section views.
    concat_input_cols = [c for c in col_names if c != "concat_3"] or col_names
    col_emb_concat = np.concatenate(
        [np.where(col_has_content[col][:, None], col_embeddings[col], 0.0)
         for col in concat_input_cols],
        axis=1,
    ).astype(np.float32)

    filter_ids = load_filter_ids(cases_txt)
    if filter_ids is not None:
        keep_idx = [i for i, cid in enumerate(case_ids) if cid in filter_ids]
        case_ids = [case_ids[i] for i in keep_idx]
        mean_embeddings = mean_embeddings[keep_idx]
        col_emb_concat = col_emb_concat[keep_idx]
        print(f"  Case filter active — evaluating {len(case_ids)} cases.")

    cancer_case_ids = _load_cancer_case_ids(annotation_csv)
    print(f"  Annotation: {len(cancer_case_ids)} cancer cases (across all splits).")

    print(f"  Loading classifier from {classifier_path}...")
    clf = CasePresenceClassifier.load(classifier_path)
    if clf.emb_dim == mean_embeddings.shape[1]:
        feats = mean_embeddings
    elif clf.emb_dim == col_emb_concat.shape[1]:
        feats = col_emb_concat
    else:
        raise ValueError(
            f"CasePresenceClassifier emb_dim={clf.emb_dim} matches neither "
            f"mean_embeddings dim={mean_embeddings.shape[1]} nor "
            f"col_emb_concat dim={col_emb_concat.shape[1]}"
        )
    cancer_probs = clf.predict_proba(torch.from_numpy(feats)).numpy()
    del clf

    n = len(case_ids)
    rows: list[dict] = []
    tp = tn = fp = fn = 0
    n_cancer = n_non_cancer = 0
    y_true = np.zeros(n, dtype=np.int8)
    for i, cid in enumerate(case_ids):
        true_pos = cid in cancer_case_ids
        pred_pos = bool(cancer_probs[i] >= threshold)
        v = _verdict(true_pos, pred_pos)
        rows.append({
            "case_id": cid,
            "cancer_prob": f"{float(cancer_probs[i]):.4f}",
            "true_label": "cancer" if true_pos else "non-cancer",
            "pred_label": "cancer" if pred_pos else "non-cancer",
            "verdict": v,
        })
        y_true[i] = 1 if true_pos else 0
        if true_pos:
            n_cancer += 1
        else:
            n_non_cancer += 1
        if v == "TP":
            tp += 1
        elif v == "TN":
            tn += 1
        elif v == "FP":
            fp += 1
        else:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, n)
    try:
        auc = float(roc_auc_score(y_true, cancer_probs)) if n_cancer > 0 and n_non_cancer > 0 else 0.0
    except ValueError:
        auc = 0.0

    fields = ["case_id", "cancer_prob", "true_label", "pred_label", "verdict"]
    eval_path = out_dir / "case_presence_evaluation.csv"
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "threshold": round(threshold, 4),
        "n_total": n,
        "n_cancer": n_cancer,
        "n_non_cancer": n_non_cancer,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
    }
    summary_path = out_dir / "case_presence_evaluation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    history_path = out_dir / "case_presence_evaluation_history.csv"
    history_fields = ["timestamp", "label", *summary.keys()]
    history_row = {
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "label": history_label,
        **summary,
    }
    new_file = not history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        if new_file:
            writer.writeheader()
        writer.writerow(history_row)

    print(
        f"\n=== Stage 1 — CasePresenceClassifier (threshold={threshold:.2f}) ===\n"
        f"  Total: {n}  (cancer={n_cancer}, non-cancer={n_non_cancer})\n"
        f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}\n"
        f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  "
        f"Accuracy={accuracy:.4f}  AUC={auc:.4f}\n"
        f"\nWrote:\n  {eval_path}\n  {summary_path}\n  {history_path}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", default=config.CASE_PRESENCE_CLASSIFIER_PT)
    parser.add_argument("--embedding-cache", default=config.EMBEDDING_CACHE_NPZ)
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument("--out-dir",
                        default=f"{config.OUTPUT_EVALUATION_DIR}/{config.BEST_PREDICTIONS_SUBDIR}")
    parser.add_argument("--test-cases", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label", default="", help="Short description for history row.")
    args = parser.parse_args()
    evaluate_case_presence(
        classifier_path=Path(args.classifier),
        embedding_cache_path=Path(args.embedding_cache),
        annotation_csv=Path(args.annotation_csv),
        out_dir=Path(args.out_dir),
        cases_txt=args.test_cases,
        threshold=args.threshold,
        history_label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
