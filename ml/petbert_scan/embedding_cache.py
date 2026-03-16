"""Save and load the PetBERT embedding cache.

Since PetBERT is frozen, the per-column report embeddings and taxonomy label
embeddings are identical every training cycle.  This module lets petbert_scan
write them once and both petbert_scan and train_classifier reload them on
subsequent runs, skipping all PetBERT inference.

Cache format (npz):
  case_ids          object  (N,)      case_id strings in row order
  mean_embeddings   float32 (N, 768)  mean across non-empty columns per case
  token_counts      int32   (N,)      total non-padding tokens across columns
  col_names         object  (C,)      column names that were embedded
  col_<safe>        float32 (N, 768)  per-column embeddings
  has_<safe>        bool    (N,)      True where the column had content
  label_texts       object  (M,)      label strings ("{term} {group}")
  label_embeddings  float32 (M, 768)  label embeddings
  model_name        object  (1,)      HF model name — for cache invalidation
  report_mtime      float64 (1,)      report CSV mtime — for cache invalidation
  labels_mtime      float64 (1,)      labels CSV mtime — for cache invalidation
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _safe(col: str) -> str:
    """Convert a column name to a key safe for npz storage."""
    return col.replace(" ", "_").replace(",", "").replace("/", "_")


def save_cache(
    path: str,
    *,
    case_ids: list[str],
    col_embeddings: dict[str, np.ndarray],
    col_has_content: dict[str, np.ndarray],
    mean_embeddings: np.ndarray,
    token_counts: np.ndarray,
    label_texts: list[str],
    label_embeddings: np.ndarray,
    model_name: str,
    report_csv_path: str,
    labels_csv_path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    col_names = list(col_embeddings.keys())
    arrays: dict[str, np.ndarray] = {
        "case_ids":         np.array(case_ids, dtype=object),
        "mean_embeddings":  mean_embeddings,
        "token_counts":     token_counts,
        "col_names":        np.array(col_names, dtype=object),
        "label_texts":      np.array(label_texts, dtype=object),
        "label_embeddings": label_embeddings,
        "model_name":       np.array([model_name], dtype=object),
        "report_mtime":     np.array([os.path.getmtime(report_csv_path)]),
        "labels_mtime":     np.array([os.path.getmtime(labels_csv_path)]),
    }
    for col in col_names:
        s = _safe(col)
        arrays[f"col_{s}"] = col_embeddings[col]
        arrays[f"has_{s}"] = col_has_content[col]
    np.savez(path, **arrays)
    print(f"Saved embedding cache ({len(case_ids)} cases, {len(label_texts)} labels): {path}")


def load_cache(
    path: str,
    *,
    model_name: str,
    report_csv_path: str,
    labels_csv_path: str,
    expected_col_names: list[str] | None = None,
) -> dict | None:
    """Load the cache and validate freshness.

    Returns a dict with the cached data, or None if the cache is missing,
    stale (source files changed), or incompatible (different model/columns).
    """
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)

        if str(data["model_name"][0]) != model_name:
            print(f"Embedding cache miss: model changed → recomputing")
            return None
        if not os.path.exists(report_csv_path) or not os.path.exists(labels_csv_path):
            return None
        if abs(float(data["report_mtime"][0]) - os.path.getmtime(report_csv_path)) > 1:
            print("Embedding cache miss: report CSV modified → recomputing")
            return None
        if abs(float(data["labels_mtime"][0]) - os.path.getmtime(labels_csv_path)) > 1:
            print("Embedding cache miss: labels CSV modified → recomputing")
            return None

        col_names: list[str] = list(data["col_names"])
        if expected_col_names is not None and col_names != list(expected_col_names):
            print(f"Embedding cache miss: columns changed → recomputing")
            return None

        col_embeddings: dict[str, np.ndarray] = {}
        col_has_content: dict[str, np.ndarray] = {}
        for col in col_names:
            s = _safe(col)
            col_embeddings[col] = data[f"col_{s}"]
            col_has_content[col] = data[f"has_{s}"]

        return {
            "case_ids":         list(data["case_ids"]),
            "mean_embeddings":  data["mean_embeddings"],
            "token_counts":     data["token_counts"],
            "col_names":        col_names,
            "col_embeddings":   col_embeddings,
            "col_has_content":  col_has_content,
            "label_texts":      list(data["label_texts"]),
            "label_embeddings": data["label_embeddings"],
        }
    except Exception as e:
        print(f"Embedding cache load failed ({e}) → recomputing")
        return None
