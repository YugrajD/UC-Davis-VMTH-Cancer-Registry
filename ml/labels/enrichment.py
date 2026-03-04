"""Enrich taxonomy label embeddings with the mean report embeddings of keyword-confirmed cases.

For each label term that has at least one keyword-confirmed case in
keyword_predictions.csv, the label embedding is blended toward the centroid of
confirmed-case report embeddings with a weight proportional to confirmation count:
  alpha = n / (n + smoothing)   (default smoothing=5)
Labels with few confirmed cases stay close to the original label embedding;
well-confirmed labels are pulled strongly toward the report centroid.

Labels with no keyword matches are unchanged.  Report embeddings are taken from
the embedding cache (mean across non-empty columns per case), so this requires no
additional PetBERT inference.

This approach is more effective than enriching with diagnosis text because the
cached report embeddings live in the same embedding space as what the presence
classifier sees at inference time — pulling each label toward the centroid of
real reports that contain that cancer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_enriched_label_embeddings(
    label_embeddings: np.ndarray,        # (M, 768) original label embeddings
    label_names: list[str],              # M term strings (e.g. "Hemangiosarcoma, NOS")
    keyword_predictions_csv: str,
    case_ids: list[str],                 # N case IDs in cache row order
    mean_report_embeddings: np.ndarray,  # (N, 768) cached mean report embeddings
    smoothing: float = 5.0,             # prior sample count; alpha = n / (n + smoothing)
) -> np.ndarray:
    """Return enriched (M, 768) label embeddings.

    For each label term that appears as ``matched_term`` in
    keyword_predictions.csv, the returned embedding is a weighted blend:

        alpha = n / (n + smoothing)
        enriched = (1 - alpha) * original_label_embedding
                 + alpha       * mean_report_embedding_for_confirmed_cases

    where n is the number of keyword-confirmed cases for that label.
    With smoothing=5: n=1 → alpha≈0.17, n=5 → alpha=0.50, n=50 → alpha≈0.91.

    Labels with no keyword matches are unchanged.  The result is float32.
    """
    df = pd.read_csv(keyword_predictions_csv)
    matched = df[df["matched_term"].notna() & (df["matched_term"] != "")]

    # Build case_id → row index for fast lookup.
    case_id_to_idx = {cid: i for i, cid in enumerate(case_ids)}

    # Group cache row indices by matched term.
    term_to_case_indices: dict[str, list[int]] = {}
    missing_cases = 0
    for _, row in matched.iterrows():
        term = str(row["matched_term"]).strip()
        case_id = str(row["case_id"]).strip() if pd.notna(row["case_id"]) else ""
        if not term or not case_id:
            continue
        idx = case_id_to_idx.get(case_id)
        if idx is None:
            missing_cases += 1
            continue
        term_to_case_indices.setdefault(term, []).append(idx)

    if missing_cases:
        print(f"  Warning: {missing_cases} keyword-matched rows had no matching case in the cache.")

    if not term_to_case_indices:
        print("  No keyword-matched cases found in cache — label embeddings unchanged.")
        return label_embeddings

    n_pairs = sum(len(v) for v in term_to_case_indices.values())
    print(
        f"  Enriching {len(term_to_case_indices)} labels using "
        f"cached report embeddings ({n_pairs} confirmed pairs)..."
    )

    label_idx = {name: i for i, name in enumerate(label_names)}
    enriched = label_embeddings.copy()
    n_enriched = 0
    for term, case_indices in term_to_case_indices.items():
        idx = label_idx.get(term)
        if idx is None:
            continue
        report_vecs = mean_report_embeddings[case_indices]  # (K, 768)
        mean_report = report_vecs.mean(axis=0)
        n = len(case_indices)
        alpha = n / (n + smoothing)
        enriched[idx] = (1.0 - alpha) * label_embeddings[idx] + alpha * mean_report
        n_enriched += 1

    print(f"  Enriched {n_enriched}/{len(label_names)} label embeddings.")
    return enriched.astype(np.float32)
