"""Enrich taxonomy label embeddings with averaged keyword-matched diagnosis embeddings.

For each label that has at least one keyword-matched diagnosis in
keyword_predictions.csv, the label embedding is replaced with the average of:
  - the original label text embedding  ("Hemangiosarcoma, NOS Blood vessel tumors")
  - the mean embedding of all keyword-matched diagnosis texts for that label

Labels with no keyword matches are unchanged.  This enrichment is computed once
(during the embedding-cache build step) and stored alongside the plain label
embeddings in the cache so that every subsequent training cycle uses the same
enriched representations without needing to re-run PetBERT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_enriched_label_embeddings(
    label_embeddings: np.ndarray,   # (M, 768) original label embeddings
    label_names: list[str],         # M term strings  (e.g. "Hemangiosarcoma, NOS")
    keyword_predictions_csv: str,
    tokenizer,
    model,
    *,
    device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Return enriched (M, 768) label embeddings.

    For each label term that appears as ``matched_term`` in
    keyword_predictions.csv, the returned embedding is:
        (original_label_embedding + mean_diagnosis_embedding) / 2

    All other labels are unchanged.  The result is float32.
    """
    from petbert_scan.embedding import embed_texts

    df = pd.read_csv(keyword_predictions_csv)
    matched = df[df["matched_term"].notna() & (df["matched_term"] != "")]

    # Group unique, non-empty diagnosis texts by matched term.
    term_to_diags: dict[str, list[str]] = {}
    for _, row in matched.iterrows():
        term = str(row["matched_term"]).strip()
        diag = str(row["diagnosis"]).strip() if pd.notna(row["diagnosis"]) else ""
        if term and diag:
            term_to_diags.setdefault(term, []).append(diag)

    if not term_to_diags:
        print("  No keyword-matched diagnoses found — label embeddings unchanged.")
        return label_embeddings

    unique_diags = list({d for diags in term_to_diags.values() for d in diags})
    print(
        f"  Enriching {len(term_to_diags)} labels using "
        f"{len(unique_diags)} unique diagnosis texts..."
    )

    diag_embs, _ = embed_texts(
        tokenizer,
        model,
        unique_diags,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        desc="Embedding diagnoses for label enrichment",
    )
    diag_emb_map = {d: diag_embs[i] for i, d in enumerate(unique_diags)}

    label_idx = {name: i for i, name in enumerate(label_names)}
    enriched = label_embeddings.copy()
    n_enriched = 0
    for term, diags in term_to_diags.items():
        idx = label_idx.get(term)
        if idx is None:
            continue
        diag_vecs = np.stack([diag_emb_map[d] for d in diags])  # (K, 768)
        mean_diag = diag_vecs.mean(axis=0)
        enriched[idx] = (label_embeddings[idx] + mean_diag) / 2.0
        n_enriched += 1

    print(f"  Enriched {n_enriched}/{len(label_names)} label embeddings.")
    return enriched.astype(np.float32)
