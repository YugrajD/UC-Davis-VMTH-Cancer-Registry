"""Output writers for scan rows, categories, neighbors, embeddings, and summary."""

import json
import os

import numpy as np
import pandas as pd

from .types import ScanOutputs, TaskMode
from .utils import ensure_dir


def build_outputs(out_dir: str, task: TaskMode) -> ScanOutputs:
    ensure_dir(out_dir)
    neighbors_csv = (
        os.path.join(out_dir, "petbert_scan_neighbors.csv")
        if task in ("neighbors", "both")
        else None
    )
    return ScanOutputs(
        rows_csv=os.path.join(out_dir, "petbert_scan_rows.csv"),
        categories_csv=os.path.join(out_dir, "petbert_scan_categories.csv"),
        neighbors_csv=neighbors_csv,
        npz=os.path.join(out_dir, "petbert_scan_embeddings.npz"),
        summary_json=os.path.join(out_dir, "petbert_scan_summary.json"),
    )


def write_rows_csv(
    *,
    path: str,
    row_count: int,
    ids: list[str],
    texts: list[str],
    id_col: str,
    text_col: str,
    char_lens: np.ndarray,
    token_counts: np.ndarray,
    final_labels: list[str],
    final_indices: list[int],
    final_scores: list[float],
    methods: list[str],
    pca_2d: np.ndarray,
    matched_terms: list[str],
    matched_groups: list[str],
    matched_codes: list[str],
    auxiliary_labels: list[str],
    original_row_indices: list[int],
    diagnosis_indices: list[int],
    original_texts: list[str],
) -> pd.DataFrame:
    """Write row-level output with one row per sub-diagnosis.

    When a clinical entry contains multiple numbered diagnoses (e.g.
    ``"1) Osteosarcoma 2) Cystitis"``), each sub-diagnosis produces its own
    output row.  The ``row_index`` column links back to the original CSV row,
    and ``diagnosis_index`` identifies the sub-diagnosis within that entry
    (1-based, matching the ``1)``, ``2)`` numbering).

    The ``text_col`` column contains the individual sub-diagnosis text that
    was actually embedded and matched, while ``original_text`` preserves the
    full unsplit clinical entry for reference.
    """
    rows_df = pd.DataFrame(
        {
            "row_index": original_row_indices,
            "diagnosis_index": diagnosis_indices,
            id_col: ids,
            text_col: texts,
            "original_text": original_texts,
            "char_len": char_lens,
            "token_count": token_counts,
            "predicted_category": final_labels,
            "predicted_term": matched_terms,
            "predicted_group": matched_groups,
            "predicted_code": matched_codes,
            "auxiliary_label": auxiliary_labels,
            "predicted_label_index": final_indices,
            "category_confidence": final_scores,
            "category_method": methods,
            "pca1": pca_2d[:, 0],
            "pca2": pca_2d[:, 1],
        }
    )
    rows_df.to_csv(path, index=False)
    return rows_df


def write_categories_csv(
    *,
    path: str,
    row_count: int,
    ids: list[str],
    texts: list[str],
    id_col: str,
    text_col: str,
    final_labels: list[str],
    final_indices: list[int],
    final_scores: list[float],
    methods: list[str],
    keyword_labels: list[str],
    keyword_scores: list[float],
    embedding_labels: np.ndarray,
    embedding_scores: np.ndarray,
    label_scores: np.ndarray,
    labels: list[str],
    matched_terms: list[str],
    matched_groups: list[str],
    matched_codes: list[str],
    auxiliary_labels: list[str],
    include_score_columns: bool = True,
    original_row_indices: list[int] | None = None,
    diagnosis_indices: list[int] | None = None,
    original_texts: list[str] | None = None,
) -> pd.DataFrame:
    """Write classification-focused output with one row per sub-diagnosis.

    Includes the same multi-diagnosis provenance columns as
    :func:`write_rows_csv` (``row_index``, ``diagnosis_index``,
    ``original_text``) so that sub-diagnoses can be traced back to
    their original clinical entry.
    """
    data: dict = {
        "row_index": (
            original_row_indices
            if original_row_indices is not None
            else list(np.arange(row_count, dtype=np.int32))
        ),
    }

    if diagnosis_indices is not None:
        data["diagnosis_index"] = diagnosis_indices

    data.update(
        {
            id_col: ids,
            text_col: texts,
        }
    )

    if original_texts is not None:
        data["original_text"] = original_texts

    data.update(
        {
            "predicted_category": final_labels,
            "predicted_term": matched_terms,
            "predicted_group": matched_groups,
            "predicted_code": matched_codes,
            "auxiliary_label": auxiliary_labels,
            "predicted_label_index": final_indices,
            "category_confidence": final_scores,
            "category_method": methods,
            "keyword_category": keyword_labels,
            "keyword_confidence": keyword_scores,
            "embedding_category": embedding_labels,
            "embedding_similarity": embedding_scores,
        }
    )

    category_df = pd.DataFrame(data)

    if include_score_columns:
        for label_idx, label_name in enumerate(labels):
            score_column = f"score_{label_name.lower().replace(' ', '_')}"
            category_df[score_column] = label_scores[:, label_idx].astype(np.float32, copy=False)
    category_df.to_csv(path, index=False)
    return category_df


def write_neighbors_csv(
    *,
    path: str,
    ids: list[str],
    texts: list[str],
    id_col: str,
    text_col: str,
    neighbor_idx: np.ndarray,
    neighbor_sim: np.ndarray,
) -> None:
    neighbor_rows = []
    for row_index in range(len(texts)):
        for rank in range(neighbor_idx.shape[1]):
            neighbor_row = int(neighbor_idx[row_index, rank])
            neighbor_rows.append(
                {
                    "row_index": row_index,
                    "neighbor_rank": rank + 1,
                    "neighbor_row_index": neighbor_row,
                    "cosine_sim": float(neighbor_sim[row_index, rank]),
                    id_col: ids[row_index],
                    f"neighbor_{id_col}": ids[neighbor_row],
                    text_col: texts[row_index],
                    f"neighbor_{text_col}": texts[neighbor_row],
                }
            )
    pd.DataFrame(neighbor_rows).to_csv(path, index=False)


def write_embeddings_npz(path: str, embeddings: np.ndarray, ids: list[str], texts: list[str]) -> None:
    np.savez_compressed(
        path,
        embeddings=embeddings,
        ids=np.array(ids, dtype=object),
        texts=np.array(texts, dtype=object),
    )


def write_summary_json(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
