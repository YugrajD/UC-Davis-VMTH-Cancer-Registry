"""Output writers for predictions, provenance, similarity scores, visualization, neighbors, embeddings, and summary."""

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
        predictions_csv=os.path.join(out_dir, "petbert_scan_predictions.csv"),
        provenance_csv=os.path.join(out_dir, "petbert_scan_provenance.csv"),
        similarity_csv=os.path.join(out_dir, "petbert_scan_similarity_scores.csv"),
        visualization_csv=os.path.join(out_dir, "petbert_scan_visualization.csv"),
        neighbors_csv=neighbors_csv,
        npz=os.path.join(out_dir, "petbert_scan_embeddings.npz"),
        summary_json=os.path.join(out_dir, "petbert_scan_summary.json"),
    )


def _concat_group(values: list[str], *, count: int) -> str:
    """Concatenate sub-diagnosis values for a single patient row.

    When *count* is 1 the value is returned as-is (no prefix).
    When *count* > 1 each value is prefixed with ``"k) "`` and all are
    joined with a space.
    """
    if count == 1:
        return values[0]
    return " ".join(f"{i}) {v}" for i, v in enumerate(values, start=1))


def write_predictions_csv(
    *,
    path: str,
    ids: list[str],
    id_col: str,
    matched_terms: list[str],
    matched_groups: list[str],
    matched_codes: list[str],
    final_scores: list[float],
    methods: list[str],
    original_row_indices: list[int],
    original_texts: list[str],
) -> pd.DataFrame:
    """Write the presentation-ready predictions file.

    Collapses sub-diagnoses back to one row per original patient.
    Multi-diagnosis entries are concatenated with ``1)``, ``2)`` prefixes;
    single-diagnosis entries are shown plain.  ``predicted_code`` is blanked
    for uncategorized predictions (method ``low_confidence`` or ``empty``).
    """
    # Build a per-sub-diagnosis frame first
    codes_cleaned = [
        code if method not in ("low_confidence", "empty") else ""
        for code, method in zip(matched_codes, methods)
    ]
    scores_str = [f"{s:.2f}" for s in final_scores]

    sub_df = pd.DataFrame({
        "row_index": original_row_indices,
        id_col: ids,
        "original_text": original_texts,
        "predicted_term": matched_terms,
        "predicted_group": matched_groups,
        "predicted_code": codes_cleaned,
        "confidence": scores_str,
        "method": methods,
    })

    # Group by original row and concatenate
    concat_cols = ["predicted_term", "predicted_group", "predicted_code", "confidence", "method"]
    rows = []
    for row_idx, group in sub_df.groupby("row_index", sort=True):
        count = len(group)
        row = {
            id_col: group[id_col].iloc[0],
            "original_text": group["original_text"].iloc[0],
        }
        for col in concat_cols:
            row[col] = _concat_group(group[col].tolist(), count=count)
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(path, index=False)
    return pred_df


def write_provenance_csv(
    *,
    path: str,
    ids: list[str],
    id_col: str,
    texts: list[str],
    char_lens: np.ndarray,
    token_counts: np.ndarray,
    final_labels: list[str],
    final_indices: list[int],
    embedding_labels: np.ndarray,
    embedding_scores: np.ndarray,
    original_row_indices: list[int],
    diagnosis_indices: list[int],
) -> pd.DataFrame:
    """Write the per-diagnosis traceability and debug file."""
    prov_df = pd.DataFrame({
        "row_index": original_row_indices,
        id_col: ids,
        "diagnosis_index": diagnosis_indices,
        "diagnosis_text": texts,
        "char_len": char_lens,
        "token_count": token_counts,
        "predicted_category": final_labels,
        "predicted_label_index": final_indices,
        "embedding_category": embedding_labels,
        "embedding_similarity": embedding_scores,
    })
    prov_df.to_csv(path, index=False)
    return prov_df


def write_similarity_csv(
    *,
    path: str,
    original_row_indices: list[int],
    diagnosis_indices: list[int],
    label_scores: np.ndarray,
    labels: list[str],
) -> None:
    """Write the per-label cosine similarity score matrix."""
    key_df = pd.DataFrame({
        "row_index": original_row_indices,
        "diagnosis_index": diagnosis_indices,
    })
    score_columns = {
        f"score_{name.lower().replace(' ', '_')}": label_scores[:, idx].astype(np.float32, copy=False)
        for idx, name in enumerate(labels)
    }
    sim_df = pd.concat([key_df, pd.DataFrame(score_columns)], axis=1)
    sim_df.to_csv(path, index=False)


def write_visualization_csv(
    *,
    path: str,
    ids: list[str],
    id_col: str,
    matched_groups: list[str],
    pca_2d: np.ndarray,
    original_row_indices: list[int],
    diagnosis_indices: list[int],
) -> None:
    """Write the PCA visualization coordinates."""
    viz_df = pd.DataFrame({
        "row_index": original_row_indices,
        "diagnosis_index": diagnosis_indices,
        id_col: ids,
        "predicted_group": matched_groups,
        "pca1": pca_2d[:, 0],
        "pca2": pca_2d[:, 1],
    })
    viz_df.to_csv(path, index=False)


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
