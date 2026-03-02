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
        column_scores_csv=os.path.join(out_dir, "petbert_scan_column_scores.csv"),
        neighbors_csv=neighbors_csv,
        npz=os.path.join(out_dir, "petbert_scan_embeddings.npz"),
        summary_json=os.path.join(out_dir, "petbert_scan_summary.json"),
    )


def write_predictions_csv(
    *,
    path: str,
    ids: list[str],
    id_col: str,
    all_k_terms: list[list[str]],
    all_k_groups: list[list[str]],
    all_k_codes: list[list[str]],
    all_k_scores: list[list[float]],
    all_k_methods: list[list[str]],
) -> pd.DataFrame:
    """Write the presentation-ready predictions file.

    One row per (case, prediction rank). ``diagnosis_index`` is the rank of the
    prediction for that case (1 = best match, up to 5). Cases where the text was
    empty produce no rows.
    """
    rows = []
    for i, patient_id in enumerate(ids):
        for rank, (term, group, code, score, method) in enumerate(
            zip(all_k_terms[i], all_k_groups[i], all_k_codes[i], all_k_scores[i], all_k_methods[i]),
            start=1,
        ):
            rows.append({
                id_col: patient_id,
                "diagnosis_index": rank,
                "predicted_term": term,
                "predicted_group": group,
                "predicted_code": code,
                "confidence": f"{score:.2f}",
                "method": method,
            })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(path, index=False)
    return pred_df


def write_column_scores_csv(
    *,
    path: str,
    ids: list[str],
    id_col: str,
    col_texts: dict[str, list[str]],
    col_top_terms: dict[str, list[str]],
    col_top_groups: dict[str, list[str]],
    col_top_codes: dict[str, list[str]],
    col_top_scores: dict[str, list[float]],
    col_decisive: dict[str, list[bool]],
) -> None:
    """Write the per-column similarity score file.

    One row per (case x column). Shows which taxonomy label each text column
    independently matched best, and which column was decisive (highest score)
    for each case.
    """
    rows = []
    for i, patient_id in enumerate(ids):
        for col_name in col_texts:
            rows.append({
                "row_index": i,
                id_col: patient_id,
                "column_name": col_name,
                "column_text": col_texts[col_name][i],
                "top_term": col_top_terms[col_name][i],
                "top_group": col_top_groups[col_name][i],
                "top_code": col_top_codes[col_name][i],
                "top_score": round(col_top_scores[col_name][i], 4),
                "was_decisive": col_decisive[col_name][i],
            })
    pd.DataFrame(rows).to_csv(path, index=False)


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
