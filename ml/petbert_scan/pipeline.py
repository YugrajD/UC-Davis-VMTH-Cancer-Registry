"""Top-level orchestration for reading input, running categorization, and writing outputs.

This is the main entry point for the data categorization pipeline. The high-level flow is:

  1.   Load and clean clinical report text from reportText.csv.
  2.   Embed each text column independently with PetBERT (768-dim vector each).
  3.   Load the Vet-ICD-O taxonomy and embed each label the same way.
  4.   Compare per-column embeddings to label embeddings via cosine similarity;
       the label with the highest score across any column wins per case.
  5.   Return the top-k qualifying labels per case (up to 5 above the confidence
       threshold), or the top-1 as "low_confidence" if none pass.
  6.   Map the chosen label indices back to ICD-O code, group, and term.
  7.   Write all results to CSV / NPZ / JSON output files.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .categorization import run_categorization
from .embedding import cosine_similarity_matrix, embed_columns_separate, embed_texts, load_tokenizer_and_model, topk_cosine_neighbors
from .io import (
    build_outputs,
    write_column_scores_csv,
    write_embeddings_npz,
    write_neighbors_csv,
    write_predictions_csv,
    write_provenance_csv,
    write_similarity_csv,
    write_summary_json,
    write_visualization_csv,
)
from labels.catalog import label_catalog_for_config
from labels.projection import resolve_taxonomy_matches
from .types import ScanConfig, ScanOutputs
from .utils import clean_text, device_from_arg, merge_report_columns


def _validate_columns(
    dataframe: pd.DataFrame,
    id_col: str,
    text_cols: tuple[str, ...],
) -> None:
    if id_col not in dataframe.columns:
        raise ValueError(f"Missing id column {id_col!r}. Available: {dataframe.columns.tolist()}")
    missing = [c for c in text_cols if c not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing text columns {missing!r}. Available: {dataframe.columns.tolist()}"
        )


def run_scan(config: ScanConfig) -> ScanOutputs:
    """Execute the full categorization pipeline end-to-end.

    Reads clinical report rows from reportText.csv, embeds each report section
    independently with PetBERT, computes cosine similarity against taxonomy labels
    for each column separately, and assigns the top-k labels whose highest
    similarity across any column exceeds the confidence threshold (up to 5 per case).
    """

    # --- Step 0: Prepare output file paths -----------------------------------
    outputs = build_outputs(config.out_dir, config.task)

    # --- Step 1: Load input data ---------------------------------------------
    dataframe = pd.read_csv(config.csv_path, encoding='latin-1')
    # Strip UTF-8 BOM from column names (e.g. ï»¿ prefix from Excel-exported CSVs)
    dataframe.columns = [col.lstrip('\ufeff').lstrip('ï»¿') for col in dataframe.columns]
    if config.max_rows is not None:
        dataframe = dataframe.head(config.max_rows).copy()
    _validate_columns(dataframe, config.id_col, config.text_cols)

    ids = dataframe[config.id_col].map(clean_text).tolist()
    cols = list(config.text_cols)
    col_texts = {col: dataframe[col].map(clean_text).tolist() for col in cols}
    n = len(ids)
    row_indices = list(range(n))

    # Merged text per row for provenance display and neighbors
    texts = dataframe.apply(lambda row: merge_report_columns(row, cols), axis=1).tolist()
    char_lens = np.array([len(t) for t in texts], dtype=np.int32)

    # --- Step 2: Embed each column independently with PetBERT ----------------
    tokenizer, model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)
    torch_device = device_from_arg(config.device)

    col_embeddings, col_has_content, token_counts = embed_columns_separate(
        tokenizer,
        model,
        col_texts,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # Compute a mean embedding per row (across non-empty columns) for
    # visualization, neighbors, and the embeddings NPZ.
    col_emb_stack = np.stack([col_embeddings[col] for col in cols], axis=0)  # (C, N, 768)
    content_mask = np.stack([col_has_content[col] for col in cols], axis=0).astype(np.float32)  # (C, N)
    col_emb_masked = col_emb_stack * content_mask[:, :, None]  # (C, N, 768)
    content_counts = np.maximum(content_mask.sum(axis=0), 1.0)  # (N,)
    embeddings = (col_emb_masked.sum(axis=0) / content_counts[:, None]).astype(np.float32)  # (N, 768)

    # --- Step 3: Build & embed taxonomy labels --------------------------------
    label_catalog = label_catalog_for_config(config)
    label_embeddings, _ = embed_texts(
        tokenizer,
        model,
        label_catalog.label_texts,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
        desc="Embedding labels",
    )

    # --- Step 4: Categorize with top-k predictions ---------------------------
    # Pass per-column embeddings; run_categorization takes the element-wise max
    # similarity across columns so the strongest column wins per (row, label) pair.
    col_emb_list = [col_embeddings[col] for col in cols]
    col_has_content_list = [col_has_content[col] for col in cols]

    categorization = run_categorization(
        texts=texts,
        text_embeddings=col_emb_list,
        label_embeddings=label_embeddings,
        labels=label_catalog.labels,
        embedding_min_sim=config.embedding_min_sim,
        col_has_content=col_has_content_list,
        max_predictions=5,
    )

    # --- Step 5: Resolve top-k label indices -> term / group / code ----------
    all_k_terms: list[list[str]] = []
    all_k_groups: list[list[str]] = []
    all_k_codes: list[list[str]] = []
    for k_idxs, k_methods in zip(categorization.top_k_indices, categorization.top_k_methods):
        terms, groups, codes = resolve_taxonomy_matches(
            k_idxs, label_catalog.labels, label_catalog.taxonomy_labels
        )
        # Blank code for low_confidence predictions
        codes = [c if m == "embedding" else "" for c, m in zip(codes, k_methods)]
        all_k_terms.append(terms)
        all_k_groups.append(groups)
        all_k_codes.append(codes)

    # Top-1 resolved info (for provenance, similarity, visualization)
    matched_terms, matched_groups, matched_codes = resolve_taxonomy_matches(
        categorization.final_indices, label_catalog.labels, label_catalog.taxonomy_labels
    )

    # --- Step 6: Compute per-column top predictions for column scores ---------
    col_top_terms: dict[str, list[str]] = {}
    col_top_groups: dict[str, list[str]] = {}
    col_top_codes: dict[str, list[str]] = {}
    col_top_scores: dict[str, list[float]] = {}

    for col in cols:
        col_sims = cosine_similarity_matrix(col_embeddings[col], label_embeddings)  # (N, M)
        col_top_idx = np.argmax(col_sims, axis=1)
        col_top_sc = col_sims[np.arange(n), col_top_idx].astype(np.float32)
        # Zero out scores for rows where this column was empty
        col_top_sc[~col_has_content[col]] = 0.0
        t, g, c = resolve_taxonomy_matches(
            col_top_idx.tolist(), label_catalog.labels, label_catalog.taxonomy_labels
        )
        col_top_terms[col] = t
        col_top_groups[col] = g
        col_top_codes[col] = c
        col_top_scores[col] = col_top_sc.tolist()

    # Mark which column had the highest score per row (decisive column)
    col_decisive: dict[str, list[bool]] = {col: [False] * n for col in cols}
    for i in range(n):
        best_col = max(cols, key=lambda c: col_top_scores[c][i])
        col_decisive[best_col][i] = True

    # --- Step 7: PCA for 2-D visualization -----------------------------------
    pca = PCA(n_components=2, random_state=0)
    pca_2d = pca.fit_transform(embeddings).astype(np.float32, copy=False)

    # --- Step 8: Write output files ------------------------------------------
    write_predictions_csv(
        path=outputs.predictions_csv,
        ids=ids,
        id_col=config.id_col,
        all_k_terms=all_k_terms,
        all_k_groups=all_k_groups,
        all_k_codes=all_k_codes,
        all_k_scores=categorization.top_k_scores,
        all_k_methods=categorization.top_k_methods,
    )

    write_provenance_csv(
        path=outputs.provenance_csv,
        ids=ids,
        id_col=config.id_col,
        texts=texts,
        char_lens=char_lens,
        token_counts=token_counts,
        final_labels=categorization.final_labels,
        final_indices=categorization.final_indices,
        embedding_labels=categorization.embedding_labels,
        embedding_scores=categorization.embedding_scores,
        original_row_indices=row_indices,
        diagnosis_indices=[1] * n,
    )

    write_similarity_csv(
        path=outputs.similarity_csv,
        original_row_indices=row_indices,
        diagnosis_indices=[1] * n,
        label_scores=categorization.label_scores,
        labels=categorization.labels,
    )

    write_visualization_csv(
        path=outputs.visualization_csv,
        ids=ids,
        id_col=config.id_col,
        matched_groups=matched_groups,
        pca_2d=pca_2d,
        original_row_indices=row_indices,
        diagnosis_indices=[1] * n,
    )

    write_column_scores_csv(
        path=outputs.column_scores_csv,
        ids=ids,
        id_col=config.id_col,
        col_texts=col_texts,
        col_top_terms=col_top_terms,
        col_top_groups=col_top_groups,
        col_top_codes=col_top_codes,
        col_top_scores=col_top_scores,
        col_decisive=col_decisive,
    )

    if outputs.neighbors_csv is not None:
        neighbor_idx, neighbor_sim = topk_cosine_neighbors(
            embeddings, k=config.neighbors_k, chunk_size=2048
        )
        write_neighbors_csv(
            path=outputs.neighbors_csv,
            ids=ids,
            texts=texts,
            id_col=config.id_col,
            text_col="merged_text",
            neighbor_idx=neighbor_idx,
            neighbor_sim=neighbor_sim,
        )

    write_embeddings_npz(outputs.npz, embeddings, ids, texts)

    summary = {
        "csv_path": config.csv_path,
        "model_name": config.model_name,
        "device": str(torch_device),
        "input_rows": int(n),
        "task": config.task,
        "text_cols": list(config.text_cols),
        "col_weights": config.col_weights,
        "id_col": config.id_col,
        "max_length": int(config.max_length),
        "batch_size": int(config.batch_size),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "labels": categorization.labels,
        "labels_csv_path": config.labels_csv_path,
        "embedding_min_sim": float(config.embedding_min_sim),
        "predicted_term_counts": pd.Series(matched_terms).value_counts().to_dict(),
        "predicted_group_counts": pd.Series(matched_groups).value_counts().to_dict(),
        "predicted_code_counts": pd.Series(matched_codes).value_counts().to_dict(),
        "prediction_method_counts": pd.Series(categorization.methods).value_counts().to_dict(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()
        if embeddings.size
        else [0.0, 0.0],
    }
    write_summary_json(outputs.summary_json, summary)
    return outputs
