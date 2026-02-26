"""Top-level orchestration for reading input, running categorization, and writing outputs.

This is the main entry point for the data categorization pipeline. The high-level flow is:

  1.   Load and clean clinical report text from reportText.csv.
  1.5  Split multi-diagnosis entries into individual sub-diagnoses so that
       each sub-diagnosis is embedded and categorized independently.
  2.   Embed every sub-diagnosis string with PetBERT (768-dim vector each).
  3.   Load the Vet-ICD-O taxonomy and embed each label the same way.
  4.   Compare diagnosis embeddings to label embeddings via cosine similarity.
  5.   Pick the closest taxonomy label for each sub-diagnosis (with a confidence threshold).
  6.   Map the chosen label index back to its ICD-O code, group, and term.
  7.   Write all results to CSV / NPZ / JSON output files.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .categorization import run_categorization
from .embedding import embed_columns_weighted, embed_texts, load_tokenizer_and_model, topk_cosine_neighbors
from .io import (
    build_outputs,
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
from .utils import clean_text, device_from_arg, merge_report_columns, split_numbered_diagnoses


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


def _expand_multi_diagnoses(
    ids: list[str], texts: list[str]
) -> tuple[list[str], list[str], list[int], list[int], list[str]]:
    """Split multi-diagnosis entries and expand all parallel lists.

    Clinical entries formatted as ``"1) Osteosarcoma 2) Cystitis"`` are split
    into individual sub-diagnosis strings.  Every other per-row list (IDs,
    texts) is expanded so that downstream pipeline steps receive a flat list
    of sub-diagnoses and can process them without knowing about the split.

    Returns:
        expanded_ids:          Patient ID repeated for each sub-diagnosis.
        expanded_texts:        Individual sub-diagnosis strings.
        original_row_indices:  Index of the original CSV row each sub-diagnosis
                               came from (0-based).
        diagnosis_indices:     Position of this sub-diagnosis within its
                               original entry (1-based, matching the ``1)``,
                               ``2)`` numbering in the source text).
        original_texts:        The full unsplit text for each sub-diagnosis
                               (duplicated so it aligns with the expanded
                               lists).
    """
    expanded_ids: list[str] = []
    expanded_texts: list[str] = []
    original_row_indices: list[int] = []
    diagnosis_indices: list[int] = []
    original_texts: list[str] = []

    for row_idx, (patient_id, text) in enumerate(zip(ids, texts)):
        sub_diagnoses = split_numbered_diagnoses(text)
        for diag_idx, sub_text in enumerate(sub_diagnoses, start=1):
            expanded_ids.append(patient_id)
            expanded_texts.append(sub_text)
            original_row_indices.append(row_idx)
            diagnosis_indices.append(diag_idx)
            original_texts.append(text)

    return expanded_ids, expanded_texts, original_row_indices, diagnosis_indices, original_texts


def run_scan(config: ScanConfig) -> ScanOutputs:
    """Execute the full categorization pipeline end-to-end.

    Reads clinical report rows from reportText.csv, splits multi-diagnosis
    entries, embeds each report section with PetBERT using a weighted average
    across columns, matches to the closest Vet-ICD-O taxonomy label by cosine
    similarity, and writes the results to disk.
    """

    # --- Step 0: Prepare output file paths -----------------------------------
    outputs = build_outputs(config.out_dir, config.task)

    # --- Step 1: Load input data ---------------------------------------------
    # Read the CSV containing patient IDs and clinical report free-text.
    dataframe = pd.read_csv(config.csv_path, encoding='latin-1')
    # Strip UTF-8 BOM from column names (e.g. ï»¿ prefix from Excel-exported CSVs)
    dataframe.columns = [col.lstrip('\ufeff').lstrip('ï»¿') for col in dataframe.columns]
    if config.max_rows is not None:
        dataframe = dataframe.head(config.max_rows).copy()
    _validate_columns(dataframe, config.id_col, config.text_cols)

    # Clean the ID column and build per-column text lists for embedding.
    ids = dataframe[config.id_col].map(clean_text).tolist()
    cols = list(config.text_cols)
    col_texts = {col: dataframe[col].map(clean_text).tolist() for col in cols}

    # Also build a merged string per row for display / provenance purposes only
    # (not used for embedding).
    texts = dataframe.apply(lambda row: merge_report_columns(row, cols), axis=1).tolist()

    # --- Step 1.5: Split multi-diagnosis entries ------------------------------
    # Clinical entries often contain multiple diagnoses numbered as
    # "1) Osteosarcoma 2) Chronic cystitis".  Split these into individual
    # sub-diagnoses so each one is categorized independently.
    # This expands N input rows into M sub-diagnosis rows (M >= N).
    (
        expanded_ids,
        expanded_texts,
        original_row_indices,
        diagnosis_indices,
        original_texts,
    ) = _expand_multi_diagnoses(ids, texts)

    # For the predictions CSV, show only the FINAL COMMENT column value instead
    # of the full merged text (which includes all report sections).
    fc_col = "FINAL COMMENT"
    if fc_col in dataframe.columns:
        fc_values = dataframe[fc_col].map(clean_text).tolist()
        original_texts = [fc_values[i] for i in original_row_indices]

    char_lens = np.array([len(t) for t in expanded_texts], dtype=np.int32)

    # --- Step 2: Embed diagnosis texts with PetBERT --------------------------
    # Load the pre-trained PetBERT tokenizer + model from HuggingFace (or local cache).
    tokenizer, model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)
    torch_device = device_from_arg(config.device)

    # Embed each column independently and combine into a single weighted-average
    # embedding per input row.  Each column gets its own full max_length token
    # budget, so no section is crowded out by another.  Empty cells are excluded
    # from each row's average automatically.
    #
    # row_embeddings shape: (N_rows, 768)
    row_embeddings, row_token_counts = embed_columns_weighted(
        tokenizer,
        model,
        col_texts,
        config.col_weights,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # Expand row-level embeddings to the M expanded sub-diagnosis rows using
    # the original_row_indices produced by _expand_multi_diagnoses.
    idx = np.array(original_row_indices, dtype=np.intp)
    embeddings = row_embeddings[idx]        # (M, 768)
    token_counts = row_token_counts[idx]    # (M,)

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

    # --- Step 4: Compare & categorize ----------------------------------------
    categorization = run_categorization(
        texts=expanded_texts,
        text_embeddings=embeddings,
        label_embeddings=label_embeddings,
        labels=label_catalog.labels,
        embedding_min_sim=config.embedding_min_sim,
    )

    # --- Step 5: Map label index -> ICD code, group, term --------------------
    matched_terms, matched_groups, matched_codes = resolve_taxonomy_matches(
        categorization.final_indices,
        label_catalog.labels,
        label_catalog.taxonomy_labels,
    )

    # --- Step 6: PCA for 2-D visualization -----------------------------------
    pca = PCA(n_components=2, random_state=0)
    pca_2d = pca.fit_transform(embeddings).astype(np.float32, copy=False)

    # --- Step 7: Write output files ------------------------------------------
    write_predictions_csv(
        path=outputs.predictions_csv,
        ids=expanded_ids,
        id_col=config.id_col,
        matched_terms=matched_terms,
        matched_groups=matched_groups,
        matched_codes=matched_codes,
        final_scores=categorization.final_scores,
        methods=categorization.methods,
        original_row_indices=original_row_indices,
        original_texts=original_texts,
    )

    write_provenance_csv(
        path=outputs.provenance_csv,
        ids=expanded_ids,
        id_col=config.id_col,
        texts=expanded_texts,
        char_lens=char_lens,
        token_counts=token_counts,
        final_labels=categorization.final_labels,
        final_indices=categorization.final_indices,
        embedding_labels=categorization.embedding_labels,
        embedding_scores=categorization.embedding_scores,
        original_row_indices=original_row_indices,
        diagnosis_indices=diagnosis_indices,
    )

    write_similarity_csv(
        path=outputs.similarity_csv,
        original_row_indices=original_row_indices,
        diagnosis_indices=diagnosis_indices,
        label_scores=categorization.label_scores,
        labels=categorization.labels,
    )

    write_visualization_csv(
        path=outputs.visualization_csv,
        ids=expanded_ids,
        id_col=config.id_col,
        matched_groups=matched_groups,
        pca_2d=pca_2d,
        original_row_indices=original_row_indices,
        diagnosis_indices=diagnosis_indices,
    )

    if outputs.neighbors_csv is not None:
        neighbor_idx, neighbor_sim = topk_cosine_neighbors(
            embeddings, k=config.neighbors_k, chunk_size=2048
        )
        write_neighbors_csv(
            path=outputs.neighbors_csv,
            ids=expanded_ids,
            texts=expanded_texts,
            id_col=config.id_col,
            text_col="merged_text",
            neighbor_idx=neighbor_idx,
            neighbor_sim=neighbor_sim,
        )

    write_embeddings_npz(outputs.npz, embeddings, expanded_ids, expanded_texts)

    summary = {
        "csv_path": config.csv_path,
        "model_name": config.model_name,
        "device": str(torch_device),
        "input_rows": int(len(texts)),
        "expanded_rows": int(len(expanded_texts)),
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
