"""Top-level orchestration for reading input, running categorization, and writing outputs.

This is the main entry point for the data categorization pipeline. The high-level flow is:

  1. Load and clean clinical diagnosis text from a CSV.
  2. Embed every diagnosis string with PetBERT (produces a 768-dim vector per row).
  3. Load the Vet-ICD-O taxonomy and embed each label the same way.
  4. Compare diagnosis embeddings to label embeddings via cosine similarity.
  5. Pick the closest taxonomy label for each row (with a confidence threshold).
  6. Optionally override predictions using auxiliary carcinoma/sarcoma patient lists.
  7. Map the chosen label index back to its ICD-O code, group, and term.
  8. Write all results to CSV / NPZ / JSON output files.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .auxiliary_policy import AuxiliaryLabelPolicy
from .categorization import run_hybrid_categorization
from .embedding import embed_texts, load_tokenizer_and_model, topk_cosine_neighbors
from .io import (
    build_outputs,
    write_categories_csv,
    write_embeddings_npz,
    write_neighbors_csv,
    write_rows_csv,
    write_summary_json,
)
from labels.catalog import label_catalog_for_config
from labels.projection import resolve_taxonomy_matches
from .types import ScanConfig, ScanOutputs
from .utils import clean_text, device_from_arg


def run_scan(config: ScanConfig) -> ScanOutputs:
    """Execute the full categorization pipeline end-to-end.

    Reads clinical diagnosis rows, embeds them with PetBERT, matches each to
    the closest Vet-ICD-O taxonomy label by cosine similarity, and writes the
    results to disk.
    """

    # --- Step 0: Prepare output file paths -----------------------------------
    outputs = build_outputs(config.out_dir, config.task)

    # --- Step 1: Load input data ---------------------------------------------
    # Read the CSV containing patient IDs and clinical diagnosis free-text.
    dataframe = pd.read_csv(config.csv_path, encoding='latin-1')
    if config.max_rows is not None:
        dataframe = dataframe.head(config.max_rows).copy()
    _validate_columns(dataframe, config.id_col, config.text_col)

    # Clean every cell (strip whitespace, coerce NaN to empty string).
    ids = dataframe[config.id_col].map(clean_text).tolist()
    texts = dataframe[config.text_col].map(clean_text).tolist()
    char_lens = np.array([len(text) for text in texts], dtype=np.int32)

    # --- Step 2: Embed diagnosis texts with PetBERT --------------------------
    # Load the pre-trained PetBERT tokenizer + model from HuggingFace (or local cache).
    tokenizer, model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)
    torch_device = device_from_arg(config.device)

    # Produce a (num_rows, 768) embedding matrix -- one 768-dim vector per row.
    # Each text is tokenized (padded/truncated to max_length=256), run through
    # PetBERT's transformer layers, and the [CLS] token's hidden state is taken
    # as the fixed-size representation of the entire input text.
    embeddings, token_counts = embed_texts(
        tokenizer,
        model,
        texts,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # --- Step 3: Build & embed taxonomy labels --------------------------------
    # Load the Vet-ICD-O-canine-1 taxonomy (~857 unique labels), each with a
    # term, group, and ICD-O code.  Convert each label to a descriptive sentence
    # like:
    #   "Veterinary diagnosis term: Hemangiosarcoma, NOS. Group: Blood vessel
    #    tumors. Code: 9120/3."
    # Then embed those sentences through the *same* PetBERT model so that the
    # diagnosis embeddings and label embeddings live in the same vector space.
    label_catalog = label_catalog_for_config(config)
    label_embeddings, _ = embed_texts(
        tokenizer,
        model,
        label_catalog.label_texts,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # --- Step 4: Compare & categorize ----------------------------------------
    # For each diagnosis embedding, compute cosine similarity against every
    # label embedding, then pick the label with the highest similarity score.
    # If that best score is below the confidence threshold (default 0.6), the
    # row is marked "Uncategorized" / method="low_confidence" instead.
    categorization = run_hybrid_categorization(
        texts=texts,
        text_embeddings=embeddings,
        label_embeddings=label_embeddings,
        labels=label_catalog.labels,
        embedding_min_sim=config.embedding_min_sim,
    )

    # --- Step 5: Optional auxiliary label override ----------------------------
    # If --use-auxiliary-labels is set, patients known to have carcinoma or
    # sarcoma (via external CSV lists of patient IDs) get their prediction
    # constrained to only labels whose term contains "carcinoma" or "sarcoma".
    # The highest-scoring label within that constrained subset is chosen.
    auxiliary_decision = AuxiliaryLabelPolicy(config, label_catalog.labels).apply(
        ids=ids,
        categorization=categorization,
    )

    # --- Step 6: Map label index -> ICD code, group, term --------------------
    # Each row's final_index points into the taxonomy list.  This step
    # resolves the index to the human-readable term, group name, and
    # Vet-ICD-O-canine-1 code (e.g. "9120/3").
    matched_terms, matched_groups, matched_codes = resolve_taxonomy_matches(
        categorization.final_indices,
        label_catalog.labels,
        label_catalog.taxonomy_labels,
    )

    # --- Step 7: PCA for 2-D visualization -----------------------------------
    pca = PCA(n_components=2, random_state=0)
    pca_2d = pca.fit_transform(embeddings).astype(np.float32, copy=False)

    # --- Step 8: Write output files ------------------------------------------
    write_rows_csv(
        path=outputs.rows_csv,
        row_count=len(texts),
        ids=ids,
        texts=texts,
        id_col=config.id_col,
        text_col=config.text_col,
        char_lens=char_lens,
        token_counts=token_counts,
        final_labels=categorization.final_labels,
        final_indices=categorization.final_indices,
        final_scores=categorization.final_scores,
        methods=categorization.methods,
        pca_2d=pca_2d,
        matched_terms=matched_terms,
        matched_groups=matched_groups,
        matched_codes=matched_codes,
        auxiliary_labels=auxiliary_decision.labels,
    )

    category_df = write_categories_csv(
        path=outputs.categories_csv,
        row_count=len(texts),
        ids=ids,
        texts=texts,
        id_col=config.id_col,
        text_col=config.text_col,
        final_labels=categorization.final_labels,
        final_indices=categorization.final_indices,
        final_scores=categorization.final_scores,
        methods=categorization.methods,
        keyword_labels=categorization.keyword_labels,
        keyword_scores=categorization.keyword_scores,
        embedding_labels=categorization.embedding_labels,
        embedding_scores=categorization.embedding_scores,
        label_scores=categorization.label_scores,
        labels=categorization.labels,
        matched_terms=matched_terms,
        matched_groups=matched_groups,
        matched_codes=matched_codes,
        auxiliary_labels=auxiliary_decision.labels,
        include_score_columns=label_catalog.include_score_columns,
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
            text_col=config.text_col,
            neighbor_idx=neighbor_idx,
            neighbor_sim=neighbor_sim,
        )

    write_embeddings_npz(outputs.npz, embeddings, ids, texts)

    summary = {
        "csv_path": config.csv_path,
        "model_name": config.model_name,
        "device": str(torch_device),
        "rows": int(len(texts)),
        "task": config.task,
        "text_col": config.text_col,
        "id_col": config.id_col,
        "max_length": int(config.max_length),
        "batch_size": int(config.batch_size),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "labels": categorization.labels,
        "labels_csv_path": config.labels_csv_path,
        "embedding_min_sim": float(config.embedding_min_sim),
        "use_auxiliary_labels": bool(config.use_auxiliary_labels),
        "carcinoma_csv_path": config.carcinoma_csv_path if config.use_auxiliary_labels else "",
        "sarcoma_csv_path": config.sarcoma_csv_path if config.use_auxiliary_labels else "",
        "predicted_term_counts": category_df["predicted_term"].value_counts().to_dict(),
        "predicted_group_counts": category_df["predicted_group"].value_counts().to_dict(),
        "predicted_code_counts": category_df["predicted_code"].value_counts().to_dict(),
        "prediction_method_counts": category_df["category_method"].value_counts().to_dict(),
        "auxiliary_label_counts": category_df["auxiliary_label"].value_counts().to_dict(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()
        if embeddings.size
        else [0.0, 0.0],
    }
    write_summary_json(outputs.summary_json, summary)
    return outputs


def _validate_columns(dataframe: pd.DataFrame, id_col: str, text_col: str) -> None:
    if id_col not in dataframe.columns:
        raise ValueError(f"Missing id column {id_col!r}. Available: {dataframe.columns.tolist()}")
    if text_col not in dataframe.columns:
        raise ValueError(
            f"Missing text column {text_col!r}. Available: {dataframe.columns.tolist()}"
        )
