"""Top-level orchestration for reading input, running categorization, and writing outputs.

This is the main entry point for the data categorization pipeline. The high-level flow is:

  1.   Load and clean clinical report text from reportText.csv.
  2.   Embed each text column independently with PetBERT (768-dim vector each).
  3.   Load the Vet-ICD-O taxonomy and embed each label the same way.
  4.   Score each (case, label) pair with the PresenceClassifier;
       select top-k labels per case using group-keyword categorization.
  5.   Map the chosen label indices back to ICD-O code, group, and term.
  6.   Write all results to CSV / NPZ / JSON output files.
"""

import json

import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA

from .categorization import run_categorization, run_categorization_group, run_categorization_group_keyword
from .embedding import embed_columns_separate, embed_texts, load_tokenizer_and_model, topk_cosine_neighbors
from .text_filters import low_confidence_label_supported_by_text, strip_tissue_lists
from model.presence_classifier import PresenceClassifier
from model.group_classifier import GroupClassifier
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
from ICD_labels import label_catalog_for_config, resolve_taxonomy_matches
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

    # Tissue-list filter: drop necropsy "(T1) lung, liver; (T2) ..." segments
    # from HISTOPATHOLOGICAL SUMMARY before embedding so they don't hijack the
    # cosine match with organ words. FINAL COMMENT (the diagnostic prose) is
    # untouched. See text_filters.strip_tissue_lists.
    if config.strip_tissue_lists and "HISTOPATHOLOGICAL SUMMARY" in col_texts:
        original_hp = col_texts["HISTOPATHOLOGICAL SUMMARY"]
        filtered_hp = [strip_tissue_lists(t) for t in original_hp]
        n_changed = sum(1 for o, f in zip(original_hp, filtered_hp) if o != f)
        col_texts["HISTOPATHOLOGICAL SUMMARY"] = filtered_hp
        print(f"Tissue-list filter: stripped list segments from {n_changed:,}/{n:,} HP summaries")

    # Merged text per row for provenance display and neighbors
    texts = dataframe.apply(lambda row: merge_report_columns(row, cols), axis=1).tolist()
    char_lens = np.array([len(t) for t in texts], dtype=np.int32)

    # --- Steps 2–3: Embed with PetBERT (or load from cache) ------------------
    torch_device = device_from_arg(config.device)
    cache = None
    if config.embedding_cache_path:
        from .embedding_cache import load_cache, save_cache
        cache = load_cache(
            config.embedding_cache_path,
            model_name=config.model_name,
            report_csv_path=config.csv_path,
            labels_csv_path=config.labels_csv_path,
            expected_col_names=cols,
        )

    if cache is not None:
        print(f"Loaded embeddings from cache: {config.embedding_cache_path}")
        # Build an index so we can slice the cache to exactly the rows in the
        # dataframe (handles --max-rows and any CSV row-order differences).
        cache_id_to_idx = {cid: i for i, cid in enumerate(cache["case_ids"])}
        sel = [cache_id_to_idx[cid] for cid in ids if cid in cache_id_to_idx]
        col_embeddings  = {col: arr[sel] for col, arr in cache["col_embeddings"].items()}
        col_has_content = {col: arr[sel] for col, arr in cache["col_has_content"].items()}
        embeddings      = cache["mean_embeddings"][sel]
        token_counts    = cache["token_counts"][sel]
        label_catalog   = label_catalog_for_config(config.labels_csv_path)
        label_embeddings = cache["label_embeddings"]
    else:
        # Step 2: Extract text representations (via PetBERT or FineTuned Model)
        # this uses direct classification
        if config.finetuned_model_path is not None:
            from .embedding import load_finetuned_classifier, predict_groups_finetuned
            print(f"Loading fine-tuned PetBERT classifier from {config.finetuned_model_path}...")
            tokenizer, ft_model, idx_to_group = load_finetuned_classifier(
                config.finetuned_model_path, local_only=config.local_only
            )
            
            print("Predicting cancer groups directly with fine-tuned model...")
            group_probs = predict_groups_finetuned(
                tokenizer,
                ft_model,
                texts,
                device=torch_device,
                batch_size=config.batch_size,
                max_length=config.max_length,
            )
            
            # keep index 0 as Uncategorized, matching the training schema
            # more testing required for independent embeddings vs classifier
            group_names = [idx_to_group[i] for i in range(len(idx_to_group))]
            
            # still need dummy embeddings/has_content for compatibility with the rest of the pipeline
            col_embeddings = {col: np.zeros((n, 768), dtype=np.float32) for col in cols}
            col_has_content = {col: np.ones(n, dtype=bool) for col in cols}
            embeddings = np.zeros((n, 768), dtype=np.float32)
            token_counts = np.zeros(n, dtype=np.int32)
            
            # base model must be loaded to embed the taxonomy labels for term selection between groups
            print("Loading base SAVSNET/PetBERT to embed taxonomy labels...")
            _, base_model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)
            
        else:
            tokenizer, model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)
            base_model = model

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

        # --- Step 3: Build & embed taxonomy labels ----------------------------
        label_catalog = label_catalog_for_config(config.labels_csv_path)
        label_embeddings, _ = embed_texts(
            tokenizer,
            base_model,
            label_catalog.label_texts,
            device=torch_device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            desc="Embedding labels",
        )

        # Save cache if a path was provided (cache miss means we just computed)
        if config.embedding_cache_path:
            save_cache(
                config.embedding_cache_path,
                case_ids=ids,
                col_embeddings=col_embeddings,
                col_has_content=col_has_content,
                mean_embeddings=embeddings,
                token_counts=token_counts,
                label_texts=label_catalog.label_texts,
                label_embeddings=label_embeddings,
                model_name=config.model_name,
                report_csv_path=config.csv_path,
                labels_csv_path=config.labels_csv_path,
            )

    # --- Step 4: Categorize with top-k predictions ---------------------------
    col_emb_list = [col_embeddings[col] for col in cols]
    col_has_content_list = [col_has_content[col] for col in cols]

    # Build per-column concat embedding (used by PresenceClassifier).
    col_emb_concat = np.concatenate(
        [np.where(col_has_content[col][:, None], col_embeddings[col], 0.0)
         for col in cols],
        axis=1,
    ).astype(np.float32)

    # --- Step 4 (cont): Choose categorization strategy ----------------------
    # Group-only mode: GroupClassifier MLP or fine-tuned model.
    if (
        config.group_classifier_path is not None
        or config.finetuned_model_path is not None
    ):
        if config.finetuned_model_path is None:
            print(f"Loading group classifier from {config.group_classifier_path}...")
            group_clf, group_names = GroupClassifier.load(config.group_classifier_path)
            group_clf.to(torch_device)
            # Older 768-dim checkpoints were trained on mean embeddings; newer
            # 2304-dim ones on per-column concat. Pick the input that matches.
            if group_clf.emb_dim == embeddings.shape[1]:
                group_input = embeddings
            elif group_clf.emb_dim == col_emb_concat.shape[1]:
                group_input = col_emb_concat
            else:
                raise ValueError(
                    f"GroupClassifier expects emb_dim={group_clf.emb_dim}, "
                    f"got mean={embeddings.shape[1]}, concat={col_emb_concat.shape[1]}"
                )
            group_probs = group_clf.predict_proba(torch.from_numpy(group_input)).numpy()
            group_clf.cpu()
            del group_clf
        else:
            # group_probs and group_names already set by fine-tuned model above
            print("Using group probabilities mapped from the fine-tuned sequence classifier.")

        categorization = run_categorization_group(
            texts=texts,
            mean_embeddings=embeddings,
            label_embeddings=label_embeddings,
            taxonomy_labels=label_catalog.taxonomy_labels,
            labels=label_catalog.labels,
            group_probs=group_probs,
            group_names=group_names,
            threshold=config.group_classifier_threshold,
            max_predictions=5,
        )

    else:
        # Binary-classifier mode — requires --presence-classifier.
        if config.presence_classifier_path is None:
            raise ValueError("--presence-classifier is required")
        print(f"Loading presence classifier from {config.presence_classifier_path}...")
        classifier = PresenceClassifier.load(config.presence_classifier_path)
        classifier.to(torch_device)
        presence_score_matrix = classifier.score_matrix(
            torch.from_numpy(col_emb_concat),
            torch.from_numpy(label_embeddings),
        ).numpy()
        classifier.cpu()
        del classifier

        if config.categorization_mode == "group-keyword":
            categorization = run_categorization_group_keyword(
                texts=texts,
                score_matrix=presence_score_matrix,
                taxonomy_labels=label_catalog.taxonomy_labels,
                labels=label_catalog.labels,
                embedding_min_sim=config.embedding_min_sim,
                max_predictions=5,
            )
        else:
            categorization = run_categorization(
                texts=texts,
                text_embeddings=col_emb_list,
                label_embeddings=label_embeddings,
                labels=label_catalog.labels,
                embedding_min_sim=config.embedding_min_sim,
                col_has_content=col_has_content_list,
                max_predictions=5,
                score_matrix=presence_score_matrix,
            )

    # --- Step 5: Optional low-confidence top-k rescue ------------------------
    #
    # Default categorization stores only the hidden top-1 candidate for
    # low-confidence rows. For this rescue layer, scan the ranked score matrix
    # and allow the first label whose term is directly supported by report text.
    # This is deliberately narrower than lowering the threshold globally.
    n_low_confidence_rescued = 0
    if config.apply_low_confidence_rescue:
        rescue_k = max(1, int(config.low_confidence_rescue_k))
        for i, source_text in enumerate(texts):
            if not categorization.top_k_methods[i]:
                continue
            if not all(method == "low_confidence" for method in categorization.top_k_methods[i]):
                continue
            scores = categorization.label_scores[i]
            ranked = np.argsort(-scores)[:rescue_k]
            for label_idx in ranked:
                label_idx = int(label_idx)
                term = label_catalog.labels[label_idx]
                if not low_confidence_label_supported_by_text(term, source_text):
                    continue
                score = float(scores[label_idx])
                method = "embedding+low_confidence_rescue"
                categorization.top_k_indices[i] = [label_idx]
                categorization.top_k_scores[i] = [score]
                categorization.top_k_methods[i] = [method]
                categorization.final_indices[i] = label_idx
                categorization.final_labels[i] = term
                categorization.final_scores[i] = score
                categorization.methods[i] = method
                n_low_confidence_rescued += 1
                break
        print(
            "Low-confidence rescue: allowed "
            f"{n_low_confidence_rescued:,} text-supported top-{rescue_k} labels through"
        )

    # --- Step 6: Resolve top-k label indices -> term / group / code ----------
    all_k_terms: list[list[str]] = []
    all_k_groups: list[list[str]] = []
    all_k_codes: list[list[str]] = []
    for k_idxs, k_methods in zip(categorization.top_k_indices, categorization.top_k_methods):
        terms, groups, codes = resolve_taxonomy_matches(
            k_idxs, label_catalog.labels, label_catalog.taxonomy_labels
        )
        all_k_terms.append(terms)
        all_k_groups.append(groups)
        all_k_codes.append(codes)

    # Blank codes only for labels still rejected as low-confidence. Rescued
    # rows keep their resolved Vet-ICD-O code.
    for i, k_methods in enumerate(categorization.top_k_methods):
        all_k_codes[i] = [
            code if method != "low_confidence" else ""
            for code, method in zip(all_k_codes[i], k_methods)
        ]

    # Top-1 resolved info (for provenance, similarity, visualization)
    matched_terms, matched_groups, matched_codes = resolve_taxonomy_matches(
        categorization.final_indices, label_catalog.labels, label_catalog.taxonomy_labels
    )

    # --- Step 6: PCA for 2-D visualization -----------------------------------
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

    # --- Cascade post-processing: kNN fallback for low-confidence predictions ---
    if config.cascade_threshold > 0:
        from .cascade import apply_cascade
        apply_cascade(
            predictions_csv=outputs.predictions_csv,
            report_embeddings=embeddings,
            label_embeddings=label_embeddings,
            case_ids=ids,
            labels_csv_path=config.labels_csv_path,
            threshold=config.cascade_threshold,
            k=config.cascade_k,
            adaptive_thresholds_path=config.cascade_adaptive_path,
        )

    # --- Rule-based gates: subtype demotion + non-neoplastic suppression ----
    # Run after the cascade so kNN-replaced rows are also gated. Subtype gate
    # runs before the non-neoplastic gate so demoted-to-NOS predictions can
    # still be suppressed when the case isn't a tumor.
    if config.apply_subtype_gate:
        from .gates import apply_subtype_gate
        apply_subtype_gate(
            predictions_csv=outputs.predictions_csv,
            reports_csv_path=config.csv_path,
            labels_csv_path=config.labels_csv_path,
            id_col=config.id_col,
            text_cols=config.text_cols,
        )
    if config.apply_non_neoplastic_gate:
        from .gates import apply_non_neoplastic_gate
        apply_non_neoplastic_gate(
            predictions_csv=outputs.predictions_csv,
            reports_csv_path=config.csv_path,
            id_col=config.id_col,
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
        "low_confidence_rescue_count": int(n_low_confidence_rescued),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()
        if embeddings.size
        else [0.0, 0.0],
    }
    write_summary_json(outputs.summary_json, summary)
    return outputs
