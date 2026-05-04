"""Top-level orchestration for reading input, running categorization, and writing outputs.

The high-level flow is:

  1.   Load and clean clinical report text from reportText.csv.
  2.   Select text using TF-IDF multi-column selector (or explicit --text-cols for debugging).
  3.   Embed the selected text with PetBERT (768-dim vector).
  4.   Production path — 3-stage:
         (a) CasePresenceClassifier gates non-cancer cases.
         (b) GroupClassifier predicts which cancer group(s) each case belongs to.
         (c) ICD-O behavior keyword matching selects the specific term within each group.
       Training-cycle path — binary (run_cycle.py only):
         PresenceClassifier scores all (case, label) pairs; argmax selects winner.
  5.   Map the chosen label indices back to ICD-O code, group, and term.
  6.   Write all results to CSV / NPZ / JSON output files.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA

from .categorization import run_categorization, run_categorization_group
from .embedding import embed_columns_separate, embed_texts, load_tokenizer_and_model, topk_cosine_neighbors
from .text_selector import get_selector, SOURCE_COLS as _TFIDF_SOURCE_COLS
from model.presence_classifier import PresenceClassifier
from model.group_classifier import GroupClassifier
from model.case_presence_classifier import CasePresenceClassifier
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
    """Execute the full categorization pipeline end-to-end."""

    # --- Step 0: Prepare output file paths -----------------------------------
    outputs = build_outputs(config.out_dir, config.task)

    # --- Step 1: Load input data ---------------------------------------------
    dataframe = pd.read_csv(config.csv_path, encoding='latin-1')
    dataframe.columns = [col.lstrip('﻿').lstrip('ï»¿') for col in dataframe.columns]
    if config.max_rows is not None:
        dataframe = dataframe.head(config.max_rows).copy()
    _validate_columns(dataframe, config.id_col, config.text_cols)

    ids = dataframe[config.id_col].map(clean_text).tolist()
    n = len(ids)
    row_indices = list(range(n))

    # --- Step 2: Select text (TF-IDF or explicit columns) --------------------
    if not config.text_cols:
        selector = get_selector(config.tfidf_vectorizer_path)
        cols = ["tfidf_selected"]
        selected_texts: list[str] = []
        for i in range(n):
            row_col_texts = {
                col: clean_text(dataframe.iloc[i].get(col, ""))
                for col in _TFIDF_SOURCE_COLS
            }
            selected_texts.append(selector.select(row_col_texts, max_tokens=512))
        col_texts = {"tfidf_selected": selected_texts}
        texts = selected_texts
    else:
        cols = list(config.text_cols)
        col_texts = {col: dataframe[col].map(clean_text).tolist() for col in cols}
        texts = dataframe.apply(lambda row: merge_report_columns(row, cols), axis=1).tolist()
    char_lens = np.array([len(t) for t in texts], dtype=np.int32)

    # --- Steps 3: Embed with PetBERT (or load from cache) --------------------
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
        cache_id_to_idx = {cid: i for i, cid in enumerate(cache["case_ids"])}
        sel = [cache_id_to_idx[cid] for cid in ids if cid in cache_id_to_idx]
        col_embeddings  = {col: arr[sel] for col, arr in cache["col_embeddings"].items()}
        col_has_content = {col: arr[sel] for col, arr in cache["col_has_content"].items()}
        embeddings      = cache["mean_embeddings"][sel]
        token_counts    = cache["token_counts"][sel]
        label_catalog   = label_catalog_for_config(config.labels_csv_path)
        label_embeddings = cache["label_embeddings"]
    else:
        tokenizer, model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)

        col_embeddings, col_has_content, token_counts = embed_columns_separate(
            tokenizer,
            model,
            col_texts,
            device=torch_device,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )

        col_emb_stack = np.stack([col_embeddings[col] for col in cols], axis=0)  # (C, N, 768)
        content_mask = np.stack([col_has_content[col] for col in cols], axis=0).astype(np.float32)
        col_emb_masked = col_emb_stack * content_mask[:, :, None]
        content_counts = np.maximum(content_mask.sum(axis=0), 1.0)
        embeddings = (col_emb_masked.sum(axis=0) / content_counts[:, None]).astype(np.float32)

        label_catalog = label_catalog_for_config(config.labels_csv_path)
        label_embeddings, _ = embed_texts(
            tokenizer,
            model,
            label_catalog.label_texts,
            device=torch_device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            desc="Embedding labels",
        )

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

    # --- Step 4: Categorize --------------------------------------------------
    col_emb_concat = np.concatenate(
        [np.where(col_has_content[col][:, None], col_embeddings[col], 0.0)
         for col in cols],
        axis=1,
    ).astype(np.float32)

    if config.group_classifier_path is not None:
        # --- Production 3-stage path -----------------------------------------
        # Stage 1: CasePresenceClassifier gate — rejects non-cancer cases.
        if config.case_presence_classifier_path is not None:
            print(f"Loading case presence classifier from {config.case_presence_classifier_path}...")
            case_clf = CasePresenceClassifier.load(config.case_presence_classifier_path)
            case_clf.to(torch_device)
            cancer_probs = case_clf.predict_proba(torch.from_numpy(embeddings)).numpy()
            case_clf.cpu()
            del case_clf
            presence_gate_mask = cancer_probs >= config.case_presence_threshold
            print(
                f"  Case presence gate (threshold={config.case_presence_threshold:.2f}): "
                f"{int(presence_gate_mask.sum())}/{n} cases pass "
                f"({presence_gate_mask.mean() * 100:.1f}%)"
            )
        else:
            presence_gate_mask = np.ones(n, dtype=bool)

        # Stage 2: GroupClassifier predicts which cancer group(s) each case belongs to.
        print(f"Loading group classifier from {config.group_classifier_path}...")
        group_clf, group_names = GroupClassifier.load(config.group_classifier_path)
        group_clf.to(torch_device)
        group_probs = group_clf.predict_proba(torch.from_numpy(col_emb_concat)).numpy()
        group_clf.cpu()
        del group_clf

        # Gate-rejected cases have group_probs zeroed → fall through to Uncategorized.
        group_probs[~presence_gate_mask] = 0.0

        # Load uncommon group list so "Uncommon" predictions route to KW correction
        # across the pooled uncommon label indices rather than falling through to
        # "Unidentified Group".
        _uncommon_path = Path(config.uncommon_groups_path)
        uncommon_groups: frozenset[str] = frozenset()
        if _uncommon_path.exists():
            uncommon_groups = frozenset(
                line.strip() for line in _uncommon_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        # Stage 3: KW correction selects the best term within each predicted group.
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
            presence_mask=presence_gate_mask,
            uncommon_groups=uncommon_groups,
            fallback_to_argmax=config.group_classifier_fallback_to_argmax,
        )

    else:
        # --- Training-cycle binary path (run_cycle.py only) ------------------
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

        categorization = run_categorization(
            texts=texts,
            text_embeddings=[col_embeddings[col] for col in cols],
            label_embeddings=label_embeddings,
            labels=label_catalog.labels,
            embedding_min_sim=config.embedding_min_sim,
            col_has_content=[col_has_content[col] for col in cols],
            max_predictions=5,
            score_matrix=presence_score_matrix,
        )

    # --- Step 5: Resolve top-k label indices -> term / group / code ----------
    all_k_terms: list[list[str]] = []
    all_k_groups: list[list[str]] = []
    all_k_codes: list[list[str]] = []
    for k_idxs, k_methods in zip(categorization.top_k_indices, categorization.top_k_methods):
        terms, groups, codes = resolve_taxonomy_matches(
            k_idxs, label_catalog.labels, label_catalog.taxonomy_labels
        )
        codes = [c if m == "embedding" else "" for c, m in zip(codes, k_methods)]
        all_k_terms.append(terms)
        all_k_groups.append(groups)
        all_k_codes.append(codes)

    matched_terms, matched_groups, matched_codes = resolve_taxonomy_matches(
        categorization.final_indices, label_catalog.labels, label_catalog.taxonomy_labels
    )

    # --- Step 6: PCA for 2-D visualization -----------------------------------
    pca = PCA(n_components=2, random_state=0)
    pca_2d = pca.fit_transform(embeddings).astype(np.float32, copy=False)

    # --- Step 7: Write output files ------------------------------------------
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
