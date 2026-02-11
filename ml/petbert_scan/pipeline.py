import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .categorization import build_label_texts, resolve_labels, run_hybrid_categorization
from .embedding import embed_texts, load_tokenizer_and_model, topk_cosine_neighbors
from .io import (
    build_outputs,
    write_categories_csv,
    write_embeddings_npz,
    write_neighbors_csv,
    write_rows_csv,
    write_summary_json,
)
from .types import ScanConfig, ScanOutputs
from .utils import clean_text, device_from_arg


def run_scan(config: ScanConfig) -> ScanOutputs:
    outputs = build_outputs(config.out_dir, config.task)

    dataframe = pd.read_csv(config.csv_path)
    if config.max_rows is not None:
        dataframe = dataframe.head(config.max_rows).copy()
    _validate_columns(dataframe, config.id_col, config.text_col)

    ids = dataframe[config.id_col].map(clean_text).tolist()
    texts = dataframe[config.text_col].map(clean_text).tolist()
    char_lens = np.array([len(text) for text in texts], dtype=np.int32)

    tokenizer, model = load_tokenizer_and_model(config.model_name, local_only=config.local_only)
    torch_device = device_from_arg(config.device)
    embeddings, token_counts = embed_texts(
        tokenizer,
        model,
        texts,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    labels = resolve_labels()
    label_texts = build_label_texts(labels)
    label_embeddings, _ = embed_texts(
        tokenizer,
        model,
        label_texts,
        device=torch_device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    categorization = run_hybrid_categorization(
        texts=texts,
        text_embeddings=embeddings,
        label_embeddings=label_embeddings,
        labels=labels,
        keyword_min_conf=config.keyword_min_conf,
        embedding_min_sim=config.embedding_min_sim,
    )

    pca = PCA(n_components=2, random_state=0)
    pca_2d = pca.fit_transform(embeddings).astype(np.float32, copy=False)

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
        final_scores=categorization.final_scores,
        methods=categorization.methods,
        pca_2d=pca_2d,
    )

    category_df = write_categories_csv(
        path=outputs.categories_csv,
        row_count=len(texts),
        ids=ids,
        texts=texts,
        id_col=config.id_col,
        text_col=config.text_col,
        final_labels=categorization.final_labels,
        final_scores=categorization.final_scores,
        methods=categorization.methods,
        keyword_labels=categorization.keyword_labels,
        keyword_scores=categorization.keyword_scores,
        embedding_labels=categorization.embedding_labels,
        embedding_scores=categorization.embedding_scores,
        label_scores=categorization.label_scores,
        labels=categorization.labels,
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
        "keyword_min_conf": float(config.keyword_min_conf),
        "embedding_min_sim": float(config.embedding_min_sim),
        "predicted_category_counts": category_df["predicted_category"].value_counts().to_dict(),
        "prediction_method_counts": category_df["category_method"].value_counts().to_dict(),
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

