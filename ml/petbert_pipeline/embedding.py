"""Model loading, text embedding, and cosine-neighbor utilities.

This module handles the core ML operations:
  - Loading the PetBERT model and tokenizer from HuggingFace.
  - Converting free-text strings into 768-dimensional embedding vectors by
    mean-pooling the attended token hidden states from PetBERT's last hidden layer.
  - Computing cosine similarity between two sets of embeddings.
  - Finding top-k nearest neighbors within an embedding matrix.
"""

import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import transformers.utils.logging as hf_logging


def load_tokenizer_and_model(
    model_name: str, *, local_only: bool
) -> tuple[AutoTokenizer, AutoModelForMaskedLM]:
    """Download (or load from cache) the PetBERT tokenizer and model.

    PetBERT (SAVSNET/PetBERT) is a BERT-style masked language model pre-trained
    on veterinary clinical text.  We use it as a feature extractor -- we never
    use the masked-LM head, only the base transformer's hidden states.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    hf_logging.set_verbosity_error()
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_only)
    hf_logging.set_verbosity_warning()
    print("[Warning]: BertForMaskedLM does not inherit from GenerationMixin; generate() will break from transformers v4.50.")
    return tokenizer, model


@torch.inference_mode()
def embed_texts(
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str = "Embedding",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of text strings into embedding vectors using PetBERT.

    For each input string the function:
      1. Tokenizes the text (padding shorter texts, truncating at max_length).
      2. Passes the token IDs through PetBERT's base transformer (skipping
         the masked-LM head) via ``model.base_model(...)``.
      3. Computes the **mean of attended token embeddings** -- the average of
         all non-padding token hidden states from the last transformer layer.
         This 768-dim vector serves as the fixed-size representation of the
         entire input text. Mean pooling outperforms [CLS] for cosine-based
         retrieval when the model has not been fine-tuned for sentence
         similarity.

    Returns:
        embeddings: ndarray of shape (num_texts, 768).
        token_counts: ndarray of shape (num_texts,) with the number of
            non-padding tokens per text.
    """
    model.eval()
    model.to(device)

    num_batches = math.ceil(len(texts) / batch_size)
    all_embeddings: list[np.ndarray] = []
    all_token_counts: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), total=num_batches, desc=desc, unit="batch"):
        batch_texts = texts[start : start + batch_size]

        # Tokenize: converts raw text to input_ids + attention_mask tensors.
        # padding=True  -> pad shorter texts in the batch to the longest one.
        # truncation=True, max_length=256 -> clip texts longer than 256 tokens.
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Run through the base transformer (not the MLM head).
        # outputs.last_hidden_state has shape (batch, seq_len, 768).
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean-pool over attended (non-padding) tokens.
        hidden = outputs.last_hidden_state               # (B, T, 768)
        mask = attention_mask.unsqueeze(-1).float()      # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)              # (B, 768)
        counts = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        mean_embedding = summed / counts                 # (B, 768)
        all_embeddings.append(
            mean_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        # Count real (non-padding) tokens per text for diagnostics.
        token_counts = attention_mask.sum(dim=1).detach().cpu().numpy()
        all_token_counts.append(token_counts.astype(np.int32, copy=False))

    return np.vstack(all_embeddings), np.concatenate(all_token_counts)


def embed_columns_separate(
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    col_texts: dict[str, list[str]],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Embed each column independently and return per-column results.

    Each column is passed through PetBERT as its own batch so it gets the full
    ``max_length`` token budget.

    Args:
        col_texts: Mapping from column name to a list of N cleaned text strings.

    Returns:
        col_embeddings:   {col: (N, 768)} embedding array per column.
        col_has_content:  {col: (N,) bool array} True where the cell is non-empty.
        total_token_counts: (N,) non-padding tokens summed across all columns.
    """
    if not col_texts:
        raise ValueError("col_texts must not be empty")

    col_embeddings: dict[str, np.ndarray] = {}
    col_has_content: dict[str, np.ndarray] = {}
    token_count_arrays: list[np.ndarray] = []

    for col, texts in col_texts.items():
        col_has_content[col] = np.array([bool(t) for t in texts], dtype=bool)

        emb, tok = embed_texts(
            tokenizer,
            model,
            texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            desc=f"Embedding [{col}]",
        )
        col_embeddings[col] = emb
        token_count_arrays.append(tok)

    total_token_counts = np.stack(token_count_arrays, axis=0).sum(axis=0)
    return col_embeddings, col_has_content, total_token_counts.astype(np.int32, copy=False)


def cosine_similarity_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two embedding matrices.

    Given query (N, D) and ref (M, D), returns an (N, M) matrix where
    entry [i, j] = cosine_similarity(query[i], ref[j]).

    Cosine similarity = dot(a, b) / (||a|| * ||b||), ranging from -1 to 1.
    """
    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    query_norms = np.where(query_norms == 0, 1.0, query_norms)
    ref_norms = np.linalg.norm(ref, axis=1, keepdims=True)
    ref_norms = np.where(ref_norms == 0, 1.0, ref_norms)
    return (query / query_norms) @ (ref / ref_norms).T


def topk_cosine_neighbors(
    embeddings: np.ndarray, *, k: int, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Find the k most similar rows for every row in the embedding matrix.

    Self-matches are excluded (a row is never its own neighbor).  Processing is
    done in chunks to limit peak memory usage for large datasets.

    Returns:
        neighbor_idx: (N, k) int64 array of neighbor row indices.
        neighbor_sim: (N, k) float32 array of cosine similarity scores.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D)")
    n, _ = embeddings.shape
    if n == 0:
        return np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float32)

    # L2-normalize once so dot product = cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = embeddings / norms
    neighbor_idx = np.empty((n, k), dtype=np.int64)
    neighbor_sim = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = unit[start:end] @ unit.T

        # Exclude self-similarity by setting diagonal entries to -inf.
        row_indices = np.arange(start, end)
        sims[np.arange(end - start), row_indices] = -np.inf

        # Partial sort to find top-k efficiently (faster than full sort).
        part = np.argpartition(-sims, kth=min(k, n - 1) - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        top = np.take_along_axis(part, order, axis=1)
        top_sims = np.take_along_axis(part_sims, order, axis=1)

        neighbor_idx[start:end] = top
        neighbor_sim[start:end] = top_sims.astype(np.float32, copy=False)

    return neighbor_idx, neighbor_sim
