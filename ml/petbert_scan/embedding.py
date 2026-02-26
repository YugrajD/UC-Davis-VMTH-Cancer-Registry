"""Model loading, text embedding, and cosine-neighbor utilities.

This module handles the core ML operations:
  - Loading the PetBERT model and tokenizer from HuggingFace.
  - Converting free-text strings into 768-dimensional embedding vectors by
    extracting the [CLS] token representation from PetBERT's last hidden layer.
  - Computing cosine similarity between two sets of embeddings.
  - Finding top-k nearest neighbors within an embedding matrix.
"""

import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_tokenizer_and_model(
    model_name: str, *, local_only: bool
) -> tuple[AutoTokenizer, AutoModelForMaskedLM]:
    """Download (or load from cache) the PetBERT tokenizer and model.

    PetBERT (SAVSNET/PetBERT) is a BERT-style masked language model pre-trained
    on veterinary clinical text.  We use it as a feature extractor -- we never
    use the masked-LM head, only the base transformer's hidden states.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_only)
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
      3. Extracts the **[CLS] token embedding** -- the first token's hidden
         state from the last transformer layer (position 0 in the sequence).
         This single 768-dim vector serves as the fixed-size representation
         of the entire input text.

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

        # Take the [CLS] token (index 0) as the sentence-level embedding.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(
            cls_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        # Count real (non-padding) tokens per text for diagnostics.
        token_counts = attention_mask.sum(dim=1).detach().cpu().numpy()
        all_token_counts.append(token_counts.astype(np.int32, copy=False))

    return np.vstack(all_embeddings), np.concatenate(all_token_counts)


def embed_columns_weighted(
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    col_texts: dict[str, list[str]],
    weights: dict[str, float],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed each column independently then return a per-row weighted average.

    Each column is passed through PetBERT as its own batch so it gets the full
    ``max_length`` token budget.  Empty cells are excluded from each row's
    weighted average so they don't dilute the signal from populated columns.

    Args:
        col_texts: Mapping from column name to a list of N cleaned text strings.
        weights: Per-column scalar weights.  Columns absent from this dict
            default to 1.0.

    Returns:
        embeddings:   (N, 768) weighted-average embedding per row.
        token_counts: (N,) total non-padding tokens summed across columns.
    """
    if not col_texts:
        raise ValueError("col_texts must not be empty")

    cols = list(col_texts.keys())

    col_embeddings: list[np.ndarray] = []
    col_token_counts: list[np.ndarray] = []
    col_has_content: list[np.ndarray] = []

    for col in cols:
        texts = col_texts[col]
        has_content = np.array([bool(t) for t in texts], dtype=np.float32)
        col_has_content.append(has_content)

        emb, tok = embed_texts(
            tokenizer,
            model,
            texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            desc=f"Embedding [{col}]",
        )
        col_embeddings.append(emb)
        col_token_counts.append(tok)

    # raw weight per column — shape (num_cols,)
    raw_weights = np.array([weights.get(col, 1.0) for col in cols], dtype=np.float32)

    # effective weight: raw_weight * has_content so empty cells contribute 0
    has_content_matrix = np.stack(col_has_content, axis=0)  # (num_cols, N)
    eff_weights = has_content_matrix * raw_weights[:, np.newaxis]  # (num_cols, N)

    # normalize per row so effective weights sum to 1
    weight_sums = eff_weights.sum(axis=0, keepdims=True)  # (1, N)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    norm_weights = eff_weights / weight_sums  # (num_cols, N)

    # weighted sum: (num_cols, N, 768) * (num_cols, N, 1) → sum over cols → (N, 768)
    emb_stack = np.stack(col_embeddings, axis=0)
    weighted_embeddings = (emb_stack * norm_weights[:, :, np.newaxis]).sum(axis=0)

    total_token_counts = np.stack(col_token_counts, axis=0).sum(axis=0)

    return weighted_embeddings.astype(np.float32, copy=False), total_token_counts.astype(np.int32, copy=False)


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
