import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_tokenizer_and_model(
    model_name: str, *, local_only: bool
) -> tuple[AutoTokenizer, AutoModelForMaskedLM]:
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
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_embeddings: list[np.ndarray] = []
    all_token_counts: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(
            cls_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        token_counts = attention_mask.sum(dim=1).detach().cpu().numpy()
        all_token_counts.append(token_counts.astype(np.int32, copy=False))

    return np.vstack(all_embeddings), np.concatenate(all_token_counts)


def cosine_similarity_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    query_norms = np.where(query_norms == 0, 1.0, query_norms)
    ref_norms = np.linalg.norm(ref, axis=1, keepdims=True)
    ref_norms = np.where(ref_norms == 0, 1.0, ref_norms)
    return (query / query_norms) @ (ref / ref_norms).T


def topk_cosine_neighbors(
    embeddings: np.ndarray, *, k: int, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D)")
    n, _ = embeddings.shape
    if n == 0:
        return np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = embeddings / norms
    neighbor_idx = np.empty((n, k), dtype=np.int64)
    neighbor_sim = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = unit[start:end] @ unit.T
        row_indices = np.arange(start, end)
        sims[np.arange(end - start), row_indices] = -np.inf

        part = np.argpartition(-sims, kth=min(k, n - 1) - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        top = np.take_along_axis(part, order, axis=1)
        top_sims = np.take_along_axis(part_sims, order, axis=1)

        neighbor_idx[start:end] = top
        neighbor_sim[start:end] = top_sims.astype(np.float32, copy=False)

    return neighbor_idx, neighbor_sim

