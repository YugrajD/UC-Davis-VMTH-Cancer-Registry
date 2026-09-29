"""The PetBERT embedding cache, keyed on content hash (not mtime).

Legacy (``production/petbert_pipeline/embedding_cache.py``) validated against
model name + report/labels CSV mtime, ±1s — a 2026-08 real incident (labels.csv
mtime moved with no content change) shows why that's fragile. This cache is
keyed on ``content_key()``: sha256 of the report CSV bytes + labels CSV bytes +
the embedding fingerprint (backbone sha256, section-spec version, max_length).
Any of those changing produces a different key, so the old entry is simply
unused rather than silently reused or invalidated by an unrelated mtime touch.

Stored under ``config.EMBEDDING_CACHE_DIR`` as ``<key>.npz``, one file per
distinct (report, labels, backbone) combination — never bundled into a
generation directory (ml-rewrite-plan.md, Artefacts).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from report_mapping.model.generation import npz_col_key


@dataclass
class EmbeddingCache:
    case_ids: list[str]
    col_embeddings: dict[str, np.ndarray]   # section columns + "concat_3", each (N, 768) or (N, 2304)
    col_has_content: dict[str, np.ndarray]  # section columns only, (N,) bool
    label_texts: list[str]
    label_embeddings: np.ndarray            # (M, 768)


def _sha256_bytes(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_key(report_csv: str | Path, labels_csv: str | Path, fingerprint: dict) -> str:
    """sha256 of report bytes + labels bytes + the embedding fingerprint, hex-digested."""
    digest = hashlib.sha256()
    digest.update(_sha256_bytes(report_csv).encode("utf-8"))
    digest.update(_sha256_bytes(labels_csv).encode("utf-8"))
    digest.update(repr(sorted(fingerprint.items())).encode("utf-8"))
    return digest.hexdigest()


def cache_path(key: str, cache_dir: str | Path | None = None) -> Path:
    """``cache_dir`` overrides ``config.EMBEDDING_CACHE_DIR`` — e.g. a scratch
    directory for an L2b re-embed run that must never touch the real
    (machine-local) cache."""
    directory = Path(cache_dir) if cache_dir is not None else config.EMBEDDING_CACHE_DIR
    return directory / f"{key}.npz"


def load(key: str, cache_dir: str | Path | None = None) -> EmbeddingCache | None:
    """None on any miss: file absent or unreadable. Never raises."""
    path = cache_path(key, cache_dir)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        col_names = [str(c) for c in data["col_names"]]
        col_embeddings, col_has_content = {}, {}
        for col in col_names:
            s = npz_col_key(col)
            col_embeddings[col] = data[f"col_{s}"]
            if f"has_{s}" in data.files:
                col_has_content[col] = data[f"has_{s}"]
        return EmbeddingCache(
            case_ids=[str(c) for c in data["case_ids"]],
            col_embeddings=col_embeddings,
            col_has_content=col_has_content,
            label_texts=[str(t) for t in data["label_texts"]],
            label_embeddings=np.asarray(data["label_embeddings"]),
        )
    except Exception as exc:  # noqa: BLE001 - a corrupt cache file is just a miss
        print(f"Embedding cache at {path} failed to load ({exc}) -> treating as a miss")
        return None


def save(key: str, cache: EmbeddingCache, cache_dir: str | Path | None = None) -> None:
    path = cache_path(key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    col_names = list(cache.col_embeddings.keys())
    arrays: dict[str, np.ndarray] = {
        "case_ids": np.array(cache.case_ids, dtype=object),
        "col_names": np.array(col_names, dtype=object),
        "label_texts": np.array(cache.label_texts, dtype=object),
        "label_embeddings": cache.label_embeddings,
    }
    for col in col_names:
        s = npz_col_key(col)
        arrays[f"col_{s}"] = cache.col_embeddings[col]
        if col in cache.col_has_content:
            arrays[f"has_{s}"] = cache.col_has_content[col]
    np.savez(path, **arrays)
