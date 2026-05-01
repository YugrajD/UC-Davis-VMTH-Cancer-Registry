"""Build a case-level binary dataset for CasePresenceClassifier.

Positive: any case with at least one confirmed cancer annotation.
Negative: cases present in the embedding cache with no cancer annotation.

Output NPZ keys: case_ids, embeddings (N, 768), targets (N,) float32.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import config


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_dataset(
    *,
    annotation_csv: str = config.ANNOTATION_CSV,
    embedding_cache: str = config.EMBEDDING_CACHE_NPZ,
    out: str = config.CASE_PRESENCE_DATASET_NPZ,
    train_cases_txt: str = "",
) -> None:
    # --- Identify cancer-positive cases from annotation CSV ---
    ann_rows = load_csv(Path(annotation_csv))
    cancer_case_ids: set[str] = {
        row["case_id"]
        for row in ann_rows
        if row.get("matched_term", "").strip()
    }
    print(f"Annotation: {len(cancer_case_ids)} cancer-positive cases")

    # --- Optionally restrict to train split ---
    train_ids: set[str] | None = None
    if train_cases_txt and Path(train_cases_txt).exists():
        with open(train_cases_txt, encoding="utf-8") as f:
            train_ids = {line.strip() for line in f if line.strip()}
        print(f"Train split active: {len(train_ids)} cases")

    # --- Load mean embeddings directly from cache NPZ ---
    cache_path = Path(embedding_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {embedding_cache}\n"
            "Run the pipeline once with --embedding-cache to build it."
        )
    raw = np.load(cache_path, allow_pickle=True)
    cached_ids: list[str] = raw["case_ids"].tolist()
    mean_embs: np.ndarray = raw["mean_embeddings"]  # (N, 768)

    # --- Build dataset ---
    out_ids: list[str] = []
    out_embs: list[np.ndarray] = []
    out_targets: list[float] = []

    for i, cid in enumerate(cached_ids):
        if train_ids is not None and cid not in train_ids:
            continue
        out_ids.append(cid)
        out_embs.append(mean_embs[i])
        out_targets.append(1.0 if cid in cancer_case_ids else 0.0)

    embeddings = np.array(out_embs, dtype=np.float32)
    targets = np.array(out_targets, dtype=np.float32)

    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    print(f"Dataset: {len(targets)} cases  ({n_pos} cancer, {n_neg} non-cancer)")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        case_ids=np.array(out_ids),
        embeddings=embeddings,
        targets=targets,
    )
    print(f"Saved to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build case-level binary dataset for CasePresenceClassifier."
    )
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument("--embedding-cache", default=config.EMBEDDING_CACHE_NPZ)
    parser.add_argument("--out", default=config.CASE_PRESENCE_DATASET_NPZ)
    parser.add_argument(
        "--train-cases",
        default="",
        help="Path to train_cases.txt. Restricts dataset to train split cases.",
    )
    args = parser.parse_args()
    build_dataset(
        annotation_csv=args.annotation_csv,
        embedding_cache=args.embedding_cache,
        out=args.out,
        train_cases_txt=args.train_cases,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
