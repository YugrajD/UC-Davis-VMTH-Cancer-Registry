"""kNN retrieval using the existing embedding cache (no additional training).

The adapted backbone was already contrastively trained on report-label pairs,
so its embeddings are suitable for nearest-neighbor retrieval. This script
uses the cached embeddings directly — no model loading, runs in seconds.

Usage:
  python ml/training/simcse/knn_from_cache.py
  python ml/training/simcse/knn_from_cache.py --test-cases ml/output/splits/test_cases.txt --k 5
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from production.petbert_pipeline.embedding_cache import load_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="kNN retrieval from embedding cache.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--out-csv", default="ml/output/production/knn/knn_predictions.csv")
    parser.add_argument("--test-cases", default="", help="Only predict for these cases")
    parser.add_argument("--labels-csv", default=config.LABELS_CSV)
    args = parser.parse_args()

    # Load embedding cache
    cache = load_cache(
        config.EMBEDDING_CACHE_NPZ,
        model_name="ml/output/checkpoints/contrastive",
        report_csv_path=config.REPORTS_CSV,
        labels_csv_path=config.LABELS_CSV,
    )
    if not cache:
        print("ERROR: Embedding cache not available. Run the pipeline first.")
        return 1

    case_ids = cache["case_ids"]
    report_embs = cache["mean_embeddings"]  # (N, 768)
    label_embs = cache["label_embeddings"]  # (M, 768)
    label_texts = cache["label_texts"]  # "term group" format

    # Load label metadata (term, group, code) from labels CSV
    # The CSV has a BOM + title row before the actual headers on line 2
    labels_meta: list[dict] = []
    with open(args.labels_csv, encoding="utf-8-sig") as f:
        lines = f.readlines()
        # Find the header line (contains "Term")
        header_idx = 0
        for i, line in enumerate(lines):
            if "Term" in line and "Group" in line:
                header_idx = i
                break
        reader = csv.DictReader(lines[header_idx:])
        for row in reader:
            term = row.get("Term", "").strip()
            group = row.get("Group", "").strip()
            code = row.get("Vet-ICD-O-canine-1 code", "").strip()
            if term and group:
                labels_meta.append({"term": term, "group": group, "code": code})

    if len(labels_meta) != label_embs.shape[0]:
        print(f"WARNING: labels_meta ({len(labels_meta)}) != label_embs ({label_embs.shape[0]})")
        min_n = min(len(labels_meta), label_embs.shape[0])
        labels_meta = labels_meta[:min_n]
        label_embs = label_embs[:min_n]

    # Filter to test cases if specified
    filter_ids = None
    if args.test_cases and Path(args.test_cases).exists():
        with open(args.test_cases) as f:
            filter_ids = {line.strip() for line in f if line.strip()}

    # Build index of case_ids to keep
    if filter_ids:
        mask = np.array([cid in filter_ids for cid in case_ids])
        case_ids_filtered = [cid for cid, keep in zip(case_ids, mask) if keep]
        report_embs_filtered = report_embs[mask]
    else:
        case_ids_filtered = case_ids
        report_embs_filtered = report_embs

    print(f"Reports: {len(case_ids_filtered)}, Labels: {len(labels_meta)}, k={args.k}")

    # L2 normalize
    r_norms = np.linalg.norm(report_embs_filtered, axis=1, keepdims=True).clip(1e-9)
    l_norms = np.linalg.norm(label_embs, axis=1, keepdims=True).clip(1e-9)
    r_unit = report_embs_filtered / r_norms
    l_unit = label_embs / l_norms

    # Compute similarities in chunks to limit memory
    chunk_size = 512
    n = len(case_ids_filtered)
    k = args.k

    predictions: list[dict] = []
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = r_unit[start:end] @ l_unit.T  # (chunk, M)

        if k == 1:
            best_idx = sims.argmax(axis=1)
            best_sim = sims[np.arange(end - start), best_idx]
            for i in range(end - start):
                meta = labels_meta[best_idx[i]]
                predictions.append({
                    "case_id": case_ids_filtered[start + i],
                    "diagnosis_index": 1,
                    "predicted_term": meta["term"],
                    "predicted_group": meta["group"],
                    "predicted_code": meta["code"],
                    "confidence": float(best_sim[i]),
                    "method": "knn_retrieval",
                })
        else:
            # k-NN with weighted group voting
            topk_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
            for i in range(end - start):
                top_i = topk_idx[i]
                top_sims = sims[i, top_i]
                order = np.argsort(-top_sims)
                top_i = top_i[order]
                top_sims = top_sims[order]

                # Weighted vote on group
                group_votes: dict[str, float] = {}
                for idx, s in zip(top_i, top_sims):
                    g = labels_meta[int(idx)]["group"]
                    group_votes[g] = group_votes.get(g, 0.0) + float(s)
                best_group = max(group_votes, key=lambda g: group_votes[g])

                # Pick highest-sim term within winning group
                for idx, s in zip(top_i, top_sims):
                    if labels_meta[int(idx)]["group"] == best_group:
                        meta = labels_meta[int(idx)]
                        predictions.append({
                            "case_id": case_ids_filtered[start + i],
                            "diagnosis_index": 1,
                            "predicted_term": meta["term"],
                            "predicted_group": meta["group"],
                            "predicted_code": meta["code"],
                            "confidence": float(s),
                            "method": "knn_retrieval",
                        })
                        break

        if (start // chunk_size) % 5 == 0:
            print(f"  {end}/{n} cases processed")

    # Write output
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "diagnosis_index", "predicted_term", "predicted_group",
                  "predicted_code", "confidence", "method"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Wrote {len(predictions)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
