"""Confidence cascade: supervised model + kNN retrieval fallback.

For each prediction from the supervised model:
  - If confidence >= threshold: keep the supervised prediction
  - If confidence < threshold: replace with kNN retrieval prediction

This combines the supervised model's strength on common cancers with
kNN's ability to retrieve rare cancer types by definition similarity.

Usage:
  python ml/training/simcse/cascade_predict.py --threshold 0.4
  python ml/training/simcse/cascade_predict.py --threshold 0.4 --test-cases ml/output/splits/test_cases.txt
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from production.petbert_pipeline.embedding_cache import load_cache


def load_predictions(csv_path: str, cases_filter: set[str] | None = None) -> dict[str, list[dict]]:
    """Load supervised predictions grouped by case_id."""
    by_case: dict[str, list[dict]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["case_id"]
            if cases_filter and cid not in cases_filter:
                continue
            by_case.setdefault(cid, []).append(row)
    return by_case


def build_knn_lookup(
    case_ids: list[str],
    report_embs: np.ndarray,
    label_embs: np.ndarray,
    labels_meta: list[dict],
    *,
    k: int = 5,
    cases_filter: set[str] | None = None,
) -> dict[str, dict]:
    """Build a case_id → kNN prediction lookup."""
    # Filter cases
    if cases_filter:
        mask = np.array([cid in cases_filter for cid in case_ids])
        filtered_ids = [cid for cid, keep in zip(case_ids, mask) if keep]
        filtered_embs = report_embs[mask]
    else:
        filtered_ids = case_ids
        filtered_embs = report_embs

    # L2 normalize
    r_norms = np.linalg.norm(filtered_embs, axis=1, keepdims=True).clip(1e-9)
    l_norms = np.linalg.norm(label_embs, axis=1, keepdims=True).clip(1e-9)
    r_unit = filtered_embs / r_norms
    l_unit = label_embs / l_norms

    lookup: dict[str, dict] = {}
    chunk_size = 512
    for start in range(0, len(filtered_ids), chunk_size):
        end = min(len(filtered_ids), start + chunk_size)
        sims = r_unit[start:end] @ l_unit.T

        topk_idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for i in range(end - start):
            top_i = topk_idx[i]
            top_sims = sims[i, top_i]
            order = np.argsort(-top_sims)
            top_i = top_i[order]
            top_sims = top_sims[order]

            # Weighted group vote
            group_votes: dict[str, float] = {}
            for idx, s in zip(top_i, top_sims):
                g = labels_meta[int(idx)]["group"]
                group_votes[g] = group_votes.get(g, 0.0) + float(s)
            best_group = max(group_votes, key=lambda g: group_votes[g])

            # Best term in winning group
            for idx, s in zip(top_i, top_sims):
                if labels_meta[int(idx)]["group"] == best_group:
                    meta = labels_meta[int(idx)]
                    lookup[filtered_ids[start + i]] = {
                        "predicted_term": meta["term"],
                        "predicted_group": meta["group"],
                        "predicted_code": meta["code"],
                        "confidence": float(s),
                        "method": "knn_fallback",
                    }
                    break

    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description="Confidence cascade: supervised + kNN fallback.")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Global confidence threshold below which kNN fallback is used")
    parser.add_argument("--adaptive-thresholds", default="",
                        help="Path to per-group thresholds JSON (overrides --threshold per group)")
    parser.add_argument("--supervised-csv",
                        default="ml/output/production/contrastive/petbert_predictions.csv")
    parser.add_argument("--out-csv",
                        default="ml/output/production/cascade/cascade_predictions.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-cases", default="")
    parser.add_argument("--labels-csv", default=config.LABELS_CSV)
    args = parser.parse_args()

    # Load filter
    filter_ids = None
    if args.test_cases and Path(args.test_cases).exists():
        with open(args.test_cases) as f:
            filter_ids = {line.strip() for line in f if line.strip()}

    # Load supervised predictions
    supervised = load_predictions(args.supervised_csv, filter_ids)
    print(f"Supervised predictions: {sum(len(v) for v in supervised.values())} rows, "
          f"{len(supervised)} cases")

    # Load embedding cache for kNN
    cache = load_cache(
        config.EMBEDDING_CACHE_NPZ,
        model_name="ml/output/checkpoints/contrastive",
        report_csv_path=config.REPORTS_CSV,
        labels_csv_path=config.LABELS_CSV,
    )
    if not cache:
        print("ERROR: Embedding cache not available.")
        return 1

    # Load labels metadata
    labels_meta: list[dict] = []
    with open(args.labels_csv, encoding="utf-8-sig") as f:
        lines = f.readlines()
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

    min_n = min(len(labels_meta), cache["label_embeddings"].shape[0])
    labels_meta = labels_meta[:min_n]
    label_embs = cache["label_embeddings"][:min_n]

    # Build kNN lookup for all cases
    knn_lookup = build_knn_lookup(
        cache["case_ids"], cache["mean_embeddings"], label_embs, labels_meta,
        k=args.k, cases_filter=filter_ids,
    )
    print(f"kNN lookup: {len(knn_lookup)} cases")

    # Load adaptive thresholds if provided
    adaptive = None
    if args.adaptive_thresholds and Path(args.adaptive_thresholds).exists():
        import json
        with open(args.adaptive_thresholds) as f:
            adaptive = json.load(f)
        print(f"Using adaptive per-group thresholds from {args.adaptive_thresholds}")

    # Apply cascade
    out_rows: list[dict] = []
    n_supervised = 0
    n_fallback = 0
    n_skipped = 0

    for cid, preds in supervised.items():
        for pred in preds:
            conf = float(pred.get("confidence", 0))
            group = pred.get("predicted_group", "")
            t = adaptive.get(group, args.threshold) if adaptive else args.threshold
            if pred["predicted_term"] == "Uncategorized":
                # Keep abstentions — don't override with kNN
                out_rows.append({**pred})
                n_supervised += 1
            elif t > 0 and conf < t:
                # Below threshold — use kNN if available
                if cid in knn_lookup:
                    knn_pred = knn_lookup[cid]
                    out_rows.append({
                        "case_id": cid,
                        "diagnosis_index": pred.get("diagnosis_index", 1),
                        **knn_pred,
                    })
                    n_fallback += 1
                else:
                    out_rows.append({**pred})
                    n_supervised += 1
            else:
                out_rows.append({**pred})
                n_supervised += 1

    print(f"Cascade (threshold={args.threshold}): "
          f"{n_supervised} supervised, {n_fallback} kNN fallback, {n_skipped} skipped")

    # Write output
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "diagnosis_index", "predicted_term", "predicted_group",
                  "predicted_code", "confidence", "method"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
