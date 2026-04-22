"""kNN retrieval-based cancer label prediction using SimCSE embeddings.

For each case report:
  1. Embed the report text using the SimCSE-finetuned model
  2. Find the k nearest label definition embeddings (cosine similarity)
  3. Predict the closest label's term and group

This serves as the self-supervised fallback in the confidence cascade:
the supervised model handles common cancers well, but for rare types where
it lacks training signal, kNN retrieval can match based on semantic
similarity to the label definition text alone.

Usage:
  python ml/training/simcse/knn_predict.py --model ml/output/checkpoints/simcse
  python ml/training/simcse/knn_predict.py --model ml/output/checkpoints/simcse \
    --test-cases ml/output/splits/test_cases.txt
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from model.constants import DEFAULT_TEXT_COLS
from production.petbert_pipeline import device_from_arg


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_batch(
    tokenizer, model, texts: list[str], *, device, max_length: int = 512, batch_size: int = 32
) -> np.ndarray:
    model.eval()
    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = _mean_pool(out.last_hidden_state, attention_mask)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def load_labels(labels_csv: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load label definitions. Returns (texts_for_embedding, terms, groups, codes)."""
    texts, terms, groups, codes = [], [], [], []
    with open(labels_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row.get("Term", "").strip()
            group = row.get("Group", "").strip()
            code = row.get("Vet-ICD-O-canine-1 code", "").strip()
            if not term or not group:
                continue
            texts.append(f"{term} {group}")
            terms.append(term)
            groups.append(group)
            codes.append(code)
    return texts, terms, groups, codes


def load_reports(
    csv_path: str, text_cols: tuple[str, ...], cases_txt: str = ""
) -> tuple[list[str], list[str]]:
    """Load report texts. Returns (case_ids, combined_texts)."""
    filter_ids = None
    if cases_txt and Path(cases_txt).exists():
        with open(cases_txt) as f:
            filter_ids = {line.strip() for line in f if line.strip()}

    case_ids, texts = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["case_id"]
            if filter_ids and cid not in filter_ids:
                continue
            parts = [row.get(col, "").strip() for col in text_cols]
            combined = " ".join(p for p in parts if p)
            case_ids.append(cid)
            texts.append(combined)
    return case_ids, texts


def knn_predict(
    report_embs: np.ndarray,
    label_embs: np.ndarray,
    label_terms: list[str],
    label_groups: list[str],
    label_codes: list[str],
    *,
    k: int = 1,
) -> list[dict]:
    """For each report, find the nearest label and return predictions."""
    # L2 normalize for cosine similarity via dot product
    r_norm = report_embs / np.linalg.norm(report_embs, axis=1, keepdims=True).clip(1e-9)
    l_norm = label_embs / np.linalg.norm(label_embs, axis=1, keepdims=True).clip(1e-9)

    predictions = []
    chunk_size = 512
    for start in tqdm(range(0, len(r_norm), chunk_size), desc="kNN search", unit="chunk"):
        end = min(len(r_norm), start + chunk_size)
        sims = r_norm[start:end] @ l_norm.T  # (chunk, M)
        if k == 1:
            best_idx = sims.argmax(axis=1)
            best_sim = sims[np.arange(end - start), best_idx]
            for i, (idx, sim) in enumerate(zip(best_idx, best_sim)):
                predictions.append({
                    "predicted_term": label_terms[idx],
                    "predicted_group": label_groups[idx],
                    "predicted_code": label_codes[idx],
                    "confidence": float(sim),
                    "method": "knn_simcse",
                })
        else:
            # k > 1: majority vote on group, pick highest-sim term within winning group
            topk_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
            for i in range(end - start):
                top_i = topk_idx[i]
                top_sims = sims[i, top_i]
                # Sort by similarity descending
                order = np.argsort(-top_sims)
                top_i = top_i[order]
                top_sims = top_sims[order]
                # Majority vote on group
                group_votes: dict[str, float] = {}
                for idx, s in zip(top_i, top_sims):
                    g = label_groups[idx]
                    group_votes[g] = group_votes.get(g, 0.0) + s
                best_group = max(group_votes, key=lambda g: group_votes[g])
                # Pick highest-sim term within winning group
                for idx, s in zip(top_i, top_sims):
                    if label_groups[idx] == best_group:
                        predictions.append({
                            "predicted_term": label_terms[idx],
                            "predicted_group": label_groups[idx],
                            "predicted_code": label_codes[idx],
                            "confidence": float(s),
                            "method": "knn_simcse",
                        })
                        break

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="kNN retrieval predictions using SimCSE model.")
    parser.add_argument("--model", default="ml/output/checkpoints/simcse",
                        help="SimCSE model checkpoint directory")
    parser.add_argument("--out-csv", default="ml/output/production/simcse/knn_predictions.csv",
                        help="Output predictions CSV")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors (default: 5)")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--test-cases", default="",
                        help="Only predict for these cases (for evaluation)")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    print(f"kNN prediction | model={args.model} | k={args.k} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, local_files_only=True)
    model.to(device)

    # Embed labels
    label_texts, label_terms, label_groups, label_codes = load_labels(config.LABELS_CSV)
    print(f"Labels: {len(label_texts)}")
    label_embs = embed_batch(tokenizer, model, label_texts, device=device, batch_size=args.batch_size)

    # Embed reports
    case_ids, report_texts = load_reports(config.REPORTS_CSV, DEFAULT_TEXT_COLS, cases_txt=args.test_cases)
    print(f"Reports: {len(case_ids)}")
    report_embs = embed_batch(tokenizer, model, report_texts, device=device, batch_size=args.batch_size)

    # kNN prediction
    predictions = knn_predict(report_embs, label_embs, label_terms, label_groups, label_codes, k=args.k)

    # Write output CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "diagnosis_index", "predicted_term", "predicted_group",
                  "predicted_code", "confidence", "method"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid, pred in zip(case_ids, predictions):
            writer.writerow({"case_id": cid, "diagnosis_index": 1, **pred})

    print(f"Wrote {len(predictions)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
