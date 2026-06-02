"""Build PyTorch Dataset for PetBERT Finetuning

Extracts ground truth labels from keyword_predictions.csv and pairs them
with report text (FINAL COMMENT, etc.) from report.csv. Unmatched cases
are treated as uncategorized (Class 0).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from labels.taxonomy import load_labels_taxonomy

def build_group_map(labels_csv: str) -> dict[str, int]:
    """Map Vet-ICD-O group name -> class index (1 to N)."""
    # Class 0 is reserved for Uncategorized/Negative
    taxonomy = load_labels_taxonomy(labels_csv)
    groups = sorted(list(set(l.group for l in taxonomy)))
    return {group: idx + 1 for idx, group in enumerate(groups)}


def build_dataset(
    *,
    reports_csv: str = "database/data/output/report.csv",
    predictions_csv: str = "ml/output/diagnoses/keyword_predictions.csv",
    labels_csv: str = "ml/labels/labels.csv",
    out_dir: str = "ml/data/finetune_dataset",
    model_name: str = "SAVSNET/PetBERT",
    text_cols: tuple[str, ...] = ("FINAL COMMENT", "HISTOPATHOLOGICAL SUMMARY", "ANCILLARY TESTS"),
    max_length: int = 512,
    val_split: float = 0.15,
    seed: int = 42,
) -> None:
    print(f"Loading taxonomy groups from {labels_csv}...")
    group_to_idx = build_group_map(labels_csv)
    idx_to_group = {idx: grp for grp, idx in group_to_idx.items()}
    idx_to_group[0] = "Uncategorized"
    print(f"  Found {len(group_to_idx)} cancer groups.")

    print(f"Loading keyword predictions from {predictions_csv}...")
    case_to_groups = {}
    with open(predictions_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"]
            if case_id not in case_to_groups:
                case_to_groups[case_id] = set()
            
            group = row.get("matched_group", "").strip()
            method = row.get("method", "").strip()
            if method != "no_match" and group:
                if group in group_to_idx:
                    case_to_groups[case_id].add(group_to_idx[group])
                else:
                    print(f"Warning: Unknown group '{group}' in predictions.")

    matched_cases = sum(1 for groups in case_to_groups.values() if len(groups) > 0)
    unmatched_cases = len(case_to_groups) - matched_cases
    print(f"  Found {len(case_to_groups)} total cases.")
    print(f"  {matched_cases} cases with cancer keywords, {unmatched_cases} Uncategorized.")

    print(f"Loading report text from {reports_csv}...")
    texts = []
    labels = []  # Single integer class per case (multi-label handled as multi-rows or simplifying for classification)
    
    # remember that for standard fine-tuning classification (CrossEntropyLoss), we need 1 label per item.
    # If a case has multiple cancer groups, we'll create duplicate entries for it,
    # one for each group. For Uncategorized, label is 0.
    
    with open(reports_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        # Ensure requested columns exist
        available_cols = set(reader.fieldnames or [])
        valid_cols = [col for col in text_cols if col in available_cols]
        if not valid_cols:
            raise ValueError(f"None of {text_cols} found in {reports_csv}. Available: {available_cols}")

        for row in reader:
            case_id = row["case_id"]
            if case_id not in case_to_groups:
                continue
                
            # Construct text: concatenate valid columns
            parts = []
            for col in valid_cols:
                content = row.get(col, "").strip()
                if content:
                    parts.append(f"[{col}] {content}")
            
            report_text = "\n\n".join(parts)
            if len(report_text) < 10:
                continue # Skip empty reports
                
            groups = case_to_groups[case_id]
            if not groups:
                # Uncategorized (0)
                texts.append(report_text)
                labels.append(0)
            else:
                # One row per cancer group
                for g in groups:
                    texts.append(report_text)
                    labels.append(g)

    print(f"Generated {len(texts)} (text, label) pairs.")

    num_classes = len(idx_to_group)
    
    # Calculate class weights
    label_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    class_weights = np.ones(num_classes, dtype=np.float32)
    for i in range(num_classes):
        count = label_counts[i]
        if count > 0:
            # inverse frequency formula
            class_weights[i] = total_samples / (num_classes * count)
        else:
            class_weights[i] = 0.0
            
    # cap maximum weight to avoid giant gradients for very rare classes
    max_weight = 20.0
    class_weights = np.clip(class_weights, 0.0, max_weight)
            
    print("Class weight sample:")
    for i in [0, 1, 2, 3]:
        print(f"  Class {i} ({idx_to_group.get(i, 'Unknown')}): count={label_counts[i]}, weight={class_weights[i]:.2f}")

    print(f"Tokenizing using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    from datasets import ClassLabel

    hf_dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "text": texts  # Keep raw text for debugging, optional
    })

    # cast labels to ClassLabel for stratification
    hf_dataset = hf_dataset.cast_column("labels", ClassLabel(num_classes=num_classes))

    print(f"Splitting into {1-val_split:.0%} train / {val_split:.0%} val (random split)...")
    split_dataset = hf_dataset.train_test_split(test_size=val_split, seed=seed)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to {out_path}...")
    split_dataset.save_to_disk(str(out_path))
    
    np.save(str(out_path / "class_weights.npy"), class_weights)
    
    import json
    with open(out_path / "config.json", "w") as f:
        json.dump({
            "num_classes": num_classes,
            "group_to_idx": group_to_idx,
            "idx_to_group": idx_to_group,
            "label_counts": label_counts.tolist(),
        }, f, indent=2)

    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-csv", default="database/data/output/report.csv")
    parser.add_argument("--predictions-csv", default="ml/output/diagnoses/keyword_predictions.csv")
    parser.add_argument("--labels-csv", default="ml/labels/labels.csv")
    parser.add_argument("--out-dir", default="ml/data/finetune_dataset")
    parser.add_argument("--model", default="SAVSNET/PetBERT")
    args = parser.parse_args()
    
    build_dataset(
        reports_csv=args.reports_csv,
        predictions_csv=args.predictions_csv,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        model_name=args.model,
    )
