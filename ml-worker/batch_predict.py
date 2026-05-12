"""Standalone PetBERT batch prediction script for GCP Batch.

Reads env vars for paths, runs the PetBERT scan pipeline, and writes
predictions.json to the output directory. No web server — just runs once
and exits.
"""

import csv
import json
import os
import sys

# The ml/ package tree is copied into /ml at build time
sys.path.insert(0, "/ml")

from petbert_scan.pipeline import run_scan
from petbert_scan.types import ScanConfig


def main() -> None:
    job_id = os.environ["JOB_ID"]
    input_csv = os.environ["INPUT_CSV_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]
    model_path = os.environ.get("MODEL_PATH", "/mnt/gcs/models/petbert")
    labels_csv = os.environ.get("LABELS_CSV_PATH", "/mnt/gcs/models/labels/labels.csv")

    print(f"[batch_predict] job={job_id} input={input_csv} output={output_dir}")

    scan_out_dir = os.path.join(output_dir, "scan_output")
    os.makedirs(scan_out_dir, exist_ok=True)

    config = ScanConfig(
        csv_path=input_csv,
        id_col="anon_id",
        text_cols=("Clinical Diagnoses",),
        col_weights={"Clinical Diagnoses": 1.0},
        model_name=model_path,
        local_only=True,
        out_dir=scan_out_dir,
        max_rows=None,
        batch_size=16,
        max_length=256,
        neighbors_k=3,
        task="categorize",
        embedding_min_sim=0.6,
        device="auto",
        labels_csv_path=labels_csv,
        presence_classifier_path=None,
    )

    print("[batch_predict] Starting PetBERT scan...")
    outputs = run_scan(config)
    print(f"[batch_predict] Scan complete: {outputs.predictions_csv}")

    # Convert predictions CSV → JSON (same format the local ml-worker returns)
    predictions = []
    with open(outputs.predictions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({
                "anon_id": row.get("anon_id", ""),
                "original_text": row.get("original_text", ""),
                "predicted_term": row.get("predicted_term", ""),
                "predicted_group": row.get("predicted_group", ""),
                "predicted_code": row.get("predicted_code", ""),
                "confidence": row.get("confidence", ""),
                "method": row.get("method", ""),
            })

    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"[batch_predict] Wrote {len(predictions)} predictions to {predictions_path}")


if __name__ == "__main__":
    main()
