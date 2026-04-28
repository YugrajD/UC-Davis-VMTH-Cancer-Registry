"""ML worker microservice — wraps PetBERT scan pipeline as an HTTP endpoint.

Accepts a CSV upload (Dataset A), runs the PetBERT categorization pipeline,
and returns structured predictions.
"""

import csv
import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException

# The ml/ directory is volume-mounted at /ml
sys.path.insert(0, "/ml")

app = FastAPI(title="VMTH PetBERT ML Worker")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run PetBERT categorization on an uploaded CSV.

    Expects CSV with columns: anon_id, Clinical Diagnoses (+ optional others).
    PetBERT processes the Clinical Diagnoses column. Returns structured predictions JSON.
    """
    if not file.filename or not file.filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="File must be a .csv or .xlsx")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "input.csv")
        out_dir = os.path.join(tmpdir, "output")

        with open(csv_path, "wb") as f:
            f.write(contents)

        # Import pipeline at call time (heavy torch import)
        try:
            from production.petbert_pipeline.pipeline import run_scan
            from production.petbert_pipeline.types import ScanConfig
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to import PetBERT pipeline: {e}",
            )

        config = ScanConfig(
            csv_path=csv_path,
            id_col="anon_id",
            text_cols=("Clinical Diagnoses",),
            col_weights={"Clinical Diagnoses": 1.0},
            model_name=os.environ.get("PETBERT_MODEL_PATH", "/ml/models/petbert"),
            local_only=True,
            out_dir=out_dir,
            max_rows=None,
            batch_size=16,
            max_length=256,
            neighbors_k=3,
            task="categorize",
            embedding_min_sim=0.6,
            device="auto",
            labels_csv_path="/ml/labels/labels.csv",
            presence_classifier_path=None,
            embedding_cache_path=None,
        )

        try:
            outputs = run_scan(config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"PetBERT pipeline error: {e}",
            )

        # Read the predictions CSV and return as structured JSON
        predictions = []
        input_rows = 0
        with open(outputs.predictions_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_rows += 1
                predictions.append({
                    "anon_id": row.get("anon_id", ""),
                    "original_text": row.get("original_text", ""),
                    "predicted_term": row.get("predicted_term", ""),
                    "predicted_group": row.get("predicted_group", ""),
                    "predicted_code": row.get("predicted_code", ""),
                    "confidence": row.get("confidence", ""),
                    "method": row.get("method", ""),
                })

    return {
        "predictions": predictions,
        "input_rows": input_rows,
    }
