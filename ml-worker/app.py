"""ML worker microservice — serves report-mapping predictions over HTTP.

Loads one report-mapping generation (the bundle) at startup and refuses to start unless its manifest
and embedding fingerprint verify. The bundle root is the parent of PETBERT_MODEL_PATH; every other
path variable that is set must point inside the same bundle (handoff/worker_format.py).
All model code is imported from ml/ (mounted or copied at /ml).
"""

import io
import os
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, "/ml")

import io_utils
from handoff import worker_format
from report_mapping.inference import predict as predict_mod
from report_mapping.model.generation import load_generation
from report_mapping.sections import clean_text

BUNDLE_ROOT = worker_format.resolve_bundle_root(os.environ, "PETBERT_MODEL_PATH")
worker_format.verify_worker_bundle(BUNDLE_ROOT)  # refuse to start on a missing or changed bundle file
GENERATION = load_generation(BUNDLE_ROOT)

app = FastAPI(title="VMTH PetBERT ML Worker")


@app.get("/health")
async def health():
    return {"status": "ok", "source_version": GENERATION.generation_id}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict codes for an uploaded Dataset A CSV (``anon_id``, ``Text``; utf-8, as the backend writes it)."""
    if not file.filename or not file.filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="File must be a .csv or .xlsx")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    upload = io_utils.read_csv(io.BytesIO(contents), encoding="utf-8", dtype=str, keep_default_na=False)
    if worker_format.UPLOAD_ID_COL not in upload.columns:
        raise HTTPException(status_code=400, detail=f"Missing column {worker_format.UPLOAD_ID_COL!r}")
    ids = upload[worker_format.UPLOAD_ID_COL].map(clean_text).tolist()
    text = upload.get(worker_format.UPLOAD_TEXT_COL, pd.Series([""] * len(upload)))

    try:
        rows = predict_mod.predict_frame(GENERATION, worker_format.upload_to_reports(upload), ids, device_arg="auto")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "predictions": worker_format.response_rows(rows, dict(zip(ids, text.map(clean_text)))),
        "input_rows": len(rows),
    }
