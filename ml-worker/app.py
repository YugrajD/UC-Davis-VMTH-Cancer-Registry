"""ML worker microservice — serves report-mapping predictions over HTTP.

Loads one report-mapping generation (the bundle) at startup and refuses to start unless its spec check passes
(files match the manifest, the generation's architecture and text spec equal this code's, the thresholds were
calibrated for its weights). The bundle root is the parent of PETBERT_MODEL_PATH (the PetBERT-era name for the
model path, kept for the backend; only its parent directory is used). Each upload row's Text is scored as one
flat text. Torch threads come from TORCH_NUM_THREADS (default min(8, CPU count)).
All model code is imported from ml/ (mounted or copied at /ml).
"""

import io
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
import torch

sys.path.insert(0, "/ml")

import io_utils
from handoff import worker_format
from report_mapping.inference import predict as predict_mod
from report_mapping.model import resolve_device

torch.set_num_threads(worker_format.torch_threads(os.environ))
DEVICE = resolve_device("auto")
BUNDLE_ROOT = worker_format.resolve_bundle_root(os.environ, "PETBERT_MODEL_PATH")
# load_generation runs the spec check first: the worker refuses to start on a bad bundle.
GENERATION = predict_mod.load_generation(BUNDLE_ROOT, DEVICE)

app = FastAPI(title="VMTH Report-Mapping ML Worker")


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
    ids, texts = worker_format.upload_to_texts(upload)

    try:
        rows = predict_mod.predict_texts(ids, texts, GENERATION, DEVICE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Report-mapping prediction failed")

    return {
        "predictions": worker_format.response_rows(rows.to_dict("records"), dict(zip(ids, texts))),
        "input_rows": len(rows),
    }
