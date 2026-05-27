"""ML worker microservice — wraps PetBERT scan pipeline as an HTTP endpoint.

Accepts a CSV upload (Dataset A), runs the PetBERT categorization pipeline,
and returns structured predictions.
"""

import csv
import json
import os
import sys
import tempfile
from collections import defaultdict

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException

# The ml/ directory is volume-mounted at /ml
sys.path.insert(0, "/ml")

app = FastAPI(title="VMTH PetBERT ML Worker")

DEFAULT_CASE_PRESENCE_THRESHOLD = 0.5
DEFAULT_GROUP_CLASSIFIER_THRESHOLD = 0.3


def _join_numbered(rows: list[dict], field: str) -> str:
    """Aggregate per-rank rows into '1) val1 2) val2' format.

    Mirrors batch_predict.py so ingestion_service.parse_predictions handles
    both pipeline paths identically.
    """
    return " ".join(f"{i + 1}) {r.get(field, '')}" for i, r in enumerate(rows))


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _build_scan_config_kwargs(
    *,
    expanded_csv_path: str,
    out_dir: str,
    model_path: str,
    labels_csv: str,
    case_presence_classifier: str | None,
    group_classifier: str | None,
    lp_thresholds_json: str | None,
    uncommon_groups: str | None,
) -> dict:
    config_kwargs: dict = dict(
        csv_path=expanded_csv_path,
        id_col="anon_id",
        model_name=model_path,
        local_only=True,
        out_dir=out_dir,
        max_rows=None,
        batch_size=16,
        max_length=256,
        neighbors_k=3,
        task="categorize",
        embedding_min_sim=0.6,
        device="auto",
        labels_csv_path=labels_csv,
        group_classifier_path=group_classifier,
        group_classifier_threshold=_float_env(
            "GROUP_CLASSIFIER_THRESHOLD", DEFAULT_GROUP_CLASSIFIER_THRESHOLD
        ),
        case_presence_classifier_path=case_presence_classifier,
        case_presence_threshold=_float_env(
            "CASE_PRESENCE_THRESHOLD", DEFAULT_CASE_PRESENCE_THRESHOLD
        ),
        label_presence_thresholds_json=lp_thresholds_json,
    )
    if uncommon_groups:
        config_kwargs["uncommon_groups_path"] = uncommon_groups
    return config_kwargs


def _load_json_file(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _compact_petbert_summary(summary: dict) -> dict:
    if not summary:
        return {}
    return {
        "input_rows": summary.get("input_rows"),
        "prediction_method_counts": summary.get("prediction_method_counts", {}),
        "predicted_group_counts": summary.get("predicted_group_counts", {}),
        "thresholds": summary.get("thresholds", {}),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run PetBERT categorization on an uploaded CSV.

    Expects CSV with columns: anon_id, Text (+ optional others).
    PetBERT processes the Text column (pathology report). Returns structured predictions JSON.
    """
    if not file.filename or not file.filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="File must be a .csv or .xlsx")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    model_path = os.environ.get("PETBERT_MODEL_PATH", "/ml/models/petbert")
    labels_csv = os.environ.get("LABELS_CSV_PATH", "/ml/ICD_labels/labels.csv")
    group_classifier = os.environ.get("GROUP_CLASSIFIER_PATH") or None

    _cp = os.environ.get("CASE_PRESENCE_CLASSIFIER_PATH") or ""
    case_presence_classifier = _cp if (_cp and os.path.exists(_cp)) else None
    _lp = os.environ.get("LP_THRESHOLDS_JSON_PATH") or ""
    lp_thresholds_json = _lp if (_lp and os.path.exists(_lp)) else None
    _ug = os.environ.get("UNCOMMON_GROUPS_PATH") or ""
    uncommon_groups = _ug if (_ug and os.path.exists(_ug)) else None

    if not group_classifier or not os.path.exists(group_classifier):
        raise HTTPException(
            status_code=500,
            detail=f"Group classifier not found at {group_classifier!r}. Set GROUP_CLASSIFIER_PATH.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_csv_path = os.path.join(tmpdir, "input_raw.csv")
        expanded_csv_path = os.path.join(tmpdir, "input.csv")
        out_dir = os.path.join(tmpdir, "output")

        with open(raw_csv_path, "wb") as f:
            f.write(contents)

        # Dataset A has a single 'Text' column. The 4-stage pipeline expects three
        # named section columns. Duplicate Text into all three so the concat-3
        # embedding shape matches classifier training — mirroring batch_predict.py.
        df = pd.read_csv(raw_csv_path, encoding="latin-1")
        text_col = df["Text"] if "Text" in df.columns else pd.Series([""] * len(df))
        for col in ("HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "ANCILLARY TESTS"):
            if col not in df.columns:
                df[col] = text_col
        df.to_csv(expanded_csv_path, index=False)

        # Build anon_id → Text lookup from the raw input for pathology report storage.
        text_by_id: dict[str, str] = {}
        with open(raw_csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                aid = (row.get("anon_id") or "").strip()
                if aid:
                    text_by_id[aid] = (row.get("Text") or "").strip()

        # Import pipeline at call time (heavy torch import)
        try:
            from production.petbert_pipeline.pipeline import run_scan
            from production.petbert_pipeline.types import ScanConfig
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PetBERT pipeline is not available",
            )

        config_kwargs = _build_scan_config_kwargs(
            expanded_csv_path=expanded_csv_path,
            out_dir=out_dir,
            model_path=model_path,
            labels_csv=labels_csv,
            case_presence_classifier=case_presence_classifier,
            group_classifier=group_classifier,
            lp_thresholds_json=lp_thresholds_json,
            uncommon_groups=uncommon_groups,
        )

        config = ScanConfig(**config_kwargs)

        try:
            outputs = run_scan(config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="PetBERT pipeline failed",
            )
        petbert_summary = _compact_petbert_summary(_load_json_file(outputs.summary_json))

        # Group per-rank rows by patient. The pipeline writes one row per
        # (patient, diagnosis_index); aggregate into numbered format so
        # ingestion_service.parse_predictions works the same as for GCP Batch.
        by_patient: dict[str, list[dict]] = defaultdict(list)
        input_rows = 0
        with open(outputs.predictions_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_rows += 1
                anon_id = (row.get("anon_id") or "").strip()
                if anon_id:
                    by_patient[anon_id].append(row)

        predictions = []
        for anon_id, rows in by_patient.items():
            rows.sort(key=lambda r: int(r.get("diagnosis_index", 1)))
            if len(rows) == 1:
                r = rows[0]
                predictions.append({
                    "anon_id": anon_id,
                    "original_text": text_by_id.get(anon_id, ""),
                    "predicted_term": r.get("predicted_term", ""),
                    "predicted_group": r.get("predicted_group", ""),
                    "predicted_code": r.get("predicted_code", ""),
                    "confidence": r.get("confidence", ""),
                    "method": r.get("method", ""),
                })
            else:
                predictions.append({
                    "anon_id": anon_id,
                    "original_text": text_by_id.get(anon_id, ""),
                    "predicted_term": _join_numbered(rows, "predicted_term"),
                    "predicted_group": _join_numbered(rows, "predicted_group"),
                    "predicted_code": _join_numbered(rows, "predicted_code"),
                    "confidence": _join_numbered(rows, "confidence"),
                    "method": _join_numbered(rows, "method"),
                })

    return {
        "predictions": predictions,
        "input_rows": input_rows,
        "petbert_summary": petbert_summary,
    }
