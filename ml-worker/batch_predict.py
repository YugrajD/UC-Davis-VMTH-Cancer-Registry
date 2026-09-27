"""Standalone report-mapping batch prediction for GCP Batch.

Reads env vars for paths, verifies the bundle (its root is the parent of MODEL_PATH; every other path
variable that is set must point inside it), predicts, and writes predictions.json to OUTPUT_DIR in the
same payload app.py returns. No web server — runs once and exits.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/ml")

import io_utils
from handoff import worker_format
from report_mapping.inference import predict as predict_mod
from report_mapping.model.generation import load_generation
from report_mapping.sections import clean_text


def main() -> None:
    job_id = os.environ["JOB_ID"]
    input_csv = os.environ["INPUT_CSV_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]

    bundle_root = worker_format.resolve_bundle_root(os.environ, "MODEL_PATH")
    worker_format.verify_worker_bundle(bundle_root)  # refuse on a missing or changed bundle file
    generation = load_generation(bundle_root)
    print(f"[batch_predict] job={job_id} input={input_csv} output={output_dir} "
          f"source_version={generation.generation_id}")

    upload = io_utils.read_csv(input_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    ids = upload[worker_format.UPLOAD_ID_COL].map(clean_text).tolist()
    text = upload.get(worker_format.UPLOAD_TEXT_COL, pd.Series([""] * len(upload)))
    rows = predict_mod.predict_frame(generation, worker_format.upload_to_reports(upload), ids, device_arg="auto")
    predictions = worker_format.response_rows(rows, dict(zip(ids, text.map(clean_text))))

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"[batch_predict] Wrote {len(predictions)} predictions to {predictions_path}")


if __name__ == "__main__":
    main()
