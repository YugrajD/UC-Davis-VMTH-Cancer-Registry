"""Standalone report-mapping batch prediction for GCP Batch.

Reads env vars for paths, loads the bundle (its root is the parent of MODEL_PATH, which only has to be named
<root>/petbert; load_generation runs the spec check and refuses a bad bundle), predicts each upload row's Text
as one flat text, and writes predictions.json to OUTPUT_DIR in the same payload app.py returns. Torch threads
come from TORCH_NUM_THREADS (default min(8, CPU count)). No web server — runs once and exits.
"""

import json
import os
import sys

import torch

for ml_path in (
    "/ml",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml")),
):
    if os.path.isdir(ml_path) and ml_path not in sys.path:
        sys.path.insert(0, ml_path)

import io_utils
from handoff import worker_format
from report_mapping.inference import predict as predict_mod
from report_mapping.model import resolve_device


def main() -> None:
    job_id = os.environ["JOB_ID"]
    input_csv = os.environ["INPUT_CSV_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]

    torch.set_num_threads(worker_format.torch_threads(os.environ))
    device = resolve_device("auto")
    bundle_root = worker_format.resolve_bundle_root(os.environ, "MODEL_PATH")
    generation = predict_mod.load_generation(bundle_root, device)  # spec check first; refuses a bad bundle
    print(f"[batch_predict] job={job_id} input={input_csv} output={output_dir} "
          f"source_version={generation.generation_id}")

    upload = io_utils.read_csv(input_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    ids, texts = worker_format.upload_to_texts(upload)
    rows = predict_mod.predict_texts(ids, texts, generation, device)
    predictions = worker_format.response_rows(rows.to_dict("records"), dict(zip(ids, texts)))

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"[batch_predict] Wrote {len(predictions)} predictions to {predictions_path}")


if __name__ == "__main__":
    main()
