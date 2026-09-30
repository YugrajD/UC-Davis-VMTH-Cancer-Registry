"""ECS Fargate entrypoint: pull inputs from S3, run batch_predict, push outputs.

Env (set by the task definition / RunTask overrides):
  S3_BUCKET, JOB_ID            required
  MODEL_FOLDER                 model bundle under models/ (default: production)
  AWS_REGION                   standard AWS SDK region
  AWS_S3_ENDPOINT_URL          optional, for S3 emulators

S3 layout mirrors the backend (app/services/s3_service.py):
  uploads/{JOB_ID}/dataset_a.csv                        input
  models/{MODEL_FOLDER}/{petbert,labels,checkpoints}/... one verified bundle
  uploads/{JOB_ID}/predictions.json                     output
  uploads/{JOB_ID}/scan_output/*                        output (diagnostics)

The whole models/{MODEL_FOLDER}/ prefix is downloaded as one recursive copy,
preserving its internal layout (petbert/, labels/, checkpoints/ — including
checkpoints/label_presence/*.pt and checkpoints/thresholds.json — and
manifest.json). There are no optional files: batch_predict.py's own
worker_format.resolve_bundle_root() + load_generation() verify the bundle's
manifest and embedding fingerprint at startup and refuse to run on anything
missing or changed, so a partial download must fail the job here rather than
produce predictions from a mismatched bundle. Only MODEL_PATH is set — every
other path the worker needs is derived from the bundle root.
"""

import os
import sys

import boto3

import batch_predict

LOCAL_DATA = "/tmp/batch_data"


def _client():
    kwargs = {}
    if os.environ.get("AWS_S3_ENDPOINT_URL"):
        kwargs["endpoint_url"] = os.environ["AWS_S3_ENDPOINT_URL"]
    return boto3.client("s3", **kwargs)


def _download_prefix(s3, bucket: str, prefix: str, dest_dir: str) -> int:
    """Download every object under prefix into dest_dir. Returns object count."""
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            target = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file(bucket, obj["Key"], target)
            count += 1
    return count


def main() -> None:
    bucket = os.environ["S3_BUCKET"]
    job_id = os.environ["JOB_ID"]
    model_folder = os.environ.get("MODEL_FOLDER") or "production"
    s3 = _client()

    model_root = os.path.join(LOCAL_DATA, "models", model_folder)
    input_csv = os.path.join(LOCAL_DATA, "dataset_a.csv")

    job_prefix = f"uploads/{job_id}"
    model_prefix = f"models/{model_folder}/"

    print(f"[entrypoint] downloading input CSV for job {job_id}")
    s3.download_file(bucket, f"{job_prefix}/dataset_a.csv", input_csv)

    print(f"[entrypoint] downloading model bundle s3://{bucket}/{model_prefix}")
    count = _download_prefix(s3, bucket, model_prefix, model_root)
    if count == 0:
        raise RuntimeError(f"No objects found under s3://{bucket}/{model_prefix}")
    print(f"[entrypoint] downloaded {count} bundle files")

    os.environ.update({
        "INPUT_CSV_PATH": input_csv,
        "OUTPUT_DIR": LOCAL_DATA,
        "MODEL_PATH": os.path.join(model_root, "petbert"),
    })

    # batch_predict.main() verifies the bundle's manifest.json and embedding
    # fingerprint before predicting, and refuses to run on a mismatch — see
    # module docstring above.
    batch_predict.main()

    print("[entrypoint] uploading outputs")
    s3.upload_file(
        os.path.join(LOCAL_DATA, "predictions.json"), bucket, f"{job_prefix}/predictions.json"
    )
    scan_dir = os.path.join(LOCAL_DATA, "scan_output")
    if os.path.isdir(scan_dir):
        for root, _dirs, files in os.walk(scan_dir):
            for fname in files:
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, scan_dir)
                s3.upload_file(path, bucket, f"{job_prefix}/scan_output/{rel}")
    else:
        print("[entrypoint] no scan_output directory produced")
    print("[entrypoint] done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # non-zero exit lets the backend see FAILED
        print(f"[entrypoint] FAILED: {exc!r}", file=sys.stderr)
        raise
