"""ECS Fargate entrypoint: pull inputs from S3, run batch_predict, push outputs.

Env (set by the task definition / RunTask overrides):
  S3_BUCKET, JOB_ID            required
  MODEL_FOLDER                 model bundle under models/ (default: production)
  AWS_REGION                   standard AWS SDK region
  AWS_S3_ENDPOINT_URL          optional, for S3 emulators

S3 layout mirrors the backend (app/services/s3_service.py):
  uploads/{JOB_ID}/dataset_a.csv        input
  models/{MODEL_FOLDER}/{petbert,labels,checkpoints}/...
  uploads/{JOB_ID}/predictions.json     output
  uploads/{JOB_ID}/scan_output/*        output (diagnostics)
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError

import batch_predict

LOCAL_DATA = "/tmp/batch_data"

# Optional bundle files: a missing key disables the matching pipeline stage.
OPTIONAL_CHECKPOINTS = (
    "group_classifier_best.pt",
    "case_presence_classifier.pt",
    "lp_thresholds.json",
    "uncommon_groups.txt",
)


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


def _download_optional(s3, bucket: str, key: str, target: str) -> None:
    try:
        s3.download_file(bucket, key, target)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            print(f"[entrypoint] optional {key} not found; skipping")
            return
        raise


def main() -> None:
    bucket = os.environ["S3_BUCKET"]
    job_id = os.environ["JOB_ID"]
    model_folder = os.environ.get("MODEL_FOLDER") or "production"
    s3 = _client()

    models_dir = os.path.join(LOCAL_DATA, "models")
    input_csv = os.path.join(LOCAL_DATA, "dataset_a.csv")
    os.makedirs(os.path.join(models_dir, "checkpoints"), exist_ok=True)

    job_prefix = f"uploads/{job_id}"
    model_prefix = f"models/{model_folder}"

    print(f"[entrypoint] downloading input CSV for job {job_id}")
    s3.download_file(bucket, f"{job_prefix}/dataset_a.csv", input_csv)

    print(f"[entrypoint] downloading model bundle s3://{bucket}/{model_prefix}/")
    if _download_prefix(s3, bucket, f"{model_prefix}/petbert/", os.path.join(models_dir, "petbert")) == 0:
        raise RuntimeError(f"No PetBERT weights under s3://{bucket}/{model_prefix}/petbert/")
    os.makedirs(os.path.join(models_dir, "labels"), exist_ok=True)
    s3.download_file(
        bucket, f"{model_prefix}/labels/labels.csv", os.path.join(models_dir, "labels", "labels.csv")
    )
    for name in OPTIONAL_CHECKPOINTS:
        _download_optional(
            s3, bucket, f"{model_prefix}/checkpoints/{name}",
            os.path.join(models_dir, "checkpoints", name),
        )
    print("[entrypoint] download complete")

    ckpt = os.path.join(models_dir, "checkpoints")
    os.environ.update({
        "INPUT_CSV_PATH": input_csv,
        "OUTPUT_DIR": LOCAL_DATA,
        "MODEL_PATH": os.path.join(models_dir, "petbert"),
        "LABELS_CSV_PATH": os.path.join(models_dir, "labels", "labels.csv"),
        "CASE_PRESENCE_CLASSIFIER_PATH": os.path.join(ckpt, "case_presence_classifier.pt"),
        "GROUP_CLASSIFIER_PATH": os.path.join(ckpt, "group_classifier_best.pt"),
        "LP_THRESHOLDS_JSON_PATH": os.path.join(ckpt, "lp_thresholds.json"),
        "UNCOMMON_GROUPS_PATH": os.path.join(ckpt, "uncommon_groups.txt"),
    })

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
