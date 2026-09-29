"""S3 upload/download helpers for job inputs, ML outputs, and report text."""

import json
import logging
from functools import lru_cache
from typing import Any, Iterator

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)

_UPLOADS_PREFIX = "uploads"
_REPORTS_PREFIX = "reports"
_MODELS_PREFIX = "models"

_LEGACY_MODEL_DIRS = frozenset({"checkpoints", "labels", "petbert"})


@lru_cache(maxsize=1)
def _get_client():
    """Return a shared boto3 S3 client (thread-safe; reused across executors)."""
    kwargs: dict[str, Any] = {"region_name": settings.AWS_REGION}
    if settings.AWS_S3_ENDPOINT_URL:
        # Floci (and most S3 emulators) need path-style addressing.
        kwargs["endpoint_url"] = settings.AWS_S3_ENDPOINT_URL
        kwargs["config"] = Config(s3={"addressing_style": "path"})
    return boto3.client("s3", **kwargs)


def job_prefix(job_id: int) -> str:
    return f"{_UPLOADS_PREFIX}/{job_id}"


def upload_csv(job_id: int, filename: str, data: bytes) -> str:
    """Upload a CSV to uploads/{job_id}/{filename}. Returns the object key."""
    key = f"{job_prefix(job_id)}/{filename}"
    _get_client().put_object(
        Bucket=settings.S3_BUCKET, Key=key, Body=data, ContentType="text/csv"
    )
    logger.info("Uploaded s3://%s/%s (%d bytes)", settings.S3_BUCKET, key, len(data))
    return key


def download_csv(job_id: int, filename: str = "dataset_a.csv") -> bytes:
    """Download uploads/{job_id}/{filename}."""
    key = f"{job_prefix(job_id)}/{filename}"
    return _get_client().get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"].read()


def csv_exists(job_id: int, filename: str = "dataset_a.csv") -> bool:
    key = f"{job_prefix(job_id)}/{filename}"
    try:
        _get_client().head_object(Bucket=settings.S3_BUCKET, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


def stream_csv(job_id: int, filename: str = "dataset_a.csv") -> Iterator[bytes]:
    """Yield the stored CSV in chunks (for StreamingResponse)."""
    key = f"{job_prefix(job_id)}/{filename}"
    body = _get_client().get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"]
    try:
        yield from body.iter_chunks(chunk_size=64 * 1024)
    finally:
        body.close()


def download_predictions(job_id: int) -> list[dict[str, Any]]:
    """Download and parse predictions.json written by the ML task."""
    key = f"{job_prefix(job_id)}/predictions.json"
    raw = _get_client().get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"].read()
    predictions = json.loads(raw)
    logger.info("Downloaded %d predictions for job %d", len(predictions), job_id)
    return predictions


def download_petbert_summary(job_id: int) -> dict[str, Any]:
    """Download PetBERT's scan summary when the ML task produced it."""
    key = f"{job_prefix(job_id)}/scan_output/petbert_summary.json"
    try:
        raw = _get_client().get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"].read()
    except Exception as exc:
        logger.info("No PetBERT summary found for job %d at %s: %s", job_id, key, exc)
        return {}
    try:
        summary = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid PetBERT summary JSON for job %d at %s: %s", job_id, key, exc)
        return {}
    logger.info(
        "Downloaded PetBERT summary for job %d: methods=%s",
        job_id, summary.get("prediction_method_counts", {}),
    )
    return summary


def cleanup_job_files(job_id: int) -> None:
    """Delete all objects under uploads/{job_id}/."""
    client = _get_client()
    prefix = f"{job_prefix(job_id)}/"
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
        objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objects:
            client.delete_objects(
                Bucket=settings.S3_BUCKET, Delete={"Objects": objects, "Quiet": True}
            )
            deleted += len(objects)
    if deleted:
        logger.info("Cleaned up %d S3 objects for job %d", deleted, job_id)


def upload_report_text(job_id: int, anon_id: str, text: str) -> str:
    """Upload a single patient's pathology report text.

    Returns the object key stored in pathology_reports.storage_path.
    """
    key = f"{_REPORTS_PREFIX}/{job_id}/{anon_id}.txt"
    _get_client().put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
    return key


def download_report_text(storage_path: str) -> str:
    """Download a patient's pathology report text by object key."""
    obj = _get_client().get_object(Bucket=settings.S3_BUCKET, Key=storage_path)
    return obj["Body"].read().decode("utf-8")


def list_model_folders() -> list[str]:
    """Return versioned model bundle names under s3://{bucket}/models/.

    Excludes the legacy flat-structure directories (checkpoints/, labels/,
    petbert/) that pre-date the versioned layout.
    """
    client = _get_client()
    prefix = f"{_MODELS_PREFIX}/"
    folders: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"].removeprefix(prefix).rstrip("/")
            if name and name not in _LEGACY_MODEL_DIRS:
                folders.append(name)
    return sorted(folders)
