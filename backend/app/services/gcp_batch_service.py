"""GCS upload/download and GCP Batch submit/poll helpers."""

import json
import logging
from typing import Any

from google.cloud import batch_v1, storage

from app.config import settings

logger = logging.getLogger(__name__)

_GCS_PREFIX = "uploads"


# ── GCS helpers ──────────────────────────────────────────────────────────


def _get_storage_client() -> storage.Client:
    return storage.Client(project=settings.GCP_PROJECT_ID)


def _get_bucket() -> storage.Bucket:
    return _get_storage_client().bucket(settings.GCS_BUCKET)


def upload_csv_to_gcs(job_id: int, filename: str, data: bytes) -> str:
    """Upload a CSV file to GCS. Returns the gs:// URI."""
    blob_path = f"{_GCS_PREFIX}/{job_id}/{filename}"
    blob = _get_bucket().blob(blob_path)
    blob.upload_from_string(data, content_type="text/csv")
    uri = f"gs://{settings.GCS_BUCKET}/{blob_path}"
    logger.info("Uploaded %s (%d bytes)", uri, len(data))
    return uri


def download_predictions_from_gcs(job_id: int) -> list[dict[str, Any]]:
    """Download and parse predictions.json from GCS."""
    blob_path = f"{_GCS_PREFIX}/{job_id}/predictions.json"
    blob = _get_bucket().blob(blob_path)
    raw = blob.download_as_bytes()
    predictions = json.loads(raw)
    logger.info("Downloaded %d predictions for job %d", len(predictions), job_id)
    return predictions


def cleanup_gcs_job_files(job_id: int) -> None:
    """Delete all blobs under uploads/{job_id}/."""
    bucket = _get_bucket()
    prefix = f"{_GCS_PREFIX}/{job_id}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)
        logger.info("Cleaned up %d GCS objects for job %d", len(blobs), job_id)


# ── GCP Batch helpers ───────────────────────────────────────────────────


def _get_batch_client() -> batch_v1.BatchServiceClient:
    return batch_v1.BatchServiceClient()


def submit_batch_job(job_id: int) -> str:
    """Submit a GCP Batch job for PetBERT inference.

    Returns the full job resource name
    (e.g. projects/{p}/locations/{l}/jobs/{j}).
    """
    client = _get_batch_client()

    gcs_mount_path = "/mnt/gcs"
    input_csv = f"{gcs_mount_path}/{_GCS_PREFIX}/{job_id}/dataset_a.csv"
    output_dir = f"{gcs_mount_path}/{_GCS_PREFIX}/{job_id}"
    model_path = f"{gcs_mount_path}/models/petbert"
    labels_csv = f"{gcs_mount_path}/models/labels/labels.csv"

    container = batch_v1.Runnable.Container(
        image_uri=settings.GCP_BATCH_IMAGE_URI,
        commands=[],
    )
    container.environment = batch_v1.Environment(
        variables={
            "JOB_ID": str(job_id),
            "INPUT_CSV_PATH": input_csv,
            "OUTPUT_DIR": output_dir,
            "MODEL_PATH": model_path,
            "LABELS_CSV_PATH": labels_csv,
        },
    )

    runnable = batch_v1.Runnable(container=container)

    task_spec = batch_v1.TaskSpec(
        runnables=[runnable],
        max_retry_count=0,
        max_run_duration=f"{settings.GCP_BATCH_TIMEOUT_HOURS * 3600}s",
    )

    # Mount the GCS bucket so the container reads/writes files directly
    gcs_volume = batch_v1.Volume(
        gcs=batch_v1.GCS(remote_path=settings.GCS_BUCKET),
        mount_path=gcs_mount_path,
    )
    task_spec.volumes = [gcs_volume]

    task_group = batch_v1.TaskGroup(
        task_count=1,
        task_spec=task_spec,
    )

    allocation = batch_v1.AllocationPolicy(
        instances=[
            batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                policy=batch_v1.AllocationPolicy.InstancePolicy(
                    machine_type=settings.GCP_BATCH_MACHINE_TYPE,
                ),
            )
        ],
    )

    batch_job = batch_v1.Job(
        task_groups=[task_group],
        allocation_policy=allocation,
        logs_policy=batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING,
        ),
    )

    parent = f"projects/{settings.GCP_PROJECT_ID}/locations/{settings.GCP_REGION}"
    batch_job_id = f"petbert-ingest-{job_id}"

    created = client.create_job(
        parent=parent,
        job_id=batch_job_id,
        job=batch_job,
    )

    logger.info("Submitted Batch job %s for ingestion job %d", created.name, job_id)
    return created.name


def get_batch_job_status(job_name: str) -> tuple[str, str | None]:
    """Poll the Batch job state.

    Returns (state_name, error_message | None).
    Possible states: QUEUED, SCHEDULED, RUNNING, SUCCEEDED, FAILED, DELETION_IN_PROGRESS.
    """
    client = _get_batch_client()
    job = client.get_job(name=job_name)
    state_name = batch_v1.JobStatus.State(job.status.state).name

    error_msg = None
    if state_name == "FAILED" and job.status.status_events:
        last_event = job.status.status_events[-1]
        error_msg = last_event.description

    return state_name, error_msg
