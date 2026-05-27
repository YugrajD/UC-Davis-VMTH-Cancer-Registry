"""GCS upload/download and GCP Batch submit/poll helpers."""

import json
import logging
import time
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


def download_petbert_summary_from_gcs(job_id: int) -> dict[str, Any]:
    """Download PetBERT's scan summary from GCS when the Batch job produced it."""
    blob_path = f"{_GCS_PREFIX}/{job_id}/scan_output/petbert_summary.json"
    blob = _get_bucket().blob(blob_path)
    try:
        raw = blob.download_as_bytes()
    except Exception as exc:
        logger.info("No PetBERT summary found for job %d at %s: %s", job_id, blob_path, exc)
        return {}
    try:
        summary = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid PetBERT summary JSON for job %d at %s: %s", job_id, blob_path, exc)
        return {}
    method_counts = summary.get("prediction_method_counts", {})
    logger.info("Downloaded PetBERT summary for job %d: methods=%s", job_id, method_counts)
    return summary


def cleanup_gcs_job_files(job_id: int) -> None:
    """Delete all blobs under uploads/{job_id}/."""
    bucket = _get_bucket()
    prefix = f"{_GCS_PREFIX}/{job_id}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)
        logger.info("Cleaned up %d GCS objects for job %d", len(blobs), job_id)


_REPORTS_PREFIX = "reports"


def upload_report_text_to_gcs(job_id: int, anon_id: str, text: str) -> str:
    """Upload a single patient's pathology report text to GCS.

    Returns the blob path (without gs:// prefix) stored in pathology_reports.gcs_path.
    """
    blob_path = f"{_REPORTS_PREFIX}/{job_id}/{anon_id}.txt"
    blob = _get_bucket().blob(blob_path)
    blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
    return blob_path


def download_report_text_from_gcs(gcs_path: str) -> str:
    """Download a patient's pathology report text from GCS."""
    blob = _get_bucket().blob(gcs_path)
    return blob.download_as_text(encoding="utf-8")


_LEGACY_MODEL_DIRS = frozenset({"checkpoints", "labels", "petbert"})


def list_model_folders() -> list[str]:
    """Return versioned model bundle names under gs://{bucket}/models/.

    Only returns proper bundle folders (e.g. production, model_a, model_b).
    Excludes the legacy flat-structure directories (checkpoints/, labels/,
    petbert/) that pre-date the versioned layout.
    Returns an empty list when no folders exist or GCS is unreachable.
    """
    client = _get_storage_client()
    iterator = client.list_blobs(
        settings.GCS_BUCKET,
        prefix="models/",
        delimiter="/",
    )
    list(iterator)  # consume iterator so .prefixes is populated
    folders = []
    for prefix in iterator.prefixes:
        # prefix is e.g. "models/production/" — extract the folder name
        name = prefix.removeprefix("models/").rstrip("/")
        if name and name not in _LEGACY_MODEL_DIRS:
            folders.append(name)
    return sorted(folders)


# ── GCP Batch helpers ───────────────────────────────────────────────────


def _get_batch_client() -> batch_v1.BatchServiceClient:
    return batch_v1.BatchServiceClient()


def submit_batch_job(job_id: int, model_folder: str = "production") -> str:
    """Submit a GCP Batch job for PetBERT inference.

    Returns the full job resource name
    (e.g. projects/{p}/locations/{l}/jobs/{j}).

    model_folder selects which GCS bundle under gs://{bucket}/models/ to use.
    Each folder must contain petbert/, labels/, and checkpoints/ subdirectories.

    Uses gsutil to download/upload files instead of gcsfuse volume mount
    to avoid compatibility issues with non-DNS-compliant bucket names.
    """
    client = _get_batch_client()

    bucket = settings.GCS_BUCKET
    local_data = "/tmp/batch_data"
    input_csv = f"{local_data}/dataset_a.csv"
    output_dir = local_data
    model_path = f"{local_data}/models/petbert"
    labels_csv = f"{local_data}/models/labels/labels.csv"
    case_presence_ckpt = f"{local_data}/models/checkpoints/case_presence_classifier.pt"
    group_ckpt = f"{local_data}/models/checkpoints/group_classifier_best.pt"
    lp_thresholds = f"{local_data}/models/checkpoints/lp_thresholds.json"
    uncommon_groups_file = f"{local_data}/models/checkpoints/uncommon_groups.txt"
    gcs_model_root = f"gs://{bucket}/models/{model_folder}"

    # Pre-task: download input CSV, model weights, labels, and classifiers from GCS
    # Uses google/cloud-sdk container since COS doesn't have gcloud installed.
    # Required files use set -e (hard failure); optional files use || echo so a
    # missing file degrades gracefully rather than aborting the job.
    setup_container = batch_v1.Runnable.Container(
        image_uri="gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
        commands=[
            "/bin/bash", "-c",
            " && ".join([
                "set -e",
                f"mkdir -p {local_data}/models/petbert {local_data}/models/labels {local_data}/models/checkpoints",
                # Required
                f"echo 'Downloading input CSV...'",
                f"gcloud storage cp 'gs://{bucket}/{_GCS_PREFIX}/{job_id}/dataset_a.csv' {input_csv}",
                f"echo 'Downloading model weights from {gcs_model_root}...'",
                f"gcloud storage cp -r '{gcs_model_root}/petbert/*' {model_path}/",
                f"echo 'Downloading labels...'",
                f"gcloud storage cp '{gcs_model_root}/labels/labels.csv' {labels_csv}",
                f"echo 'Downloading group classifier (required)...'",
                f"gcloud storage cp '{gcs_model_root}/checkpoints/group_classifier_best.pt' {group_ckpt}",
                # Optional — missing files disable the corresponding pipeline stage
                f"echo 'Downloading optional checkpoints...'",
                f"gcloud storage cp '{gcs_model_root}/checkpoints/case_presence_classifier.pt' {case_presence_ckpt} || echo 'No case_presence_classifier.pt; Stage 1 gate disabled.'",
                f"gcloud storage cp '{gcs_model_root}/checkpoints/lp_thresholds.json' {lp_thresholds} || echo 'No lp_thresholds.json; using global LP threshold.'",
                f"gcloud storage cp '{gcs_model_root}/checkpoints/uncommon_groups.txt' {uncommon_groups_file} || echo 'No uncommon_groups.txt; using empty set.'",
                f"echo 'Download complete.'",
            ]),
        ],
        volumes=[f"{local_data}:{local_data}"],
    )
    setup_runnable = batch_v1.Runnable(container=setup_container)

    # Main task: run PetBERT inference container
    main_container = batch_v1.Runnable.Container(
        image_uri=settings.GCP_BATCH_IMAGE_URI,
        commands=[],
        volumes=[f"{local_data}:{local_data}"],
    )

    main_runnable = batch_v1.Runnable(
        container=main_container,
        environment=batch_v1.Environment(
            variables={
                "JOB_ID": str(job_id),
                "INPUT_CSV_PATH": input_csv,
                "OUTPUT_DIR": output_dir,
                "MODEL_PATH": model_path,
                "LABELS_CSV_PATH": labels_csv,
                "CASE_PRESENCE_CLASSIFIER_PATH": case_presence_ckpt,
                "GROUP_CLASSIFIER_PATH": group_ckpt,
                "LP_THRESHOLDS_JSON_PATH": lp_thresholds,
                "UNCOMMON_GROUPS_PATH": uncommon_groups_file,
                "CASE_PRESENCE_THRESHOLD": str(settings.CASE_PRESENCE_THRESHOLD),
                "GROUP_CLASSIFIER_THRESHOLD": str(settings.GROUP_CLASSIFIER_THRESHOLD),
            },
        ),
    )

    # Post-task: upload predictions back to GCS
    upload_container = batch_v1.Runnable.Container(
        image_uri="gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
        commands=[
            "/bin/bash", "-c",
            (
                f"set -e && "
                f"echo 'Uploading predictions...' && "
                f"gcloud storage cp {local_data}/predictions.json "
                f"'gs://{bucket}/{_GCS_PREFIX}/{job_id}/predictions.json' && "
                f"if [ -d {local_data}/scan_output ]; then "
                f"echo 'Uploading scan output diagnostics...' && "
                f"gcloud storage cp -r {local_data}/scan_output/* "
                f"'gs://{bucket}/{_GCS_PREFIX}/{job_id}/scan_output/'; "
                f"else echo 'No scan_output directory produced.'; fi && "
                f"echo 'Upload complete.'"
            ),
        ],
        volumes=[f"{local_data}:{local_data}"],
    )
    upload_runnable = batch_v1.Runnable(container=upload_container)

    task_spec = batch_v1.TaskSpec(
        runnables=[setup_runnable, main_runnable, upload_runnable],
        max_retry_count=0,
        max_run_duration=f"{settings.GCP_BATCH_TIMEOUT_HOURS * 3600}s",
    )

    task_group = batch_v1.TaskGroup(
        task_count=1,
        task_spec=task_spec,
    )

    sa_email = settings.GCP_BATCH_SERVICE_ACCOUNT or None
    allocation = batch_v1.AllocationPolicy(
        instances=[
            batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                policy=batch_v1.AllocationPolicy.InstancePolicy(
                    machine_type=settings.GCP_BATCH_MACHINE_TYPE,
                ),
            )
        ],
        service_account=batch_v1.ServiceAccount(email=sa_email) if sa_email else None,
    )

    batch_job = batch_v1.Job(
        task_groups=[task_group],
        allocation_policy=allocation,
        logs_policy=batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING,
        ),
    )

    parent = f"projects/{settings.GCP_PROJECT_ID}/locations/{settings.GCP_REGION}"
    # Include timestamp to avoid ALREADY_EXISTS on re-runs of the same job_id
    batch_job_id = f"petbert-ingest-{job_id}-{int(time.time())}"

    created = client.create_job(
        parent=parent,
        job_id=batch_job_id,
        job=batch_job,
    )

    logger.info("Submitted Batch job %s for ingestion job %d", created.name, job_id)
    return created.name


def cancel_batch_job(job_name: str) -> None:
    """Cancel a running GCP Batch job.

    Uses the Batch cancel API which stops execution without deleting the job
    record, preserving logs in Cloud Logging.
    Silently ignores errors (e.g. job already finished) so callers don't need
    to handle races between polling and cancellation.
    """
    try:
        client = _get_batch_client()
        client.cancel_job(name=job_name)
        logger.info("Cancelled Batch job %s", job_name)
    except Exception:
        logger.warning("Could not cancel Batch job %s (may have already finished)", job_name, exc_info=True)


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
