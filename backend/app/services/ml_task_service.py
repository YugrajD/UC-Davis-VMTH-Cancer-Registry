"""ECS Fargate RunTask helpers for the on-demand PetBERT inference task."""

import logging
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

# Terminal/non-terminal states returned by get_task_status, normalised from
# ECS lastStatus so the poll loop in job_processor stays small.
STATE_PENDING = "PENDING"
STATE_RUNNING = "RUNNING"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"

_PENDING_STATUSES = frozenset({"PROVISIONING", "PENDING", "ACTIVATING"})

# Retry transient AWS/network errors only - our own RuntimeErrors are final.
_retry_aws = retry(
    retry=retry_if_exception_type((ClientError, BotoCoreError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True,
)


@lru_cache(maxsize=1)
def _get_client():
    return boto3.client("ecs", region_name=settings.AWS_REGION)


@_retry_aws
def submit_ml_task(job_id: int, model_folder: str = "production") -> str:
    """Start the ML Fargate task for an ingestion job. Returns the task ARN.

    The container downloads uploads/{job_id}/dataset_a.csv and the model bundle
    models/{model_folder}/ from S3 as one verified unit (manifest.json checked
    at worker startup — see ml-worker/s3_batch_entrypoint.py), then uploads
    predictions.json back.
    """
    response = _get_client().run_task(
        cluster=settings.ECS_CLUSTER_ARN,
        taskDefinition=settings.ML_TASK_DEFINITION_ARN,
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": settings.ml_task_subnet_ids_list,
                "securityGroups": [settings.ML_TASK_SECURITY_GROUP_ID],
                "assignPublicIp": "DISABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": settings.ML_TASK_CONTAINER_NAME,
                    "environment": [
                        {"name": "JOB_ID", "value": str(job_id)},
                        {"name": "MODEL_FOLDER", "value": model_folder},
                    ],
                }
            ]
        },
        tags=[{"key": "ingestion_job_id", "value": str(job_id)}],
    )
    failures = response.get("failures") or []
    if failures or not response.get("tasks"):
        reason = failures[0].get("reason", "unknown") if failures else "no task returned"
        raise RuntimeError(f"Could not start ML task: {reason}")
    task_arn = response["tasks"][0]["taskArn"]
    logger.info("Started ML task %s for ingestion job %d", task_arn, job_id)
    return task_arn


@_retry_aws
def get_task_status(task_arn: str) -> tuple[str, str | None]:
    """Return (state, error_message | None); state is PENDING/RUNNING/SUCCEEDED/FAILED."""
    response = _get_client().describe_tasks(cluster=settings.ECS_CLUSTER_ARN, tasks=[task_arn])
    if not response.get("tasks"):
        failures = response.get("failures") or []
        reason = failures[0].get("reason", "task not found") if failures else "task not found"
        return STATE_FAILED, reason

    task = response["tasks"][0]
    status = task.get("lastStatus", "")
    if status in _PENDING_STATUSES:
        return STATE_PENDING, None
    if status != "STOPPED":
        return STATE_RUNNING, None

    container = next(
        (c for c in task.get("containers", []) if c.get("name") == settings.ML_TASK_CONTAINER_NAME),
        None,
    )
    exit_code = container.get("exitCode") if container else None
    if exit_code == 0:
        return STATE_SUCCEEDED, None
    reason = (container or {}).get("reason") or task.get("stoppedReason") or "no details"
    return STATE_FAILED, f"exit code {exit_code}: {reason}"


def stop_ml_task(task_arn: str, reason: str = "Cancelled by user") -> None:
    """Stop a running ML task. Errors are swallowed (the task may have finished)."""
    try:
        _get_client().stop_task(cluster=settings.ECS_CLUSTER_ARN, task=task_arn, reason=reason)
        logger.info("Stopped ML task %s", task_arn)
    except Exception:
        logger.warning("Could not stop ML task %s (may have already finished)", task_arn, exc_info=True)
