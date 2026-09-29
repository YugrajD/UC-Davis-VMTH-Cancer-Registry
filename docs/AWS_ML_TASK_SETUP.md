# AWS ML task & S3 setup

PetBERT inference runs as an on-demand **ECS Fargate task** (no Service). The backend
starts it with `ecs:RunTask` when a reviewer approves an ingestion job, polls
`DescribeTasks`, then reads the results back from S3. Infrastructure is defined in
`infra/lib/app-stack.ts` (task definition, roles, ECR repos) and `infra/lib/data-stack.ts` (bucket).

## Flow

1. Upload: the backend writes `uploads/{job_id}/dataset_a.csv` to S3.
2. Approve: `job_processor._process_via_ecs_task` calls `run_task` with `JOB_ID` and `MODEL_FOLDER` overrides.
3. The container (`ml-worker/s3_batch_entrypoint.py`) downloads the CSV and `models/{MODEL_FOLDER}/`
   (`petbert/`, `labels/labels.csv`, optional `checkpoints/*`), runs `batch_predict.py`, and uploads
   `uploads/{job_id}/predictions.json` and `uploads/{job_id}/scan_output/*`.
4. The backend downloads the predictions and ingests them; report text is written to
   `reports/{job_id}/{anon_id}.txt` and referenced by `pathology_reports.storage_path`.

Sizing: 4 vCPU / 16 GB, CPU only (GCP ran `n1-standard-4`, no GPU; a full run takes ~10 h), 50 GiB
ephemeral storage for the ~12 GB of weights. The backend enforces `ML_TIMEOUT_HOURS` (default 12)
because ECS has no max run time.

## Backend configuration

Set by CDK on the backend service: `S3_BUCKET`, `AWS_REGION`, `USE_ECS_ML=true`, `ECS_CLUSTER_ARN`,
`ML_TASK_DEFINITION_ARN`, `ML_TASK_SUBNET_IDS`, `ML_TASK_SECURITY_GROUP_ID`. Optional:
`ML_POLL_INTERVAL` (60), `ML_TIMEOUT_HOURS` (12), `ML_CLEANUP_JOB_FILES` (false — keeps
`scan_output/` for diagnostics). Leave `AWS_S3_ENDPOINT_URL` unset in AWS.

## Build and push the images

The ECR repos are created by the foundation stack, so images can be pushed before the app stack
is deployed (the backend service and ML task definition both reference `:latest`, so deploy the
app stack only after both images exist):

```bash
cdk deploy cancer-registry-prod-foundation cancer-registry-prod-data   # creates the repos (no domain needed)
scripts/push-images.sh all prod                                         # or: backend | ml-worker
cdk deploy -c apiDomainName=api-dev.<your-domain> cancer-registry-prod-app
```

The app stack is skipped unless `apiDomainName` is set: the API is HTTPS-only (ACM certificate,
port 443, HTTP redirects to HTTPS). During the app deploy the certificate stays in
`CREATE_IN_PROGRESS` until you add its DNS validation CNAME (ACM console → the new certificate)
at your DNS provider. Afterwards CNAME `apiDomainName` to the `BackendAlbDnsName` stack output.

The Amplify frontend is opt-in (`-c deployFrontend=true`). Before using it, store a GitHub token
(fine-grained PAT with repo read access) in Secrets Manager as plain text under
`cancer-registry-<env>-github-token`. The GitHub owner/repo are set in `infra/config/constants.ts`.

The script builds `linux/amd64` (Fargate's default architecture, so Apple Silicon builds work) and
tags `:<git-sha>` and `:latest`. The ML image is built from the repo root with
`ml-worker/Dockerfile.batch`; model weights are not baked in (see below).

## One-time GCS → S3 data copy

Copy `models/` and `reports/` (uploads/ is transient and is not needed). Check the size and egress
cost first: `gcloud storage du -s gs://<gcs-bucket>/models gs://<gcs-bucket>/reports`.

```bash
# from a machine with both gcloud and AWS credentials
rclone copy gcs:<gcs-bucket>/models s3:<s3-bucket>/models --progress
rclone copy gcs:<gcs-bucket>/reports s3:<s3-bucket>/reports --progress
```

Expected model layout: `models/<folder>/{petbert/,labels/labels.csv,checkpoints/...}` (e.g. `production`).
Object keys are unchanged, so after running migration `032` the existing `storage_path` values
(formerly `gcs_path`) resolve against the S3 bucket as-is. Verify a sample row by fetching its key.

## Local development

Local dev keeps `USE_ECS_ML=false` and uses the `ml-worker` HTTP container; CSVs and report text go
through Floci S3 (`AWS_S3_ENDPOINT_URL=http://floci:4566`, bucket `vmth-cancer-registry-storage-local`
created by `database/docker/floci/floci-init.sh`).
