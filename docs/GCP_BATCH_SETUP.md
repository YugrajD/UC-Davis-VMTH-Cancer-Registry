# GCP Batch Setup for PetBERT ML Inference

This guide walks through the one-time GCP setup required to run PetBERT
inference via GCP Batch instead of the local `ml-worker` container.

## Why GCP Batch?

The PetBERT inference step takes ~10 hours for a full dataset. GCP Batch
provisions a VM on demand, runs the container, and tears it down. Cost is
~$1–2 per run vs. ~$100+/month for a 24/7 VM.

## Prerequisites

- A GCP project with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed locally

## 1. Enable APIs

```bash
gcloud services enable \
  batch.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  --project=YOUR_PROJECT_ID
```

## 2. Create a GCS Bucket

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME \
  --project=YOUR_PROJECT_ID \
  --location=us-central1 \
  --uniform-bucket-level-access
```

## 3. Upload PetBERT Model to GCS

The model weights (~12 GB) are read from GCS at runtime, not baked into the
Docker image.

```bash
gsutil -m cp -r /path/to/local/petbert gs://YOUR_BUCKET_NAME/models/petbert
```

## 4. Upload ML Labels to GCS

```bash
gsutil -m cp -r ml/labels gs://YOUR_BUCKET_NAME/models/labels
```

## 5. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create vmth \
  --repository-format=docker \
  --location=us-central1 \
  --project=YOUR_PROJECT_ID
```

## 6. Build and Push the ML Worker Image

Build from the repository root (not from `ml-worker/`):

```bash
# Authenticate Docker with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build
docker build -f ml-worker/Dockerfile.batch \
  -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/vmth/petbert-batch:latest .

# Push
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/vmth/petbert-batch:latest
```

## 7. Create a Service Account

```bash
gcloud iam service-accounts create vmth-batch \
  --display-name="VMTH Batch Runner" \
  --project=YOUR_PROJECT_ID

SA_EMAIL=vmth-batch@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Grant roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/batch.jobsEditor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.reader"
```

## 8. Download the Service Account Key

```bash
gcloud iam service-accounts keys create secrets/gcp-sa-key.json \
  --iam-account=$SA_EMAIL
```

The `secrets/` directory is git-ignored. **Never commit this file.**

## 9. Configure Environment Variables

Copy the GCP section from `.env.example` into your `.env` and fill in:

```env
USE_GCP_BATCH=true
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCS_BUCKET=your-gcs-bucket-name
GCP_BATCH_IMAGE_URI=us-central1-docker.pkg.dev/your-project/vmth/petbert-batch:latest
GCP_BATCH_MACHINE_TYPE=n1-standard-4
GCP_BATCH_POLL_INTERVAL=60
GCP_BATCH_TIMEOUT_HOURS=12
GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-sa-key.json
```

## Verification

1. **Local dev** (`USE_GCP_BATCH=false`): Existing flow via `ml-worker` works identically.
2. **GCP Batch**: Set `USE_GCP_BATCH=true`, approve an upload, then monitor:
   - Backend logs for periodic `Batch status = RUNNING` messages
   - GCP Console → Batch → Jobs for the running job
3. **Failure handling**: Kill the Batch job in GCP Console → backend detects `FAILED` → job marked failed with error.
4. **Restart recovery**: Restart backend during a Batch run → stale job marked failed with `batch_job_name` logged.
5. **DB check**: `SELECT id, status, batch_job_name FROM ingestion_jobs;`
