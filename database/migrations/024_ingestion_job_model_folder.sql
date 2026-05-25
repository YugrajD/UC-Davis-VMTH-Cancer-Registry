-- Add model_folder to ingestion_jobs so admins can select which GCS model
-- bundle was used for a given job. NULL means the default ("production").
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS model_folder VARCHAR(255);
