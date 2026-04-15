ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS batch_job_name VARCHAR(500);
