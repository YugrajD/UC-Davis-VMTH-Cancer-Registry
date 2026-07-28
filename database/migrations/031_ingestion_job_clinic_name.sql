-- 031_ingestion_job_clinic_name.sql
-- Add clinic_name to ingestion_jobs (backend/app/models/models.py:240).

ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS clinic_name VARCHAR(255);
