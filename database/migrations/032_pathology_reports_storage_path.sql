-- Rename pathology_reports.gcs_path to storage_path now that report text
-- lives in S3 (key format unchanged: reports/{job_id}/{anon_id}.txt).
ALTER TABLE pathology_reports RENAME COLUMN gcs_path TO storage_path;
