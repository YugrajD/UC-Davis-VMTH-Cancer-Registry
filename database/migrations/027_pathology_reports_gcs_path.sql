-- Move pathology report text out of PostgreSQL into GCS.
-- Replaces the report_text TEXT column with a gcs_path pointer.
-- Existing rows (backfilled from case_diagnoses.original_text in migration 026)
-- are cleared — text for those rows is unavailable in GCS (legacy behaviour).
ALTER TABLE pathology_reports DROP COLUMN IF EXISTS report_text;
ALTER TABLE pathology_reports ADD COLUMN IF NOT EXISTS gcs_path VARCHAR(1000);
