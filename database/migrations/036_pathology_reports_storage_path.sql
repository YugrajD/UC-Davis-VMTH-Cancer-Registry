-- 036_pathology_reports_storage_path.sql
-- Renames pathology_reports.gcs_path to storage_path. The column holds a
-- cloud-provider-agnostic locator for the report's stored text — a GCS blob
-- path today, an S3 object key once USE_ECS_ML is used (see
-- backend/app/services/s3_service.py). Both storage backends coexist behind
-- their respective flags; the column just needs a name that isn't GCP-specific.

ALTER TABLE pathology_reports RENAME COLUMN gcs_path TO storage_path;
