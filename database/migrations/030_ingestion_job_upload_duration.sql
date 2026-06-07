-- Migration 030: track server-side upload processing time on ingestion jobs.
-- Captures milliseconds spent receiving + normalizing an upload before it
-- enters the review queue, for upload-pipeline time analysis.
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS upload_duration_ms INTEGER;
