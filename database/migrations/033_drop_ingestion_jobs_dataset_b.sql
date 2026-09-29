-- Dataset B was removed from the upload flow; ingestion_jobs.dataset_b_filename
-- is unused but still NOT NULL from 011, which rejects every new job insert.
ALTER TABLE ingestion_jobs DROP COLUMN IF EXISTS dataset_b_filename;
