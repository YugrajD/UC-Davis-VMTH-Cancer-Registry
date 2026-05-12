-- Store per-job model accuracy summary produced after ingestion.
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS result_summary JSONB;
