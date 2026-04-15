-- Add processing_stage to ingestion_jobs so the UI can display which step
-- of the pipeline a job is currently at while status = 'processing'.
--
-- Possible values (set by job_processor.py):
--   Local ML worker path:
--     reading_files, running_ml_worker, ingesting
--   GCP Batch path:
--     uploading_to_gcs, submitting_batch_job, batch_queued,
--     batch_scheduled, batch_running, downloading_predictions, ingesting

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS processing_stage VARCHAR(50);
