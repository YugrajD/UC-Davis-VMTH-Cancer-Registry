-- Add 'cancelled' as a valid job status.
-- PostgreSQL CHECK constraints cannot be modified in-place;
-- drop the old one and replace it.

ALTER TABLE ingestion_jobs
    DROP CONSTRAINT IF EXISTS ingestion_jobs_status_check;

ALTER TABLE ingestion_jobs
    ADD CONSTRAINT ingestion_jobs_status_check
    CHECK (status IN (
        'pending_review',
        'approved',
        'rejected',
        'processing',
        'completed',
        'failed',
        'cancelled'
    ));
