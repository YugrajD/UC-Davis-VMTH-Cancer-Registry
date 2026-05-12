-- Link case_diagnoses to the ingestion_job that created them.
ALTER TABLE case_diagnoses ADD COLUMN IF NOT EXISTS ingestion_job_id INTEGER
    REFERENCES ingestion_jobs(id);

CREATE INDEX IF NOT EXISTS idx_case_diagnoses_job ON case_diagnoses (ingestion_job_id);
