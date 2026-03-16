-- Ingestion jobs table for admin approval workflow
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id SERIAL PRIMARY KEY,
    uploaded_by_email VARCHAR(255) NOT NULL,
    uploaded_by_sub VARCHAR(255) NOT NULL,
    dataset_a_filename VARCHAR(255) NOT NULL,
    dataset_b_filename VARCHAR(255) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending_review'
        CHECK (status IN ('pending_review','approved','rejected','processing','completed','failed')),
    reviewed_by_email VARCHAR(255),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    rejection_reason TEXT,
    ingestion_log_id INTEGER REFERENCES ingestion_logs(id),
    processing_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
