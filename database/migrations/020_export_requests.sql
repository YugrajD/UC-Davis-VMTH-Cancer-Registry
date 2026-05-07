CREATE TABLE IF NOT EXISTS export_requests (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'denied', 'downloaded')),
    reason TEXT,
    resolved_by_email VARCHAR(255),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_export_requests_email ON export_requests (LOWER(email));
CREATE INDEX IF NOT EXISTS idx_export_requests_status ON export_requests (status) WHERE status = 'pending';
