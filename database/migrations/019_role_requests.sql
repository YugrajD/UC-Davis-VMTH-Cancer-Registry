-- Role request system: users can request uploader/reviewer roles.
CREATE TABLE IF NOT EXISTS role_requests (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    requested_role VARCHAR(20) NOT NULL CHECK (requested_role IN ('uploader', 'reviewer')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'denied')),
    reason TEXT,
    resolved_by_email VARCHAR(255),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (email, requested_role, status)
);

-- Efficient lookups
CREATE INDEX IF NOT EXISTS idx_role_requests_email ON role_requests (LOWER(email));
CREATE INDEX IF NOT EXISTS idx_role_requests_status ON role_requests (status) WHERE status = 'pending';
