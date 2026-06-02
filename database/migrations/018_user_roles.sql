-- 018_user_roles.sql
-- Move per-user role assignments from env vars into a managed table so
-- admins can grant/revoke roles through the UI.
--
-- ADMIN_EMAILS / UPLOADER_EMAILS / REVIEWER_EMAILS env vars become a
-- startup seed only — DB rows take precedence at request time. Emails
-- not present in this table fall through to env-list membership.

CREATE TABLE IF NOT EXISTS user_roles (
    email VARCHAR(255) PRIMARY KEY,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    is_uploader BOOLEAN NOT NULL DEFAULT FALSE,
    is_reviewer BOOLEAN NOT NULL DEFAULT FALSE,
    updated_by_email VARCHAR(255),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Lower-case email matching is the convention everywhere else
-- (Patient.anon_id, ingestion_jobs.uploaded_by_email). Index on the
-- raw email; admins should normalize input on the way in.
CREATE INDEX IF NOT EXISTS idx_user_roles_email_lower
    ON user_roles (LOWER(email));
