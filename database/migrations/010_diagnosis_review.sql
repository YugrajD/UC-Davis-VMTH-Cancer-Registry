-- 010_diagnosis_review.sql
-- Per-diagnosis manual review workflow.
--
-- Adds review state and an audit trail to case_diagnoses, plus a
-- review_events table that records every state change so multiple
-- reviewers can collaborate without losing history.

-- 1. Per-diagnosis review state and correction audit columns.
ALTER TABLE case_diagnoses
    ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'confirmed',
    ADD COLUMN IF NOT EXISTS reviewed_by_email VARCHAR(255),
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reviewer_notes TEXT,
    ADD COLUMN IF NOT EXISTS original_cancer_type_id INTEGER REFERENCES cancer_types(id),
    ADD COLUMN IF NOT EXISTS original_icd_o_code VARCHAR(20),
    ADD COLUMN IF NOT EXISTS original_predicted_term TEXT,
    ADD COLUMN IF NOT EXISTS top2_margin NUMERIC(4,2);

ALTER TABLE case_diagnoses
    DROP CONSTRAINT IF EXISTS case_diagnoses_review_status_check;
ALTER TABLE case_diagnoses
    ADD CONSTRAINT case_diagnoses_review_status_check
    CHECK (review_status IN ('pending', 'confirmed', 'corrected', 'rejected'));

CREATE INDEX IF NOT EXISTS idx_case_diagnoses_review_status
    ON case_diagnoses (review_status)
    WHERE review_status <> 'confirmed';

-- 2. Append-only event log of state changes for the audit trail.
CREATE TABLE IF NOT EXISTS diagnosis_review_events (
    id SERIAL PRIMARY KEY,
    case_diagnosis_id INTEGER NOT NULL
        REFERENCES case_diagnoses(id) ON DELETE CASCADE,
    actor_email VARCHAR(255) NOT NULL,
    action VARCHAR(20) NOT NULL,
    -- 'flagged' (auto, at ingest), 'confirm', 'correct', 'reject', 'reopen'
    from_status VARCHAR(20),
    to_status VARCHAR(20) NOT NULL,
    cancer_type_id_before INTEGER REFERENCES cancer_types(id),
    cancer_type_id_after INTEGER REFERENCES cancer_types(id),
    icd_o_code_before VARCHAR(20),
    icd_o_code_after VARCHAR(20),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_diagnosis_review_events_case
    ON diagnosis_review_events (case_diagnosis_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_diagnosis_review_events_actor
    ON diagnosis_review_events (actor_email);

-- 3. Auto-created cancer types may need admin confirmation. Existing rows
--    are treated as confirmed since they were curated manually.
ALTER TABLE cancer_types
    ADD COLUMN IF NOT EXISTS confirmed BOOLEAN NOT NULL DEFAULT TRUE;

CREATE INDEX IF NOT EXISTS idx_cancer_types_unconfirmed
    ON cancer_types (confirmed) WHERE confirmed = FALSE;
