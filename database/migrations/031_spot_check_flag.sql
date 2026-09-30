-- 031_spot_check_flag.sql
-- Lets an admin flag specific cases (by CASE_ID / anon_id) as needing
-- manual spot-check via a CSV upload. Filterable in the Diagnosis Review
-- queue alongside the existing status/year/patient/clinic/type filters.

ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS needs_spot_check BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS spot_check_flagged_by_email VARCHAR(255),
    ADD COLUMN IF NOT EXISTS spot_check_flagged_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_patients_needs_spot_check
    ON patients (needs_spot_check)
    WHERE needs_spot_check = TRUE;
