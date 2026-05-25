-- Normalize pathology report text out of case_diagnoses into a dedicated table.
-- The old pathology_reports table (migration 005) referenced cancer_cases and
-- was never populated by the active ingestion pipeline. Drop and recreate it
-- with the correct schema, then migrate existing original_text data across.

-- 1. Remove old table (CASCADE drops the stale FK on any leftover references)
DROP TABLE IF EXISTS pathology_reports CASCADE;

-- 2. Create normalized table — one row per patient per report
CREATE TABLE pathology_reports (
    id          SERIAL PRIMARY KEY,
    patient_id  INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    report_text TEXT    NOT NULL,
    report_date DATE,
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pathology_reports_patient_id ON pathology_reports(patient_id);

-- 3. Add FK from case_diagnoses to pathology_reports
ALTER TABLE case_diagnoses
    ADD COLUMN IF NOT EXISTS pathology_report_id INTEGER
        REFERENCES pathology_reports(id) ON DELETE SET NULL;

-- 4. Backfill: one pathology_report per patient from case_diagnoses.original_text.
--    DISTINCT ON picks the lowest-id diagnosis row per patient as the source.
INSERT INTO pathology_reports (patient_id, report_text)
SELECT DISTINCT ON (patient_id) patient_id, original_text
FROM case_diagnoses
WHERE original_text IS NOT NULL AND original_text <> ''
ORDER BY patient_id, id;

-- 5. Link each case_diagnosis to its new pathology_report
UPDATE case_diagnoses cd
SET pathology_report_id = pr.id
FROM pathology_reports pr
WHERE cd.patient_id = pr.patient_id;

-- 6. Drop the now-redundant column
ALTER TABLE case_diagnoses DROP COLUMN IF EXISTS original_text;
