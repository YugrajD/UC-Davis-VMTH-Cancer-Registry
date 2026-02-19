-- 009_one_case_per_patient.sql
-- One registry case per dog; all cancer predictions stored in case_diagnoses.

-- Allow one case per patient without a single cancer_type (types live in case_diagnoses)
ALTER TABLE cancer_cases ALTER COLUMN cancer_type_id DROP NOT NULL;

-- Table: one row per PetBERT prediction, grouped under one cancer_case per patient
CREATE TABLE IF NOT EXISTS case_diagnoses (
    id SERIAL PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cancer_cases(id) ON DELETE CASCADE,
    cancer_type_id INTEGER NOT NULL REFERENCES cancer_types(id),
    icd_o_code VARCHAR(20),
    predicted_term TEXT,
    original_text TEXT,
    confidence NUMERIC(4,2),
    prediction_method VARCHAR(20),
    source_row_index INTEGER,
    diagnosis_index INTEGER
);

CREATE INDEX IF NOT EXISTS idx_case_diagnoses_case ON case_diagnoses (case_id);
CREATE INDEX IF NOT EXISTS idx_case_diagnoses_cancer_type ON case_diagnoses (cancer_type_id);
CREATE INDEX IF NOT EXISTS idx_case_diagnoses_icd ON case_diagnoses (icd_o_code) WHERE icd_o_code IS NOT NULL;
