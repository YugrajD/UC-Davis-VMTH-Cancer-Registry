-- 007_ingestion_schema.sql
-- Schema changes to support ingestion of pre-classified PetBERT data

-- ---------------------------------------------------------------
-- 1. Patient columns for CSV-ingested records
-- ---------------------------------------------------------------
ALTER TABLE patients ADD COLUMN IF NOT EXISTS anon_id VARCHAR(100);
ALTER TABLE patients ADD COLUMN IF NOT EXISTS zip_code VARCHAR(10);
ALTER TABLE patients ADD COLUMN IF NOT EXISTS data_source VARCHAR(20) DEFAULT 'mock';

CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_anon_id
    ON patients (anon_id) WHERE anon_id IS NOT NULL;

-- Make demographic columns nullable for CSV patients
ALTER TABLE patients ALTER COLUMN species_id DROP NOT NULL;
ALTER TABLE patients ALTER COLUMN breed_id DROP NOT NULL;
ALTER TABLE patients ALTER COLUMN age_years DROP NOT NULL;
ALTER TABLE patients ALTER COLUMN county_id DROP NOT NULL;
ALTER TABLE patients ALTER COLUMN registered_date DROP NOT NULL;
ALTER TABLE patients ALTER COLUMN sex DROP NOT NULL;

ALTER TABLE patients DROP CONSTRAINT IF EXISTS patients_sex_check;
ALTER TABLE patients ADD CONSTRAINT patients_sex_check
    CHECK (sex IS NULL OR sex IN ('Male', 'Female', 'Neutered Male', 'Spayed Female'));

-- ---------------------------------------------------------------
-- 2. Cancer cases: nullable county/date, add PetBERT classification data
-- ---------------------------------------------------------------
ALTER TABLE cancer_cases ALTER COLUMN county_id DROP NOT NULL;
ALTER TABLE cancer_cases ALTER COLUMN diagnosis_date DROP NOT NULL;

ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS source_row_index INTEGER;
ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS diagnosis_index INTEGER;
ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS icd_o_code VARCHAR(20);
ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS predicted_term TEXT;
ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS original_text TEXT;
ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS confidence NUMERIC(4,2);
ALTER TABLE cancer_cases ADD COLUMN IF NOT EXISTS prediction_method VARCHAR(20);

CREATE INDEX IF NOT EXISTS idx_cases_source_row
    ON cancer_cases (source_row_index) WHERE source_row_index IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cases_icd_o_code
    ON cancer_cases (icd_o_code) WHERE icd_o_code IS NOT NULL;

-- ---------------------------------------------------------------
-- 3. Seed "Unknown" / "Dog" lookups for ingested records
-- ---------------------------------------------------------------
INSERT INTO species (name) VALUES ('Unknown') ON CONFLICT (name) DO NOTHING;
INSERT INTO species (name) VALUES ('Dog') ON CONFLICT (name) DO NOTHING;

INSERT INTO breeds (species_id, name)
    SELECT s.id, 'Unknown'
    FROM species s WHERE s.name = 'Unknown'
    AND NOT EXISTS (SELECT 1 FROM breeds WHERE name = 'Unknown')
    LIMIT 1;

INSERT INTO cancer_types (name, description)
    VALUES ('Unknown', 'Unclassified or unknown cancer type')
    ON CONFLICT (name) DO NOTHING;

-- ---------------------------------------------------------------
-- 4. Ingestion logs table
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id SERIAL PRIMARY KEY,
    dataset_a_filename VARCHAR(255),
    dataset_b_filename VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    rows_processed INTEGER DEFAULT 0,
    rows_inserted INTEGER DEFAULT 0,
    rows_skipped INTEGER DEFAULT 0,
    rows_errored INTEGER DEFAULT 0,
    errors JSONB DEFAULT '[]'::JSONB,
    warnings JSONB DEFAULT '[]'::JSONB
);
