-- Migration: Merge cancer_cases into patients + case_diagnoses
-- Eliminates the pass-through cancer_cases table (1:1 with patients)
-- Moves diagnosis_date, outcome to patients
-- Points case_diagnoses and pathology_reports directly to patients
--
-- Run on Supabase BEFORE deploying code changes.

BEGIN;

-- Step 1: Add diagnosis_date and outcome columns to patients
ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS diagnosis_date DATE,
    ADD COLUMN IF NOT EXISTS outcome VARCHAR(20);

-- Step 2: Copy diagnosis_date and outcome from cancer_cases to patients (1:1)
UPDATE patients p
SET diagnosis_date = cc.diagnosis_date,
    outcome        = cc.outcome
FROM cancer_cases cc
WHERE cc.patient_id = p.id;

-- Step 3: Add patient_id to case_diagnoses, populate from cancer_cases join
ALTER TABLE case_diagnoses
    ADD COLUMN IF NOT EXISTS patient_id INTEGER;

UPDATE case_diagnoses cd
SET patient_id = cc.patient_id
FROM cancer_cases cc
WHERE cc.id = cd.case_id;

-- Make patient_id NOT NULL and add FK
ALTER TABLE case_diagnoses
    ALTER COLUMN patient_id SET NOT NULL;

ALTER TABLE case_diagnoses
    ADD CONSTRAINT fk_case_diagnoses_patient
    FOREIGN KEY (patient_id) REFERENCES patients(id);

-- Drop old case_id column
ALTER TABLE case_diagnoses DROP COLUMN case_id;

-- Step 4: Add patient_id to pathology_reports, populate from cancer_cases join
ALTER TABLE pathology_reports
    ADD COLUMN IF NOT EXISTS patient_id INTEGER;

UPDATE pathology_reports pr
SET patient_id = cc.patient_id
FROM cancer_cases cc
WHERE cc.id = pr.case_id;

-- Make patient_id NOT NULL and add FK
ALTER TABLE pathology_reports
    ALTER COLUMN patient_id SET NOT NULL;

ALTER TABLE pathology_reports
    ADD CONSTRAINT fk_pathology_reports_patient
    FOREIGN KEY (patient_id) REFERENCES patients(id);

-- Drop old case_id column
ALTER TABLE pathology_reports DROP COLUMN case_id;

-- Step 5: Drop cancer_cases table
DROP TABLE cancer_cases CASCADE;

-- Step 6: Recreate materialized views with new schema
DROP MATERIALIZED VIEW IF EXISTS mv_county_cancer_incidence CASCADE;
CREATE MATERIALIZED VIEW mv_county_cancer_incidence AS
SELECT p.county_id, co.name AS county_name, ct.id AS cancer_type_id,
       ct.name AS cancer_type_name,
       EXTRACT(YEAR FROM p.diagnosis_date)::INTEGER AS year,
       COUNT(*) AS case_count, s.name AS species_name
FROM case_diagnoses cd
JOIN patients p ON cd.patient_id = p.id
JOIN counties co ON p.county_id = co.id
JOIN cancer_types ct ON cd.cancer_type_id = ct.id
JOIN species s ON p.species_id = s.id
GROUP BY p.county_id, co.name, ct.id, ct.name,
         EXTRACT(YEAR FROM p.diagnosis_date), s.name;

DROP MATERIALIZED VIEW IF EXISTS mv_yearly_trends CASCADE;
CREATE MATERIALIZED VIEW mv_yearly_trends AS
SELECT EXTRACT(YEAR FROM p.diagnosis_date)::INTEGER AS year,
       ct.id AS cancer_type_id, ct.name AS cancer_type_name,
       s.id AS species_id, s.name AS species_name,
       COUNT(*) AS case_count,
       COUNT(*) FILTER (WHERE p.outcome = 'deceased') AS deceased_count,
       COUNT(*) FILTER (WHERE p.outcome = 'alive') AS alive_count
FROM case_diagnoses cd
JOIN patients p ON cd.patient_id = p.id
JOIN cancer_types ct ON cd.cancer_type_id = ct.id
JOIN species s ON p.species_id = s.id
GROUP BY EXTRACT(YEAR FROM p.diagnosis_date), ct.id, ct.name, s.id, s.name;

COMMIT;
