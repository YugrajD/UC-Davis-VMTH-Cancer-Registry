-- 004_core_tables.sql
-- Patients and cancer cases core tables

CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    species_id INTEGER NOT NULL REFERENCES species(id),
    breed_id INTEGER NOT NULL REFERENCES breeds(id),
    sex VARCHAR(20) NOT NULL CHECK (sex IN ('Male', 'Female', 'Neutered Male', 'Spayed Female')),
    age_years NUMERIC(5, 1) NOT NULL,
    weight_kg NUMERIC(6, 2),
    county_id INTEGER NOT NULL REFERENCES counties(id),
    registered_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS cancer_cases (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    cancer_type_id INTEGER NOT NULL REFERENCES cancer_types(id),
    diagnosis_date DATE NOT NULL,
    stage VARCHAR(5) CHECK (stage IN ('I', 'II', 'III', 'IV')),
    outcome VARCHAR(20) CHECK (outcome IN ('alive', 'deceased', 'unknown')),
    county_id INTEGER NOT NULL REFERENCES counties(id)
);

CREATE INDEX IF NOT EXISTS idx_patients_species ON patients (species_id);
CREATE INDEX IF NOT EXISTS idx_patients_county ON patients (county_id);
CREATE INDEX IF NOT EXISTS idx_patients_registered ON patients (registered_date);

CREATE INDEX IF NOT EXISTS idx_cases_cancer_type ON cancer_cases (cancer_type_id);
CREATE INDEX IF NOT EXISTS idx_cases_county ON cancer_cases (county_id);
CREATE INDEX IF NOT EXISTS idx_cases_diagnosis_date ON cancer_cases (diagnosis_date);
CREATE INDEX IF NOT EXISTS idx_cases_patient ON cancer_cases (patient_id);
