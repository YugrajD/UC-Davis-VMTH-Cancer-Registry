-- Migration: Rebuild materialized views to include age_group dimension
-- Age = EXTRACT(YEAR FROM diagnosis_date) - EXTRACT(YEAR FROM birth_date)
-- Buckets: young 0-2, juvenile 3-5, adult 6-8, old 9-11, senior >=12

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS mv_county_cancer_incidence CASCADE;
CREATE MATERIALIZED VIEW mv_county_cancer_incidence AS
SELECT
    p.county_id,
    co.name        AS county_name,
    ct.id          AS cancer_type_id,
    ct.name        AS cancer_type_name,
    s.id           AS species_id,
    s.name         AS species_name,
    COALESCE(p.sex, 'Unknown') AS sex,
    EXTRACT(YEAR FROM p.diagnosis_date)::INTEGER AS year,
    CASE
        WHEN p.birth_date IS NULL OR p.diagnosis_date IS NULL THEN 'Unknown'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 0 AND 2 THEN 'young'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 3 AND 5 THEN 'juvenile'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 6 AND 8 THEN 'adult'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 9 AND 11 THEN 'old'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) >= 12 THEN 'senior'
        ELSE 'Unknown'
    END AS age_group,
    COUNT(*)       AS case_count
FROM case_diagnoses cd
JOIN patients     p  ON cd.patient_id     = p.id
JOIN counties     co ON p.county_id       = co.id
JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
JOIN species      s  ON p.species_id      = s.id
WHERE p.data_source = 'petbert'
  AND cd.review_status IN ('confirmed', 'corrected')
GROUP BY p.county_id, co.name, ct.id, ct.name,
         s.id, s.name, COALESCE(p.sex, 'Unknown'),
         EXTRACT(YEAR FROM p.diagnosis_date),
         CASE
             WHEN p.birth_date IS NULL OR p.diagnosis_date IS NULL THEN 'Unknown'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 0 AND 2 THEN 'young'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 3 AND 5 THEN 'juvenile'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 6 AND 8 THEN 'adult'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 9 AND 11 THEN 'old'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) >= 12 THEN 'senior'
             ELSE 'Unknown'
         END;

CREATE UNIQUE INDEX idx_mv_county_cancer
    ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year, age_group);


DROP MATERIALIZED VIEW IF EXISTS mv_yearly_trends CASCADE;
CREATE MATERIALIZED VIEW mv_yearly_trends AS
SELECT
    EXTRACT(YEAR FROM p.diagnosis_date)::INTEGER AS year,
    ct.id          AS cancer_type_id,
    ct.name        AS cancer_type_name,
    s.id           AS species_id,
    s.name         AS species_name,
    p.county_id,
    co.name        AS county_name,
    COALESCE(p.sex, 'Unknown') AS sex,
    CASE
        WHEN p.birth_date IS NULL OR p.diagnosis_date IS NULL THEN 'Unknown'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 0 AND 2 THEN 'young'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 3 AND 5 THEN 'juvenile'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 6 AND 8 THEN 'adult'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 9 AND 11 THEN 'old'
        WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) >= 12 THEN 'senior'
        ELSE 'Unknown'
    END AS age_group,
    COUNT(*)       AS case_count,
    COUNT(*) FILTER (WHERE p.outcome = 'deceased') AS deceased_count,
    COUNT(*) FILTER (WHERE p.outcome = 'alive')    AS alive_count
FROM case_diagnoses cd
JOIN patients     p  ON cd.patient_id     = p.id
JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
JOIN species      s  ON p.species_id      = s.id
JOIN counties     co ON p.county_id       = co.id
WHERE p.data_source = 'petbert'
  AND cd.review_status IN ('confirmed', 'corrected')
GROUP BY EXTRACT(YEAR FROM p.diagnosis_date),
         ct.id, ct.name, s.id, s.name,
         p.county_id, co.name, COALESCE(p.sex, 'Unknown'),
         CASE
             WHEN p.birth_date IS NULL OR p.diagnosis_date IS NULL THEN 'Unknown'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 0 AND 2 THEN 'young'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 3 AND 5 THEN 'juvenile'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 6 AND 8 THEN 'adult'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) BETWEEN 9 AND 11 THEN 'old'
             WHEN (EXTRACT(YEAR FROM p.diagnosis_date) - EXTRACT(YEAR FROM p.birth_date)) >= 12 THEN 'senior'
             ELSE 'Unknown'
         END;

CREATE UNIQUE INDEX idx_mv_yearly_trends
    ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex, age_group);

COMMIT;
