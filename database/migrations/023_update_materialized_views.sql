-- Migration: Update materialized views with proper filters and dimensions
-- Adds data_source and review_status filtering, sex and county dimensions

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
         EXTRACT(YEAR FROM p.diagnosis_date);

CREATE UNIQUE INDEX idx_mv_county_cancer
    ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year);


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
         p.county_id, co.name, COALESCE(p.sex, 'Unknown');

CREATE UNIQUE INDEX idx_mv_yearly_trends
    ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex);

COMMIT;
