-- 006_materialized_views.sql
-- Pre-computed views for dashboard performance

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_county_cancer_incidence AS
SELECT
    c.county_id,
    co.name AS county_name,
    ct.id AS cancer_type_id,
    ct.name AS cancer_type_name,
    EXTRACT(YEAR FROM c.diagnosis_date)::INTEGER AS year,
    COUNT(*) AS case_count,
    s.name AS species_name
FROM cancer_cases c
JOIN counties co ON c.county_id = co.id
JOIN cancer_types ct ON c.cancer_type_id = ct.id
JOIN patients p ON c.patient_id = p.id
JOIN species s ON p.species_id = s.id
GROUP BY c.county_id, co.name, ct.id, ct.name,
         EXTRACT(YEAR FROM c.diagnosis_date), s.name;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_county_cancer
    ON mv_county_cancer_incidence (county_id, cancer_type_id, year, species_name);


CREATE MATERIALIZED VIEW IF NOT EXISTS mv_yearly_trends AS
SELECT
    EXTRACT(YEAR FROM c.diagnosis_date)::INTEGER AS year,
    ct.id AS cancer_type_id,
    ct.name AS cancer_type_name,
    s.id AS species_id,
    s.name AS species_name,
    COUNT(*) AS case_count,
    COUNT(*) FILTER (WHERE c.outcome = 'deceased') AS deceased_count,
    COUNT(*) FILTER (WHERE c.outcome = 'alive') AS alive_count
FROM cancer_cases c
JOIN cancer_types ct ON c.cancer_type_id = ct.id
JOIN patients p ON c.patient_id = p.id
JOIN species s ON p.species_id = s.id
GROUP BY EXTRACT(YEAR FROM c.diagnosis_date), ct.id, ct.name, s.id, s.name;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_yearly_trends
    ON mv_yearly_trends (year, cancer_type_id, species_id);
