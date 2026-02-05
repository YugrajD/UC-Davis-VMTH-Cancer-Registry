-- 005_pathology_reports.sql
-- Pathology report text storage with BERT classification

CREATE TABLE IF NOT EXISTS pathology_reports (
    id SERIAL PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cancer_cases(id),
    report_text TEXT NOT NULL,
    classification VARCHAR(100),
    confidence_score NUMERIC(5, 4),
    report_date DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_case ON pathology_reports (case_id);
CREATE INDEX IF NOT EXISTS idx_reports_classification ON pathology_reports (classification);
CREATE INDEX IF NOT EXISTS idx_reports_text_trgm ON pathology_reports USING GIN (report_text gin_trgm_ops);
