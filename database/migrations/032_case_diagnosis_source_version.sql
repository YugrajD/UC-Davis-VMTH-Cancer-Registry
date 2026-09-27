-- 032_case_diagnosis_source_version.sql
-- Records which report-mapping generation produced each predicted code (the
-- worker's source_version, e.g. "gen-20260927T003905Z"), so a code can be
-- traced back to the model that made it. See ml/documentation/icd-mapping-strategy.md
-- and ml/documentation/ml-worker-change-request.md.

ALTER TABLE case_diagnoses
    ADD COLUMN IF NOT EXISTS source_version VARCHAR(80);
