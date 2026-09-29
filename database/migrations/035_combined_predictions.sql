-- 035_combined_predictions.sql
-- Schema for loading ML's combined_predictions_<run>.csv (gold > silver > bronze,
-- ml/coding/combine.py) as the registry's code of record, and review_queue_<run>.csv
-- for cases awaiting review. See ml/documentation/audit-list-change-request.md
-- section 1.
--
-- Design notes:
--   - code_source / source_confidence / ml_review_status are additive columns on
--     case_diagnoses; source_version already exists (migration 032). A
--     combined_predictions load deletes and reinserts a case's case_diagnoses
--     rows outright (not soft-superseded) — this does cascade-delete any
--     diagnosis_review_events history for that case, a deliberate accepted
--     tradeoff, not an oversight (see project memory / session decision log).
--   - NO_CANCER is recorded case-level on patients (registry_no_cancer), not as
--     a case_diagnoses row — a case coded cancer-free has zero code rows, same
--     as a case that's simply never been coded, distinguished by this flag.
--     Every existing dashboard/incidence/trends/geo/export query already INNER
--     JOINs case_diagnoses, so a patient with zero rows is already excluded from
--     every cancer count — this flag needs no changes to those consumers.
--   - registry_awaiting_review marks a case present in review_queue_<run>.csv
--     with no combined_predictions row yet (queued, not coded) — excluded from
--     counts the same way, via having zero case_diagnoses rows.

ALTER TABLE case_diagnoses
    ADD COLUMN IF NOT EXISTS code_source VARCHAR(20),
    ADD COLUMN IF NOT EXISTS source_confidence TEXT,
    ADD COLUMN IF NOT EXISTS ml_review_status VARCHAR(20);

ALTER TABLE case_diagnoses
    ADD CONSTRAINT case_diagnoses_code_source_check
        CHECK (code_source IS NULL OR code_source IN ('manual', 'diagnosis', 'report'));

ALTER TABLE case_diagnoses
    ADD CONSTRAINT case_diagnoses_ml_review_status_check
        CHECK (ml_review_status IS NULL OR ml_review_status IN ('confirmed', 'auto_accepted', 'queued'));

ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS registry_no_cancer BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS registry_no_cancer_source_version VARCHAR(80),
    ADD COLUMN IF NOT EXISTS registry_awaiting_review BOOLEAN NOT NULL DEFAULT FALSE;
