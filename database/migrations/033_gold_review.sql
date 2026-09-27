-- 033_gold_review.sql
-- Schema for the dashboard review worklist (audit list) and gold export, per
-- ml/documentation/audit-list-change-request.md. The specialist reviews a case
-- once from a worklist ML sends, recording its complete cancer-code set (or
-- "no reportable cancer"); an admin later exports completed reviews as a
-- gold_<export_id>.csv per reviewer, sent back to ML.

-- Reference data: Vet-ICD-O-Canine-1 (group, term) pairs, seeded from
-- ml/taxonomy/labels.csv. Not read live from disk — production has no /ml
-- mount (backend/Dockerfile only COPYs ./backend), so this table is the
-- picker's and the export validator's only source of truth.
CREATE TABLE IF NOT EXISTS taxonomy_terms (
    id SERIAL PRIMARY KEY,
    vet_icd_o_code VARCHAR(20),
    taxonomy_group VARCHAR(255) NOT NULL,
    taxonomy_term VARCHAR(255) NOT NULL,
    UNIQUE (taxonomy_group, taxonomy_term)
);

-- One row per imported audit_list_<list_id>.txt. Only one list is ever
-- active (the specialist's current worklist); older lists' header rows are
-- kept for traceability but is_active is flipped off.
CREATE TABLE IF NOT EXISTS audit_lists (
    id SERIAL PRIMARY KEY,
    list_id VARCHAR(100) NOT NULL UNIQUE,
    imported_by_email VARCHAR(255) NOT NULL,
    imported_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    sha256 VARCHAR(64) NOT NULL,
    case_count INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);
-- Only one active list at a time.
CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_lists_one_active
    ON audit_lists (is_active) WHERE is_active;

-- case_id + position within a list, kept across every list ever imported —
-- not just the active one. A case that drops off a newer list is still safe
-- to export (ML still has it in audit_list_ledger.csv); a case that was
-- never on ANY list makes ML refuse the whole gold file (see change
-- request's refusal rules), so export eligibility checks this table's full
-- history, not just the active list.
CREATE TABLE IF NOT EXISTS audit_list_cases (
    id SERIAL PRIMARY KEY,
    audit_list_id INTEGER NOT NULL REFERENCES audit_lists(id) ON DELETE CASCADE,
    -- Exactly as ML sent it — never normalized. The gold export echoes this
    -- back verbatim; normalize_anon_id is applied only when joining to
    -- patients.anon_id to show report text, never when storing or exporting.
    case_id VARCHAR(100) NOT NULL,
    position INTEGER NOT NULL,
    UNIQUE (audit_list_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_audit_list_cases_case_id ON audit_list_cases (case_id);
CREATE INDEX IF NOT EXISTS idx_audit_list_cases_list_position ON audit_list_cases (audit_list_id, position);

-- One export batch. export_id must be new each time (date + counter).
CREATE TABLE IF NOT EXISTS gold_exports (
    id SERIAL PRIMARY KEY,
    export_id VARCHAR(100) NOT NULL UNIQUE,
    reviewer_email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    case_count INTEGER NOT NULL
);

-- One row per case ever reviewed. "No-cancer or codes, never both" is
-- enforced in the router (consistent with how this codebase enforces other
-- review-state rules), not here.
--
-- Editable while locked = false. Exporting sets locked = true, gold_export_id
-- and exported_at (the most recent export this case was part of — ML is the
-- system of record for prior gold versions, per its audit_list_ledger.csv, so
-- this column isn't a version history, just "what's currently sent"). An
-- admin can explicitly reopen a locked review (locked = false, gold_export_id
-- and exported_at left as history of the last export); editing and
-- re-exporting then replaces ML's copy, per the change request's "re-sending
-- a case replaces its earlier review on ML's side."
CREATE TABLE IF NOT EXISTS case_reviews (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(100) NOT NULL UNIQUE,
    no_cancer BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_by_email VARCHAR(255) NOT NULL,
    reviewed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    gold_export_id INTEGER REFERENCES gold_exports(id) ON DELETE SET NULL,
    exported_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_case_reviews_reviewer_locked
    ON case_reviews (reviewed_by_email, locked);

-- The complete code set for a case (only when no_cancer = false). Each
-- (group, term) must exist in taxonomy_terms — validated in the router when
-- the review is saved, and re-validated at export time in case the taxonomy
-- changed in between.
CREATE TABLE IF NOT EXISTS case_review_codes (
    id SERIAL PRIMARY KEY,
    case_review_id INTEGER NOT NULL REFERENCES case_reviews(id) ON DELETE CASCADE,
    taxonomy_group VARCHAR(255) NOT NULL,
    taxonomy_term VARCHAR(255) NOT NULL,
    UNIQUE (case_review_id, taxonomy_group, taxonomy_term)
);
