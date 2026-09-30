-- 034_drop_spot_check_flag.sql
-- Reverts 031_spot_check_flag.sql. Spot-check was built as a stopgap for
-- "seed which cases need a human to audit next" before ML's actual audit-list
-- contract existed (see ml/documentation/audit-list-change-request.md,
-- database/migrations/033_gold_review.sql) — the Audit Worklist now does that
-- job properly, sourced from ML's authoritative list instead of an admin's
-- ad hoc CSV guess. Nothing else reads these columns.

DROP INDEX IF EXISTS idx_patients_needs_spot_check;

ALTER TABLE patients
    DROP COLUMN IF EXISTS needs_spot_check,
    DROP COLUMN IF EXISTS spot_check_flagged_by_email,
    DROP COLUMN IF EXISTS spot_check_flagged_at;
