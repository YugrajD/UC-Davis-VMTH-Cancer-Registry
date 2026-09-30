-- 036_taxonomy_term_level.sql
-- Exposes labels.csv's `level` column (Preferred / Synonym / Related) on
-- taxonomy_terms, so the review screen's code picker can tell which term is
-- a code's Preferred autofill target. See ml/documentation for the picker's
-- change request. Backfilled by re-running
-- database/seed/seed_taxonomy_terms.py after this migration is applied —
-- nullable until then, since existing rows predate this column.

ALTER TABLE taxonomy_terms
    ADD COLUMN IF NOT EXISTS term_level VARCHAR(20);
