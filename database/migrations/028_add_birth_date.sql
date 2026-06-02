-- Migration: Add birth_date to patients table for age calculation
BEGIN;

ALTER TABLE patients ADD COLUMN IF NOT EXISTS birth_date DATE;

COMMIT;
