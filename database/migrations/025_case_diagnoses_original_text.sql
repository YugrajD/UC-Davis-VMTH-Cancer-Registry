-- Add original_text to case_diagnoses so reviewers can see the raw source
-- text PetBERT classified (the "Clinical Diagnoses" cell from the upload).
ALTER TABLE case_diagnoses
    ADD COLUMN IF NOT EXISTS original_text TEXT;
