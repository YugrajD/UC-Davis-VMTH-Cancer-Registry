-- Enable Row Level Security on all tables.
--
-- The frontend Supabase client is only used for authentication (sign in/out).
-- All data queries go through the backend API, which connects as the postgres
-- superuser and bypasses RLS. Enabling RLS with no permissive policies means
-- anyone using the anon or authenticated key directly cannot read or write
-- any table.

ALTER TABLE species ENABLE ROW LEVEL SECURITY;
ALTER TABLE breeds ENABLE ROW LEVEL SECURITY;
ALTER TABLE cancer_types ENABLE ROW LEVEL SECURITY;
ALTER TABLE counties ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE case_diagnoses ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE calenviroscreen ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;

-- Note: cancer_cases and pathology_reports were removed by later migrations
-- (003_merge_cancer_cases.sql). If they exist in your database, run:
--   ALTER TABLE cancer_cases ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE pathology_reports ENABLE ROW LEVEL SECURITY;
