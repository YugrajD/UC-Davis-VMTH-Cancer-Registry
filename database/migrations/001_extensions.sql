-- 001_extensions.sql
-- Enable PostGIS and other required extensions

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search on pathology reports
