--
-- PostgreSQL database dump
--

\restrict xGJFNIr8hS3AwxCcDbtC4EIQpJcYoVJLXVqsHe2A41JwcrHapxiTXEsKTn2vtan

-- Dumped from database version 17.6
-- Dumped by pg_dump version 18.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA public;


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


--
-- Name: rls_auto_enable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.rls_auto_enable() RETURNS event_trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'pg_catalog'
    AS $$
DECLARE
  cmd record;
BEGIN
  FOR cmd IN
    SELECT *
    FROM pg_event_trigger_ddl_commands()
    WHERE command_tag IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
      AND object_type IN ('table','partitioned table')
  LOOP
     IF cmd.schema_name IS NOT NULL AND cmd.schema_name IN ('public') AND cmd.schema_name NOT IN ('pg_catalog','information_schema') AND cmd.schema_name NOT LIKE 'pg_toast%' AND cmd.schema_name NOT LIKE 'pg_temp%' THEN
      BEGIN
        EXECUTE format('alter table if exists %s enable row level security', cmd.object_identity);
        RAISE LOG 'rls_auto_enable: enabled RLS on %', cmd.object_identity;
      EXCEPTION
        WHEN OTHERS THEN
          RAISE LOG 'rls_auto_enable: failed to enable RLS on %', cmd.object_identity;
      END;
     ELSE
        RAISE LOG 'rls_auto_enable: skip % (either system schema or not in enforced list: %.)', cmd.object_identity, cmd.schema_name;
     END IF;
  END LOOP;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: audit_list_cases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.audit_list_cases (
    id integer NOT NULL,
    audit_list_id integer NOT NULL,
    case_id character varying(100) NOT NULL,
    "position" integer NOT NULL
);


--
-- Name: audit_list_cases_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.audit_list_cases_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: audit_list_cases_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.audit_list_cases_id_seq OWNED BY public.audit_list_cases.id;


--
-- Name: audit_lists; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.audit_lists (
    id integer NOT NULL,
    list_id character varying(100) NOT NULL,
    imported_by_email character varying(255) NOT NULL,
    imported_at timestamp with time zone DEFAULT now() NOT NULL,
    sha256 character varying(64) NOT NULL,
    case_count integer NOT NULL,
    is_active boolean DEFAULT false NOT NULL
);


--
-- Name: audit_lists_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.audit_lists_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: audit_lists_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.audit_lists_id_seq OWNED BY public.audit_lists.id;


--
-- Name: breeds; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.breeds (
    id integer NOT NULL,
    species_id integer NOT NULL,
    name character varying(100) NOT NULL
);


--
-- Name: breeds_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.breeds_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: breeds_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.breeds_id_seq OWNED BY public.breeds.id;


--
-- Name: calenviroscreen; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.calenviroscreen (
    id integer NOT NULL,
    county_id integer NOT NULL,
    ces_score numeric(6,2),
    pollution_burden numeric(6,2),
    ozone numeric(6,2),
    pm25 numeric(6,2),
    diesel_pm numeric(6,2),
    pesticides numeric(6,2),
    toxic_releases numeric(6,2),
    traffic numeric(6,2),
    drinking_water numeric(6,2),
    lead numeric(6,2),
    cleanup_sites numeric(6,2),
    groundwater_threats numeric(6,2),
    hazardous_waste numeric(6,2),
    solid_waste numeric(6,2),
    impaired_water numeric(6,2),
    pop_characteristics numeric(6,2),
    asthma numeric(6,2),
    low_birth_weight numeric(6,2),
    cardiovascular numeric(6,2),
    poverty numeric(6,2),
    unemployment numeric(6,2),
    housing_burden numeric(6,2),
    education numeric(6,2),
    linguistic_isolation numeric(6,2)
);


--
-- Name: calenviroscreen_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.calenviroscreen_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: calenviroscreen_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.calenviroscreen_id_seq OWNED BY public.calenviroscreen.id;


--
-- Name: cancer_types; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cancer_types (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    description text,
    confirmed boolean DEFAULT true NOT NULL
);


--
-- Name: cancer_types_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cancer_types_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cancer_types_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cancer_types_id_seq OWNED BY public.cancer_types.id;


--
-- Name: case_diagnoses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_diagnoses (
    id integer NOT NULL,
    cancer_type_id integer NOT NULL,
    icd_o_code character varying(20),
    predicted_term text,
    confidence numeric(4,2),
    prediction_method character varying(50),
    source_row_index integer,
    diagnosis_index integer,
    patient_id integer NOT NULL,
    review_status character varying(20) DEFAULT 'confirmed'::character varying NOT NULL,
    reviewed_by_email character varying(255),
    reviewed_at timestamp with time zone,
    reviewer_notes text,
    original_cancer_type_id integer,
    original_icd_o_code character varying(20),
    original_predicted_term text,
    top2_margin numeric(4,2),
    ingestion_job_id integer,
    pathology_report_id integer,
    source_version character varying(80),
    code_source character varying(20),
    source_confidence text,
    ml_review_status character varying(20),
    CONSTRAINT case_diagnoses_code_source_check CHECK (((code_source IS NULL) OR ((code_source)::text = ANY ((ARRAY['manual'::character varying, 'diagnosis'::character varying, 'report'::character varying])::text[])))),
    CONSTRAINT case_diagnoses_ml_review_status_check CHECK (((ml_review_status IS NULL) OR ((ml_review_status)::text = ANY ((ARRAY['confirmed'::character varying, 'auto_accepted'::character varying, 'queued'::character varying])::text[])))),
    CONSTRAINT case_diagnoses_review_status_check CHECK (((review_status)::text = ANY ((ARRAY['pending'::character varying, 'confirmed'::character varying, 'corrected'::character varying, 'rejected'::character varying])::text[])))
);


--
-- Name: case_diagnoses_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_diagnoses_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: case_diagnoses_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_diagnoses_id_seq OWNED BY public.case_diagnoses.id;


--
-- Name: case_review_codes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_review_codes (
    id integer NOT NULL,
    case_review_id integer NOT NULL,
    taxonomy_group character varying(255) NOT NULL,
    taxonomy_term character varying(255) NOT NULL
);


--
-- Name: case_review_codes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_review_codes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: case_review_codes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_review_codes_id_seq OWNED BY public.case_review_codes.id;


--
-- Name: case_reviews; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_reviews (
    id integer NOT NULL,
    case_id character varying(100) NOT NULL,
    no_cancer boolean DEFAULT false NOT NULL,
    reviewed_by_email character varying(255) NOT NULL,
    reviewed_at timestamp with time zone DEFAULT now() NOT NULL,
    locked boolean DEFAULT false NOT NULL,
    gold_export_id integer,
    exported_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: case_reviews_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_reviews_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: case_reviews_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_reviews_id_seq OWNED BY public.case_reviews.id;


--
-- Name: counties; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.counties (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    fips_code character varying(5) NOT NULL,
    geom public.geometry(MultiPolygon,4326),
    population integer,
    area_sq_miles numeric(10,2),
    is_catchment boolean DEFAULT false NOT NULL
);


--
-- Name: counties_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.counties_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: counties_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.counties_id_seq OWNED BY public.counties.id;


--
-- Name: diagnosis_review_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.diagnosis_review_events (
    id integer NOT NULL,
    case_diagnosis_id integer NOT NULL,
    actor_email character varying(255) NOT NULL,
    action character varying(20) NOT NULL,
    from_status character varying(20),
    to_status character varying(20) NOT NULL,
    cancer_type_id_before integer,
    cancer_type_id_after integer,
    icd_o_code_before character varying(20),
    icd_o_code_after character varying(20),
    notes text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: diagnosis_review_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.diagnosis_review_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: diagnosis_review_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.diagnosis_review_events_id_seq OWNED BY public.diagnosis_review_events.id;


--
-- Name: export_requests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.export_requests (
    id integer NOT NULL,
    email character varying(255) NOT NULL,
    status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    reason text,
    resolved_by_email character varying(255),
    resolved_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT export_requests_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'approved'::character varying, 'denied'::character varying, 'downloaded'::character varying])::text[])))
);


--
-- Name: export_requests_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.export_requests_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: export_requests_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.export_requests_id_seq OWNED BY public.export_requests.id;


--
-- Name: gold_exports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gold_exports (
    id integer NOT NULL,
    export_id character varying(100) NOT NULL,
    reviewer_email character varying(255) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    case_count integer NOT NULL
);


--
-- Name: gold_exports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.gold_exports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: gold_exports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.gold_exports_id_seq OWNED BY public.gold_exports.id;


--
-- Name: ingestion_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingestion_jobs (
    id integer NOT NULL,
    uploaded_by_email character varying(255) NOT NULL,
    uploaded_by_sub character varying(255) NOT NULL,
    dataset_a_filename character varying(255) NOT NULL,
    dataset_b_filename character varying(255),
    storage_path character varying(500) NOT NULL,
    status character varying(20) DEFAULT 'pending_review'::character varying NOT NULL,
    reviewed_by_email character varying(255),
    reviewed_at timestamp with time zone,
    rejection_reason text,
    ingestion_log_id integer,
    processing_error text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    batch_job_name character varying(500),
    processing_stage character varying(50),
    result_summary jsonb,
    model_folder character varying(255),
    clinic_name character varying(255),
    upload_duration_ms integer,
    CONSTRAINT ingestion_jobs_status_check CHECK (((status)::text = ANY ((ARRAY['pending_review'::character varying, 'approved'::character varying, 'rejected'::character varying, 'processing'::character varying, 'completed'::character varying, 'failed'::character varying, 'cancelled'::character varying])::text[])))
);


--
-- Name: ingestion_jobs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ingestion_jobs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ingestion_jobs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ingestion_jobs_id_seq OWNED BY public.ingestion_jobs.id;


--
-- Name: ingestion_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingestion_logs (
    id integer NOT NULL,
    dataset_a_filename character varying(255),
    dataset_b_filename character varying(255),
    started_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone,
    rows_processed integer DEFAULT 0,
    rows_inserted integer DEFAULT 0,
    rows_skipped integer DEFAULT 0,
    rows_errored integer DEFAULT 0,
    errors jsonb DEFAULT '[]'::jsonb,
    warnings jsonb DEFAULT '[]'::jsonb
);


--
-- Name: ingestion_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ingestion_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ingestion_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ingestion_logs_id_seq OWNED BY public.ingestion_logs.id;


--
-- Name: patients; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.patients (
    id integer NOT NULL,
    species_id integer,
    breed_id integer,
    sex character varying(20),
    county_id integer,
    anon_id character varying(100),
    zip_code character varying(10),
    data_source character varying(20) DEFAULT 'mock'::character varying,
    diagnosis_date date,
    outcome character varying(20),
    birth_date date,
    registry_no_cancer boolean DEFAULT false NOT NULL,
    registry_no_cancer_source_version character varying(80),
    registry_awaiting_review boolean DEFAULT false NOT NULL,
    CONSTRAINT patients_sex_check CHECK (((sex IS NULL) OR ((sex)::text = ANY (ARRAY[('Male'::character varying)::text, ('Female'::character varying)::text, ('Neutered Male'::character varying)::text, ('Spayed Female'::character varying)::text]))))
);


--
-- Name: species; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.species (
    id integer NOT NULL,
    name character varying(50) NOT NULL
);


--
-- Name: mv_county_cancer_incidence; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.mv_county_cancer_incidence AS
 SELECT p.county_id,
    co.name AS county_name,
    ct.id AS cancer_type_id,
    ct.name AS cancer_type_name,
    s.id AS species_id,
    s.name AS species_name,
    COALESCE(p.sex, 'Unknown'::character varying) AS sex,
    (EXTRACT(year FROM p.diagnosis_date))::integer AS year,
    count(*) AS case_count
   FROM ((((public.case_diagnoses cd
     JOIN public.patients p ON ((cd.patient_id = p.id)))
     LEFT JOIN public.counties co ON ((p.county_id = co.id)))
     JOIN public.cancer_types ct ON ((cd.cancer_type_id = ct.id)))
     JOIN public.species s ON ((p.species_id = s.id)))
  WHERE (((p.data_source)::text = 'petbert'::text) AND ((cd.review_status)::text = ANY ((ARRAY['confirmed'::character varying, 'corrected'::character varying])::text[])) AND ((ct.name)::text <> 'Non-Cancer'::text))
  GROUP BY p.county_id, co.name, ct.id, ct.name, s.id, s.name, COALESCE(p.sex, 'Unknown'::character varying), (EXTRACT(year FROM p.diagnosis_date))
  WITH NO DATA;


--
-- Name: mv_yearly_trends; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.mv_yearly_trends AS
 SELECT (EXTRACT(year FROM p.diagnosis_date))::integer AS year,
    ct.id AS cancer_type_id,
    ct.name AS cancer_type_name,
    s.id AS species_id,
    s.name AS species_name,
    p.county_id,
    co.name AS county_name,
    COALESCE(p.sex, 'Unknown'::character varying) AS sex,
    count(*) AS case_count,
    count(*) FILTER (WHERE ((p.outcome)::text = 'deceased'::text)) AS deceased_count,
    count(*) FILTER (WHERE ((p.outcome)::text = 'alive'::text)) AS alive_count
   FROM ((((public.case_diagnoses cd
     JOIN public.patients p ON ((cd.patient_id = p.id)))
     JOIN public.cancer_types ct ON ((cd.cancer_type_id = ct.id)))
     JOIN public.species s ON ((p.species_id = s.id)))
     LEFT JOIN public.counties co ON ((p.county_id = co.id)))
  WHERE (((p.data_source)::text = 'petbert'::text) AND ((cd.review_status)::text = ANY ((ARRAY['confirmed'::character varying, 'corrected'::character varying])::text[])) AND ((ct.name)::text <> 'Non-Cancer'::text))
  GROUP BY (EXTRACT(year FROM p.diagnosis_date)), ct.id, ct.name, s.id, s.name, p.county_id, co.name, COALESCE(p.sex, 'Unknown'::character varying)
  WITH NO DATA;


--
-- Name: pathology_reports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pathology_reports (
    id integer NOT NULL,
    patient_id integer NOT NULL,
    report_date date,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    storage_path character varying(1000),
    source_diagnosis text
);


--
-- Name: pathology_reports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pathology_reports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pathology_reports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pathology_reports_id_seq OWNED BY public.pathology_reports.id;


--
-- Name: patients_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.patients_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: patients_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.patients_id_seq OWNED BY public.patients.id;


--
-- Name: role_requests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.role_requests (
    id integer NOT NULL,
    email character varying(255) NOT NULL,
    requested_role character varying(20) NOT NULL,
    status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    reason text,
    resolved_by_email character varying(255),
    resolved_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT role_requests_requested_role_check CHECK (((requested_role)::text = ANY ((ARRAY['uploader'::character varying, 'reviewer'::character varying])::text[]))),
    CONSTRAINT role_requests_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'approved'::character varying, 'denied'::character varying])::text[])))
);


--
-- Name: role_requests_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.role_requests_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: role_requests_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.role_requests_id_seq OWNED BY public.role_requests.id;


--
-- Name: species_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.species_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: species_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.species_id_seq OWNED BY public.species.id;


--
-- Name: taxonomy_terms; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.taxonomy_terms (
    id integer NOT NULL,
    vet_icd_o_code character varying(20),
    taxonomy_group character varying(255) NOT NULL,
    taxonomy_term character varying(255) NOT NULL,
    term_level character varying(20)
);


--
-- Name: taxonomy_terms_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.taxonomy_terms_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: taxonomy_terms_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.taxonomy_terms_id_seq OWNED BY public.taxonomy_terms.id;


--
-- Name: user_roles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_roles (
    email character varying(255) NOT NULL,
    is_admin boolean DEFAULT false NOT NULL,
    is_uploader boolean DEFAULT false NOT NULL,
    is_reviewer boolean DEFAULT false NOT NULL,
    updated_by_email character varying(255),
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: audit_list_cases id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_list_cases ALTER COLUMN id SET DEFAULT nextval('public.audit_list_cases_id_seq'::regclass);


--
-- Name: audit_lists id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_lists ALTER COLUMN id SET DEFAULT nextval('public.audit_lists_id_seq'::regclass);


--
-- Name: breeds id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.breeds ALTER COLUMN id SET DEFAULT nextval('public.breeds_id_seq'::regclass);


--
-- Name: calenviroscreen id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calenviroscreen ALTER COLUMN id SET DEFAULT nextval('public.calenviroscreen_id_seq'::regclass);


--
-- Name: cancer_types id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cancer_types ALTER COLUMN id SET DEFAULT nextval('public.cancer_types_id_seq'::regclass);


--
-- Name: case_diagnoses id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses ALTER COLUMN id SET DEFAULT nextval('public.case_diagnoses_id_seq'::regclass);


--
-- Name: case_review_codes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_review_codes ALTER COLUMN id SET DEFAULT nextval('public.case_review_codes_id_seq'::regclass);


--
-- Name: case_reviews id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_reviews ALTER COLUMN id SET DEFAULT nextval('public.case_reviews_id_seq'::regclass);


--
-- Name: counties id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counties ALTER COLUMN id SET DEFAULT nextval('public.counties_id_seq'::regclass);


--
-- Name: diagnosis_review_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis_review_events ALTER COLUMN id SET DEFAULT nextval('public.diagnosis_review_events_id_seq'::regclass);


--
-- Name: export_requests id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.export_requests ALTER COLUMN id SET DEFAULT nextval('public.export_requests_id_seq'::regclass);


--
-- Name: gold_exports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gold_exports ALTER COLUMN id SET DEFAULT nextval('public.gold_exports_id_seq'::regclass);


--
-- Name: ingestion_jobs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_jobs ALTER COLUMN id SET DEFAULT nextval('public.ingestion_jobs_id_seq'::regclass);


--
-- Name: ingestion_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_logs ALTER COLUMN id SET DEFAULT nextval('public.ingestion_logs_id_seq'::regclass);


--
-- Name: pathology_reports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pathology_reports ALTER COLUMN id SET DEFAULT nextval('public.pathology_reports_id_seq'::regclass);


--
-- Name: patients id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patients ALTER COLUMN id SET DEFAULT nextval('public.patients_id_seq'::regclass);


--
-- Name: role_requests id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.role_requests ALTER COLUMN id SET DEFAULT nextval('public.role_requests_id_seq'::regclass);


--
-- Name: species id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.species ALTER COLUMN id SET DEFAULT nextval('public.species_id_seq'::regclass);


--
-- Name: taxonomy_terms id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.taxonomy_terms ALTER COLUMN id SET DEFAULT nextval('public.taxonomy_terms_id_seq'::regclass);


--
-- Name: audit_list_cases audit_list_cases_audit_list_id_case_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_list_cases
    ADD CONSTRAINT audit_list_cases_audit_list_id_case_id_key UNIQUE (audit_list_id, case_id);


--
-- Name: audit_list_cases audit_list_cases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_list_cases
    ADD CONSTRAINT audit_list_cases_pkey PRIMARY KEY (id);


--
-- Name: audit_lists audit_lists_list_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_lists
    ADD CONSTRAINT audit_lists_list_id_key UNIQUE (list_id);


--
-- Name: audit_lists audit_lists_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_lists
    ADD CONSTRAINT audit_lists_pkey PRIMARY KEY (id);


--
-- Name: breeds breeds_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.breeds
    ADD CONSTRAINT breeds_pkey PRIMARY KEY (id);


--
-- Name: breeds breeds_species_id_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.breeds
    ADD CONSTRAINT breeds_species_id_name_key UNIQUE (species_id, name);


--
-- Name: calenviroscreen calenviroscreen_county_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calenviroscreen
    ADD CONSTRAINT calenviroscreen_county_id_key UNIQUE (county_id);


--
-- Name: calenviroscreen calenviroscreen_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calenviroscreen
    ADD CONSTRAINT calenviroscreen_pkey PRIMARY KEY (id);


--
-- Name: cancer_types cancer_types_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cancer_types
    ADD CONSTRAINT cancer_types_name_key UNIQUE (name);


--
-- Name: cancer_types cancer_types_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cancer_types
    ADD CONSTRAINT cancer_types_pkey PRIMARY KEY (id);


--
-- Name: case_diagnoses case_diagnoses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses
    ADD CONSTRAINT case_diagnoses_pkey PRIMARY KEY (id);


--
-- Name: case_review_codes case_review_codes_case_review_id_taxonomy_group_taxonomy_te_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_review_codes
    ADD CONSTRAINT case_review_codes_case_review_id_taxonomy_group_taxonomy_te_key UNIQUE (case_review_id, taxonomy_group, taxonomy_term);


--
-- Name: case_review_codes case_review_codes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_review_codes
    ADD CONSTRAINT case_review_codes_pkey PRIMARY KEY (id);


--
-- Name: case_reviews case_reviews_case_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_reviews
    ADD CONSTRAINT case_reviews_case_id_key UNIQUE (case_id);


--
-- Name: case_reviews case_reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_reviews
    ADD CONSTRAINT case_reviews_pkey PRIMARY KEY (id);


--
-- Name: counties counties_fips_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counties
    ADD CONSTRAINT counties_fips_code_key UNIQUE (fips_code);


--
-- Name: counties counties_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counties
    ADD CONSTRAINT counties_name_key UNIQUE (name);


--
-- Name: counties counties_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counties
    ADD CONSTRAINT counties_pkey PRIMARY KEY (id);


--
-- Name: diagnosis_review_events diagnosis_review_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis_review_events
    ADD CONSTRAINT diagnosis_review_events_pkey PRIMARY KEY (id);


--
-- Name: export_requests export_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.export_requests
    ADD CONSTRAINT export_requests_pkey PRIMARY KEY (id);


--
-- Name: gold_exports gold_exports_export_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gold_exports
    ADD CONSTRAINT gold_exports_export_id_key UNIQUE (export_id);


--
-- Name: gold_exports gold_exports_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gold_exports
    ADD CONSTRAINT gold_exports_pkey PRIMARY KEY (id);


--
-- Name: ingestion_jobs ingestion_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_jobs
    ADD CONSTRAINT ingestion_jobs_pkey PRIMARY KEY (id);


--
-- Name: ingestion_logs ingestion_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_logs
    ADD CONSTRAINT ingestion_logs_pkey PRIMARY KEY (id);


--
-- Name: pathology_reports pathology_reports_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pathology_reports
    ADD CONSTRAINT pathology_reports_pkey PRIMARY KEY (id);


--
-- Name: patients patients_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT patients_pkey PRIMARY KEY (id);


--
-- Name: role_requests role_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.role_requests
    ADD CONSTRAINT role_requests_pkey PRIMARY KEY (id);


--
-- Name: species species_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.species
    ADD CONSTRAINT species_name_key UNIQUE (name);


--
-- Name: species species_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.species
    ADD CONSTRAINT species_pkey PRIMARY KEY (id);


--
-- Name: taxonomy_terms taxonomy_terms_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.taxonomy_terms
    ADD CONSTRAINT taxonomy_terms_pkey PRIMARY KEY (id);


--
-- Name: taxonomy_terms taxonomy_terms_taxonomy_group_taxonomy_term_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.taxonomy_terms
    ADD CONSTRAINT taxonomy_terms_taxonomy_group_taxonomy_term_key UNIQUE (taxonomy_group, taxonomy_term);


--
-- Name: user_roles user_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_roles
    ADD CONSTRAINT user_roles_pkey PRIMARY KEY (email);


--
-- Name: idx_audit_list_cases_case_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_list_cases_case_id ON public.audit_list_cases USING btree (case_id);


--
-- Name: idx_audit_list_cases_list_position; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_list_cases_list_position ON public.audit_list_cases USING btree (audit_list_id, "position");


--
-- Name: idx_audit_lists_one_active; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_audit_lists_one_active ON public.audit_lists USING btree (is_active) WHERE is_active;


--
-- Name: idx_cancer_types_unconfirmed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cancer_types_unconfirmed ON public.cancer_types USING btree (confirmed) WHERE (confirmed = false);


--
-- Name: idx_case_diagnoses_cancer_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_diagnoses_cancer_type ON public.case_diagnoses USING btree (cancer_type_id);


--
-- Name: idx_case_diagnoses_icd; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_diagnoses_icd ON public.case_diagnoses USING btree (icd_o_code) WHERE (icd_o_code IS NOT NULL);


--
-- Name: idx_case_diagnoses_job; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_diagnoses_job ON public.case_diagnoses USING btree (ingestion_job_id);


--
-- Name: idx_case_diagnoses_review_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_diagnoses_review_status ON public.case_diagnoses USING btree (review_status) WHERE ((review_status)::text <> 'confirmed'::text);


--
-- Name: idx_case_reviews_reviewer_locked; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_reviews_reviewer_locked ON public.case_reviews USING btree (reviewed_by_email, locked);


--
-- Name: idx_counties_geom; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_counties_geom ON public.counties USING gist (geom);


--
-- Name: idx_counties_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_counties_name ON public.counties USING btree (name);


--
-- Name: idx_diagnosis_review_events_actor; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_diagnosis_review_events_actor ON public.diagnosis_review_events USING btree (actor_email);


--
-- Name: idx_diagnosis_review_events_case; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_diagnosis_review_events_case ON public.diagnosis_review_events USING btree (case_diagnosis_id, created_at DESC);


--
-- Name: idx_ingestion_jobs_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ingestion_jobs_status ON public.ingestion_jobs USING btree (status);


--
-- Name: idx_mv_county_cancer; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_mv_county_cancer ON public.mv_county_cancer_incidence USING btree (county_id, cancer_type_id, species_id, sex, year) NULLS NOT DISTINCT;


--
-- Name: idx_mv_yearly_trends; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_mv_yearly_trends ON public.mv_yearly_trends USING btree (year, cancer_type_id, species_id, county_id, sex) NULLS NOT DISTINCT;


--
-- Name: idx_pathology_reports_patient_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pathology_reports_patient_id ON public.pathology_reports USING btree (patient_id);


--
-- Name: idx_patients_anon_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_patients_anon_id ON public.patients USING btree (anon_id);


--
-- Name: idx_patients_county; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_patients_county ON public.patients USING btree (county_id);


--
-- Name: idx_patients_species; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_patients_species ON public.patients USING btree (species_id);


--
-- Name: idx_role_requests_email; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_role_requests_email ON public.role_requests USING btree (lower((email)::text));


--
-- Name: idx_role_requests_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_role_requests_status ON public.role_requests USING btree (status) WHERE ((status)::text = 'pending'::text);


--
-- Name: idx_user_roles_email_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_roles_email_lower ON public.user_roles USING btree (lower((email)::text));


--
-- Name: audit_list_cases audit_list_cases_audit_list_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_list_cases
    ADD CONSTRAINT audit_list_cases_audit_list_id_fkey FOREIGN KEY (audit_list_id) REFERENCES public.audit_lists(id) ON DELETE CASCADE;


--
-- Name: breeds breeds_species_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.breeds
    ADD CONSTRAINT breeds_species_id_fkey FOREIGN KEY (species_id) REFERENCES public.species(id);


--
-- Name: calenviroscreen calenviroscreen_county_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calenviroscreen
    ADD CONSTRAINT calenviroscreen_county_id_fkey FOREIGN KEY (county_id) REFERENCES public.counties(id);


--
-- Name: case_diagnoses case_diagnoses_cancer_type_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses
    ADD CONSTRAINT case_diagnoses_cancer_type_id_fkey FOREIGN KEY (cancer_type_id) REFERENCES public.cancer_types(id);


--
-- Name: case_diagnoses case_diagnoses_ingestion_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses
    ADD CONSTRAINT case_diagnoses_ingestion_job_id_fkey FOREIGN KEY (ingestion_job_id) REFERENCES public.ingestion_jobs(id);


--
-- Name: case_diagnoses case_diagnoses_original_cancer_type_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses
    ADD CONSTRAINT case_diagnoses_original_cancer_type_id_fkey FOREIGN KEY (original_cancer_type_id) REFERENCES public.cancer_types(id);


--
-- Name: case_diagnoses case_diagnoses_pathology_report_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses
    ADD CONSTRAINT case_diagnoses_pathology_report_id_fkey FOREIGN KEY (pathology_report_id) REFERENCES public.pathology_reports(id) ON DELETE SET NULL;


--
-- Name: case_review_codes case_review_codes_case_review_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_review_codes
    ADD CONSTRAINT case_review_codes_case_review_id_fkey FOREIGN KEY (case_review_id) REFERENCES public.case_reviews(id) ON DELETE CASCADE;


--
-- Name: case_reviews case_reviews_gold_export_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_reviews
    ADD CONSTRAINT case_reviews_gold_export_id_fkey FOREIGN KEY (gold_export_id) REFERENCES public.gold_exports(id) ON DELETE SET NULL;


--
-- Name: diagnosis_review_events diagnosis_review_events_cancer_type_id_after_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis_review_events
    ADD CONSTRAINT diagnosis_review_events_cancer_type_id_after_fkey FOREIGN KEY (cancer_type_id_after) REFERENCES public.cancer_types(id);


--
-- Name: diagnosis_review_events diagnosis_review_events_cancer_type_id_before_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis_review_events
    ADD CONSTRAINT diagnosis_review_events_cancer_type_id_before_fkey FOREIGN KEY (cancer_type_id_before) REFERENCES public.cancer_types(id);


--
-- Name: diagnosis_review_events diagnosis_review_events_case_diagnosis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.diagnosis_review_events
    ADD CONSTRAINT diagnosis_review_events_case_diagnosis_id_fkey FOREIGN KEY (case_diagnosis_id) REFERENCES public.case_diagnoses(id) ON DELETE CASCADE;


--
-- Name: case_diagnoses fk_case_diagnoses_patient; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_diagnoses
    ADD CONSTRAINT fk_case_diagnoses_patient FOREIGN KEY (patient_id) REFERENCES public.patients(id);


--
-- Name: ingestion_jobs ingestion_jobs_ingestion_log_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_jobs
    ADD CONSTRAINT ingestion_jobs_ingestion_log_id_fkey FOREIGN KEY (ingestion_log_id) REFERENCES public.ingestion_logs(id);


--
-- Name: pathology_reports pathology_reports_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pathology_reports
    ADD CONSTRAINT pathology_reports_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- Name: patients patients_breed_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT patients_breed_id_fkey FOREIGN KEY (breed_id) REFERENCES public.breeds(id);


--
-- Name: patients patients_county_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT patients_county_id_fkey FOREIGN KEY (county_id) REFERENCES public.counties(id);


--
-- Name: patients patients_species_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT patients_species_id_fkey FOREIGN KEY (species_id) REFERENCES public.species(id);


--
-- Name: breeds Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.breeds FOR SELECT USING (true);


--
-- Name: cancer_types Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.cancer_types FOR SELECT USING (true);


--
-- Name: counties Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.counties FOR SELECT USING (true);


--
-- Name: species Enable read access for all users; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Enable read access for all users" ON public.species FOR SELECT USING (true);


--
-- Name: audit_list_cases; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.audit_list_cases ENABLE ROW LEVEL SECURITY;

--
-- Name: audit_lists; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.audit_lists ENABLE ROW LEVEL SECURITY;

--
-- Name: breeds; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.breeds ENABLE ROW LEVEL SECURITY;

--
-- Name: calenviroscreen; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.calenviroscreen ENABLE ROW LEVEL SECURITY;

--
-- Name: cancer_types; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cancer_types ENABLE ROW LEVEL SECURITY;

--
-- Name: case_diagnoses; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.case_diagnoses ENABLE ROW LEVEL SECURITY;

--
-- Name: case_review_codes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.case_review_codes ENABLE ROW LEVEL SECURITY;

--
-- Name: case_reviews; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.case_reviews ENABLE ROW LEVEL SECURITY;

--
-- Name: counties; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.counties ENABLE ROW LEVEL SECURITY;

--
-- Name: diagnosis_review_events; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.diagnosis_review_events ENABLE ROW LEVEL SECURITY;

--
-- Name: export_requests; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.export_requests ENABLE ROW LEVEL SECURITY;

--
-- Name: gold_exports; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.gold_exports ENABLE ROW LEVEL SECURITY;

--
-- Name: ingestion_jobs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;

--
-- Name: ingestion_logs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.ingestion_logs ENABLE ROW LEVEL SECURITY;

--
-- Name: pathology_reports; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.pathology_reports ENABLE ROW LEVEL SECURITY;

--
-- Name: patients; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

--
-- Name: role_requests; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.role_requests ENABLE ROW LEVEL SECURITY;

--
-- Name: species; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.species ENABLE ROW LEVEL SECURITY;

--
-- Name: taxonomy_terms; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.taxonomy_terms ENABLE ROW LEVEL SECURITY;

--
-- Name: user_roles; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

--
-- PostgreSQL database dump complete
--

\unrestrict xGJFNIr8hS3AwxCcDbtC4EIQpJcYoVJLXVqsHe2A41JwcrHapxiTXEsKTn2vtan

