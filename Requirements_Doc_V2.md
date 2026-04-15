# **Product Requirements Document V2**

Team 14: UC Davis VMTH Cancer Registry
Authors: [Yugraj Dhillon](mailto:ydhillon@ucdavis.edu), [David Estrella](mailto:dtestrella@ucdavis.edu), [Chun Ho Li](mailto:lchli@ucdavis.edu), [Justin Pak](mailto:jetpak@ucdavis.edu)

## Table of Contents {#table-of-contents}

[Introduction](#introduction)

[Competitive Analysis](#competitive-analysis)

[Technology Survey](#technology-survey)

[System Architecture Overview](#system-architecture-overview)

[Requirements](#requirements)

[Technologies Employed](#technologies-employed)

[Real-World Constraint Analysis](#real-world-constraint-analysis)

[Social/Legal Aspect of the Product](#social/legal-aspect-of-the-product)

[Glossary of Terms](#glossary-of-terms)

[References](#references)

## Introduction  {#introduction}

There exists refined and expansive databases for human cancer cases, but the same cannot be said for animals such as dogs and cats. Efforts have been made to create one, but nothing definitive has yet to be created. UC Davis, one of the number one Veterinarian programs in the world has 30+ years of free text clinical data that is unusable currently due to it being unorganized.

What we aim to accomplish is to organize the clinical data, and to create a cancer registry with a dashboard for researchers to use to identify geographical data to help us understand cancer risks in certain regions. It is not only helpful to know that cancer rates are high or low in a region for dogs and cats, but it can also reveal cancer risks that may lay for humans as well (e.g., we could explore if environmental factors are affecting both humans and animals in an area). This registry will be one of a kind in California, and we hope that other states in this nation will adopt this as well.

## Competitive Analysis {#competitive-analysis}

We analyzed five similar existing cancer registries:

### Take C.H.A.R.G.E {#take-c.h.a.r.g.e}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Allows the public to upload their dog's medical records anonymously Informs users of average costs and treatment options for specific cancers in their region Uses the Vet-ICD-O-canine-1 coding system to allow comparison between dogs and humans | Allowing the public to upload records includes a risk of erroneous or incomplete medical records Funded by pharmaceutical company (Jaguar Health) which brings unease as to what is done with the data |

### VMDB {#vmdb}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Receives data from only Veterinary Teaching Hospitals, ensuring accurate diagnoses Operating since 1964 so it has a history of data that can be used to track cancer trends | Uses the SNOMED coding system which requires specific formatting, not scalable to other Veterinarian facilities Requires formal requests and fees to access the database Data is skewed to pets whose owners could afford university specialists |

### SAVSNET {#savsnet}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Receives data in real-time through the electronic health records of participating private vet clinics and labs Uses NLP (PetBERT) and Text Mining to map data to codes Reduces data bias as less specialized clinics can contribute data | Uses the VeNom coding system (UK standard) and SNOMED-CT which are not as scalable Uses data only from the UK |

### Swiss Canine and Feline Cancer Registry {#swiss-canine-and-feline-cancer-registry}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Tracks all cases in a specific geographic population, allowing calculation of true incidence rates Uses the Vet-ICD-O-canine-1 coding system to allow comparison between dogs and humans | Restricted to Switzerland, lack of scalability Maintenance is expensive and labor-intensive due to level of detail |

### ACARCinom {#acarcinom}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Gathers data in a convenient way (through biopsy) which reduces additional work required Uses the Vet-ICD-O-canine-1 coding system to allow comparison between dogs and humans | Requires tumor biopsy for data, leading to undercounting of cancer if a pet is not biopsied |

## Technology Survey {#technology-survey}

### Front-end {#front-end}

Conclusion: We evaluated both React and Angular and we came to the conclusion that React would be the most logical choice. It's the most supported frontend framework out there. It has strong support for data visualization libraries, it's easy to integrate with REST APIs, and it's widely adopted in research environments over angular.

| React |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Commonly used for websites with interactive elements Relatively easy to pick up | Might be too heavy and can easily end up creating a slow website |

| Angular |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Strong TypeScript support | Steep learning curve Probably overkill for our use case |

### Back-end {#back-end}

Conclusion: NLP pipelines and data processing are primarily Python-based. Coupled with our plan to use BERT, Python is an obvious inclusion for our project (at least on the business logic side). We also plan on using FastAPI for backend development since we felt the FastAPI's strong points aligned with our team's goals. Lastly, for the website itself, our plan is to code the website using TypeScript due to its benefits over JavaScript (type safety, tooling support, error catching, etc) making it well suited for a team project.

| FastAPI |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Native support for NLP (BERT, SpaCy, HuggingFace) Easy REST API creation Supports powerful libraries like Pandas and NumPy Widely used in research environments | Slightly slower than node for high-throughput workloads (relevant given the amount of data we're handling) |

| Node.js/Bun |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Good frontend tooling support Good native ecosystem Since we have no tech debt we don't have to worry about trying to adopt new tech | New tech so there might be issues Limited ML/NLP support as far as we know Self-contained ecosystem doesn't matter as much when we still need Python anyways for BERT |

### Database {#database}

Conclusion: We went with PostgreSQL hosted on Supabase. PostgreSQL suits our use case with PostGIS for geospatial queries and strong relational data integrity. Supabase provides managed hosting, built-in authentication, and an admin dashboard for non-technical team members.

| MySQL |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Simple to setup Good performance for read-heavy workloads Native full-text search capabilities | Weaker geospatial support since can't use PostGIS Less flexible with complex data types |

| PostgreSQL |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Geospatial support via PostGIS extension Native full-text search capabilities Strong data integrity with ACID compliance | Slightly more complex setup than MySQL |

| MongoDB |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Flexible schema, good for inconsistent medical record formats Easy to iterate on data models | Not as good for doing relational queries (e.g., linking patient records, diagnoses, and outcomes to each other) Geospatial features less mature compared to PostGIS extension |

| Elastisearch |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Excellent for doing processes involving full-text search (e.g., pathology reports) Fast aggregations and filtering Good for exploratory data analysis | Complex to maintain |

### API Structuring {#api-structuring}

Conclusion: The clear choice is RESTful APIs to connect the frontend with the backend, it just provides the most simplicity with good performance. It'll be easy to debug and it'll provide the analytics endpoints necessary to create the dashboard.

| REST |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Simple Native support in FastAPI Works well with React Minimal learning curve | Can overfetch in some scenarios (mostly a non-issue given our use case). |

| GraphQL |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Flexible queries | More complex, higher learning curves |

| gRPC |  |
| ----- | ----- |
| **Pros** | **Cons** |
| High performance Real time streaming | Overkill for our use case |

### NLP/Data Extraction {#nlp/data-extraction}

Conclusion: The client wants BERT, so the best option for doing NLP is using BERT. Also using an LLM like ChatGPT for example presents many privacy issues that we do not want to go down. We use PetBERT, a fine-tuned BERT model trained on veterinary clinical text, deployed as a standalone ML worker microservice.

| BERT |  |
| ----- | ----- |
| **Pros** | **Cons** |
| It's what the client wants us to use. The client is familiar with BERT as well so they'll have an easier time using the final product when we deliver it to them. | The entire team has to take some time get familiar with BERT (not really a downside so much as a required aspect). |

| LLM APIs |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Strong extractions Minimal | Cost Privacy Concerns |

| SpaCy NER |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Lightweight Easy to customize | Lower accuracy |

### Geospatial Analysis {#geospatial-analysis}

Conclusion: We decided that using both for this project ended up being the best choice. GeoPandas for geospatial data processing, and PostGIS for spatial storage and queries. This will allow us to do spatial joins between cancer cases and census tracts. Frontend visualization will be done through react mapping libraries.

| PostGIS (PostgreSQL Extension) |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Stores geometry directly in the database Enables spatial queries Allows aggregations by census tract or region | Lengthy setup |

| GeoPandas |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Python-native Supports spatial joins Integrates easily with Pandas workflow | Slower when working with big datasets |

### Project Management Software {#project-management-software}

Conclusion: We will be using GitHub Projects for now due to its tight integrations, but if it fails to fulfill our needs then we can quickly and easily pivot to a Trello-based solution

| Trello |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Easy to setup More familiar | Some features we want might be paywalled Requires setup for automation (e.g., PM needs to write GitHub Actions) |

| GitHub Projects |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Can be linked with issues on GitHub Comes built in to GitHub Tight integration with GitHub issues | Less familiar (more onboarding/setup) Less features Might not have the features we want |

## System Architecture Overview  {#system-architecture-overview}

The system follows a three-layer architecture: a React frontend, a FastAPI backend with a separate ML worker, and a PostgreSQL database hosted on Supabase. All services are containerized with Docker Compose.

### Components

**Frontend (React + TypeScript + Vite)**

The frontend is a single-page application that provides six tabs for data exploration and two tabs for data management. It communicates with the backend exclusively through RESTful API calls. Authentication is handled via Supabase Auth (JWT-based), with the Supabase JS client used only for sign-in/sign-out flows. The frontend never queries the database directly.

Tabs:
- **Overview** — Summary statistics, county-level data table, and an interactive choropleth map of California. Supports filtering by species, cancer type, breed, sex, and year range.
- **Breed Disparities** — Autocomplete breed search with per-breed breakdown of cancer types, sex distribution, and geographic distribution.
- **Cancer Types** — Bar chart of the top 10 cancer types by case count, responsive to active filters.
- **Analysis** — Side-by-side comparison of three choropleth maps: VMTH veterinary cancer incidence, CalEnviroScreen 4.0 environmental health indicators (25 selectable metrics), and human cancer registry rates (California Cancer Registry 2017–2021). Supports 2-map and 3-map layout modes with a map pair selector.
- **Data Upload** — CSV upload form requiring two files (Dataset A: clinical diagnoses, Dataset B: demographics). Uploads are submitted for admin review. Authenticated users see a "My Uploads" table showing their submission history and status.
- **Review Queue** (admin only) — Lists all ingestion jobs with status badges. Admins can preview uploaded CSVs, approve jobs (triggering the NLP pipeline), or reject them with an optional reason. Auto-polls every 10 seconds when jobs are processing.

**Backend (FastAPI + Python 3.11)**

The backend serves as the central API layer. It handles data queries, file uploads, authentication verification, and orchestrates the ingestion pipeline. It connects to PostgreSQL via SQLAlchemy (async) and to the ML worker via HTTP.

API surface:

| Route group | Endpoints | Auth |
|---|---|---|
| Dashboard | `GET /summary`, `GET /filters` | Public |
| Incidence | `GET /incidence`, `/by-cancer-type`, `/by-species`, `/by-breed`, `/breed-detail` | Public |
| Geo | `GET /counties` (GeoJSON), `GET /counties/{id}`, `GET /calenviroscreen` | Public |
| Trends | `GET /yearly`, `GET /by-cancer-type` | Public |
| Ingestion | `POST /upload`, `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/preview/{dataset}`, `POST /jobs/{id}/review` | Mixed (upload is optional auth, jobs require auth, review requires admin) |
| Auth | `GET /me` | Required |

All public endpoints filter to `data_source = 'petbert'` to exclude mock/seed data from production views.

**ML Worker (FastAPI microservice)**

A standalone container running PetBERT, a fine-tuned BERT model for veterinary cancer classification. Exposes two endpoints:

- `GET /health` — Health check
- `POST /predict` — Accepts a CSV upload (columns: `anon_id`, `Clinical Diagnoses`), runs PetBERT inference, and returns structured predictions with `predicted_term`, `predicted_code`, `confidence`, and `method` per row.

Configuration: batch_size=16, max_length=256, neighbors_k=3, embedding_min_sim=0.6, automatic CPU/GPU device selection.

**Database (PostgreSQL 16 + PostGIS 3.4 on Supabase)**

Core tables:
- `patients` — Species, breed, sex, county, ZIP code, anonymized ID, data source, diagnosis date, outcome
- `case_diagnoses` — One row per predicted cancer per patient, with ICD-O code, predicted term, confidence score, and prediction method
- `counties` — All 58 California counties with PostGIS MULTIPOLYGON geometries and a `is_catchment` flag for the 16 UCD catchment counties
- `calenviroscreen` — CalEnviroScreen 4.0 county-level environmental health indicators (25 metrics)
- `ingestion_jobs` — Upload lifecycle tracking (pending_review → processing → completed/failed/rejected)
- `ingestion_logs` — Audit trail for ingestion runs (rows processed, inserted, skipped, errors)

Lookup tables: `species`, `breeds`, `cancer_types`

Materialized views: `mv_county_cancer_incidence` (county × cancer type × year), `mv_yearly_trends` (year × cancer type × species)

Row Level Security is enabled on all tables. Sensitive tables (patients, case_diagnoses, ingestion_jobs, ingestion_logs, calenviroscreen) have no permissive policies, blocking direct access via the Supabase anon key. Lookup tables allow public SELECT only.

### Data Ingestion Pipeline

1. User uploads Dataset A (clinical notes CSV) and Dataset B (demographics CSV) via the Data Upload tab
2. Backend creates an `IngestionJob` with status `pending_review` and saves files to disk
3. Admin reviews the upload in the Review Queue, previews the CSV contents, and approves or rejects
4. On approval, the backend spawns an async background task that:
   - POSTs Dataset A to the ML worker for PetBERT classification
   - Parses ML predictions and demographics by `anon_id`
   - Upserts patients with ZIP-to-county geocoding
   - Inserts `case_diagnoses` with ICD-O codes and confidence scores
   - Refreshes materialized views
   - Updates job status to `completed` or `failed`
5. The user can track job progress in the Data Upload tab; the admin can monitor in the Review Queue

### Dependencies

| Component | Key dependencies |
|---|---|
| Backend | FastAPI 0.109, SQLAlchemy 2.0 (async), asyncpg, GeoAlchemy2, Pydantic 2, PyJWT, httpx, pandas, openpyxl, zipcodes |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS v4, react-simple-maps, d3-scale, @supabase/supabase-js |
| ML Worker | FastAPI, transformers (HuggingFace), torch, scikit-learn, pandas |
| Infrastructure | Docker Compose, PostgreSQL 16, PostGIS 3.4, Supabase (hosting + auth) |

## Requirements {#requirements}

### User Stories (Updated) {#user-stories}

**Priority 1 — Core functionality (implemented)**

1. As a researcher, I want to view cancer cases on an interactive California county map so I can identify geographic patterns in cancer incidence.
   - *Acceptance test:* Load the Overview tab. The choropleth map renders all 58 CA counties. Hovering over a county shows case count. Counties with more cases appear darker.

2. As a researcher, I want to filter data by species, breed, sex, cancer type, and year range so I can investigate specific risk factors.
   - *Acceptance test:* Select "Golden Retriever" from the breed filter and "Lymphoma" from cancer type. All charts, tables, and the map update to show only matching records.

3. As a researcher, I want to compare veterinary cancer incidence with environmental health data so I can explore potential correlations between cancer and environmental factors.
   - *Acceptance test:* Open the Analysis tab. Three maps display side-by-side: VMTH cancer incidence, CalEnviroScreen 4.0 (select "Pollution Burden" from the dropdown), and human cancer rates. Toggle to 2-map mode and select a different pair.

4. As a data manager, I want to upload clinical CSV files that are processed by the NLP pipeline so new patient records are added to the registry without manual data entry.
   - *Acceptance test:* Go to Data Upload, select a clinical notes CSV (Dataset A) and a demographics CSV (Dataset B), click "Submit for Review." The upload appears in "My Uploads" with status "Pending Review."

5. As an admin, I want to review and approve or reject uploaded datasets before they are ingested so I can ensure data quality.
   - *Acceptance test:* Sign in as admin. Open the Review Queue. Click "Preview A" on a pending job to inspect the CSV. Click "Approve." The status changes to "Processing" and eventually "Completed." Verify new records appear in the dashboard.

6. As a researcher, I want to view breed-specific cancer breakdowns so I can identify which breeds are predisposed to which cancers.
   - *Acceptance test:* Open the Breed Disparities tab. Search for "Boxer." The view shows total cases, a bar chart of the top cancer types for Boxers, sex breakdown, and a county distribution map.

7. As a researcher, I want to view cancer type distributions across the registry so I can understand the relative prevalence of different cancers.
   - *Acceptance test:* Open the Cancer Types tab. A bar chart shows the top 10 cancer types ordered by count. Apply a species filter; the chart updates.

8. As a user, I want all patient data to be de-identified so that no individual animal or owner can be traced.
   - *Acceptance test:* Query the `patients` table. No owner names, addresses, or contact info are stored. Patients are identified only by `anon_id`. Row Level Security blocks direct database access from the browser.

**Priority 2 — Authentication and access control (implemented)**

9. As a user, I want to sign in with my email and password so my uploads are tracked to my account.
   - *Acceptance test:* Click "Sign In" in the navigation bar. Enter email and password. After successful login, the user's email appears in the top bar and "My Uploads" shows only their submissions.

10. As an admin, I want only my approved admin accounts to access the Review Queue so unauthorized users cannot approve or reject data.
    - *Acceptance test:* Sign in as a non-admin user. The "Review Queue" tab does not appear. Attempting to call `POST /jobs/{id}/review` directly returns HTTP 403.

11. As an admin, I want non-admin users to be rate-limited to 3 uploads per day so the system is not abused.
    - *Acceptance test:* Sign in as a regular user. Submit 3 uploads. Attempt a 4th upload. The system returns HTTP 429 with the message "Upload limit reached (3 per day)."

**Priority 3 — Comparative analysis (implemented)**

12. As a researcher, I want to compare human cancer rates with veterinary cancer rates by county so I can identify geographic overlap that may suggest shared environmental risk factors.
    - *Acceptance test:* Open the Analysis tab. The Human Cancer Registry map shows age-adjusted rates per 100K from the California Cancer Registry (2017–2021). Compare visually with the VMTH map.

13. As a researcher, I want to view CalEnviroScreen 4.0 environmental health indicators by county so I can overlay environmental burden with cancer data.
    - *Acceptance test:* On the Analysis tab, select different indicators from the CalEnviroScreen dropdown (e.g., Ozone, PM2.5, Poverty). The map recolors to reflect the selected metric.

**Priority 4 — Future enhancements (not yet implemented)**

14. As a researcher, I want to view trend lines over time so I can determine if the prevalence of specific cancers is increasing or decreasing.
    - *Acceptance test:* Open a Trends tab. A line chart shows yearly case counts. Apply a cancer type filter; the chart shows trend lines per selected type.

15. As a researcher, I want to query the registry using natural language so I can find data without knowledge of SQL or coding systems.
    - *Acceptance test:* Enter a free-text query like "lymphoma cases in Sacramento County." The system returns matching records without requiring SQL.

16. As a researcher, I want the system to flag low-confidence NLP classifications for manual review so ambiguous diagnoses are not silently discarded.
    - *Acceptance test:* After ingestion, cases with confidence below a threshold appear in a review queue with the original text and predicted classification. A reviewer can accept or override.

17. As a user, I want to view diagnoses coded with the ICD-O coding system so I can compare veterinary and human cancer data using a standardized classification.
    - *Acceptance test:* Case diagnoses display their ICD-O code alongside the predicted cancer term. Researchers can filter or group by ICD-O code.

18. As a researcher, I want to search pathology reports by keyword so I can find specific cases or terminology across the registry.
    - *Acceptance test:* Enter "lymphoma" in a search field. The system returns pathology reports containing that keyword with highlighted matches and pagination.

## Prototyping Code {#prototyping-code}

Github URL: [https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry](https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry)

## Technologies Employed {#technologies-employed}

**Programming Languages:** Python 3.11 (backend, NLP, data processing), TypeScript (frontend)
**Backend Framework:** FastAPI 0.109 with SQLAlchemy 2.0 (async) and Pydantic v2
**Frontend Framework:** React 18 with Vite, Tailwind CSS v4, react-simple-maps, d3-scale
**Database:** PostgreSQL 16 with PostGIS 3.4, hosted on Supabase
**Authentication:** Supabase Auth (JWT with ES256/HS256), PyJWT for backend verification
**NLP/Data Extraction:** PetBERT (fine-tuned BERT model via HuggingFace transformers + PyTorch)
**Geospatial Analysis:** PostGIS for spatial storage and queries, county boundary GeoJSON for frontend rendering
**API Structuring:** REST (FastAPI auto-generates OpenAPI/Swagger docs at `/docs`)
**Infrastructure:** Docker Compose (6 services: backend, frontend, ml-worker, seed, ingest, geo-seed)
**Project Management:** GitHub Projects

## Real-World Constraint Analysis {#real-world-constraint-analysis}

### Cost {#cost}

The majority of project costs are incurred during the development phase, primarily due to the computational resources required to train the PetBERT model on client-provided data. Supabase provides a free tier for database hosting and authentication. During deployment, costs will consist of hosting the backend and ML worker on a cloud VM (estimated ~$12–15/month for a GCP Compute Engine e2-medium instance or equivalent). The Supabase free tier includes 500MB of database storage and 1GB of file storage, sufficient for current data volumes. As the platform expands to include additional clinics or datasets, periodic retraining of the model will be necessary, resulting in additional computational expenses.

### Space {#space}

The server infrastructure is hosted using cloud services (Supabase for the database, a cloud VM for the backend and ML worker), eliminating the need for on-site hardware. The PetBERT model checkpoint requires approximately 1GB of storage. Uploaded CSV files are stored on a Docker volume on the hosting VM. The current dataset (~395 patients, ~2,348 case diagnoses) is small; the architecture can scale to tens of thousands of records without architectural changes.

### Security {#security}

Authentication is handled through Supabase Auth with JWT tokens (ES256). The backend verifies tokens on protected endpoints using FastAPI dependency injection. Admin access is controlled via an `ADMIN_EMAILS` environment variable. Row Level Security (RLS) is enabled on all database tables, blocking direct access from the browser via the Supabase anon key. All data access flows through the authenticated backend API. The upload endpoint is rate-limited to 3 submissions per day for non-admin users. Environment secrets (database credentials, JWT secrets) are stored in `.env` files that are gitignored and never committed to version control.

### Privacy {#privacy}

All patient data is de-identified prior to ingestion. Patients are identified only by an anonymized `anon_id`. No owner names, addresses, phone numbers, or other personally identifiable information (PII) are stored in the database. CSV files uploaded for review are stored server-side and accessible only to admin users through authenticated API endpoints.

### Scalability {#scalability}

While we have access to extensive records from a single clinic for training, we anticipate a potential decrease in model accuracy when applying the system to clinics from different locations. This is likely due to variations in diagnostic writing styles among veterinarians. To mitigate this issue, the model would need to be retrained or fine-tuned using data from new clinics. The ML worker is deployed as a separate microservice, allowing it to scale independently from the API layer. Materialized views pre-compute aggregations for dashboard performance.

### Maintainability {#maintainability}

The codebase is organized into modular layers: routers, services, models, and schemas on the backend; components, contexts, and API client on the frontend. Docker Compose allows any developer to reproduce the full environment with a single command. Database schema changes are tracked as numbered SQL migration files. A comprehensive README documents the full setup process, including environment configuration, database migrations, authentication setup, and troubleshooting.

## Social/Legal Aspect of the Product  {#social/legal-aspect-of-the-product}

The cancer registry for cats and dogs provides meaningful social value by offering researchers in the veterinary field access to an interactive dashboard with a geospatial interface. This platform represents the first comprehensive cancer registry for cats and dogs in California. By centralizing and visualizing veterinary cancer data, the registry supports researchers in identifying spatial and temporal patterns in cancer cases among companion animals. The Analysis tab enables exploratory comparison between veterinary cancer incidence, human cancer rates from the California Cancer Registry, and environmental health indicators from CalEnviroScreen 4.0, contributing to broader public and comparative health research.

From a legal perspective, the client retains full ownership and rights to all data used for model training and analysis. All records included in the registry are de-identified prior to use, and no personally identifiable patient information is stored or made publicly accessible under any circumstances. Row Level Security on the database ensures that even if the Supabase public key is extracted from the frontend, no sensitive data can be queried directly. The upload approval workflow ensures that an admin reviews all data before it enters the registry, providing a human check against erroneous or inappropriate submissions.

## Glossary of Terms {#glossary-of-terms}

- **API (Application Programming Interface)** — A set of rules that allows different software components to communicate with each other.
- **Anon ID** — An anonymized identifier assigned to each patient record, replacing any personally identifiable information.
- **Angular** — A development platform, built on TypeScript, for building scalable web applications.
- **Asynchronous Processing** — A technique where long running tasks are executed separately so the main application remains responsive.
- **BERT (Bidirectional Encoder Representations from Transformers)** — A machine learning model used for understanding and extracting information from natural language text.
- **CalEnviroScreen 4.0** — A screening tool developed by California OEHHA that ranks communities based on pollution exposure and population vulnerability, using 25 environmental and socioeconomic indicators.
- **Case Diagnosis** — A single predicted cancer classification for a patient, including the predicted term, ICD-O code, and confidence score.
- **Docker Compose** — A tool for defining and running multi-container Docker applications using a YAML configuration file.
- **Elasticsearch** — A distributed search and analytics engine optimized for speed and relevance on production-scale workloads.
- **FastAPI** — A Python-based web framework used to create backend APIs, well suited for data-driven applications.
- **GeoJSON** — An open standard format for encoding geographic data structures using JSON.
- **GeoPandas** — An open-source Python library that extends pandas data structures for working with geospatial vector data.
- **GitHub Projects** — An adaptable collection of items that you can view as a table, a kanban board, or a roadmap and that stays up-to-date with GitHub data.
- **GraphQL** — An open-source query language and server-side runtime that specifies how clients should interact with APIs.
- **ICD-O (International Classification of Diseases for Oncology)** — A standardized coding system for classifying tumors by topography (site) and morphology (type).
- **Ingestion Job** — A tracked upload that moves through a lifecycle: pending_review → processing → completed/failed/rejected.
- **JWT (JSON Web Token)** — A compact, URL-safe token format used for securely transmitting authentication claims between parties.
- **Materialized View** — A database object that stores the result of a query physically, allowing fast reads of pre-computed aggregations.
- **MongoDB** — An open source, nonrelational database management system that uses flexible documents instead of tables and rows.
- **NLP (Natural Language Processing)** — A field of artificial intelligence focused on extracting meaning from human language text.
- **NLP Worker** — A separate processing component (microservice) responsible for running NLP models asynchronously without blocking user requests.
- **Node.js** — An open-source, cross-platform JavaScript runtime environment.
- **PetBERT** — A fine-tuned BERT model trained on veterinary clinical text for cancer classification, developed by SAVSNET.
- **PostGIS** — A PostgreSQL extension that adds support for geographic objects and spatial queries.
- **PostgreSQL** — An open-source relational database used to store structured application data.
- **React** — A JavaScript library used to build interactive web user interfaces.
- **REST (Representational State Transfer)** — A common architectural style for web APIs that uses standard HTTP methods to exchange data between a client and server.
- **RLS (Row Level Security)** — A PostgreSQL feature that restricts which rows a given database role can access, providing fine-grained access control.
- **Supabase** — An open-source Firebase alternative providing managed PostgreSQL hosting, authentication, storage, and real-time subscriptions.
- **Trello** — A web-based, kanban-style, list-making application for team project management.
- **Vite** — A fast build tool and development server for modern web applications.

## References {#references}

[Take C.H.A.R.G.E](https://takechargeregistry.com/)
[VMDB](https://vmdl.missouri.edu/)
[SAVSNET](https://www.liverpool.ac.uk/savsnet/)
[Swiss Canine and Feline Cancer Registry](https://www.zora.uzh.ch/entities/publication/cdca5d34-1aad-4a6f-a73a-53cf232e53a8)
[ACARCinom](https://veterinary-science.uq.edu.au/australian-companion-animal-registry-cancers)
[UCSF Catchment Area Dashboard](https://cancer.ucsf.edu/catchment-area-dashboard)
[CalEnviroScreen 4.0](https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40)
[California Cancer Registry](https://www.californiahealthmaps.org/)
[State Cancer Profiles](https://statecancerprofiles.cancer.gov/)
[PetBERT (SAVSNET)](https://github.com/SAVSNET/PetBERT)
[Supabase Documentation](https://supabase.com/docs)
[FastAPI Documentation](https://fastapi.tiangolo.com/)
[React Documentation](https://react.dev/)
[PostGIS Documentation](https://postgis.net/documentation/)
[IEEE 830-1998 — Recommended Practice for Software Requirements Specifications](https://standards.ieee.org/ieee/830/1222/)
[PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
[Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)
