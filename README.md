# UC Davis VMTH Cancer Registry

A full-stack veterinary cancer registry for UC Davis VMTH researchers. Pathology reports are uploaded, classified by a BERT-based NLP model against the Vet-ICD-O-canine-1 taxonomy, and visualized on an interactive California county choropleth alongside human-cancer and environmental-exposure data.

> **Documentation**
> - `docs/handoff/HANDOFF.md` — project handoff guide (architecture, what's implemented, remaining work)
> - `docs/handoff/future_plans.md` — scaling plan and deferred-feature roadmap
> - `docs/current-architecture.md` — security layers, data flow, API endpoint summary
> - `docs/GCP_BATCH_SETUP.md` — GCP Batch ML pipeline setup
> - `docs/DATA_PIPELINE.md` — ingestion + PetBERT classification details

## Tech Stack

- **Frontend**: React 19 + TypeScript (Vite), Tailwind CSS v4, deck.gl, d3-scale, react-simple-maps
- **Backend**: Python 3.11 + FastAPI, SQLAlchemy async, Pydantic v2, gunicorn + uvicorn workers
- **Database**: PostgreSQL 16 + PostGIS 3.4 (hosted on Supabase)
- **ML/NLP**: PetBERT (110M-param BERT pretrained on veterinary EHR data) with Vet-ICD-O-canine-1 classification
- **ML inference**: GCP Batch (production) or local `ml-worker` container (development)
- **Auth**: Supabase Auth — email/password, Google OAuth, PKCE-flow password reset (JWT HS256/ES256)
- **Frontend hosting**: Vercel
- **Backend hosting**: GCP Cloud Run
- **CI/CD**: GitHub Actions (412 tests: 117 backend pytest + 295 frontend vitest)
- **Local orchestration**: Docker Compose

## Features

| Tab | Description | Access |
|-----|-------------|--------|
| **Overview** | Summary stats, top-level metrics, species/breed breakdown, county choropleth | Public |
| **Cancer Types** | Vet-ICD-O-canine-1 cancer-type breakdown filtered by category (Non-Cancer excluded) | Public |
| **Cancer by Age** | Cancer case distribution by age group with sex and breed filters | Public |
| **Breed Disparities** | Breed-level case counts and demographic comparisons (Non-Cancer excluded) | Public |
| **Analysis** | Multi-map comparison (VMTH vs CalEnviroScreen vs human cancer vs pesticides), correlation scatter plot, yearly cancer trend chart, pesticide trend chart (real API data) | Public |
| **Data Upload** | CSV/XLSX upload with file-content validation, rate limiting, and friendly column display names | Uploader / Admin |
| **Review Queue** | Admin-only queue to preview, approve, or reject ingestion jobs | Reviewer / Admin |
| **Diagnosis Review** | Per-diagnosis review queue with cancer group filter, source-text panel, and audit log | Reviewer / Admin |
| **User Management** | DB-backed user role assignment + role-request approval queue | Admin |
| **Data Export** | Filtered CSV download with admin-approval workflow (one-time-use approvals) | Authenticated + approved |

## API Endpoints

11 routers expose the surface below. Full details in `docs/current-architecture.md`.

```
auth:             GET  /api/v1/auth/me
dashboard:        GET  /api/v1/dashboard/{summary,filters}
incidence:        GET  /api/v1/incidence
                  GET  /api/v1/incidence/{by-cancer-type,by-species,by-breed,breed-detail}
geo:              GET  /api/v1/geo/counties
                  GET  /api/v1/geo/counties/{county_id}
trends:           GET  /api/v1/trends/{yearly,by-cancer-type}
search:           POST /api/v1/search/classify              (auth required)
                  GET  /api/v1/search/reports               (auth required)
ingest:           POST /api/v1/ingest/upload                (auth required)
                  GET  /api/v1/ingest/{status,jobs}
                  GET  /api/v1/ingest/jobs/{id}/preview     (reviewer)
                  POST /api/v1/ingest/jobs/{id}/{review,cancel}  (reviewer)
diagnoses:        GET  /api/v1/diagnoses/pending            (reviewer)
                  GET  /api/v1/diagnoses/{id}               (reviewer)
                  POST /api/v1/diagnoses/{id}/review        (reviewer)
admin-users:      GET  /api/v1/admin/users/{email}/roles    (admin)
                  PUT  /api/v1/admin/users/{email}/roles    (admin)
                  GET  /api/v1/admin/users/roles            (admin)
admin:            POST /api/v1/admin/refresh-views          (admin)
role-requests:    POST /api/v1/role-requests/               (auth required)
                  GET  /api/v1/role-requests/               (auth required)
                  POST /api/v1/role-requests/{id}/resolve   (admin)
export-requests:  POST /api/v1/export-requests/             (auth required)
                  POST /api/v1/export-requests/{id}/resolve (admin)
                  GET  /api/v1/export-requests/download     (approved user)
health:           GET  /health
```

In production (`DEBUG=false`), `/docs`, `/redoc`, and `/openapi.json` return 404 to avoid leaking the API surface. Set `DEBUG=true` locally to expose them.

---

## Setup

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v4.0+)
- [Git](https://git-scm.com/)
- Access to the team's Supabase project

> **Need access?** Contact **dtestrella@ucdavis.edu** to be invited to the Supabase project.

### 1. Clone the Repository

```bash
git clone https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry.git
cd UC-Davis-VMTH-Cancer-Registry
```

### 2. Access the Supabase Project

The Supabase project is already set up. To find the credentials you'll need for the next step:

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard) and sign in with the account that was invited to the project
2. Select the **UC Davis VMTH Cancer Registry** project from the project list
3. You'll use the Supabase dashboard in the next step to locate API keys and database connection strings

If you don't see the project in your dashboard, you haven't been invited yet — contact **dtestrella@ucdavis.edu** for access.

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in each value. Here's where to find them:

#### Database URLs

1. Supabase Dashboard → **Settings** → **Database** → **Connection string** → **URI**
2. Copy the connection string and replace `[YOUR-PASSWORD]` with your database password and `[PROJECT-REF]` with the project reference shown in the URI
3. For `DATABASE_URL`, add `+asyncpg` after `postgresql` (e.g. `postgresql+asyncpg://...`)
4. For `DATABASE_URL_SYNC`, use the URI as-is with `postgresql://...`

#### Supabase Auth Keys

All found at: Supabase Dashboard → **Settings** → **API**

| `.env` variable | Where to find it |
|---|---|
| `SUPABASE_JWT_SECRET` | JWT Settings → **JWT Secret**; leave blank for newer ES256/JWKS projects |
| `SUPABASE_URL` | Project URL (e.g. `https://abcdef.supabase.co`) |
| `VITE_SUPABASE_URL` | Same as `SUPABASE_URL` |
| `VITE_SUPABASE_ANON_KEY` | Project API Keys → **anon** / **public** / **publishable** |
| `VITE_API_URL` | Deployed FastAPI backend base URL for static frontend deployments; leave blank for local Vite proxy |

#### Role allow-lists (env-var bootstrap)

Three comma-separated env vars seed the `user_roles` table on startup. They're only used for first-boot bootstrapping; ongoing role management happens through the **User Management** tab (admins) which writes to the DB directly.

```
ADMIN_EMAILS=alice@example.com,bob@example.com
UPLOADER_EMAILS=charlie@example.com
REVIEWER_EMAILS=dana@example.com
```

Admins implicitly hold uploader and reviewer privileges, so `UPLOADER_EMAILS` / `REVIEWER_EMAILS` only need to list users who don't also appear in `ADMIN_EMAILS`. Emails must exactly match the accounts registered in Supabase Auth (see step 5).

**Never commit `.env` to git.**

### 4. Run Database Migrations

The migration SQL files are in `database/migrations/` and must be run in numeric order (currently 001 through 029). Run them through the Supabase SQL Editor:

1. Supabase Dashboard → **SQL Editor**
2. Open each migration file locally, copy its contents, paste into the SQL Editor, and click **Run**
3. Run them in numeric order: `001_extensions.sql`, `002_lookup_tables.sql`, …, `029_add_age_group_to_mvs.sql`

Alternatively, you can run them from the command line using `psql`:

```bash
# Install psql if you don't have it (macOS)
brew install libpq && brew link --force libpq

# Run all migrations in order
for f in database/migrations/0*.sql; do
  echo "Running $f..."
  psql "$DATABASE_URL_SYNC" -f "$f"
done
```

Replace `$DATABASE_URL_SYNC` with your actual sync connection string (or `source .env` first).

### 5. Create User Accounts in Supabase Auth

#### Create an admin account

1. Supabase Dashboard → **Authentication** → **Users**
2. Click **Add User** → **Create new user**
3. Enter the email and a password
4. Make sure this email is listed in `ADMIN_EMAILS` in your `.env`

Roles (most-privileged at the top — each implies the ones below it):

| Role | Permissions |
|---|---|
| **Admin** | Everything below + user-role management, refresh materialized views, resolve role/export requests |
| **Reviewer** | Approve/reject ingestion jobs, work the Diagnosis Review queue |
| **Uploader** | Submit CSV/XLSX uploads via Data Upload |
| **Authenticated** | View dashboards, request an export, request a role upgrade |
| **Anonymous** | View public dashboards only (rate-limited at 30 req/min/IP) |

Anonymous *uploads* are not allowed. All write endpoints require auth.

### 6. Start the Application

```bash
docker compose up --build
```

Or use the helper script:

```bash
./start.sh
```

This starts three services:

| Service | URL | Description |
|---|---|---|
| **backend** | http://localhost:8000 | FastAPI API server |
| **frontend** | http://localhost:5173 | React dashboard |
| **ml-worker** | http://localhost:8001 | PetBERT ML classification service |

Wait until you see log output from all three containers before opening the browser.

### 7. Load Data

#### Option A: Mock data (for development/testing)

Generates ~5,000 synthetic cancer cases:

```bash
docker compose --profile seed run seed
```

#### Option B: Real PetBERT data

Place your data files in `database/data/`:
- `petbert_scan_predictions.csv` — PetBERT classification output
- `All_deidentified_K9.xlsx` — Dog visit demographics

Then run:

```bash
docker compose --profile ingest run ingest
```

#### Load county boundaries (required for map visualizations)

```bash
docker compose --profile geo-seed run geo-seed
```

This loads all 58 California county boundaries into PostGIS.

### 8. Verify Everything Works

1. Open http://localhost:5173 — you should see the dashboard
2. Check the API docs at http://localhost:8000/docs (only available when `DEBUG=true`; 404 in production)
3. Click **Sign In** in the top-right corner and log in with your admin account
4. You should see **Review Queue**, **Diagnosis Review**, and **User Management** tabs appear in the navigation
5. Go to **Data Upload**, select a CSV file, and click **Submit for Review**
6. Switch to **Review Queue** to see the pending upload with approve/reject options
7. After approval, switch to **Diagnosis Review** to see PetBERT predictions with their source-text context

---

## Development

### Running Without Docker (Frontend Only)

For faster hot-reload during frontend development:

```bash
# Start only the backend + ML worker in Docker
docker compose up backend ml-worker

# In a separate terminal
cd frontend
npm install --legacy-peer-deps
npx vite
```

The Vite dev server proxies API requests to `http://localhost:8000` automatically.

### Supabase Table Editor

Non-technical team members can view and edit data directly through the Supabase web UI:

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard)
2. Sign in with the account that was invited to the project
3. Select the project
4. Use the **Table Editor** in the left sidebar to browse tables

---

## Troubleshooting

### "API error 500" in the browser

Check the backend logs:
```bash
docker compose logs backend --tail 30
```

### Backend can't connect to the database

- Verify `DATABASE_URL` in `.env` uses `+asyncpg` (e.g. `postgresql+asyncpg://...`)
- Check that your database password has no special characters that need URL-encoding
- Confirm the project reference and region match your Supabase project
- Make sure the pooler hostname matches your project's region

### "Invalid token" errors when signing in

- Newer Supabase projects use **ES256** tokens (not HS256). The backend auto-detects the algorithm, but `SUPABASE_URL` must be set correctly so it can fetch the JWKS public keys
- Verify `SUPABASE_URL` matches the **Project URL** in Supabase Dashboard → Settings → API
- Check backend logs: `docker compose logs backend --tail 30`

### Review Queue tab doesn't appear after signing in

- Confirm the email you signed in with is listed in `ADMIN_EMAILS` in `.env`
- `ADMIN_EMAILS` is case-sensitive — the email must match exactly
- Restart the backend after changing `.env`: `docker compose restart backend`

### Frontend shows a blank page

- Check the browser console (F12 → Console) for errors
- If you see a `createClient` error, verify `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` are set in `.env`
- Restart the frontend container: `docker compose restart frontend`

### Upload rate limit (429 error)

Uploads share the global rate limit defined by `RATE_LIMIT_WRITE` in `backend/app/config.py` (default 10/minute per IP). Authenticated users see `RATE_LIMIT_DEFAULT` (120/minute) on other endpoints; anonymous IPs get `RATE_LIMIT_ANONYMOUS` (30/minute). All values are env-tunable.

### Password reset email arrives but link says "expired"

This is almost always an email-link pre-fetch issue (Gmail / Outlook scanners follow the URL on receipt). The app is configured for Supabase's PKCE flow which is resistant to this, but the Supabase **email template** must also be set to use `{{ .TokenHash }}` not the default `{{ .ConfirmationURL }}`. See `docs/handoff/HANDOFF.md` and the password-reset template snippet in the project's Supabase dashboard.

### Docker build fails

```bash
docker compose down
docker compose up --build --force-recreate
```

If that doesn't work, try clearing Docker's cache:

```bash
docker system prune -f
docker compose up --build
```

---

## Quick Reference

| Task | Command |
|---|---|
| Start all services | `docker compose up --build` |
| Stop all services | `docker compose down` |
| Load mock data | `docker compose --profile seed run seed` |
| Load real PetBERT data | `docker compose --profile ingest run ingest` |
| Load county boundaries | `docker compose --profile geo-seed run geo-seed` |
| View backend logs | `docker compose logs backend --tail 50` |
| View frontend logs | `docker compose logs frontend --tail 50` |
| Restart a service | `docker compose restart backend` |
| Open API docs | http://localhost:8000/docs |
| Open dashboard | http://localhost:5173 |
