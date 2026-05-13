# UC Davis VMTH Cancer Registry

A full-stack veterinary cancer catchment area dashboard modeled after the UCSF Helen Diller Comprehensive Cancer Center dashboard, adapted for UC Davis VMTH pet cancer data.

## Tech Stack

- **Frontend**: React 18 + TypeScript (Vite), Tailwind CSS, react-simple-maps, d3-scale
- **Backend**: Python 3.12 + FastAPI, SQLAlchemy + GeoAlchemy2
- **Database**: PostgreSQL 16 + PostGIS 3.4
- **ML/NLP**: PetBERT cancer classification pipeline
- **Auth**: Supabase Auth (JWT)
- **Orchestration**: Docker Compose

## Features

| Tab | Description |
|-----|-------------|
| **Overview** | Summary stats cards, top-level metrics, species breakdown |
| **Map** | Interactive choropleth of Northern CA catchment area (16 counties) |
| **Incidence** | Cancer incidence rates by type with bar charts |
| **Trends** | Time series with multi-line charts |
| **Species & Breed** | Pie charts and horizontal bar charts for demographic breakdowns |
| **County Data** | County-level stacked bar comparisons and data tables |
| **Data Upload** | CSV upload with admin approval workflow |
| **Review Queue** | Admin-only queue to preview, approve, or reject uploads |

## API Endpoints

- `GET /api/v1/dashboard/summary` — Dashboard summary stats
- `GET /api/v1/dashboard/filters` — Available filter options
- `GET /api/v1/incidence` — Cancer incidence with filters
- `GET /api/v1/incidence/by-cancer-type` — Grouped by cancer type
- `GET /api/v1/incidence/by-species` — Grouped by species
- `GET /api/v1/incidence/by-breed` — Grouped by breed
- `GET /api/v1/geo/counties` — GeoJSON FeatureCollection with case counts
- `GET /api/v1/trends/yearly` — Yearly case trends
- `GET /api/v1/trends/by-cancer-type` — Trends by cancer type
- `POST /api/v1/ingest/upload` — Upload CSV datasets for review
- `GET /api/v1/ingest/jobs` — List ingestion jobs
- `POST /api/v1/ingest/jobs/{id}/review` — Approve or reject a job (admin)
- `GET /api/v1/auth/me` — Current user info

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

#### Admin Emails

Set `ADMIN_EMAILS` to a comma-separated list of email addresses that should have admin access:

```
ADMIN_EMAILS=alice@example.com,bob@example.com
```

These emails must exactly match the accounts registered in Supabase Auth (see step 5).

**Never commit `.env` to git.**

### 4. Run Database Migrations

The migration SQL files are in `database/migrations/` and must be run in order (001 through 011). Run them through the Supabase SQL Editor:

1. Supabase Dashboard → **SQL Editor**
2. Open each migration file locally, copy its contents, paste into the SQL Editor, and click **Run**
3. Run them in numeric order: `001_extensions.sql`, `002_lookup_tables.sql`, ..., `011_ingestion_jobs.sql`

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

Admin users can:
- Access the **Review Queue** tab
- Preview, approve, or reject uploads from any user
- Upload without rate limits

#### Create regular user accounts (optional)

Same process as above, but don't add the email to `ADMIN_EMAILS`. Regular users can:
- Upload CSV files (limited to 3 per day)
- View their own upload history

Anonymous uploads (without signing in) are also allowed.

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
2. Check the API docs at http://localhost:8000/docs
3. Click **Sign In** in the top-right corner and log in with your admin account
4. You should see the **Review Queue** tab appear in the navigation
5. Go to **Data Upload**, select two CSV files, and click **Submit for Review**
6. Switch to **Review Queue** to see the pending upload with approve/reject options

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

Non-admin users are limited to 3 uploads per day. Admin accounts have no limit. To increase or remove the limit, modify the check in `backend/app/routers/ingest.py`.

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
