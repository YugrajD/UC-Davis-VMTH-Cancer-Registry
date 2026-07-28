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
- **Database**: PostgreSQL 16 + PostGIS 3.4 (local Docker container for dev; migrating to RDS for PostgreSQL in production — see `docs/aws-migration-plan.md`)
- **ML/NLP**: PetBERT (110M-param BERT pretrained on veterinary EHR data) with Vet-ICD-O-canine-1 classification
- **ML inference**: GCP Batch (current production) or local `ml-worker` container (development) — migrating to AWS Batch
- **Auth**: Amazon Cognito — self-hosted `cognito-local` emulator for dev, a real Cognito User Pool in production — email/password with confirmation codes, Google OAuth (Hosted UI, prod only), JWT RS256
- **Frontend hosting**: Vercel (current), migrating to AWS Amplify Hosting
- **Backend hosting**: GCP Cloud Run (current), migrating to AWS App Runner
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

Everything below runs entirely on your machine — Postgres/PostGIS and a self-hosted Cognito-compatible auth emulator ([cognito-local](https://github.com/jagregory/cognito-local)) are provisioned as Docker Compose services alongside the app. No AWS account or third-party account is required for local development.

### 1. Clone the Repository

```bash
git clone https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry.git
cd UC-Davis-VMTH-Cancer-Registry
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

The defaults in `.env.example` already point at the local Postgres and cognito-local containers defined in `docker-compose.yml` — no edits are required to get running. You only need to change `ADMIN_EMAILS` (and optionally `UPLOADER_EMAILS`/`REVIEWER_EMAILS`) to match the accounts you'll create in step 5.

#### Role allow-lists (env-var bootstrap)

Three comma-separated env vars seed the `user_roles` table on startup. They're only used for first-boot bootstrapping; ongoing role management happens through the **User Management** tab (admins) which writes to the DB directly.

```
ADMIN_EMAILS=alice@example.com,bob@example.com
UPLOADER_EMAILS=charlie@example.com
REVIEWER_EMAILS=dana@example.com
```

Admins implicitly hold uploader and reviewer privileges, so `UPLOADER_EMAILS` / `REVIEWER_EMAILS` only need to list users who don't also appear in `ADMIN_EMAILS`. Emails must exactly match the accounts registered in Auth (see step 5).

**Never commit `.env` to git.**

### 3. Start Postgres and Run Migrations

```bash
docker compose up -d postgres
docker compose run --rm migrate
```

This starts the local PostGIS-enabled Postgres container and applies every file in `database/migrations/` in order. The `migrate` service is safe to re-run any time — migrations use `IF NOT EXISTS` guards throughout.

### 4. Start the Auth Server

```bash
docker compose up -d cognito-local
```

This runs [cognito-local](https://github.com/jagregory/cognito-local), a self-hosted Cognito emulator, at `http://localhost:9229`. A fixed User Pool (`local_vmthdev`) and App Client (`vmthcancerregistryweb`) are pre-seeded from `database/docker/cognito-local/seed/` on first run, so the IDs in `.env.example` always match — no manual pool creation needed.

### 5. Create User Accounts

With the auth server running, use the AWS CLI against the local endpoint to sign up and confirm an admin account (swap in your own email/password):

```bash
AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local aws --endpoint http://localhost:9229 --region us-east-1 \
  cognito-idp sign-up --client-id vmthcancerregistryweb \
  --username admin@example.com --password 'ChangeMe123!' \
  --user-attributes Name=email,Value=admin@example.com
```

cognito-local auto-generates a confirmation code and stores it in plaintext in its local state file — read it and confirm the account:

```bash
docker exec vmth_cancer_cognito_local cat /app/.cognito/db/local_vmthdev.json \
  | python3 -c "import json,sys; print(list(json.load(sys.stdin)['Users'].values())[0]['ConfirmationCode'])"

AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local aws --endpoint http://localhost:9229 --region us-east-1 \
  cognito-idp confirm-sign-up --client-id vmthcancerregistryweb \
  --username admin@example.com --confirmation-code <code-from-above>
```

Or just sign up through the app's UI (Sign Up → enter the code shown in the container's state file, same as above).

Alternatively, skip the confirmation code entirely with `admin-confirm-sign-up` (also handy if you fat-fingered a code and want to unstick an account without deleting/recreating it):

```bash
AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local aws --endpoint http://localhost:9229 --region us-east-1 \
  cognito-idp admin-confirm-sign-up --user-pool-id local_vmthdev --username admin@example.com
```

Make sure this email is also listed in `ADMIN_EMAILS` in your `.env`.

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

This starts the app services (postgres and cognito-local from steps 3–4 keep running alongside them):

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

### Browsing the local database

Connect any Postgres client (e.g. `psql`, TablePlus, DBeaver) to `localhost:5432` using the credentials in `.env` (`postgres` / `POSTGRES_PASSWORD`). In production, RDS Query Editor or pgAdmin serves the same purpose (see `docs/aws-migration-plan.md`).

---

## Troubleshooting

### "API error 500" in the browser

Check the backend logs:
```bash
docker compose logs backend --tail 30
```

### Backend can't connect to the database

- Verify `DATABASE_URL` in `.env` uses `+asyncpg` (e.g. `postgresql+asyncpg://...`)
- Confirm the `postgres` container is healthy: `docker compose ps postgres`
- Make sure migrations have been applied: `docker compose run --rm migrate`

### "Invalid token" errors when signing in

- Confirm `cognito-local` is running: `docker compose ps cognito-local`
- Verify `COGNITO_USER_POOL_ID` and `COGNITO_CLIENT_ID` match between the `backend` and `frontend` services in `.env` (both default to `local_vmthdev` / `vmthcancerregistryweb`, matching the seeded pool)
- Check backend logs: `docker compose logs backend --tail 30`

### Review Queue tab doesn't appear after signing in

- Confirm the email you signed in with is listed in `ADMIN_EMAILS` in `.env`
- `ADMIN_EMAILS` is case-sensitive — the email must match exactly
- Restart the backend after changing `.env`: `docker compose restart backend`

### Frontend shows a blank page

- Check the browser console (F12 → Console) for errors
- Verify `VITE_COGNITO_USER_POOL_ID` and `VITE_COGNITO_CLIENT_ID` are set in `.env`
- Restart the frontend container: `docker compose restart frontend`

### Upload rate limit (429 error)

Uploads share the global rate limit defined by `RATE_LIMIT_WRITE` in `backend/app/config.py` (default 10/minute per IP). Authenticated users see `RATE_LIMIT_DEFAULT` (120/minute) on other endpoints; anonymous IPs get `RATE_LIMIT_ANONYMOUS` (30/minute). All values are env-tunable.

### Sign-up/reset confirmation code never arrives

cognito-local has no SMTP configured — it doesn't send real emails. The confirmation code is written in plaintext to its local state file instead:

```bash
docker exec vmth_cancer_cognito_local cat /app/.cognito/db/local_vmthdev.json \
  | python3 -c "import json,sys; print(list(json.load(sys.stdin)['Users'].values())[0]['ConfirmationCode'])"
```

In production, real Cognito sends the code by email (Cognito's default email sending, or a configured SES identity).

Or skip the code entirely with `admin-confirm-sign-up` — see [Step 5](#5-create-user-accounts).

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
