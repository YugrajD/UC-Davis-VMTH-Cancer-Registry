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
- **Database**: PostgreSQL 16 + PostGIS 3.4 in production (Supabase currently, migrating to RDS — see `docs/aws-migration-plan.md`); local dev runs PostgreSQL 18 + PostGIS via Floci's RDS emulation (version differs from prod for local-dev-tooling reasons — see `docs/floci-local-dev-migration.md`)
- **ML/NLP**: PetBERT (110M-param BERT pretrained on veterinary EHR data) with Vet-ICD-O-canine-1 classification
- **ML inference**: GCP Batch (current production) or local `ml-worker` container (development) — migrating to AWS Batch
- **Auth**: Amazon Cognito — self-hosted [Floci](https://github.com/floci/floci) AWS-emulator for dev, a real Cognito User Pool in production — email/password with confirmation codes, Google OAuth (Hosted UI, prod only), JWT RS256
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

Everything below runs entirely on your machine — Postgres/PostGIS (via a real RDS-emulated container) and Cognito are provisioned through [Floci](https://github.com/floci/floci), a self-hosted AWS API emulator, as Docker Compose services alongside the app. No AWS account or third-party account is required for local development. See `docs/floci-local-dev-migration.md` for how this is wired together and its current caveats.

### 1. Clone the Repository

```bash
git clone https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry.git
cd UC-Davis-VMTH-Cancer-Registry
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

The `DATABASE_URL`/`COGNITO_USER_POOL_ID`/`COGNITO_CLIENT_ID` defaults in `.env.example` are placeholders — Floci assigns the real RDS endpoint and Cognito pool/client IDs at provisioning time (step 3) and every service picks up the real values automatically, so no edits are needed there either. You only need to change `ADMIN_EMAILS` (and optionally `UPLOADER_EMAILS`/`REVIEWER_EMAILS`) to match the accounts you'll create in step 5.

#### Role allow-lists (env-var bootstrap)

Three comma-separated env vars seed the `user_roles` table on startup. They're only used for first-boot bootstrapping; ongoing role management happens through the **User Management** tab (admins) which writes to the DB directly.

```
ADMIN_EMAILS=alice@example.com,bob@example.com
UPLOADER_EMAILS=charlie@example.com
REVIEWER_EMAILS=dana@example.com
```

Admins implicitly hold uploader and reviewer privileges, so `UPLOADER_EMAILS` / `REVIEWER_EMAILS` only need to list users who don't also appear in `ADMIN_EMAILS`. Emails must exactly match the accounts registered in Auth (see step 5).

**Never commit `.env` to git.**

### 3. Start Floci and Provision RDS + Cognito

```bash
docker compose up -d floci
docker compose run --rm floci-init
docker compose run --rm floci-postgis-init
docker compose run --rm migrate
```

`floci` runs the AWS API emulator at `http://localhost:4566`. `floci-init` is a one-shot step that creates the RDS Postgres instance and a Cognito User Pool + App Client against it, then writes the real `DATABASE_URL`/`COGNITO_USER_POOL_ID`/`COGNITO_CLIENT_ID` to a shared volume every other service reads at startup (see `docs/floci-local-dev-migration.md`) — unlike the old fixed-ID cognito-local setup, these IDs are assigned by Floci and differ every time you re-provision (e.g. after `docker compose down -v`). `floci-postgis-init` installs the PostGIS extension package into the actual Postgres container Floci spun up (its default `postgres:16-alpine` image doesn't ship PostGIS). Both are safe to re-run. `migrate` then applies every file in `database/migrations/` in order and is also safe to re-run any time — migrations use `IF NOT EXISTS` guards throughout.

### 4. Start the Application

```bash
docker compose up --build backend frontend ml-worker
```

Or use the helper script, which runs steps 3–4 for you:

```bash
./start.sh
```

Wait until you see log output from all three containers before continuing.

### 5. Create User Accounts

Read the pool/client ID Floci assigned (from the shared volume, via any running container that mounts it):

```bash
docker compose exec backend cat /shared/floci-outputs.env
```

Use the `COGNITO_USER_POOL_ID` and `COGNITO_CLIENT_ID` values from that output in place of `<pool-id>` / `<client-id>` below. Sign up and confirm an admin account (swap in your own email/password):

```bash
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test aws --endpoint http://localhost:4566 --region us-east-1 \
  cognito-idp sign-up --client-id <client-id> \
  --username admin@example.com --password 'ChangeMe123!' \
  --user-attributes Name=email,Value=admin@example.com
```

Skip the confirmation-code step entirely with `admin-confirm-sign-up`:

```bash
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test aws --endpoint http://localhost:4566 --region us-east-1 \
  cognito-idp admin-confirm-sign-up --user-pool-id <pool-id> --username admin@example.com
```

If you signed up through the app's UI instead, `admin-confirm-sign-up` still works the same way — it confirms by username/pool regardless of how the user was created. Substitute the pool ID and the email you signed up with:

```bash
docker run --rm --network cancer_registry_default \
  -e AWS_ACCESS_KEY_ID=test -e AWS_SECRET_ACCESS_KEY=test -e AWS_DEFAULT_REGION=us-east-1 \
  amazon/aws-cli:2.17.62 --endpoint-url http://floci:4566 cognito-idp admin-confirm-sign-up \
  --user-pool-id <pool-id> --username admin@example.com
```

Floci has no SMTP configured, so confirmation codes are never actually generated or delivered anywhere (not even to a local file) — `admin-confirm-sign-up` is the only way to confirm a local signup.

Make sure this email is also listed in `ADMIN_EMAILS` in your `.env`, then restart the backend so it picks up the change: `docker compose restart backend`.

To delete a user and start over (e.g. you mistyped the email, or want to re-test the sign-up flow):

```bash
docker run --rm --network cancer_registry_default \
  -e AWS_ACCESS_KEY_ID=test -e AWS_SECRET_ACCESS_KEY=test -e AWS_DEFAULT_REGION=us-east-1 \
  amazon/aws-cli:2.17.62 --endpoint-url http://floci:4566 cognito-idp admin-delete-user \
  --user-pool-id <pool-id> --username admin@example.com
```

Use the `floci` network alias, not `vmth_cancer_floci` — the aws-cli's endpoint URL validation rejects hostnames with underscores.

Roles (most-privileged at the top — each implies the ones below it):

| Role | Permissions |
|---|---|
| **Admin** | Everything below + user-role management, refresh materialized views, resolve role/export requests |
| **Reviewer** | Approve/reject ingestion jobs, work the Diagnosis Review queue |
| **Uploader** | Submit CSV/XLSX uploads via Data Upload |
| **Authenticated** | View dashboards, request an export, request a role upgrade |
| **Anonymous** | View public dashboards only (rate-limited at 30 req/min/IP) |

Anonymous *uploads* are not allowed. All write endpoints require auth.

Once running (step 4), the app services are:

| Service | URL | Description |
|---|---|---|
| **backend** | http://localhost:8000 | FastAPI API server |
| **frontend** | http://localhost:5173 | React dashboard |
| **ml-worker** | http://localhost:8001 | PetBERT ML classification service |

### 6. Load Data

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

### 7. Verify Everything Works

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

# In a separate terminal — grab the Cognito IDs Floci assigned first
docker compose exec backend cat /shared/floci-outputs.env

cd frontend
VITE_COGNITO_USER_POOL_ID=<pool-id> VITE_COGNITO_CLIENT_ID=<client-id> \
  npm install --legacy-peer-deps && npx vite
```

The Vite dev server proxies API requests to `http://localhost:8000` automatically. Unlike the Docker Compose `frontend` service, running Vite directly doesn't source `floci-outputs.env` for you, so the Cognito IDs must be passed explicitly.

### Browsing the local database

Get the RDS-emulated endpoint Floci assigned:

```bash
docker compose exec backend cat /shared/floci-outputs.env
```

Connect any Postgres client (e.g. `psql`, TablePlus, DBeaver) to the `DATABASE_URL_SYNC` host/port from that output, using the `postgres`/`postgres` credentials. **Note:** whether that host/port is reachable from outside the Docker network (vs. only from other containers) hasn't been verified yet — see the open items in `docs/floci-local-dev-migration.md`. In production, RDS Query Editor or pgAdmin serves the same purpose (see `docs/aws-migration-plan.md`).

---

## Troubleshooting

### "API error 500" in the browser

Check the backend logs:
```bash
docker compose logs backend --tail 30
```

### Backend (or `migrate`) can't connect to the database — "connection refused"

Floci gets a new docker-network IP every time its container restarts, even if the underlying RDS instance/data persisted (e.g. after `docker compose down` + `up`, without `-v`). If you restarted `floci` without re-running `floci-init` afterward, the `DATABASE_URL` other services are using is stale. Fix: re-run provisioning to refresh it, then retry:

```bash
docker compose run --rm floci-init
docker compose restart backend
```

- Confirm `floci` is running: `docker compose ps floci`
- Check what `floci-init` actually wrote: `docker compose exec backend cat /shared/floci-outputs.env` (a missing/empty file means provisioning never ran — run `docker compose run --rm floci-init`)
- Make sure migrations have been applied: `docker compose run --rm migrate`

### "Network error" popup when signing in

Check the browser console — if you see a CORS error on a request to `:4566` or `/cognito`, verify `VITE_COGNITO_ENDPOINT` in `.env` is the relative path `/cognito`, not an absolute `http://localhost:4566` URL. Floci's Cognito emulation doesn't support CORS, so calling it directly cross-origin from the browser fails outright — it must go through Vite's dev-server proxy (`frontend/vite.config.ts`'s `/cognito` rule) to stay same-origin. If you set up `.env` before this project migrated to Floci, diff it against `.env.example` — a pre-existing `.env` value always wins over `docker-compose.yml`'s defaults, even after `docker compose up --force-recreate`, so this and similar Floci-related settings (`COGNITO_ISSUER_URL`, `DATABASE_URL`/`DATABASE_URL_SYNC`) can silently stay stale.

### "Invalid token" errors when signing in

- Confirm `floci` is running: `docker compose ps floci`
- Verify the `COGNITO_USER_POOL_ID`/`COGNITO_CLIENT_ID` the backend actually loaded match what you used to sign up: `docker compose exec backend cat /shared/floci-outputs.env` (these are assigned by Floci at provisioning time, not the static placeholders in `.env.example`)
- If the backend logs show a JWKS fetch DNS error (e.g. "Name or service not known"), your `.env`'s `COGNITO_ISSUER_URL` is likely stale — see above
- Check backend logs: `docker compose logs backend --tail 30`

### Review Queue tab doesn't appear after signing in

- Confirm the email you signed in with is listed in `ADMIN_EMAILS` in `.env`
- `ADMIN_EMAILS` is case-sensitive — the email must match exactly
- Restart the backend after changing `.env`: `docker compose restart backend`

### Frontend shows a blank page

- Check the browser console (F12 → Console) for errors
- Verify the frontend container actually sourced Floci's values: `docker compose logs frontend --tail 30` (it sources `/shared/floci-outputs.env` on startup — see `docker-compose.yml`)
- Restart the frontend container: `docker compose restart frontend`

### Upload rate limit (429 error)

Uploads share the global rate limit defined by `RATE_LIMIT_WRITE` in `backend/app/config.py` (default 10/minute per IP). Authenticated users see `RATE_LIMIT_DEFAULT` (120/minute) on other endpoints; anonymous IPs get `RATE_LIMIT_ANONYMOUS` (30/minute). All values are env-tunable.

### Sign-up/reset confirmation code never arrives

Floci's Cognito emulation has no SMTP configured — it never generates or delivers a real confirmation code, whether you signed up via the aws-cli or the app's UI. Use `admin-confirm-sign-up` to confirm the account directly instead — see [Step 5](#5-create-user-accounts). In production, real Cognito sends the code by email (Cognito's default email sending, or a configured SES identity).

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
| View Floci-assigned DB/Cognito values | `docker compose exec backend cat /shared/floci-outputs.env` |
| Load mock data | `docker compose --profile seed run seed` |
| Load real PetBERT data | `docker compose --profile ingest run ingest` |
| Load county boundaries | `docker compose --profile geo-seed run geo-seed` |
| View backend logs | `docker compose logs backend --tail 50` |
| View frontend logs | `docker compose logs frontend --tail 50` |
| Restart a service | `docker compose restart backend` |
| Open API docs | http://localhost:8000/docs |
| Open dashboard | http://localhost:5173 |
