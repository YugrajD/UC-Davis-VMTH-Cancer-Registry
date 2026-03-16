# Developer Setup Guide

Quick-start guide for developers joining the project. For the full setup walkthrough — including Supabase project creation, database migrations, authentication, admin accounts, and data loading — see **[SETUP_GUIDE.md](./SETUP_GUIDE.md)**.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)
- Access to the team's Supabase project (ask the project owner for an invite)

## 1. Clone the Repository

```bash
git clone https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry.git
cd UC-Davis-VMTH-Cancer-Registry
```

## 2. Create the `.env` File

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder values. Each variable has a comment explaining where to find it in the Supabase dashboard. Ask a team member for credentials if needed — **never commit this file to git**.

At minimum you need the database URLs. For authentication and admin features you also need the Supabase auth keys and `ADMIN_EMAILS`. See [SETUP_GUIDE.md](./SETUP_GUIDE.md#3-configure-environment-variables) for details on every variable.

## 3. Start the Application

```bash
docker compose up --build
```

This starts three containers:

| Service | URL | Description |
|---|---|---|
| **backend** | http://localhost:8000 | FastAPI server |
| **frontend** | http://localhost:5173 | React dashboard |
| **ml-worker** | http://localhost:8001 | PetBERT ML classification service |

The backend connects to the shared Supabase database — no local database setup is needed.

## 4. Verify It Works

Open http://localhost:5173 in your browser. You should see the cancer registry dashboard with data loaded from Supabase.

## Running Without Docker (Frontend Only)

If you want to run the frontend locally for faster hot-reload:

```bash
# Start only the backend in Docker
docker compose up backend ml-worker

# In a separate terminal, run the frontend locally
cd frontend
npm install --legacy-peer-deps
npx vite
```

The frontend Vite config automatically proxies API requests to `http://localhost:8000`.

## Loading Data

```bash
# Mock data (~5,000 synthetic cases)
docker compose --profile seed run seed

# County boundaries (required for maps)
docker compose --profile geo-seed run geo-seed

# Real PetBERT data (requires files in database/data/)
docker compose --profile ingest run ingest
```

See [SETUP_GUIDE.md](./SETUP_GUIDE.md#7-load-data) for details on each option.

## Authentication & Admin Setup

The project uses Supabase Auth for login and an `ADMIN_EMAILS` env var to grant admin privileges. See [SETUP_GUIDE.md](./SETUP_GUIDE.md#5-create-user-accounts-in-supabase-auth) for how to create accounts and configure admin access.

## Supabase Table Editor

Non-technical team members can view and edit data directly through the Supabase web UI:

1. Go to https://supabase.com/dashboard
2. Sign in with the account that was invited to the project
3. Select the project
4. Use the **Table Editor** in the left sidebar to browse tables

## Troubleshooting

### "API error 500" in the browser
Check the backend logs:
```bash
docker compose logs backend --tail 30
```

### Backend can't connect to Supabase
- Verify your `.env` credentials are correct (no `[brackets]` around values)
- Make sure the `DATABASE_URL` uses the `+asyncpg` driver prefix
- Make sure the pooler hostname matches your project's region

### Frontend shows blank page
- Check that the backend is running: `docker compose ps`
- Check the browser console for errors (F12 > Console)

### Docker build fails on `node:22-alpine`
Restart Docker Desktop and retry:
```bash
docker compose up --build
```

For more troubleshooting scenarios see [SETUP_GUIDE.md](./SETUP_GUIDE.md#troubleshooting).
