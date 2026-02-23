# Developer Setup Guide

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

Copy the example and fill in the Supabase credentials:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder values with the actual Supabase connection strings. Ask a team member for the credentials — **never commit this file to git**.

Your `.env` should look like:

```
DATABASE_URL=postgresql+asyncpg://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-us-west-2.pooler.supabase.com:5432/postgres
DATABASE_URL_SYNC=postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-us-west-2.pooler.supabase.com:5432/postgres
CORS_ORIGINS=["http://localhost:5173"]
```

## 3. Start the Application

```bash
docker compose up --build
```

This starts two containers:

| Service | URL | Description |
|---|---|---|
| **backend** | http://localhost:8000 | FastAPI server |
| **frontend** | http://localhost:5173 | React dashboard |

The backend connects to the shared Supabase database — no local database setup is needed.

## 4. Verify It Works

Open http://localhost:5173 in your browser. You should see the cancer registry dashboard with data loaded from Supabase.

## Running Without Docker (Frontend Only)

If you want to run the frontend locally for faster hot-reload:

```bash
# Start only the backend in Docker
docker compose up backend

# In a separate terminal, run the frontend locally
cd frontend
npm install --legacy-peer-deps
npx vite
```

The frontend Vite config automatically proxies API requests to `http://localhost:8000`.

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
