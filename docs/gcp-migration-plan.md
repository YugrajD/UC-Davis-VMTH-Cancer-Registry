# GCP Migration Plan: Supabase + Vercel → 100% GCP

**Status:** Proposed
**Scope:** Move all remaining non-GCP resources — Supabase Postgres/PostGIS, Supabase Auth, and Vercel frontend hosting — onto GCP-native services. Cloud Run (backend), Cloud Batch + Artifact Registry (PetBERT ML), and GCS (uploads/reports) are already on GCP and are out of scope except where they need reconfiguration (env vars, CORS, IAM).

---

## Target end-state

| Component | Current | Target |
|---|---|---|
| Frontend hosting | Vercel | **Firebase Hosting** |
| Auth | Supabase Auth | **GCP Identity Platform** |
| Database | Supabase Postgres 16 + PostGIS 3.4 | **Cloud SQL for PostgreSQL 16 + PostGIS** |
| Backend | Cloud Run | Cloud Run (unchanged) |
| ML inference | Cloud Batch | Cloud Batch (unchanged) |
| File/model storage | GCS + Artifact Registry | GCS + Artifact Registry (unchanged) |
| Admin data browsing | Supabase Table Editor | Cloud SQL Studio / pgAdmin (new) |

Firebase Hosting is paired with Identity Platform deliberately — both live under the same Firebase/GCP project, which simplifies auth config reuse.

## Key design decision: RLS stays a no-op

`database/migrations/012_enable_rls.sql` enables Row Level Security on every table but defines **no permissive policies**. The backend connects as the Postgres superuser and bypasses RLS; all real authorization happens in `backend/app/auth.py` against the `user_roles` table. This migration preserves that pattern — RLS is re-applied as-is on Cloud SQL for defense-in-depth (protects against direct DB access with a leaked lower-privilege credential) but is **not** used to enforce authorization. No new RLS policy work is in scope here. (If/when the multi-tenant clinic-scoping work in `docs/handoff/future_plans.md` §4.2 happens, that would be the time to add real policies.)

## Cutover strategy: big-bang

All three replacements (DB, Auth, Frontend hosting) cut over together in a single maintenance window, rather than phased independently. This is faster overall but has less rollback granularity — mitigated by keeping Supabase and Vercel projects live-but-idle for a rollback window after cutover.

---

## Phase 0 — Prep (no production risk)

1. Provision target infra without touching production traffic:
   - **Cloud SQL for PostgreSQL 16** with the `postgis` extension enabled. Verify PostGIS 3.4 parity (or nearest supported minor version) before relying on it.
   - **GCP Identity Platform**: enable Email/Password + Google OAuth providers. Mirror Supabase's redirect URLs and the PKCE-safe password-reset email template (Supabase requires `{{ .TokenHash }}` instead of `{{ .ConfirmationURL }}` — carry the equivalent setting into the new provider).
   - **Firebase Hosting**: create the project (same Firebase project as Identity Platform), configure the custom domain and SSL cert provisioning ahead of DNS cutover.
2. Build the full cutover checklist by inventorying every touchpoint:
   - **Vercel**: no `vercel.json` in the repo — build settings, env vars, and domain are configured entirely via the Vercel dashboard. Nothing to port from-repo; must be manually replicated into Firebase Hosting config and CI.
   - **Supabase — env vars**: `DATABASE_URL`, `DATABASE_URL_SYNC`, `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
   - **Supabase — backend code**: `backend/app/auth.py` (JWKS client, HS256/ES256 detection, `audience="authenticated"` check)
   - **Supabase — frontend code**: `frontend/src/lib/supabase.ts`, `frontend/src/contexts/AuthContext.tsx`, `frontend/src/components/LoginModal/LoginModal.tsx` (and their `.test.tsx` files)
   - **Database migrations**: `database/migrations/*.sql` (029 files, must run in numeric order against Cloud SQL)
   - **Docker Compose**: `seed`, `ingest`, `geo-seed` profiles that run against `DATABASE_URL_SYNC`
   - **CI**: GitHub Actions workflows (pytest/vitest) — check whether they spin up a real Postgres service container or mock the Supabase client
   - **Non-technical workflow**: Supabase Table Editor is used by non-technical team members to browse/edit data directly — needs a replacement before Supabase is decommissioned
   - **CORS**: FastAPI CORS config must allow the new Firebase Hosting origin

## Phase 1 — Database migration (Supabase Postgres → Cloud SQL)

1. `pg_dump` the Supabase database (schema + data), including PostGIS geometry columns and materialized views (`mv_county_cancer_incidence`, `mv_yearly_trends`).
2. Restore into Cloud SQL. Re-run `012_enable_rls.sql` as-is (RLS enabled, no policies).
3. Validate: row counts match source, PostGIS geometry queries (`ST_*`) return correct results, materialized views refresh correctly via `POST /api/v1/admin/refresh-views`.
4. Point Cloud Run's `DATABASE_URL` (asyncpg) / `DATABASE_URL_SYNC` at Cloud SQL via the Cloud SQL Auth Proxy or a private IP + Serverless VPC Connector.
5. Update local Docker Compose dev setup — likely minimal change since local dev already runs `postgis/postgis:16-3.4` directly rather than hitting Supabase for the DB.

## Phase 2 — Auth migration (Supabase Auth → GCP Identity Platform)

1. **User migration**: export Supabase Auth users (email, password hash where portable, Google OAuth linkages). Bcrypt hashes generally aren't portable across auth providers — plan for either a bulk import via Identity Platform's `identitytoolkit` import API (if hash format is supported) or a forced one-time password-reset email for affected users post-cutover.
2. **Backend (`backend/app/auth.py`)**:
   - Replace the Supabase JWKS URL (`{SUPABASE_URL}/auth/v1/.well-known/jwks.json`) with Identity Platform's JWKS endpoint.
   - Identity Platform tokens are RS256, already within `_ALLOWED_ASYMMETRIC_ALGS`. Drop the HS256 code path once no legacy Supabase HS256 tokens remain in flight.
   - Update claim reads (`email`, `sub`) — Identity Platform/Firebase claim names may differ slightly; verify against actual issued tokens.
   - Update the `audience` check (currently hardcoded `"authenticated"`) to Identity Platform's expected audience (GCP project ID).
3. **Frontend**: replace `@supabase/supabase-js` in `lib/supabase.ts` with the Firebase Auth SDK; rewrite sign-in/sign-out/Google OAuth/password-reset calls in `AuthContext.tsx` and `LoginModal.tsx`. Re-verify the password-reset flow preserves the same email-prefetch-safety property the current PKCE flow provides (see `docs/handoff/HANDOFF.md` password-reset section).
4. `user_roles` table logic (`backend/app/models/models.py`, role checks in `auth.py`) is unaffected — it's keyed by email and independent of the JWT issuer.

## Phase 3 — Frontend hosting migration (Vercel → Firebase Hosting)

1. Add `firebase.json` with a catch-all SPA rewrite (`"source": "**"` → `/index.html`) since this is a client-routed Vite app.
2. Move frontend env vars into the Firebase Hosting build/CI pipeline: `VITE_API_URL`, plus renamed Firebase config vars replacing `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`.
3. Replace Vercel's GitHub integration with a GitHub Actions deploy step (`firebase deploy --only hosting`) authenticated via a GCP service account key or Workload Identity Federation.
4. Migrate the custom domain: lower DNS TTL ahead of cutover, repoint to Firebase Hosting, verify managed SSL cert issuance before flipping traffic.
5. Confirm FastAPI CORS config allows the new Firebase Hosting origin; remove the Vercel origin after cutover completes.

## Phase 4 — Cutover (single maintenance window)

1. Freeze writes (uploads, ingestion, role/export requests).
2. Run a final delta `pg_dump`/restore from Supabase → Cloud SQL to capture anything written since the Phase 1 snapshot.
3. Deploy simultaneously: backend to Cloud Run (new `DATABASE_URL`, Identity Platform config), frontend to Firebase Hosting, DNS flip.
4. Smoke test: sign-in (password + Google OAuth), `GET /api/v1/auth/me`, upload → review → diagnosis-review flow, choropleth map load (PostGIS-backed `geo` endpoints), export-request download.
5. Keep the Supabase project and Vercel project intact but idle for a rollback window (1–2 weeks) before decommissioning.

## Phase 5 — Decommission & cleanup

1. Stand up a replacement for the Supabase Table Editor workflow (Cloud SQL Studio or pgAdmin) for non-technical staff before removing Supabase access.
2. After the rollback window passes with no issues: delete the Supabase project, delete the Vercel project, confirm both are no longer billing.
3. Update documentation to remove Supabase/Vercel references and reflect the new stack: `README.md`, `docs/current-architecture.md`, `docs/handoff/HANDOFF.md`, `.env.example`.

---

## Open items requiring a decision before/during execution

- **PostGIS version parity** on Cloud SQL — confirm exact version match to avoid `ST_*` function behavior drift.
- **Password migration approach** — decide between forced mass password-reset vs. attempting hash migration; affects Phase 2 timeline and user communications.
- **CI service containers** — confirm whether GitHub Actions tests use a real ephemeral Postgres or a mocked Supabase client, and update fixtures accordingly.
