# AWS Migration Plan: Supabase + Vercel + GCP → AWS

**Status:** Proposed
**Scope:** Campus IT does not provision GCP, so this supersedes `docs/gcp-migration-plan.md` (deleted). Everything currently on GCP or on third-party services (Supabase, Vercel) moves to AWS: backend compute, PetBERT batch inference, file storage, container images, Postgres/PostGIS, auth, and frontend hosting.

---

## Target end-state

| Component | Current | Target |
|---|---|---|
| Frontend hosting | Vercel | **S3 + CloudFront** |
| Auth | Supabase Auth | **Amazon Cognito** |
| Database | Supabase Postgres 16 + PostGIS 3.4 | **RDS for PostgreSQL 16 + PostGIS** |
| Backend compute | Cloud Run | **App Runner** |
| ML inference | GCP Batch + GCS | **AWS Batch + S3** |
| File/report storage | GCS | **S3** |
| Container images | Artifact Registry | **ECR** |
| Admin data browsing | Supabase Table Editor | pgAdmin / RDS Query Editor (new) |

App Runner is preferred over ECS Fargate for the backend: it's the closest analog to Cloud Run (fully managed, scale-to-zero-ish, deploy-from-image, no cluster/VPC networking to hand-manage for a single service). If autoscaling/concurrency limits or the sidecar-free constraint become a problem, fall back to ECS Fargate + ALB — flag this as an open decision in Phase 0.

## Key design decision: RLS stays a no-op

`database/migrations/012_enable_rls.sql` enables Row Level Security on every table but defines **no permissive policies**. The backend connects as the Postgres superuser and bypasses RLS; all real authorization happens in `backend/app/auth.py` against the `user_roles` table. This migration preserves that pattern — RLS is re-applied as-is on RDS for defense-in-depth (protects against direct DB access with a leaked lower-privilege credential) but is **not** used to enforce authorization. No new RLS policy work is in scope here.

## Cutover strategy: big-bang

All replacements (DB, Auth, Frontend hosting, backend compute, ML batch, storage) cut over together in a single maintenance window. Mitigated by keeping Supabase, Vercel, and GCP resources live-but-idle for a rollback window after cutover.

---

## Phase 0 — Prep (no production risk)

1. Provision target infra without touching production traffic:
   - **RDS for PostgreSQL 16** with the `postgis` extension enabled (RDS supports PostGIS as a managed extension via `CREATE EXTENSION postgis`). Verify PostGIS 3.4 parity against the RDS-supported version before relying on it.
   - **Amazon Cognito User Pool**: enable email/password + Google OAuth (as a federated IdP) sign-in. Mirror Supabase's redirect URLs. Cognito's hosted UI and password-reset flow differ from Supabase's PKCE `verifyOtp(token_hash)` approach — confirm the equivalent anti-prefetch property (Cognito's confirmation-code flow doesn't embed a clickable link by default, which may already avoid the email-scanner problem; verify before assuming parity).
   - **S3 buckets**: one for pathology report text / uploads (replaces GCS `uploads/`, `reports/`, `models/` prefixes), one for frontend static assets.
   - **CloudFront distribution** in front of the frontend S3 bucket, with the custom domain and ACM cert provisioned ahead of DNS cutover.
   - **ECR repository** for the backend image and the PetBERT batch image (replaces Artifact Registry).
   - **App Runner service** (or ECS Fargate — decide here) pointed at the ECR image, sized to match `backend/service.yaml`'s current resource limits (0.5–1 vCPU, 256–512Mi).
   - **AWS Batch compute environment + job queue + job definition** for PetBERT inference (replaces GCP Batch). Job definition mirrors the 3-runnable structure in `gcp_batch_service.py`: pull model/CSV from S3, run PetBERT container, push predictions back to S3 — but AWS Batch typically does this with one container plus S3-mounted volumes or explicit `aws s3 cp` steps in the entrypoint rather than GCP Batch's runnable list, so the ml-worker entrypoint script needs rework, not just a job-spec swap.
2. Build the full cutover checklist by inventorying every touchpoint:
   - **Vercel**: no `vercel.json` in the repo — build settings, env vars, and domain are configured entirely via the Vercel dashboard. Nothing to port from-repo; must be manually replicated into the S3/CloudFront + CI deploy pipeline.
   - **Supabase — env vars**: `DATABASE_URL`, `DATABASE_URL_SYNC`, `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
   - **Supabase — backend code**: `backend/app/auth.py` (JWKS client, HS256/ES256 detection, `audience="authenticated"` check)
   - **Supabase — frontend code**: `frontend/src/lib/supabase.ts`, `frontend/src/contexts/AuthContext.tsx`, `frontend/src/components/LoginModal/LoginModal.tsx` (and their `.test.tsx` files)
   - **GCP — backend code**: `backend/app/services/gcp_batch_service.py` (GCS + Batch client calls), `backend/app/services/ingestion_service.py:486-529` (uploads report text to GCS), `backend/app/services/job_processor.py:23-33,125-314` (Batch job orchestration/polling), `backend/app/routers/ingest.py:406-418` (lists GCS model folders), `backend/app/models/models.py:165` (`gcs_path` column — needs rename/repurpose to a generic storage path or `s3_key`)
   - **GCP — deploy config**: `backend/service.yaml` (Cloud Run spec), `backend/cloudbuild.yaml` (Cloud Build), `ml-worker/Dockerfile.batch` (Batch job image), `docs/GCP_BATCH_SETUP.md`
   - **GCP — packages**: `backend/requirements.txt` — `google-cloud-batch`, `google-cloud-storage` → replace with `boto3`
   - **Database migrations**: `database/migrations/*.sql` (029 files, must run in numeric order against RDS)
   - **Docker Compose**: `seed`, `ingest`, `geo-seed` profiles that run against `DATABASE_URL_SYNC`
   - **CI**: `.github/workflows/{ci.yml,pages.yml,update-npm-packages.yml}` — none currently deploy to GCP (deploys are manual via `gcloud builds submit`/`gcloud run services replace`), so this is a net-new CI deploy pipeline to build, not a migration of an existing one
   - **Non-technical workflow**: Supabase Table Editor is used by non-technical team members to browse/edit data directly — needs a replacement before Supabase is decommissioned
   - **CORS**: FastAPI CORS config must allow the new CloudFront origin

## Phase 1 — Database migration (Supabase Postgres → RDS)

1. `pg_dump` the Supabase database (schema + data), including PostGIS geometry columns and materialized views (`mv_county_cancer_incidence`, `mv_yearly_trends`).
2. Restore into RDS. Re-run `012_enable_rls.sql` as-is (RLS enabled, no policies).
3. Validate: row counts match source, PostGIS geometry queries (`ST_*`) return correct results, materialized views refresh correctly via `POST /api/v1/admin/refresh-views`.
4. Point the backend's `DATABASE_URL` (asyncpg) / `DATABASE_URL_SYNC` at RDS, either directly (RDS is reachable over VPC/public endpoint with security-group restriction) or via RDS Proxy if connection pooling under App Runner's concurrency needs it.
5. Update local Docker Compose dev setup — likely minimal change since local dev already runs `postgis/postgis:16-3.4` directly rather than hitting Supabase for the DB.

## Phase 2 — Auth migration (Supabase Auth → Cognito)

1. **User migration**: export Supabase Auth users (email, password hash where portable, Google OAuth linkages). Bcrypt hashes aren't directly importable into Cognito (Cognito doesn't expose a raw password-hash import API the way Identity Platform does) — plan for a forced one-time password-reset/verification email for affected users post-cutover, or a Cognito **migration-user Lambda trigger** that verifies the password against Supabase's auth API on first login and creates the Cognito user transparently (avoids a mass-reset email blast, keeps password continuity). Recommend the Lambda-trigger path given the mass-reset UX cost.
2. **Backend (`backend/app/auth.py`)**:
   - Replace the Supabase JWKS URL (`{SUPABASE_URL}/auth/v1/.well-known/jwks.json`) with Cognito's JWKS endpoint (`https://cognito-idp.{region}.amazonaws.com/{userPoolId}/.well-known/jwks.json`).
   - Cognito tokens are RS256, already within `_ALLOWED_ASYMMETRIC_ALGS`. Drop the HS256 code path once no legacy Supabase HS256 tokens remain in flight.
   - Update claim reads (`email`, `sub`) — Cognito's claim names largely match OIDC conventions but verify `email` is present (requires the `email` scope/attribute to be requested) and check `token_use`/`aud` (Cognito access tokens don't carry `aud`; ID tokens do — pick whichever token type the frontend forwards).
   - Update the `audience` check (currently hardcoded `"authenticated"`) to Cognito's App Client ID (for ID tokens) or drop it in favor of `client_id` validation (for access tokens).
3. **Frontend**: replace `@supabase/supabase-js` in `lib/supabase.ts` with `amazon-cognito-identity-js` or AWS Amplify Auth; rewrite sign-in/sign-out/Google OAuth/password-reset calls in `AuthContext.tsx` and `LoginModal.tsx`. Re-verify the password-reset flow preserves the same email-prefetch-safety property the current PKCE flow provides (see `docs/handoff/HANDOFF.md` password-reset section) — Cognito's default confirmation-code flow needs a manual check against this requirement.
4. `user_roles` table logic (`backend/app/models/models.py`, role checks in `auth.py`) is unaffected — it's keyed by email and independent of the JWT issuer.

## Phase 3 — Storage & ML batch migration (GCS + GCP Batch → S3 + AWS Batch)

1. **Storage (`backend/app/services/gcp_batch_service.py`)**: rewrite as an S3-backed service using `boto3`. Map GCS prefixes 1:1 — `uploads/{job_id}/` (CSV uploads), `reports/{job_id}/{anon_id}.txt` (pathology report text), `models/` (PetBERT model bundles). Rename `gcs_path` column on `pathology_reports` (`backend/app/models/models.py:165`) to a generic `storage_path` or `s3_key` and add a migration; update all readers/writers (`ingestion_service.py`, `job_processor.py`, `ingest.py`).
2. **Batch job image**: rebuild `ml-worker/Dockerfile.batch` for AWS Batch — model weights still fetched from S3 at runtime rather than baked into the image (~12 GB). AWS Batch job definitions don't support GCP Batch's multi-runnable (setup/main/upload) pattern natively; either use a single container whose entrypoint script does `aws s3 cp` before/after the PetBERT run, or split into a multi-container job definition if AWS Batch's version supports it — needs a decision during Phase 0 provisioning.
3. **Config (`backend/app/config.py:32-42`)**: replace `USE_GCP_BATCH`/`GCP_PROJECT_ID`/`GCP_REGION`/`GCS_BUCKET`/`GCP_BATCH_*` with `USE_AWS_BATCH`, `AWS_REGION`, `S3_BUCKET`, `AWS_BATCH_JOB_QUEUE`, `AWS_BATCH_JOB_DEFINITION`, `AWS_BATCH_POLL_INTERVAL`, `AWS_BATCH_TIMEOUT_HOURS`. Update `.env.example` and `docker-compose.yml` accordingly (currently pass `GCS_BUCKET`/`GOOGLE_APPLICATION_CREDENTIALS`).
4. **Packages**: `backend/requirements.txt` — drop `google-cloud-batch`, `google-cloud-storage`; add `boto3`.
5. **Orchestration (`backend/app/services/job_processor.py`)**: swap `submit_batch_job`/`get_batch_job_status`/`cancel_batch_job` calls for AWS Batch's `submit_job`/`describe_jobs`/`terminate_job` equivalents; map AWS Batch job states (`SUBMITTED`/`RUNNABLE`/`STARTING`/`RUNNING`/`SUCCEEDED`/`FAILED`) to the existing `processing_stage` values.
6. Retire `docs/GCP_BATCH_SETUP.md`, write `docs/AWS_BATCH_SETUP.md` covering: IAM role for the Batch job (S3 read/write, ECR pull), compute environment sizing (equivalent to `n1-standard-4`), and the ECR push flow for the batch image.

## Phase 4 — Backend compute migration (Cloud Run → App Runner)

1. Replace `backend/service.yaml` (Knative/Cloud Run spec) with an App Runner service config: source = ECR image, port 8000, CPU/memory matching current limits (0.5–1 vCPU / 256–512Mi), auto-scaling min/max matching `autoscaling.knative.dev/{min,max}Scale` (0–10).
2. Replace `backend/cloudbuild.yaml` (Cloud Build → Artifact Registry) with a CI step that builds the image and pushes to ECR (`docker build` + `aws ecr get-login-password` + `docker push`), tagging both `:<git-sha>` and `:latest` as the current file does.
3. Secrets: move `DATABASE_URL`, `SUPABASE_JWT_SECRET`→Cognito equivalents, etc. from Google Secret Manager references (`service.yaml`'s `secretKeyRef` blocks) to AWS Secrets Manager or SSM Parameter Store, referenced in the App Runner service's environment secrets config.
4. `FORWARDED_ALLOW_IPS` currently trusts GFE (Google Front End) IPs for correct client-IP resolution behind Cloud Run's proxy — verify App Runner's equivalent proxy behavior (it terminates TLS and forwards `X-Forwarded-For`; confirm trusted-proxy config in the rate-limiting/IP-tracking code in `backend/app/main.py` or wherever `slowapi`/brute-force tracking reads client IP).
5. Re-evaluate `timeoutSeconds: 300` / `containerConcurrency: 80` against App Runner's request-timeout and concurrency-per-instance settings — App Runner's defaults and tuning knobs differ from Knative's.

## Phase 5 — Frontend hosting migration (Vercel → S3 + CloudFront)

1. Configure the S3 bucket for static website hosting or as a CloudFront origin (CloudFront + OAC to a private bucket is the more secure/current-practice option over public static website hosting).
2. Add a CloudFront function or S3 error-document rewrite for SPA client-side routing (all paths → `/index.html`), since this is a client-routed Vite app — equivalent to the `firebase.json` catch-all rewrite considered in the earlier GCP plan.
3. Move frontend env vars into the CI build pipeline: `VITE_API_URL`, plus Cognito config vars replacing `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`.
4. Replace Vercel's GitHub integration with a GitHub Actions deploy step (`aws s3 sync` + CloudFront invalidation), authenticated via an IAM role and GitHub's OIDC provider (avoids long-lived AWS keys in CI).
5. Migrate the custom domain: lower DNS TTL ahead of cutover, repoint to CloudFront, verify the ACM cert (must be in `us-east-1` for CloudFront) issuance before flipping traffic.
6. Confirm FastAPI CORS config allows the new CloudFront origin; remove the Vercel origin after cutover completes.

## Phase 6 — Cutover (single maintenance window)

1. Freeze writes (uploads, ingestion, role/export requests).
2. Run a final delta `pg_dump`/restore from Supabase → RDS to capture anything written since the Phase 1 snapshot.
3. Deploy simultaneously: backend to App Runner (new `DATABASE_URL`, Cognito config, S3/AWS Batch config), frontend to S3/CloudFront, DNS flip.
4. Smoke test: sign-in (password + Google OAuth), `GET /api/v1/auth/me`, upload → review → diagnosis-review flow (exercises S3 + AWS Batch), choropleth map load (PostGIS-backed `geo` endpoints), export-request download.
5. Keep the Supabase project, Vercel project, and GCP project intact but idle for a rollback window (1–2 weeks) before decommissioning.

## Phase 7 — Decommission & cleanup

1. Stand up a replacement for the Supabase Table Editor workflow (pgAdmin or RDS Query Editor) for non-technical staff before removing Supabase access.
2. After the rollback window passes with no issues: delete the Supabase project, delete the Vercel project, delete the GCP project (Cloud Run service, GCS buckets, Artifact Registry repos, Batch job definitions), confirm none are still billing.
3. Update documentation to remove Supabase/Vercel/GCP references and reflect the new stack: `README.md`, `docs/current-architecture.md`, `docs/handoff/HANDOFF.md`, `.env.example`, `docs/GCP_BATCH_SETUP.md` (delete, replaced by `docs/AWS_BATCH_SETUP.md`).

---

## Open items requiring a decision before/during execution

- **App Runner vs. ECS Fargate** for backend compute — App Runner is the closer Cloud Run analog (less to manage) but has less control over networking/concurrency tuning; confirm App Runner meets the `timeoutSeconds`/`containerConcurrency` needs before committing.
- **PostGIS version parity** on RDS — confirm exact version match to avoid `ST_*` function behavior drift.
- **Password migration approach** — decide between forced mass password-reset vs. a Cognito migration Lambda trigger; affects Phase 2 timeline and user communications.
- **AWS Batch job definition shape** — single container with entrypoint-script S3 transfers, vs. multi-container, to replace GCP Batch's 3-runnable (setup/main/upload) structure.
- **CI service containers** — confirm whether GitHub Actions tests use a real ephemeral Postgres or a mocked Supabase client, and update fixtures accordingly.
- **Cognito password-reset flow** — verify it preserves the anti-email-prefetch property that Supabase's PKCE `verifyOtp(token_hash)` flow provides (see `docs/handoff/HANDOFF.md`).
