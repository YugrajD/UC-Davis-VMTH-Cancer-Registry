# AWS Migration Plan: Supabase + Vercel + GCP → AWS

**Status:** Proposed
**Scope:** Campus IT does not provision GCP, so this supersedes `docs/gcp-migration-plan.md` (deleted). Everything currently on GCP or on third-party services (Supabase, Vercel) moves to AWS: backend compute, PetBERT batch inference, file storage, container images, Postgres/PostGIS, auth, and frontend hosting.

---

## Target end-state

| Component | Current | Target |
|---|---|---|
| Frontend hosting | Vercel | **AWS Amplify Hosting** |
| Auth | Supabase Auth | **Amazon Cognito** |
| Database | Supabase Postgres 16 + PostGIS 3.4 | **RDS for PostgreSQL 16 + PostGIS** |
| Backend compute | Cloud Run | **ECS Fargate + ALB** |
| ML inference | GCP Batch + GCS | **ECS Fargate RunTask + S3** |
| File/report storage | GCS | **S3** |
| Container images | Artifact Registry | **ECR** |
| Admin data browsing | Supabase Table Editor | pgAdmin / RDS Query Editor (new) |

**ECS Fargate + ALB** is the target for backend compute, not App Runner. App Runner was the initial pick as the closest Cloud Run analog (fully managed, deploy-from-image, no cluster/VPC networking to hand-manage), but Fargate gives more control over request timeouts, per-instance concurrency, and networking — worth the extra setup (task definition, service, ALB, target group) for a service we'll be running long-term. No cluster capacity to manage (Fargate is serverless compute for ECS), so the operational overhead vs. App Runner is mostly in the one-time IaC/config, not ongoing ops.

## Key design decision: RLS stays a no-op

`database/migrations/012_enable_rls.sql` enables Row Level Security on every table but defines **no permissive policies**. The backend connects as the Postgres superuser and bypasses RLS; all real authorization happens in `backend/app/auth.py` against the `user_roles` table. This migration preserves that pattern — RLS is re-applied as-is on RDS for defense-in-depth (protects against direct DB access with a leaked lower-privilege credential) but is **not** used to enforce authorization. No new RLS policy work is in scope here.

## Cutover strategy: big-bang

All replacements (DB, Auth, Frontend hosting, backend compute, ML batch, storage) cut over together in a single maintenance window. Mitigated by keeping Supabase, Vercel, and GCP resources live-but-idle for a rollback window after cutover.

---

## Phase 0 — Prep (no production risk)

1. Provision target infra without touching production traffic:
   - **RDS for PostgreSQL 16** with the `postgis` extension enabled (RDS supports PostGIS as a managed extension via `CREATE EXTENSION postgis`). Verify PostGIS 3.4 parity against the RDS-supported version before relying on it.
   - **Amazon Cognito User Pool**: enable email/password + Google OAuth (as a federated IdP) sign-in. Mirror Supabase's redirect URLs. Cognito's hosted UI and password-reset flow differ from Supabase's PKCE `verifyOtp(token_hash)` approach — confirm the equivalent anti-prefetch property (Cognito's confirmation-code flow doesn't embed a clickable link by default, which may already avoid the email-scanner problem; verify before assuming parity).
   - **S3 bucket**: for pathology report text / uploads (replaces GCS `uploads/`, `reports/`, `models/` prefixes).
   - **Amplify Hosting app**: connect the frontend's GitHub repo, configure the build (Vite build settings), and set up the custom domain and managed SSL cert ahead of DNS cutover.
   - **ECR repository** for the backend image and the PetBERT batch image (replaces Artifact Registry).
   - **ECS Fargate service** (cluster, task definition, service, ALB + target group) pointed at the ECR image, sized to match `backend/service.yaml`'s current resource limits (0.5–1 vCPU, 256–512Mi).
   - **ECS Fargate ML task definition** (no Service) for PetBERT inference (replaces GCP Batch), started on demand by the backend via `ecs:RunTask` — see `infra/lib/app-stack.ts`. AWS Batch was considered and dropped: no GPU is used today (GCP runs `n1-standard-4`, CPU-only, ~10 h/run), and RunTask avoids the extra compute-environment/queue resources. The container entrypoint (`ml-worker/s3_batch_entrypoint.py`) downloads inputs from S3, runs PetBERT, and uploads results back.
2. Build the full cutover checklist by inventorying every touchpoint:
   - **Vercel**: no `vercel.json` in the repo — build settings, env vars, and domain are configured entirely via the Vercel dashboard. Nothing to port from-repo; must be manually replicated into Amplify Hosting's build settings and env var config.
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
   - **CORS**: FastAPI CORS config must allow the new Amplify Hosting origin

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

## Phase 3 — Storage & ML batch migration (GCS + GCP Batch → S3 + ECS Fargate RunTask)

Status: **code implemented** (see `docs/AWS_ML_TASK_SETUP.md`); remaining work is the one-time data copy and a staging run.

1. **Storage (`backend/app/services/s3_service.py`)**: `boto3`-backed replacement for the GCS helpers, same prefixes — `uploads/{job_id}/` (CSV uploads + ML outputs), `reports/{job_id}/{anon_id}.txt` (pathology report text), `models/` (PetBERT bundles). Uploaded CSVs now go straight to S3 (local `UPLOAD_DIR` is gone, so multiple backend tasks are safe). `pathology_reports.gcs_path` was renamed to `storage_path` (`database/migrations/032_pathology_reports_storage_path.sql`).
   - **Local dev — S3 via Floci**: `AWS_S3_ENDPOINT_URL=http://floci:4566` (path-style addressing); the bucket is created in `floci-init.sh`. Local ML inference still uses the `ml-worker` HTTP container (`USE_ECS_ML=false`).
2. **ML task image**: `ml-worker/Dockerfile.batch` now uses `s3_batch_entrypoint.py` — a single container that downloads the CSV and model bundle (~12 GB) from S3, runs `batch_predict.py`, and uploads `predictions.json` + `scan_output/*`. The Fargate task has 50 GiB ephemeral storage.
3. **Config (`backend/app/config.py`)**: `S3_BUCKET`, `AWS_REGION`, `AWS_S3_ENDPOINT_URL`, `USE_ECS_ML`, `ECS_CLUSTER_ARN`, `ML_TASK_DEFINITION_ARN`, `ML_TASK_CONTAINER_NAME`, `ML_TASK_SUBNET_IDS`, `ML_TASK_SECURITY_GROUP_ID`, `ML_POLL_INTERVAL`, `ML_TIMEOUT_HOURS`, `ML_CLEANUP_JOB_FILES`. `.env.example` and `docker-compose.yml` updated.
4. **Packages**: dropped `google-cloud-*`; added `boto3` and `tenacity`.
5. **Orchestration (`job_processor.py`, `ml_task_service.py`)**: `_process_via_ecs_task` calls `run_task`, polls `describe_tasks` (AWS calls wrapped with `tenacity` retries on transient errors), maps ECS states to `processing_stage` (`batch_queued` → `batch_scheduled` → `batch_running`), enforces `ML_TIMEOUT_HOURS` itself (ECS has no max run time), and cancels via `stop_task`. The task ARN is stored in the existing `ingestion_jobs.batch_job_name` column.
6. **Docs**: `docs/GCP_BATCH_SETUP.md` retired; `docs/AWS_ML_TASK_SETUP.md` covers image push, model upload, the GCS → S3 data copy, and IAM.

## Phase 4 — Backend compute migration (Cloud Run → ECS Fargate)

1. Replace `backend/service.yaml` (Knative/Cloud Run spec) with ECS artifacts: a task definition (container = ECR image, port 8000, CPU/memory matching current limits — 0.5–1 vCPU / 256–512Mi), a Fargate service, an ALB + target group in front of it, and a service auto-scaling policy (target tracking on CPU or request count) matching `autoscaling.knative.dev/{min,max}Scale` (0–10). Note ECS/Fargate has no true scale-to-zero like Cloud Run — minimum task count will be 1+ unless idle-shutdown is scripted separately.
2. Replace `backend/cloudbuild.yaml` (Cloud Build → Artifact Registry) with a CI step that builds the image and pushes to ECR (`docker build` + `aws ecr get-login-password` + `docker push`), tagging both `:<git-sha>` and `:latest` as the current file does, then updates the ECS service (`aws ecs update-service --force-new-deployment` or a task-def revision).
3. Secrets: move `DATABASE_URL`, `SUPABASE_JWT_SECRET`→Cognito equivalents, etc. from Google Secret Manager references (`service.yaml`'s `secretKeyRef` blocks) to AWS Secrets Manager or SSM Parameter Store, referenced in the ECS task definition's `secrets` block.
4. `FORWARDED_ALLOW_IPS` currently trusts GFE (Google Front End) IPs for correct client-IP resolution behind Cloud Run's proxy — with an ALB in front of Fargate, `X-Forwarded-For` is appended by the ALB itself; update the trusted-proxy config in `backend/app/main.py`/the rate-limiting/IP-tracking code to trust the ALB instead of GFE IPs.
5. Re-evaluate `timeoutSeconds: 300` / `containerConcurrency: 80` against ALB idle-timeout and target-group settings — these map more directly from Knative than App Runner's would (ALB has an explicit idle timeout setting; per-task concurrency is just however many connections the container can handle, not a platform-enforced knob).
6. VPC networking: unlike App Runner, Fargate tasks need an explicit VPC/subnet/security-group setup (reuse the RDS VPC from Phase 1) and the ALB needs public subnets — this is net-new infra work that App Runner would have avoided.

## Phase 5 — Frontend hosting migration (Vercel → Amplify Hosting)

1. Connect the frontend's GitHub repo to Amplify Hosting and configure the build spec (`amplify.yml`): install, `npm run build`, publish `frontend/dist`. This replaces Vercel's auto-detected build entirely — Amplify's build settings UI/`amplify.yml` is the equivalent of the Vercel dashboard config that currently has no in-repo file.
2. Amplify Hosting handles SPA rewrites for client-side routing (all paths → `/index.html`) via a rewrite rule in its console/config — simpler than hand-rolling a CloudFront function, no `firebase.json`-style catch-all needed in-repo.
3. Move frontend env vars into Amplify's environment variables (per-branch): `VITE_API_URL`, plus Cognito config vars replacing `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`.
4. Amplify Hosting's Git integration replaces Vercel's directly — push-to-branch triggers a build/deploy automatically, no separate GitHub Actions deploy step or IAM/OIDC wiring needed for the frontend (unlike the S3+CloudFront approach).
5. Migrate the custom domain: add it in Amplify's domain management, lower DNS TTL ahead of cutover, verify Amplify's managed SSL cert issuance before flipping traffic.
6. Confirm FastAPI CORS config allows the new Amplify Hosting origin (default `*.amplifyapp.com` during setup, then the custom domain); remove the Vercel origin after cutover completes.

## Phase 6 — Cutover (single maintenance window)

1. Freeze writes (uploads, ingestion, role/export requests).
2. Run a final delta `pg_dump`/restore from Supabase → RDS to capture anything written since the Phase 1 snapshot.
3. Deploy simultaneously: backend to App Runner (new `DATABASE_URL`, Cognito config, S3/ECS ML task config), frontend to Amplify Hosting, DNS flip.
4. Smoke test: sign-in (password + Google OAuth), `GET /api/v1/auth/me`, upload → review → diagnosis-review flow (exercises S3 + the ECS ML task), choropleth map load (PostGIS-backed `geo` endpoints), export-request download.
5. Keep the Supabase project, Vercel project, and GCP project intact but idle for a rollback window (1–2 weeks) before decommissioning.

## Phase 7 — Decommission & cleanup

1. Stand up a replacement for the Supabase Table Editor workflow (pgAdmin or RDS Query Editor) for non-technical staff before removing Supabase access.
2. After the rollback window passes with no issues: delete the Supabase project, delete the Vercel project, delete the GCP project (Cloud Run service, GCS buckets, Artifact Registry repos, Batch job definitions), confirm none are still billing.
3. Update documentation to remove Supabase/Vercel/GCP references and reflect the new stack: `README.md`, `docs/current-architecture.md`, `docs/handoff/HANDOFF.md`, `.env.example`, `docs/handoff/HANDOFF.md` / `doc-site/src/handoff.md` (still describe `USE_GCP_BATCH`, `gcs_path`) (`docs/GCP_BATCH_SETUP.md` is already replaced by `docs/AWS_ML_TASK_SETUP.md`).

---

## Open items requiring a decision before/during execution

- **PostGIS version parity** on RDS — confirm exact version match to avoid `ST_*` function behavior drift.
- **Password migration approach** — decide between forced mass password-reset vs. a Cognito migration Lambda trigger; affects Phase 2 timeline and user communications.
- **ML task shape — resolved**: single Fargate container whose entrypoint does the S3 transfers (`ml-worker/s3_batch_entrypoint.py`). Open: whether 16 GB RAM is enough (GCP's 15 GB worked) and whether Fargate platform patching interrupting a ~10 h task is acceptable.
- **CI service containers** — confirm whether GitHub Actions tests use a real ephemeral Postgres or a mocked Supabase client, and update fixtures accordingly.
- **Cognito password-reset flow** — verify it preserves the anti-email-prefetch property that Supabase's PKCE `verifyOtp(token_hash)` flow provides (see `docs/handoff/HANDOFF.md`).

## Cleanup opportunities while touching this code (not blocking, but cheap to fold in)

- **`backend/app/auth.py:46-96`** — a hand-rolled, per-process in-memory brute-force limiter (`_failed_attempts`) duplicates `slowapi`, which is already wired up elsewhere in the app (`app/rate_limit.py`, `app/main.py`). Worth consolidating onto `slowapi` while Phase 2 is already touching `auth.py` — avoids maintaining two rate-limiting mechanisms and fixes the noted single-worker limitation.
- **ML task polling (Phase 3) — done**: `tenacity` retries wrap the ECS calls in `ml_task_service.py`. The fixed-interval poll loop is kept; a backend restart mid-job still loses the watcher (startup recovery marks the job failed). EventBridge task-state events would fix that if it becomes a problem.
