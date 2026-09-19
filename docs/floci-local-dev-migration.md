# Local dev migration: postgis + cognito-local → Floci

**Status:** Implemented and verified live end-to-end (`floci` → `floci-init` → `floci-postgis-init` → `migrate` → `backend`, real HTTP request round-tripping through the RDS-emulated Postgres). See "Status" section below for what's confirmed vs. still open.
**Scope:** Replace the local dev stack's `postgres` (postgis/postgis) and `cognito-local`/`cognito-seed` containers with a single [Floci](https://github.com/floci/floci) container emulating RDS, Cognito, S3, Batch, Lambda, etc. behind one AWS API endpoint (`:4566`). This makes local dev exercise the same AWS SDK calls (`boto3`, Cognito Identity SDK) that production code will use post-`aws-migration-plan.md`, instead of two unrelated emulators (a real Postgres container + a Cognito-specific mock).

## Current state

- `postgres`: `postgis/postgis:16-3.4` container, direct TCP on 5432, migrations run against it via `migrate`.
- `cognito-seed` + `cognito-local`: `jagregory/cognito-local` with a pre-baked JSON seed (`database/docker/cognito-local/seed/`) giving a **fixed** pool ID (`local_vmthdev`) and client ID (`vmthcancerregistryweb`) that survive `docker compose down -v` because the seed step re-copies the JSON into the named volume if absent.

The fixed-ID property is what makes today's `.env.example` work: `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_ID`, and `DATABASE_URL`'s host are all static values checked into example config.

## Target state

- One `floci` service, `FLOCI_STORAGE_MODE: hybrid`, docker-socket mount (required because Floci's RDS emulation spins up a real Postgres container under the hood), state persisted to a named volume so IDs survive `docker compose down` (but not `-v`, same tradeoff as today's `cognito_local_data`).
- A one-shot `floci-init` service (same role as today's `cognito-seed`) that runs **after** `floci` is healthy and, only if not already provisioned:
  1. Creates the RDS Postgres instance (`aws rds create-db-instance`), polls until available, and resolves the actual host:port Floci assigns it (`aws rds describe-db-instances`).
  2. Creates the Cognito User Pool + App Client (`aws cognito-idp create-user-pool` / `create-user-pool-client`) — pool/client IDs are **assigned by the API call**, not pinned, since Floci is a real API emulator rather than a static-file mock.
  3. Writes the discovered values to a shared file on a named volume (e.g. `/shared/floci-outputs.env`): `DATABASE_URL`, `DATABASE_URL_SYNC`, `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_ID`.
- Every consumer of those values (`migrate`, `backend`, `seed`, `ingest`, `geo-seed`) mounts the shared volume and its entrypoint sources `floci-outputs.env` before exec'ing its real command, overriding whatever static `DATABASE_URL`/`COGNITO_*` came from `.env`.
- Frontend: same problem as today's `VITE_COGNITO_ENDPOINT` (browser must reach Floci on `localhost:4566`, not the docker-network hostname) plus now also needs the dynamically-assigned pool/client ID — likely served via a small `/config` endpoint from the backend (which already has the values from its own sourced env) rather than baked into Vite env vars at container build time.

## Why this is more involved than the cognito-local swap

`cognito-local` was a drop-in — same fixed IDs the app already expected. Floci is a real API emulator: IDs come back from the API call itself, so the "pin a known ID in `.env.example`" pattern goes away entirely. This is arguably more realistic (matches how the app will behave against real AWS Cognito/RDS, where pool ID and RDS endpoint also aren't chosen by us), but it does mean:

- `.env.example`'s static `DATABASE_URL`/`COGNITO_USER_POOL_ID`/`COGNITO_CLIENT_ID` become **defaults used only until `floci-init` overrides them** — worth a comment explaining that, so a future reader isn't confused about why the app doesn't seem to respect `.env`.
- Every service needs an entrypoint change (source-then-exec), not just an env var swap.

## Future direction: CDK, not hand-written provisioning scripts

Plan is to eventually pull infra provisioning (RDS instance, Cognito user pool, S3 buckets, Batch, etc. — both this local-dev stack and the real `aws-migration-plan.md` Phase 0 resources) into a **CDK project**, deployed against Floci locally (`cdklocal` or an endpoint override pointed at `:4566`) and against real AWS in prod from the same stack definitions. That collapses "provision for Floci" and "provision for AWS" into one IaC source of truth instead of maintaining parallel `aws-cli` scripts and Terraform/console steps.

Practical implication for this doc: don't invest in the hand-rolled `floci-init.sh`/`aws rds create-db-instance` script below as a long-lived artifact — treat it as a throwaway bridge to unblock local dev *now*, and expect it to be replaced once the CDK stack exists. Confirmed detail that simplifies the CDK side later: Floci's RDS emulation spins up an actual Postgres docker container (via the mounted docker socket) rather than pure API mocking, so `cdklocal`-deployed RDS constructs should behave close to the real thing rather than needing Floci-specific stubbing.

## Open items requiring hands-on verification (can't confirm without running Floci)

- **PostGIS on emulated RDS — resolved, verified live.** Floci's RDS emulation runs `postgres:16-alpine` — actual server is Docker's official v16.15 build (under `/usr/local/...`), while Alpine 3.24's `apk add postgis` installs into Alpine's *own* package tree (`/usr/lib/postgresqlNN/...`, `/usr/share/postgresqlNN/...`), which the real server never looks at, regardless of matching version numbers. Worse, Alpine 3.24 only ships one `postgis` build at all, hard-linked against Alpine's own `postgresql18` package — **there's no pg16-targeted postgis build in this Alpine release, period.** Verified fix (tested live against a real Floci container on the dev machine): provision the RDS instance as **engine-version 18** (matching what Alpine 3.24 actually ships), `apk add postgis`, then copy the resulting `.so`/`.control`/`.sql` files from Alpine's package tree into the real server's actual `pg_config --pkglibdir`/`--sharedir` (paths discovered dynamically, not hardcoded). `CREATE EXTENSION postgis` and `postgis_full_version()` both confirmed working this way. **Tradeoff accepted**: local dev now runs Postgres 18, while `aws-migration-plan.md` targets RDS Postgres 16 in production (matching Supabase's currently pinned version per this repo's own docs) — a deliberate, accepted gap for this throwaway local-dev bridge, not something to silently forget when the CDK rework happens.
- **Network reachability of the emulated RDS endpoint — resolved.** Confirmed live: the RDS container isn't published to a host port at all; it's reachable from other containers on the same compose network directly at the IP:port `describe-db-instances` returns (Floci's own network IP proxies the connection through to the actual backing container). Also confirmed: Floci labels that container `io.floci.resource-id=<db-instance-identifier>` — used as the discovery mechanism in `install-postgis.sh` instead of guessing from ports.
- **`create-db-instance` provisioning latency** — confirmed fast in practice (a few seconds) during live testing, not the minutes real RDS takes.
- **Idempotency across restarts — resolved (after a real bug fix).** After a `docker compose down`/`up` cycle (no `-v`), Floci's own container gets a **new docker-network IP** on every restart, even though the RDS instance identifier / data / Cognito pool all persist server-side (confirmed: migrations showed "already applied" for all 31 files after a restart — actual PGDATA survives, keyed by Floci internally off the resource ID, not tied to the ephemeral docker container hash). The bug: `floci-init.sh` originally skipped all work if `/shared/floci-outputs.env` already existed, so it kept serving a **stale cached IP** after restart → `psql: connection refused` from `migrate`. Fixed: `floci-init.sh` now always re-resolves RDS instance / Cognito pool / client (reusing them via `describe-db-instances`/`list-user-pools`/`list-user-pool-clients` if they already exist, creating only if missing) and always rewrites `floci-outputs.env` with the current values, every run. Verified live: RDS IP changed across a restart (`192.168.97.2` → `192.168.97.3`), old cached value failed, re-running `floci-init` picked up the new IP and `migrate`/`backend` worked immediately after.
  - **Caveat**: this only helps if `floci-init` actually gets *re-run* after a `floci` restart. `docker compose up`'s dependency graph won't force that on its own — Compose just checks whether the existing (already-`Exited`) `floci-init` container satisfies `condition: service_completed_successfully`; it doesn't re-execute a completed one-shot container just because a sibling service restarted. That's why `start.sh` and the README's setup steps explicitly run `docker compose run --rm floci-init` (which always creates a fresh container and re-runs the script) rather than relying on `up` alone. Anyone who skips that and just runs bare `docker compose up` after a restart can still hit the stale-IP symptom.
- **Whether the RDS host/port is reachable from the *host* machine** (not just other containers) — not yet tested; the "Browsing the local database" README section still just says to check.
- **Health check endpoint — still unresolved.** Floci's own Dockerfile documents `HEALTHCHECK ... wget -q --spider http://localhost:4566/_floci/health`, but that endpoint turned out not to exist on the version actually running (looked like stale/too-new docs). Dropped the compose-level healthcheck; `floci-init` depends on `floci` with `condition: service_started` only, and `floci-init.sh` polls readiness itself via `aws sts get-caller-identity` in a retry loop before doing anything else. Revisit if a real health endpoint is confirmed later.
- **Browser sign-in "network error" — resolved.** Floci's Cognito emulation returns `405 Method Not Allowed` with **no CORS headers** on the `OPTIONS` preflight the browser sends before any cross-origin call (frontend on `:5173` calling Floci on `:4566` directly is cross-origin). `amazon-cognito-identity-js` does a plain `fetch()` — the browser blocks it outright, surfacing as "network error" in the UI. Fix: added a `/cognito` proxy rule to `frontend/vite.config.ts` (forwards to `http://floci:4566` server-side, stripping the `/cognito` prefix) and pointed `VITE_COGNITO_ENDPOINT` at that relative path instead of Floci's absolute URL — same-origin browser requests don't trigger a CORS preflight at all, sidestepping the problem entirely. Verified live end-to-end: real sign-up → sign-in → JWT → `GET /api/v1/auth/me` all succeeded through the proxy.
- **Stale `.env` values — a real trap, not a Floci bug.** While debugging the above, found that `docker-compose.yml`'s `${VAR:-default}` fallbacks only apply when `VAR` is *unset* — an already-checked-out `.env` (gitignored, never touched by our `.env.example` edits) with old literal values (`COGNITO_ISSUER_URL=http://cognito-local:9229`, `VITE_COGNITO_ENDPOINT=http://localhost:9229`, `DATABASE_URL=...@postgres:5432/...`) silently wins over every compose default we changed, even after `--force-recreate`. This caused the backend's JWKS fetch to fail with a DNS lookup error (`cognito-local` no longer exists on the network). Anyone with a pre-existing `.env` from before this migration needs to manually re-diff it against `.env.example` and update the stale `COGNITO_ISSUER_URL`/`VITE_COGNITO_ENDPOINT`/`DATABASE_URL`/`DATABASE_URL_SYNC` lines — there's no automatic migration for this.

## Implementation sketch (compose)

```yaml
services:
  floci:
    image: floci/floci:latest
    environment:
      FLOCI_STORAGE_MODE: hybrid
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - floci_data:/app/data
    ports:
      - "4566:4566"
    # No compose healthcheck — see "Health check endpoint" open item above.

  floci-init:
    image: amazon/aws-cli:2.17.62
    environment:
      AWS_ACCESS_KEY_ID: test
      AWS_SECRET_ACCESS_KEY: test
      AWS_DEFAULT_REGION: us-east-1
      AWS_ENDPOINT_URL: http://floci:4566
    entrypoint: ["sh", "/init/floci-init.sh"]
    volumes:
      - ./database/docker/floci/floci-init.sh:/init/floci-init.sh:ro
      - floci_outputs:/shared
    depends_on:
      floci:
        condition: service_started
    # floci-init.sh polls `aws sts get-caller-identity` in a retry loop
    # before doing anything else, since compose can't health-gate here.

  migrate:
    # ...
    volumes:
      - floci_outputs:/shared:ro
    entrypoint: ["sh", "/run_migrations.sh"]
    depends_on:
      floci-init:
        condition: service_completed_successfully

volumes:
  floci_data:
  floci_outputs:
```

## Status: implemented and verified live end-to-end

Implemented per the "ship now, replace with CDK later" sequencing decision, then verified against a real running Floci instance on the dev machine — not just written and assumed to work:

- `database/docker/floci/floci-init.sh` — the provisioning script described above. Provisions engine-version 18 (see the PostGIS item above for why). Not actually "one-shot" — safe and cheap to re-run any time (e.g. after `docker compose down`/`up`), and *should* be re-run whenever `floci` restarts, since it always refreshes `floci-outputs.env` with the current live endpoint (see the "Idempotency across restarts" item above for why that matters).
- `database/docker/floci/install-postgis.sh` — installs PostGIS into the actual container Floci's RDS emulation spins up and copies the resulting files into the real server's actual `pkglibdir`/`sharedir` (see the PostGIS item above for why a plain `apk add` isn't enough). Finds the container via its `io.floci.resource-id` label, not port-guessing. Run via the `floci-postgis-init` compose service (mounts the docker socket), between `floci-init` and `migrate` in the dependency chain.
- `docker-compose.yml` — `floci` + `floci-init` + `floci-postgis-init` services replace `postgres`/`cognito-seed`/`cognito-local`; `migrate`/`backend`/`seed`/`ingest`/`geo-seed`/`frontend` all mount `floci_outputs` and depend (transitively) on `floci-init`.
- Rather than a separate `with-floci-env.sh` wrapper, sourcing was added directly where it was simplest:
  - `backend/docker-entrypoint.sh` sources `/shared/floci-outputs.env` (if present) before `exec gosu app "$@"` — covers `backend`, `seed`, `ingest`, `geo-seed`, since they all build from the same image/entrypoint.
  - `database/scripts/run_migrations.sh` sources it directly at the top.
  - `frontend`'s compose `command:` sources it inline before `npx vite`, since Vite reads `import.meta.env` from process env at server startup and there's no existing entrypoint script to hook into.
- No backend `/api/v1/config` endpoint was added — the frontend wrapper approach above made it unnecessary for now. Revisit if the CDK rework wants a cleaner separation.
- `database/docker/cognito-local/` and the `postgres_data`/`cognito_local_data` volumes were removed.
- `README.md`/`start.sh`/`.env.example` updated to match; account creation now reads the Floci-assigned pool/client ID via `docker compose exec backend cat /shared/floci-outputs.env` instead of a hardcoded pool ID.

**Confirmed working via a real run**: `docker compose run --rm floci-init` → `floci-postgis-init` → `migrate` (all 31 migrations applied, including `CREATE EXTENSION postgis`) → `docker compose up -d backend` → `GET /health` and `GET /api/v1/dashboard/summary` both returned real responses from the RDS-emulated Postgres over the docker network.

**Still not verified**: idempotency across a `docker compose down`/`up` cycle without `-v`, and whether the RDS host/port is reachable from the *host* machine (only container-to-container reachability has been confirmed).

## Next steps

1. Confirm idempotency across restarts and host-machine DB reachability (the two still-open items above).
2. Once the CDK project exists, replace `floci-init.sh`'s hand-rolled `aws-cli` provisioning (and the equivalent hand-provisioned steps in `aws-migration-plan.md`'s Phase 0) with CDK stacks deployable against both Floci and real AWS.
