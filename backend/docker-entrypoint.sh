#!/bin/sh
# Fix /app/uploads ownership before dropping privileges to `app`.
#
# `/app/uploads` is bind-mounted from the host (./backend/uploads/).  On
# Linux the host directory may be owned by root or another UID, so the
# non-root `app` user can't write inside it without this chown.  This
# entrypoint runs as root just long enough to fix ownership, then exec's
# the real command as `app` via gosu.  The `|| true` means a permission
# failure (e.g. on macOS where bind-mount ownership is managed by Docker
# Desktop) is silently ignored.
set -e

if [ -d /app/uploads ]; then
    chown -R app:app /app/uploads || true
fi

# THROWAWAY (see docs/floci-local-dev-migration.md): dev-only override of
# DATABASE_URL/COGNITO_* with values discovered by floci-init.sh, since
# Floci assigns the RDS endpoint and Cognito pool/client IDs at creation
# time rather than us pinning them in .env. Absent in prod, where this
# file is never mounted.
if [ -f /shared/floci-outputs.env ]; then
    . /shared/floci-outputs.env
fi

exec gosu app "$@"
