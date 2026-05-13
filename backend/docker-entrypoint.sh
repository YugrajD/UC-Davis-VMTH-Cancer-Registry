#!/bin/sh
# Fix /app/uploads ownership before dropping privileges to `app`.
#
# `/app/uploads` is mounted from a Docker named volume; the kernel creates
# the mount point as root-owned regardless of what the image had at build
# time, so the non-root `app` user can't write inside it.  This entrypoint
# runs as root just long enough to chown the directory tree, then exec's
# the real command as `app` via gosu.
set -e

if [ -d /app/uploads ]; then
    chown -R app:app /app/uploads || true
fi

exec gosu app "$@"
