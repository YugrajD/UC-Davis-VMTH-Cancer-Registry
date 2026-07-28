#!/bin/sh
# Apply database/migrations/*.sql files in order against DATABASE_URL_SYNC,
# skipping ones already recorded as applied. Per-statement IF NOT EXISTS
# guards aren't enough on their own — some migrations drop/rename columns
# that earlier files reference, so re-running an already-applied file
# against current-state schema can fail even though the file is "idempotent"
# in isolation. Tracking applied filenames avoids ever re-running them.
set -e

psql "$DATABASE_URL_SYNC" -v ON_ERROR_STOP=1 -c \
  "CREATE TABLE IF NOT EXISTS schema_migrations (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW());"

for f in /database/migrations/*.sql; do
    name=$(basename "$f")
    already_applied=$(psql "$DATABASE_URL_SYNC" -tA -c \
      "SELECT 1 FROM schema_migrations WHERE filename = '$name';")
    if [ "$already_applied" = "1" ]; then
        echo "Skipping $name (already applied)"
        continue
    fi
    echo "Applying $name"
    psql "$DATABASE_URL_SYNC" -v ON_ERROR_STOP=1 -f "$f"
    psql "$DATABASE_URL_SYNC" -v ON_ERROR_STOP=1 -c \
      "INSERT INTO schema_migrations (filename) VALUES ('$name');"
done
