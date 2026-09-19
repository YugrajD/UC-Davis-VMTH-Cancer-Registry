#!/bin/sh
# THROWAWAY (see docs/floci-local-dev-migration.md): the container Floci's
# RDS emulation spins up has two independent Postgres installs — the actual
# running server is Docker's official build (under /usr/local/...), while
# Alpine's `apk add postgis` installs into Alpine's own package tree (under
# /usr/lib/postgresqlNN/..., /usr/share/postgresqlNN/...). The two never
# overlap, so `apk add postgis` alone leaves `CREATE EXTENSION postgis`
# unable to find it regardless of matching major version. This installs the
# apk package, then copies the resulting files into the real server's
# actual pkglibdir/sharedir (found via docker exec + pg_config). Confirmed
# working live against a real Floci RDS container before landing this.
set -e

. /shared/floci-outputs.env

if [ -z "$FLOCI_DB_ID" ]; then
    echo "install-postgis: FLOCI_DB_ID not set — was floci-init.sh run first?" >&2
    exit 1
fi

echo "install-postgis: looking for the container labeled io.floci.resource-id=$FLOCI_DB_ID..."
CONTAINER_ID=""
for i in $(seq 1 30); do
  CONTAINER_ID=$(docker ps --filter "label=io.floci.resource-id=${FLOCI_DB_ID}" --format '{{.ID}}' | head -n1)
  if [ -n "$CONTAINER_ID" ]; then
    break
  fi
  sleep 2
done

if [ -z "$CONTAINER_ID" ]; then
  echo "install-postgis: could not find a container labeled io.floci.resource-id=${FLOCI_DB_ID}." >&2
  echo "install-postgis: run 'docker ps --filter label=io.floci' manually and adjust this script — see docs/floci-local-dev-migration.md open items." >&2
  exit 1
fi

echo "install-postgis: found container $CONTAINER_ID, installing postgis via apk..."
docker exec "$CONTAINER_ID" apk add --no-cache postgis > /dev/null

echo "install-postgis: copying postgis files into the real server's pkglibdir/sharedir..."
docker exec "$CONTAINER_ID" sh -c '
  set -e
  apk_libdir=$(find /usr/lib -maxdepth 1 -type d -name "postgresql*" | sort -V | tail -n1)
  apk_extdir=$(find /usr/share -maxdepth 1 -type d -name "postgresql*" | sort -V | tail -n1)/extension
  pg_libdir=$(pg_config --pkglibdir)
  pg_sharedir=$(pg_config --sharedir)/extension
  cp "$apk_libdir"/postgis*.so "$pg_libdir"/
  cp "$apk_extdir"/postgis*.control "$apk_extdir"/postgis*.sql "$pg_sharedir"/
'

echo "install-postgis: done"
