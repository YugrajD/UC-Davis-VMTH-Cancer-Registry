#!/usr/bin/env python3
"""
Seed taxonomy_terms from ml/taxonomy/labels.csv — the Vet-ICD-O-Canine-1
(group, term) pairs the audit-list review screen's code picker and gold
export validator read. Production has no /ml mount (backend/Dockerfile only
COPYs ./backend), so this table is the only place the backend can read the
taxonomy from at runtime; re-run this whenever labels.csv changes.

Usage:
  docker compose run --rm seed python /database/seed/seed_taxonomy_terms.py
"""

import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL_SYNC")
if not DATABASE_URL:
    sys.exit("ERROR: DATABASE_URL_SYNC environment variable is required")

LABELS_CSV = Path(os.getenv("LABELS_CSV_PATH", "/ml/taxonomy/labels.csv"))

sys.path.insert(0, "/ml")
try:
    from taxonomy.taxonomy import load_labels_taxonomy
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
    from taxonomy.taxonomy import load_labels_taxonomy


def run() -> None:
    if not LABELS_CSV.exists():
        sys.exit(f"ERROR: taxonomy CSV not found at {LABELS_CSV}")

    labels = load_labels_taxonomy(str(LABELS_CSV))
    if not labels:
        sys.exit(f"ERROR: {LABELS_CSV} parsed to zero rows")

    # (group, term) is unique in the CSV (verified against the live file —
    # no code collides two groups/terms into the same pair), matching
    # taxonomy_terms' UNIQUE(taxonomy_group, taxonomy_term).
    rows = [(label.code, label.group, label.term) for label in labels]

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    execute_values(
        cur,
        """INSERT INTO taxonomy_terms (vet_icd_o_code, taxonomy_group, taxonomy_term)
           VALUES %s
           ON CONFLICT (taxonomy_group, taxonomy_term)
           DO UPDATE SET vet_icd_o_code = EXCLUDED.vet_icd_o_code""",
        rows,
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM taxonomy_terms")
    total = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"Done!")
    print(f"  Source: {LABELS_CSV}")
    print(f"  Rows seeded/updated this run: {len(rows)}")
    print(f"  Total rows in taxonomy_terms: {total}")


if __name__ == "__main__":
    run()
