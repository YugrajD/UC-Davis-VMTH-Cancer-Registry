"""Cloud file contracts: land inbox exports, write outbox exports, build the worker bundle.

Usage:
  python ml/next/scripts/handoff.py import-pending --csv PATH --export-id ID
  python ml/next/scripts/handoff.py import-gold     --csv PATH --export-id ID --reviewer "Dr. Smith"
  python ml/next/scripts/handoff.py export-silver   --silver-id SID
  python ml/next/scripts/handoff.py export-coding   --run-id RUN
  python ml/next/scripts/handoff.py export-bundle   [--generation current]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handoff import exports, imports


def _cmd_import_pending(args: argparse.Namespace) -> int:
    result = imports.import_pending_diagnoses(args.csv, args.export_id)
    print(f"Imported {result['imported_rows']} row(s) across {result['imported_cases']} case(s) "
          f"({result['replaced_cases']} replaced); landing table now holds "
          f"{result['merged_total_rows']} row(s) across {result['merged_total_cases']} case(s).")
    return 0


def _cmd_import_gold(args: argparse.Namespace) -> int:
    result = imports.import_gold(
        args.csv, args.export_id, reviewer=args.reviewer,
        upload_period=args.upload_period, slice_rate=args.slice_rate,
    )
    print(f"Gold: {result['added_cases']} new case(s), {result['replaced_cases']} re-reviewed; "
          f"store now holds {result['total_rows']} row(s) across {result['total_cases']} case(s).")
    return 0


def _cmd_export_silver(args: argparse.Namespace) -> int:
    out_path = exports.export_silver(args.silver_id)
    print(f"wrote {out_path}")
    return 0


def _cmd_export_coding(args: argparse.Namespace) -> int:
    result = exports.export_coding(args.run_id)
    print(f"wrote {result['adopted_codes_path']} ({result['adopted_rows']} rows)")
    print(f"wrote {result['review_queue_path']} ({result['review_queue_rows']} rows)")
    return 0


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    result = exports.export_bundle(args.generation)
    print(f"wrote {result['tarball_path']} (sha256 {result['sha256']}) for generation {result['generation_id']}")
    print(f"wrote {result['sha256_path']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import-pending", help="Land + merge a pending-diagnoses export.")
    p.add_argument("--csv", required=True)
    p.add_argument("--export-id", required=True)

    p = sub.add_parser("import-gold", help="Land a gold export and ingest it via manual_audit.gold.")
    p.add_argument("--csv", required=True)
    p.add_argument("--export-id", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--upload-period", default="")
    p.add_argument("--slice-rate", default="")

    p = sub.add_parser("export-silver", help="Write silver_codes_<silver_id>.csv to the outbox.")
    p.add_argument("--silver-id", required=True)

    p = sub.add_parser("export-coding", help="Write adopted_codes_<run>.csv + review_queue_<run>.csv to the outbox.")
    p.add_argument("--run-id", required=True)

    p = sub.add_parser("export-bundle", help="Tar a report-mapping generation + sha256 for ml-worker.")
    p.add_argument("--generation", default="current")

    args = parser.parse_args()
    dispatch = {
        "import-pending": _cmd_import_pending,
        "import-gold": _cmd_import_gold,
        "export-silver": _cmd_export_silver,
        "export-coding": _cmd_export_coding,
        "export-bundle": _cmd_export_bundle,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
