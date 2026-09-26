"""Split generations: import the legacy splits, create a three-way split, run the leakage guards.

Usage:
  python ml/next/scripts/split.py import-legacy
  python ml/next/scripts/split.py create --parent legacy-80-20 --id three-way-v1
  python ml/next/scripts/split.py check --split three-way-v1 [--labels-csv PATH]

Note: `check` cannot exercise the calibration-inputs guard; that one runs inside `calibrate`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generations.guards import GuardViolation, check_all
from generations.manifest import ManifestError
from generations.splits import create_three_way, import_legacy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("import-legacy", help="Import the legacy 80/20 and temporal splits (byte-identical).")
    create = sub.add_parser("create", help="Create a three-way split from a two-way parent.")
    create.add_argument("--parent", required=True)
    create.add_argument("--id", required=True)
    check = sub.add_parser("check", help="Run every applicable leakage guard.")
    check.add_argument("--split", required=True)
    check.add_argument("--labels-csv", default=None, help="Train-only labels table to check (optional).")
    args = parser.parse_args()

    if args.command == "import-legacy":
        for manifest in import_legacy():
            print(f"{manifest['split_id']}: {manifest['counts']}")
    elif args.command == "create":
        manifest = create_three_way(args.parent, args.id)
        print(f"{manifest['split_id']} (parent {manifest['parent']}): {manifest['counts']}")
    else:
        try:
            passed = check_all(args.split, labels_csv=args.labels_csv)
        except (GuardViolation, ManifestError) as violation:
            print(f"FAIL: {violation}")
            return 1
        print(f"PASS: {args.split}: {', '.join(passed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
