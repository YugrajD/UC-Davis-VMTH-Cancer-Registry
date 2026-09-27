"""Coding-rule outputs: combined codes, corrected annotations, the review queue.

Usage:
  python ml/scripts/code_cases.py combine   --silver SID --split SPLIT --predictions PATH [--generation-id ID] [--out PATH]
  python ml/scripts/code_cases.py corrected --silver SID --split SPLIT [--out PATH]
  python ml/scripts/code_cases.py queue     --silver SID --split SPLIT --predictions PATH [--generation-id ID] [--out PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coding.combine import write_combined_codes
from coding.corrected import write_corrected_annotations
from coding.queue import write_review_queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    combine = sub.add_parser("combine", help="Write the combined-codes table.")
    combine.add_argument("--silver", required=True)
    combine.add_argument("--split", required=True)
    combine.add_argument("--predictions", required=True)
    combine.add_argument("--generation-id", default=None)
    combine.add_argument("--out", default=None)

    corrected = sub.add_parser("corrected", help="Write the corrected-annotations table (train partition only).")
    corrected.add_argument("--silver", required=True)
    corrected.add_argument("--split", required=True)
    corrected.add_argument("--out", default=None)

    queue = sub.add_parser("queue", help="Write the review queue.")
    queue.add_argument("--silver", required=True)
    queue.add_argument("--split", required=True)
    queue.add_argument("--predictions", required=True)
    queue.add_argument("--generation-id", default=None)
    queue.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.command == "combine":
        out_path = write_combined_codes(
            args.silver, args.split, args.predictions,
            generation_id=args.generation_id, out_csv=args.out,
        )
    elif args.command == "corrected":
        out_path = write_corrected_annotations(args.silver, args.split, out_csv=args.out)
    else:
        out_path = write_review_queue(
            args.silver, args.split, args.predictions,
            generation_id=args.generation_id, out_csv=args.out,
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
