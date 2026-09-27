"""Manual audit: Tier-3 sample/pilot/ingest, eval batches, gold ingest, cause pass.

Usage:
  python ml/scripts/audit.py tier3-sample --silver-id silver-0-legacy --test-cases-txt PATH --batch 1
  python ml/scripts/audit.py tier3-pilot --review-csv PATH --key-csv PATH
  python ml/scripts/audit.py tier3-ingest --review-csv PATH --key-csv PATH --batch 1 --reviewer "Dr. Smith"
  python ml/scripts/audit.py eval-batch --batch-id eval-batch-1 --silver-id silver-0-legacy --fraction 0.5
  python ml/scripts/audit.py ingest-sheet --sheet PATH --batch-id eval-batch-1 --reviewer "Dr. Smith"
  python ml/scripts/audit.py ingest-gold --rows-csv PATH --origin eval_batch --reviewer "Dr. Smith"
  python ml/scripts/audit.py cause-sheet --verdicts-csv PATH --out-csv PATH
  python ml/scripts/audit.py ingest-cause --filled-csv PATH --reviewer "Dr. Smith"

``tier3-pilot`` and ``tier3-ingest``'s ``--review-csv``/``--key-csv`` default to
this rewrite's own sheet paths, but point them at the already-issued legacy
sheets to ingest those unchanged, e.g.:
  --review-csv ml/output/annotation/tier3_audit_review.csv
  --key-csv    ml/output/annotation/tier3_audit_key.csv
(or the ``_pilot_``/``_remainder_`` variants next to them).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from manual_audit import cause_pass, eval_batch, gold, tier3_audit


def _cmd_tier3_sample(args: argparse.Namespace) -> int:
    test_case_ids = {line.strip() for line in Path(args.test_cases_txt).read_text(encoding="utf-8").splitlines() if line.strip()}
    exclude = set()
    for path in args.exclude_cases or []:
        exclude.update(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    result = tier3_audit.sample(
        silver_id=args.silver_id, test_case_ids=test_case_ids, labels_csv=args.labels_csv,
        batch=args.batch, n_rows=args.n_rows, seed=args.seed, exclude_case_ids=exclude,
    )
    print(f"Sampled {result['n_rows']} rows across {result['n_cases']} cases -> {result['review_csv']}")
    return 0


def _cmd_tier3_pilot(args: argparse.Namespace) -> int:
    result = tier3_audit.pilot(
        review_csv=args.review_csv, key_csv=args.key_csv,
        out_pilot_csv=args.out_pilot, out_remainder_csv=args.out_remainder,
        n_rows=args.n_rows, min_per_stratum=args.min_per_stratum,
        out_instructions_md=args.out_instructions,
    )
    print(f"Pilot {result['pilot_rows']} rows, remainder {result['remainder_rows']} rows.")
    return 0


def _cmd_tier3_ingest(args: argparse.Namespace) -> int:
    result = tier3_audit.ingest(
        review_csv=args.review_csv, key_csv=args.key_csv, batch=args.batch,
        reviewer=args.reviewer, labels_csv=args.labels_csv,
    )
    print(f"Ingested {result['ingested']} row(s): {result['added']} new, {result['replaced']} re-reviewed. "
          f"Audit store now holds {result['total_rows']} row(s).")
    return 0


def _cmd_eval_batch(args: argparse.Namespace) -> int:
    result = eval_batch.generate_batch(
        batch_id=args.batch_id, silver_id=args.silver_id, fraction=args.fraction, split_id=args.split_id,
        seed=args.seed, target_codes_per_group=args.target_codes_per_group,
        big_group_share=args.big_group_share, rare_stratum_n=args.rare_stratum_n,
        no_cancer_target=args.no_cancer_target,
    )
    print(f"Batch {result['batch_id']}: {result['total_cases']} cases, "
          f"excluded {result['excluded_count']} -> {result['sheet_path']}")
    for stratum, n_h in result["stratum_counts"].items():
        print(f"  {stratum:<40} {n_h:>4} drawn / target {result['targets'][stratum]:>4} "
              f"/ population {result['stratum_populations'][stratum]}")
    return 0


def _cmd_ingest_sheet(args: argparse.Namespace) -> int:
    result = eval_batch.ingest_sheet(
        sheet=args.sheet, batch_id=args.batch_id, reviewer=args.reviewer,
        split_id=args.split_id, labels_csv=args.labels_csv,
    )
    print(f"Gold: {result['added_cases']} new case(s), {result['replaced_cases']} re-reviewed; "
          f"store now holds {result['total_rows']} row(s) across {result['total_cases']} case(s).")
    return 0


def _cmd_ingest_gold(args: argparse.Namespace) -> int:
    rows = pd.read_csv(args.rows_csv, dtype=str, keep_default_na=False)
    if "origin" in rows.columns:
        if args.origin is not None and not (rows["origin"] == args.origin).all():
            print(f"ERROR: --origin {args.origin!r} conflicts with the 'origin' column already "
                  f"in {args.rows_csv} — pass one or the other, not both.", file=sys.stderr)
            return 1
    else:
        if args.origin is None:
            print(f"ERROR: {args.rows_csv} has no 'origin' column; pass --origin.", file=sys.stderr)
            return 1
        rows["origin"] = args.origin
    result = gold.ingest_gold(
        rows, reviewer=args.reviewer, source_path=args.rows_csv, labels_csv=args.labels_csv,
        batch_or_export_id=args.batch_or_export_id, upload_period=args.upload_period, slice_rate=args.slice_rate,
    )
    print(f"Gold: {result['added_cases']} new case(s), {result['replaced_cases']} re-reviewed; "
          f"store now holds {result['total_rows']} row(s) across {result['total_cases']} case(s).")
    return 0


def _cmd_cause_sheet(args: argparse.Namespace) -> int:
    verdicts = pd.read_csv(args.verdicts_csv, dtype=str, keep_default_na=False)
    result = cause_pass.build_misses_sheet(verdicts, out_csv=args.out_csv)
    print(f"Wrote {result['rows']} miss(es) to {args.out_csv}")
    return 0


def _cmd_ingest_cause(args: argparse.Namespace) -> int:
    result = cause_pass.ingest_cause_sheet(args.filled_csv, reviewer=args.reviewer)
    print(f"Ingested {result['ingested']} row(s): {result['added']} new, {result['replaced']} re-reviewed. "
          f"Cause store now holds {result['total_rows']} row(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tier3-sample", help="Draw a row-level Tier-3 audit sample.")
    p.add_argument("--silver-id", required=True, help="Silver generation id, loaded via diagnosis_mapping.silver.load_silver.")
    p.add_argument("--test-cases-txt", required=True)
    p.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--n-rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-cases", nargs="*", default=[])

    p = sub.add_parser("tier3-pilot", help="Split a review CSV into a stratified pilot + remainder.")
    p.add_argument("--review-csv", default=str(config.TIER3_AUDIT_REVIEW_CSV),
                   help="Point this at the real legacy sheet to split it unchanged, e.g. "
                        "ml/output/annotation/tier3_audit_review.csv.")
    p.add_argument("--key-csv", default=str(config.TIER3_AUDIT_KEY_CSV),
                   help="The key CSV written alongside --review-csv, e.g. "
                        "ml/output/annotation/tier3_audit_key.csv.")
    p.add_argument("--out-pilot", default=str(config.TIER3_AUDIT_PILOT_CSV))
    p.add_argument("--out-remainder", default=str(config.TIER3_AUDIT_REMAINDER_CSV))
    p.add_argument("--out-instructions", default=str(config.TIER3_AUDIT_PILOT_INSTRUCTIONS_MD))
    p.add_argument("--n-rows", type=int, default=30)
    p.add_argument("--min-per-stratum", type=int, default=3)

    p = sub.add_parser("tier3-ingest", help="Merge a filled Tier-3 review CSV into the audit store.")
    p.add_argument("--review-csv", default=str(config.TIER3_AUDIT_REVIEW_CSV),
                   help="The filled sheet to ingest. For the already-issued legacy batch, point this at "
                        "ml/output/annotation/tier3_audit_review.csv (or its pilot/remainder variant).")
    p.add_argument("--key-csv", default=str(config.TIER3_AUDIT_KEY_CSV),
                   help="The key CSV written alongside --review-csv by the legacy or new `sample`, e.g. "
                        "ml/output/annotation/tier3_audit_key.csv.")
    p.add_argument("--batch", type=int, required=True,
                   help="Which batch this filled sheet is (no default: ingesting under the wrong "
                        "batch number silently mislabels every row's audit-store 'batch' column).")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--labels-csv", default=str(config.LABELS_CSV))

    p = sub.add_parser("eval-batch", help="Draw one batch of the case-level eval-batch series.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--silver-id", required=True, help="Silver generation id, loaded via diagnosis_mapping.silver.load_silver.")
    p.add_argument("--fraction", type=float, required=True,
                   help="Share of each stratum's REMAINING target to draw this call (e.g. 0.5).")
    p.add_argument("--split-id", default=config.DEFAULT_SPLIT_ID)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-codes-per-group", type=int, default=30)
    p.add_argument("--big-group-share", type=float, default=0.01)
    p.add_argument("--rare-stratum-n", type=int, default=2)
    p.add_argument("--no-cancer-target", type=int, default=100)

    p = sub.add_parser("ingest-sheet", help="Ingest a filled eval-batch sheet into the gold store.")
    p.add_argument("--sheet", required=True)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--split-id", default=None, help="Defaults to the split_id the ledger recorded for --batch-id.")
    p.add_argument("--labels-csv", default=str(config.LABELS_CSV))

    p = sub.add_parser("ingest-gold", help="Ingest case-level gold rows directly (bypassing a sheet).")
    p.add_argument("--rows-csv", required=True, help="CSV with columns case_id, origin, and term and/or code.")
    p.add_argument("--origin", default=None,
                   help="Origin for every row, if --rows-csv has no origin column. Refused if it "
                        "conflicts with an 'origin' column already present.")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p.add_argument("--batch-or-export-id", default="")
    p.add_argument("--upload-period", default="")
    p.add_argument("--slice-rate", default="")

    p = sub.add_parser("cause-sheet", help="Build a misses sheet from a verdict table.")
    p.add_argument("--verdicts-csv", required=True, help="CSV with case_id, gold_code, method, source_version.")
    p.add_argument("--out-csv", required=True)

    p = sub.add_parser("ingest-cause", help="Ingest a filled cause-pass sheet.")
    p.add_argument("--filled-csv", required=True)
    p.add_argument("--reviewer", required=True)

    args = parser.parse_args()
    dispatch = {
        "tier3-sample": _cmd_tier3_sample,
        "tier3-pilot": _cmd_tier3_pilot,
        "tier3-ingest": _cmd_tier3_ingest,
        "eval-batch": _cmd_eval_batch,
        "ingest-sheet": _cmd_ingest_sheet,
        "ingest-gold": _cmd_ingest_gold,
        "cause-sheet": _cmd_cause_sheet,
        "ingest-cause": _cmd_ingest_cause,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
