"""Tier-3 audit tooling — sample, ingest, check-split.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Subcommands
-----------
  sample        Draw a row-level Tier-2/Tier-3 audit review CSV (+ instructions
                and taxonomy sidecars).
  ingest        Read a filled review CSV and write the gold annotation store.
  check-split   Assert gold IDs ⊆ test_cases.txt and ∩ train_cases.txt = ∅.

Usage
-----
  # Batch 1 (200 rows). Hand the .csv (+ sidecars) to the professional to fill in.
  python ml/scripts/run_gold_annotation.py sample
  python ml/scripts/run_gold_annotation.py sample --n-rows 200 --batch 1

  # Later batches exclude cases already audited:
  python ml/scripts/run_gold_annotation.py sample --batch 2 \\
      --exclude-cases ml/output/annotation/tier3_audit_batch1_cases.txt

  python ml/scripts/run_gold_annotation.py ingest --verified-by "Dr. Smith"
  python ml/scripts/run_gold_annotation.py check-split

NOTE: this is a **row-level** sample, so it is deliberately NOT wired into
`run_evaluation.py`. evaluate.py scores per case — it compares predictions
against a case's whole annotated term set — and the un-sampled rows of a
partially-covered case would read as false positives. Score this store per row:
per-stratum rates, weighted by `sample_weight` to reach the Tier-3 population.
"""

import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import config
from annotation.gold.sample import sample
from annotation.gold.ingest import ingest
from annotation.gold.check_split import check_split


def _batch_ledger_path(batch: int) -> str:
    return f"{config.ANNOTATION_DIR}/tier3_audit_batch{batch}_cases.txt"


def _cmd_sample(args: argparse.Namespace) -> int:
    ledger = args.batch_cases_out or _batch_ledger_path(args.batch)
    sample(
        annotation_csv=args.annotation_csv,
        test_cases_txt=args.test_cases,
        labels_csv=args.labels_csv,
        out_csv=args.out_csv,
        out_instructions_md=args.out_instructions,
        out_taxonomy_csv=args.out_taxonomy,
        batch_cases_out=ledger,
        n_rows=args.n_rows,
        seed=args.seed,
        exclude_cases=args.exclude_cases,
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    ingest(
        review_csv=args.review_csv,
        out_csv=args.out_csv,
        verified_by=args.verified_by,
        labels_csv=args.labels_csv,
        provenance=args.provenance,
    )
    return 0


def _cmd_check_split(args: argparse.Namespace) -> int:
    check_split(
        gold_csv=args.gold_csv,
        test_cases_txt=args.test_cases,
        train_cases_txt=args.train_cases,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gold annotation tooling: sample, ingest, check-split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------
    p_sample = sub.add_parser(
        "sample",
        help="Draw a row-level Tier-2/Tier-3 audit review CSV from test_cases.txt.",
    )
    p_sample.add_argument("--annotation-csv", default=config.ANNOTATION_CSV,
                          help=f"Silver annotation CSV to join cascade predictions from "
                               f"(default: {config.ANNOTATION_CSV})")
    p_sample.add_argument("--test-cases", default=config.TEST_CASES_TXT,
                          help=f"One case_id per line; the sampling pool "
                               f"(default: {config.TEST_CASES_TXT})")
    p_sample.add_argument("--labels-csv", default=config.LABELS_CSV,
                          help=f"Taxonomy CSV the reference sidecar is built from "
                               f"(default: {config.LABELS_CSV})")
    p_sample.add_argument("--out-csv", default=config.TIER3_AUDIT_REVIEW_CSV,
                          help=f"Destination review CSV "
                               f"(default: {config.TIER3_AUDIT_REVIEW_CSV})")
    p_sample.add_argument("--out-instructions", default=config.TIER3_AUDIT_INSTRUCTIONS_MD,
                          help=f"Destination reviewer instructions "
                               f"(default: {config.TIER3_AUDIT_INSTRUCTIONS_MD})")
    p_sample.add_argument("--out-taxonomy", default=config.TIER3_AUDIT_TAXONOMY_CSV,
                          help=f"Destination (Group, Term, Code) reference "
                               f"(default: {config.TIER3_AUDIT_TAXONOMY_CSV})")
    p_sample.add_argument("--batch", type=int, default=1,
                          help="Batch number (used for the cases-ledger filename; default: 1)")
    p_sample.add_argument("--batch-cases-out", default=None,
                          help="Where to write the sampled case-ID ledger "
                               "(default: ml/output/annotation/tier3_audit_batch<N>_cases.txt)")
    p_sample.add_argument("--exclude-cases", nargs="*", default=[],
                          help="Ledger file(s) of case IDs from earlier batches to exclude.")
    p_sample.add_argument("--n-rows", type=int, default=200,
                          help="Target number of diagnosis rows to review (default: 200)")
    p_sample.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility (default: 42)")

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    p_ingest = sub.add_parser(
        "ingest",
        help="Read a filled review CSV and write the gold annotation store.",
    )
    p_ingest.add_argument("--review-csv", default=config.TIER3_AUDIT_REVIEW_CSV,
                          help=f"Filled review CSV produced by `sample` "
                               f"(default: {config.TIER3_AUDIT_REVIEW_CSV})")
    p_ingest.add_argument("--out-csv", default=config.GOLD_ANNOTATION_CSV,
                          help=f"Destination gold annotation CSV "
                               f"(default: {config.GOLD_ANNOTATION_CSV})")
    p_ingest.add_argument("--labels-csv", default=config.LABELS_CSV,
                          help=f"Taxonomy CSV used to derive ICD-O codes "
                               f"(default: {config.LABELS_CSV})")
    p_ingest.add_argument("--verified-by", required=True,
                          help="Name or identifier of the professional who confirmed the labels.")
    p_ingest.add_argument("--provenance", default="tier3_audit",
                          help="Round/provenance tag written to every gold row. Distinguishes these "
                               "row-level audit rows from any later per-case gold-eval batch "
                               "(default: tier3_audit)")

    # ------------------------------------------------------------------
    # check-split
    # ------------------------------------------------------------------
    p_check = sub.add_parser(
        "check-split",
        help="Assert gold case IDs are a subset of test_cases.txt and disjoint from train_cases.txt.",
    )
    p_check.add_argument("--gold-csv", default=config.GOLD_ANNOTATION_CSV,
                         help=f"Gold annotation CSV to validate "
                              f"(default: {config.GOLD_ANNOTATION_CSV})")
    p_check.add_argument("--test-cases", default=config.TEST_CASES_TXT,
                         help=f"test_cases.txt (default: {config.TEST_CASES_TXT})")
    p_check.add_argument("--train-cases", default=config.TRAIN_CASES_TXT,
                         help=f"train_cases.txt (default: {config.TRAIN_CASES_TXT})")

    args = parser.parse_args()

    dispatch = {
        "sample":       _cmd_sample,
        "ingest":       _cmd_ingest,
        "check-split":  _cmd_check_split,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    raise SystemExit(main())
