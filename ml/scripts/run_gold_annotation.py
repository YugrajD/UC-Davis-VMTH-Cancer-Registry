"""Gold annotation tooling — sample, ingest, consistency, check-split.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Subcommands
-----------
  sample        Draw a stratified-random gold-EVAL review workbook (Excel).
  ingest        Read a filled review workbook and write the gold annotation store.
  consistency   Compute intra-annotator Cohen's kappa on duplicate rows.
  check-split   Assert gold IDs ⊆ test_cases.txt and ∩ train_cases.txt = ∅.

Usage
-----
  # Batch 1 (200 cases). Hand the .xlsx to the professional to fill in.
  python ml/scripts/run_gold_annotation.py sample
  python ml/scripts/run_gold_annotation.py sample --n-cases 200 --batch 1

  # Later batches exclude already-sampled cases:
  python ml/scripts/run_gold_annotation.py sample --batch 2 \\
      --exclude-cases ml/output/annotation/gold_eval_batch1_cases.txt

  python ml/scripts/run_gold_annotation.py ingest --verified-by "Dr. Smith"
  python ml/scripts/run_gold_annotation.py consistency
  python ml/scripts/run_gold_annotation.py check-split

After ingesting, evaluate inference against the gold store:
  python ml/scripts/run_evaluation.py \\
      --annotation-csv ml/output/annotation/gold_annotation.csv \\
      --test-cases ml/output/annotation/gold_eval_batch1_cases.txt
"""

import sys
from pathlib import Path

# Add ml/ to sys.path so all packages are importable without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import config
from annotation.gold.sample import sample
from annotation.gold.ingest import ingest
from annotation.gold.consistency import consistency
from annotation.gold.check_split import check_split


def _batch_ledger_path(batch: int) -> str:
    return f"{config.ANNOTATION_DIR}/gold_eval_batch{batch}_cases.txt"


def _cmd_sample(args: argparse.Namespace) -> int:
    ledger = args.batch_cases_out or _batch_ledger_path(args.batch)
    sample(
        annotation_csv=args.annotation_csv,
        test_cases_txt=args.test_cases,
        reports_csv=args.reports_csv,
        labels_csv=args.labels_csv,
        out_xlsx=args.out_xlsx,
        batch_cases_out=ledger,
        n_cases=args.n_cases,
        dup_frac=args.dup_frac,
        seed=args.seed,
        exclude_cases=args.exclude_cases,
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    ingest(
        review_xlsx=args.review_xlsx,
        out_csv=args.out_csv,
        verified_by=args.verified_by,
        labels_csv=args.labels_csv,
        provenance=args.provenance,
    )
    return 0


def _cmd_consistency(args: argparse.Namespace) -> int:
    consistency(review_xlsx=args.review_xlsx)
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
        description="Gold annotation tooling: sample, ingest, consistency, check-split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------
    p_sample = sub.add_parser(
        "sample",
        help="Draw a stratified-random gold-EVAL review workbook from test_cases.txt.",
    )
    p_sample.add_argument("--annotation-csv", default=config.ANNOTATION_CSV,
                          help=f"Silver annotation CSV to join cascade predictions from "
                               f"(default: {config.ANNOTATION_CSV})")
    p_sample.add_argument("--test-cases", default=config.TEST_CASES_TXT,
                          help=f"One case_id per line; the sampling pool "
                               f"(default: {config.TEST_CASES_TXT})")
    p_sample.add_argument("--reports-csv", default=config.REPORTS_CSV,
                          help=f"Report CSV for reviewer context columns "
                               f"(default: {config.REPORTS_CSV})")
    p_sample.add_argument("--labels-csv", default=config.LABELS_CSV,
                          help=f"Taxonomy CSV for dropdown lists "
                               f"(default: {config.LABELS_CSV})")
    p_sample.add_argument("--out-xlsx", default=config.GOLD_EVAL_REVIEW_XLSX,
                          help=f"Destination review workbook "
                               f"(default: {config.GOLD_EVAL_REVIEW_XLSX})")
    p_sample.add_argument("--batch", type=int, default=1,
                          help="Batch number (used for the cases-ledger filename; default: 1)")
    p_sample.add_argument("--batch-cases-out", default=None,
                          help="Where to write the sampled case-ID ledger "
                               "(default: ml/output/annotation/gold_eval_batch<N>_cases.txt)")
    p_sample.add_argument("--exclude-cases", nargs="*", default=[],
                          help="Ledger file(s) of case IDs from earlier batches to exclude.")
    p_sample.add_argument("--n-cases", type=int, default=200,
                          help="Target number of unique cases to sample (default: 200)")
    p_sample.add_argument("--dup-frac", type=float, default=0.08,
                          help="Fraction of cases re-listed as duplicates for "
                               "self-consistency (default: 0.08)")
    p_sample.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility (default: 42)")

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    p_ingest = sub.add_parser(
        "ingest",
        help="Read a filled review workbook and write the gold annotation store.",
    )
    p_ingest.add_argument("--review-xlsx", default=config.GOLD_EVAL_REVIEW_XLSX,
                          help=f"Filled review workbook produced by `sample` "
                               f"(default: {config.GOLD_EVAL_REVIEW_XLSX})")
    p_ingest.add_argument("--out-csv", default=config.GOLD_ANNOTATION_CSV,
                          help=f"Destination gold annotation CSV "
                               f"(default: {config.GOLD_ANNOTATION_CSV})")
    p_ingest.add_argument("--labels-csv", default=config.LABELS_CSV,
                          help=f"Taxonomy CSV used to derive ICD-O codes "
                               f"(default: {config.LABELS_CSV})")
    p_ingest.add_argument("--verified-by", required=True,
                          help="Name or identifier of the professional who confirmed the labels.")
    p_ingest.add_argument("--provenance", default="round0",
                          help="Round/provenance tag written to every gold row (default: round0)")

    # ------------------------------------------------------------------
    # consistency
    # ------------------------------------------------------------------
    p_cons = sub.add_parser(
        "consistency",
        help="Compute intra-annotator Cohen's kappa on duplicate rows in a filled workbook.",
    )
    p_cons.add_argument("--review-xlsx", default=config.GOLD_EVAL_REVIEW_XLSX,
                        help=f"Filled review workbook with both dup_pass=1 and =2 rows "
                             f"(default: {config.GOLD_EVAL_REVIEW_XLSX})")

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
        "consistency":  _cmd_consistency,
        "check-split":  _cmd_check_split,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    raise SystemExit(main())
