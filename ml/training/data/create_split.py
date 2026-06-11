"""Generate a case-level train/test split for the cancer registry dataset.

Run once before starting a new training run. The split files are stable inputs —
do not regenerate them unless you explicitly want a new random split, as that
would invalidate any previously trained checkpoints.

Cancer-positive cases are split with stratification by label group so that every
group has both train and test examples. Non-cancer cases are split randomly.

Outputs (one case_id per line, sorted):
  ml/output/splits/train_cases.txt
  ml/output/splits/test_cases.txt

Usage:
  python ml/training/data/create_split.py
  python ml/training/data/create_split.py --test-frac 0.2 --seed 42
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def create_split(
    *,
    annotation_csv: str = config.ANNOTATION_CSV,
    reports_csv: str = config.REPORTS_CSV,
    train_out: str = config.TRAIN_CASES_TXT,
    test_out: str = config.TEST_CASES_TXT,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    """Generate train/test split files. Returns (n_train, n_test)."""
    random.seed(seed)

    # --- Cancer-positive cases grouped by label group -----------------------
    case_to_group: dict[str, str] = {}
    with open(annotation_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            grp = row.get("matched_group", "").strip()
            if grp:
                case_to_group.setdefault(row["case_id"].strip(), grp)

    group_to_cases: dict[str, list[str]] = {}
    for cid, grp in case_to_group.items():
        group_to_cases.setdefault(grp, []).append(cid)

    train_ids: list[str] = []
    test_ids: list[str] = []

    print("Cancer-positive cases (stratified by group):")
    for grp in sorted(group_to_cases):
        cases = group_to_cases[grp]
        random.shuffle(cases)
        n_test = max(1, round(len(cases) * test_frac))
        test_ids.extend(cases[:n_test])
        train_ids.extend(cases[n_test:])
        print(f"  {grp:<50} total={len(cases):>4}  train={len(cases)-n_test:>4}  test={n_test:>3}")

    # --- Non-cancer cases (all cases not in annotation) ---------------------
    with open(reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [c.lstrip("\ufeff").lstrip("ï»¿") for c in reader.fieldnames]
        all_case_ids = {row["case_id"].strip() for row in reader if row.get("case_id", "").strip()}

    neg_cases = sorted(all_case_ids - set(case_to_group.keys()))
    random.shuffle(neg_cases)
    n_test_neg = round(len(neg_cases) * test_frac)
    test_ids.extend(neg_cases[:n_test_neg])
    train_ids.extend(neg_cases[n_test_neg:])
    print(f"\nNon-cancer cases:                                          total={len(neg_cases):>4}  train={len(neg_cases)-n_test_neg:>4}  test={n_test_neg:>3}")

    # --- Write output -------------------------------------------------------
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(train_ids)) + "\n")
    with open(test_out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(test_ids)) + "\n")

    n_train, n_test = len(train_ids), len(test_ids)
    total = n_train + n_test
    print(f"\nTotal: {total}  ->  train={n_train} ({n_train/total*100:.0f}%)  test={n_test} ({n_test/total*100:.0f}%)")
    print(f"Wrote {train_out}")
    print(f"Wrote {test_out}")
    return n_train, n_test


def create_temporal_split(
    *,
    cutoff_year: int,
    annotation_csv: str = config.ANNOTATION_CSV,
    reports_csv: str = config.REPORTS_CSV,
    demographics_csv: str = config.DEMOGRAPHICS_CSV,
    train_out: str = config.TRAIN_CASES_TEMPORAL_TXT,
    test_out: str = config.TEST_CASES_TEMPORAL_TXT,
) -> tuple[int, int]:
    """Generate a pure temporal holdout split keyed on demographics DtOfRq.

    Cases whose report year (DtOfRq) is >= cutoff_year go to test, all others to
    train. No stratification: rare label groups may end up train-only or
    test-sparse (per-group test coverage is printed so this is visible).

    Cases absent from demographics or with an unparseable DtOfRq fall back to
    train (their recency cannot be validated, so they cannot serve as a recency
    test target).
    """
    # --- Report year per case_id (DtOfRq is ISO "YYYY-MM-DD") ----------------
    case_to_year: dict[str, int] = {}
    with open(demographics_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id", "").strip()
            dt = row.get("DtOfRq", "").strip()
            if cid and len(dt) >= 4 and dt[:4].isdigit():
                case_to_year[cid] = int(dt[:4])

    # --- Label group per case_id (for coverage reporting) --------------------
    case_to_group: dict[str, str] = {}
    with open(annotation_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            grp = row.get("matched_group", "").strip()
            if grp:
                case_to_group.setdefault(row["case_id"].strip(), grp)

    # --- All case_ids from reports -------------------------------------------
    with open(reports_csv, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [c.lstrip("﻿").lstrip("ï»¿") for c in reader.fieldnames]
        all_case_ids = {row["case_id"].strip() for row in reader if row.get("case_id", "").strip()}

    train_ids: list[str] = []
    test_ids: list[str] = []
    missing_date = 0
    for cid in all_case_ids:
        year = case_to_year.get(cid)
        if year is None:
            missing_date += 1
            train_ids.append(cid)
        elif year >= cutoff_year:
            test_ids.append(cid)
        else:
            train_ids.append(cid)

    # --- Per-group test coverage ---------------------------------------------
    test_set = set(test_ids)
    group_test_counts: dict[str, int] = {}
    group_total_counts: dict[str, int] = {}
    for cid, grp in case_to_group.items():
        group_total_counts[grp] = group_total_counts.get(grp, 0) + 1
        if cid in test_set:
            group_test_counts[grp] = group_test_counts.get(grp, 0) + 1

    cancer_test = sum(1 for cid in test_ids if cid in case_to_group)
    print(f"Temporal split: cutoff_year={cutoff_year} (test = report year >= {cutoff_year})")
    if missing_date:
        print(f"  {missing_date} case(s) had no parseable DtOfRq -> assigned to train")
    print(f"\nPer-group TEST coverage (test_count / total_annotated):")
    for grp in sorted(group_total_counts):
        tc = group_test_counts.get(grp, 0)
        flag = "  <-- no test examples" if tc == 0 else ("  <-- sparse" if tc < 5 else "")
        print(f"  {grp:<55} {tc:>4} / {group_total_counts[grp]:>5}{flag}")

    # --- Write output --------------------------------------------------------
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(train_ids)) + "\n")
    with open(test_out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(test_ids)) + "\n")

    n_train, n_test = len(train_ids), len(test_ids)
    total = n_train + n_test
    print(f"\nTotal: {total}  ->  train={n_train} ({n_train/total*100:.0f}%)  "
          f"test={n_test} ({n_test/total*100:.0f}%, {cancer_test} cancer-positive)")
    print(f"Wrote {train_out}")
    print(f"Wrote {test_out}")
    return n_train, n_test


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a case-level train/test split."
    )
    parser.add_argument("--annotation-csv", default=config.ANNOTATION_CSV)
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV)
    parser.add_argument("--train-out", default=None,
                        help="Override train output path (defaults depend on split mode).")
    parser.add_argument("--test-out", default=None,
                        help="Override test output path (defaults depend on split mode).")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of cases to hold out as test set (random mode only; default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-cutoff-year", type=int, default=None,
                        help="Enable pure temporal holdout: cases with DtOfRq year >= this "
                             "go to test, rest to train. Writes to the temporal split files.")
    parser.add_argument("--demographics-csv", default=config.DEMOGRAPHICS_CSV,
                        help="Date source for temporal mode (DtOfRq column).")
    args = parser.parse_args()

    if args.temporal_cutoff_year is not None:
        create_temporal_split(
            cutoff_year=args.temporal_cutoff_year,
            annotation_csv=args.annotation_csv,
            reports_csv=args.reports_csv,
            demographics_csv=args.demographics_csv,
            train_out=args.train_out or config.TRAIN_CASES_TEMPORAL_TXT,
            test_out=args.test_out or config.TEST_CASES_TEMPORAL_TXT,
        )
    else:
        create_split(
            annotation_csv=args.annotation_csv,
            reports_csv=args.reports_csv,
            train_out=args.train_out or config.TRAIN_CASES_TXT,
            test_out=args.test_out or config.TEST_CASES_TXT,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
