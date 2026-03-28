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
    annotation_csv: str = config.KEYWORD_ANNOTATION_CSV,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a case-level train/test split."
    )
    parser.add_argument("--annotation-csv", default=config.KEYWORD_ANNOTATION_CSV)
    parser.add_argument("--reports-csv", default=config.REPORTS_CSV)
    parser.add_argument("--train-out", default=config.TRAIN_CASES_TXT)
    parser.add_argument("--test-out", default=config.TEST_CASES_TXT)
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of cases to hold out as test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_split(
        annotation_csv=args.annotation_csv,
        reports_csv=args.reports_csv,
        train_out=args.train_out,
        test_out=args.test_out,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
