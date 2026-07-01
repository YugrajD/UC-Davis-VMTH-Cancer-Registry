"""Guard the train/eval split for the gold annotation store.

Asserts:
  (a) every gold case ID is in test_cases.txt
  (b) no gold case ID is in train_cases.txt

Prints PASS or FAIL with any offending case IDs.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _load_ids(txt_path: str) -> set[str]:
    with open(txt_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _load_gold_case_ids(gold_csv: str) -> set[str]:
    ids: set[str] = set()
    with open(gold_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id", "").strip()
            if cid:
                ids.add(cid)
    return ids


def check_split(
    gold_csv: str,
    test_cases_txt: str,
    train_cases_txt: str,
) -> bool:
    """Return True if all assertions pass."""
    gold_ids = _load_gold_case_ids(gold_csv)
    test_ids = _load_ids(test_cases_txt)
    train_ids = _load_ids(train_cases_txt)

    not_in_test = gold_ids - test_ids
    in_train = gold_ids & train_ids

    passed = True

    if not_in_test:
        passed = False
        print(f"FAIL: {len(not_in_test)} gold case(s) NOT in test_cases.txt:")
        for cid in sorted(not_in_test)[:20]:
            print(f"  {cid}")
        if len(not_in_test) > 20:
            print(f"  ... and {len(not_in_test) - 20} more")
    else:
        print(f"PASS: all {len(gold_ids)} gold cases are in test_cases.txt")

    if in_train:
        passed = False
        print(f"FAIL: {len(in_train)} gold case(s) appear in train_cases.txt (split contamination):")
        for cid in sorted(in_train)[:20]:
            print(f"  {cid}")
        if len(in_train) > 20:
            print(f"  ... and {len(in_train) - 20} more")
    else:
        print(f"PASS: no gold cases overlap with train_cases.txt")

    if not passed:
        sys.exit(1)
    return True
