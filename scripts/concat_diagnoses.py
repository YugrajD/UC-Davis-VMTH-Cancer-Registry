"""
Concatenate clinical diagnoses rows by patient ID and index as visits.

Usage:
    python scripts/concat_diagnoses.py INPUT.csv OUTPUT.csv
"""

import csv
import sys
from collections import OrderedDict


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/concat_diagnoses.py INPUT.csv OUTPUT.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Group rows by anon_id, preserving insertion order
    patients: OrderedDict[str, list[str]] = OrderedDict()

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            anon_id = row["anon_id"]
            diagnosis = row["Clinical Diagnoses"]
            patients.setdefault(anon_id, []).append(diagnosis)

    # Write output: one row per patient, diagnoses concatenated with visit index
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["anon_id", "Clinical Diagnoses"])

        for anon_id, diagnoses in patients.items():
            if len(diagnoses) == 1:
                combined = diagnoses[0]
            else:
                parts = [f"[VISIT {i}] {d}" for i, d in enumerate(diagnoses, start=1)]
                combined = ", ".join(parts)
            writer.writerow([anon_id, combined])

    # Summary
    total_rows = sum(len(d) for d in patients.values())
    multi_visit = sum(1 for d in patients.values() if len(d) > 1)
    print(f"Processed {total_rows} rows across {len(patients)} unique patients")
    print(f"  {multi_visit} patients have multiple visits")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
