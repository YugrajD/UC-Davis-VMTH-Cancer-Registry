"""Quick post-inference sanity checks for the 36-year archive predictions.

Reads ml/output/production/historical_archive/petbert_predictions.csv and
ml/data/report_36yr.csv, prints:
  1. Method distribution (embedding / low_confidence / knn_fallback)
  2. Top-1 predicted group distribution overall and per decade
  3. Confidence distribution by decade
  4. Random spot-check cases per decade — case_id, FINAL COMMENT excerpt,
     top-1 prediction. Eyeball these.
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="ml/output/production/historical_archive/petbert_predictions.csv",
    )
    parser.add_argument("--reports", default="ml/data/report_36yr.csv")
    parser.add_argument("--per-decade-spot", type=int, default=3,
                        help="Random spot-check cases per decade")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.predictions} ...")
    preds_all = pd.read_csv(args.predictions)
    print(f"  rows: {len(preds_all):,}")

    # Top-1 prediction per case (the highest-confidence row per case_id).
    top1 = (
        preds_all.sort_values(["case_id", "confidence"], ascending=[True, False])
        .drop_duplicates("case_id", keep="first")
        .reset_index(drop=True)
    )
    print(f"  unique cases: {len(top1):,}")

    print(f"Loading {args.reports} ...")
    reports = pd.read_csv(args.reports, low_memory=False)
    reports["year"] = pd.to_numeric(reports["year"], errors="coerce").fillna(0).astype(int)
    reports["decade"] = (reports["year"] // 10 * 10).astype(int)

    # Merge for per-decade slicing
    merged = top1.merge(
        reports[["case_id", "year", "decade", "FINAL COMMENT", "HISTOPATHOLOGICAL SUMMARY"]],
        on="case_id", how="left",
    )

    # ---- 1. Method distribution ----
    print()
    print("=== Method distribution (top-1 per case) ===")
    for method, n in Counter(merged["method"]).most_common():
        print(f"  {method:<20} {n:>6,}  ({n/len(merged)*100:.1f}%)")

    # ---- 2. Group distribution ----
    print()
    print("=== Top-1 group distribution (all years, top 15) ===")
    for g, n in Counter(merged["predicted_group"]).most_common(15):
        print(f"  {g:<45} {n:>6,}  ({n/len(merged)*100:.1f}%)")

    print()
    print("=== Top-5 groups per decade ===")
    for decade, sub in merged.groupby("decade"):
        if decade == 0:
            continue
        print(f"  {decade}s  (n={len(sub):,})")
        for g, n in Counter(sub["predicted_group"]).most_common(5):
            print(f"    {g:<43} {n:>5,}  ({n/len(sub)*100:.1f}%)")

    # ---- 3. Confidence by decade ----
    print()
    print("=== Confidence distribution by decade ===")
    print(f"  {'Decade':>7}  {'n':>6}  {'mean':>5}  {'median':>6}  {'<0.4':>5}  {'<0.4 %':>7}")
    for decade, sub in merged.groupby("decade"):
        if decade == 0:
            continue
        conf = pd.to_numeric(sub["confidence"], errors="coerce")
        below = (conf < 0.4).sum()
        print(f"  {decade:>5}s  {len(sub):>6,}  {conf.mean():.2f}  {conf.median():>6.2f}  "
              f"{below:>5,}  {below/len(sub)*100:>5.1f}%")

    # ---- 4. Spot-check ----
    print()
    print(f"=== Random spot-check ({args.per_decade_spot} cases per decade) ===")
    print("(Read the FINAL COMMENT excerpt and judge whether the predicted_term matches.)")
    for decade, sub in merged.groupby("decade"):
        if decade == 0:
            continue
        # Filter out Uncategorized for spot-checks (less interesting)
        sub_real = sub[sub["predicted_group"] != "Uncategorized"]
        if len(sub_real) == 0:
            continue
        picks = sub_real.sample(min(args.per_decade_spot, len(sub_real)),
                                random_state=args.seed + int(decade)).reset_index(drop=True)
        print()
        print(f"--- {decade}s ---")
        for _, r in picks.iterrows():
            fc = str(r.get("FINAL COMMENT", ""))[:300]
            print(f"  case {r['case_id']} (year {r['year']})  "
                  f"method={r['method']}  conf={r['confidence']:.2f}")
            print(f"    PREDICTED: {r['predicted_term']!r}  ({r['predicted_group']})")
            print(f"    REPORT:    {fc!r}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
