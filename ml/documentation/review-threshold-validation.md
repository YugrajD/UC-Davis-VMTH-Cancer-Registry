# Review Threshold Validation

This document records how the per-diagnosis manual-review thresholds were
calibrated for the backend ingestion gate in `backend/app/config.py`.

## Current Thresholds

| Setting | Value | Meaning |
|---|---:|---|
| `REVIEW_AUTO_ACCEPT_CONFIDENCE` | `0.23` | Predictions below this confidence are routed to manual review |
| `REVIEW_AUTO_ACCEPT_MARGIN` | `0.15` | Rank-1 predictions with top-1 minus top-2 below this value are routed to manual review |

At ingest time, `backend/app/services/ingestion_service.py` auto-confirms a
prediction only when the confidence gate passes and the margin gate does not
flag the row. Rows emitted as `low_confidence` by the ML pipeline always remain
pending.

## Validation Data

The checked-in seed sample lives at:

```text
ml/data/validation/review_threshold_validation.csv
```

The sample was derived from the existing evaluated PetBERT output:

```text
ml/output/finetuned_eval/evaluation.csv
ml/output/finetuned_eval/petbert_predictions.csv
```

The seed contains 189 labelled prediction rows. It includes up to 25 examples
from each lower confidence bin and all examples at or above `0.20`, so the
high-confidence auto-accept band is not under-sampled. `ground_truth_correct`
is mapped from the existing evaluation verdict:

- `good` and `slightly_off` are treated as correct for review-threshold
  calibration because they preserve the clinically relevant cancer group.
- `completely_off` and `false_positive` are treated as incorrect.
- False-negative rows without a PetBERT prediction are excluded because the
  ingest review gate only operates on emitted predictions.

This seed is useful for validating the workflow and initial defaults. Replace
or extend it with clinician-reviewed rows when a hand-labelled sample is
available.

## Reproduce The Analysis

Build the seed sample again:

```bash
python3 scripts/analyze_review_thresholds.py build-sample
```

Run the threshold analysis:

```bash
python3 scripts/analyze_review_thresholds.py analyze
```

The script reports:

- precision by confidence bin
- cumulative precision at or above each confidence threshold
- precision by top-1/top-2 margin bucket
- recommended config values for the backend review gate

## Seed Results

The seed sample produced these confidence-bin results:

| Confidence bin | Rows | Correct | Precision |
|---|---:|---:|---:|
| `0.00-0.05` | 25 | 0 | 0.0% |
| `0.05-0.10` | 25 | 2 | 8.0% |
| `0.10-0.15` | 25 | 5 | 20.0% |
| `0.15-0.20` | 25 | 9 | 36.0% |
| `0.20-0.25` | 62 | 50 | 80.6% |
| `0.25-0.30` | 27 | 27 | 100.0% |

Cumulative precision crosses the 95% auto-accept target at `0.23`:

| Threshold | Rows at/above | Correct | Precision |
|---|---:|---:|---:|
| `0.19` | 90 | 77 | 85.6% |
| `0.20` | 89 | 77 | 86.5% |
| `0.22` | 71 | 63 | 88.7% |
| `0.23` | 56 | 54 | 96.4% |
| `0.24` | 41 | 41 | 100.0% |

The margin analysis supports a conservative `0.15` gate:

| Margin bucket | Rows | Correct | Precision |
|---|---:|---:|---:|
| `<0.05` | 14 | 3 | 21.4% |
| `0.05-0.10` | 20 | 7 | 35.0% |
| `>=0.10` | 49 | 39 | 79.6% |

The script's margin recommendation considers candidate cutoffs and chooses the
largest threshold where below-threshold rows remain low precision while enough
above-threshold rows remain to compare. On this seed, `0.15` is the first
candidate where the above-margin band reaches roughly the review-priority target.

## Applying Threshold Changes To Existing Rows

New ingests use `backend/app/config.py`. Existing unreviewed diagnoses can be
re-flagged with:

```text
database/scripts/reflag_review_thresholds.sql
```

The SQL script intentionally updates only system-managed diagnoses:

- `review_status` is `pending` or `confirmed`
- `reviewed_by_email` is null
- `reviewed_at` is null

It inserts `diagnosis_review_events` rows for every status change and leaves
`corrected`, `rejected`, and human-reviewed diagnoses untouched.

## Future Validation

When clinician-reviewed labels are available, keep the same CSV schema and
replace the seed rows or append a larger labelled sample. Re-run the analysis
after any material change to:

- PetBERT checkpoint or classifier architecture
- taxonomy size or grouping
- prediction confidence semantics
- report text columns used by production inference
- review policy, such as requiring exact term correctness instead of
  group-level correctness
