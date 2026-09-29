# Recency sample-weighting on a temporal holdout

**Date:** 2026-05-29 · **Verdict:** inconclusive (weak signal, not promoted)

## Hypothesis
Recent pathology reports better represent the future reports the system will code, so weighting training examples by recency (`w = 0.5^(age_years / H)`, normalised to mean 1, applied in the BCE loss of all three heads) should improve accuracy on new reports.

## Setup
- Pure temporal holdout: train on reports up to 2024, test on 2025 (1,408 cases, 634 cancer-positive), because a random split would hide any recency effect.
- Baseline: uniform weights. H=7 and H=3 years compared; tail gate on.
- Caveat: the production backbone had been adapted on the random split, so it saw 2025 pairs (mild leakage constant across arms); absolute 2025 numbers are not comparable with 62.1%, only the deltas are.

## Result
| 2025 test | Good | Slight | G+S | Off | FP | FN |
|---|---|---|---|---|---|---|
| Uniform | 46.0% | 16.8% | 62.8% | 16.8% | 3.0% | 17.4% |
| H=7 | 48.4% | 15.9% | 64.3% | 13.7% | 2.9% | 19.2% |
| H=3 | 48.6% | 14.5% | 63.1% | 12.4% | 3.4% | 21.1% |

## Why
- The mechanism is real and consistent: recency shifts Off toward Good. As H falls Off drops (16.8 to 12.4) and Good rises, but Slight falls and FN rises.
- The net G+S gain (+0.3 to +1.5 pp) is inside the noise: the 2025 slice gives about a plus or minus 3-4 pp interval.
- H could not be tuned reliably: the winner on a 2024 validation slice (H=3) generalised worse on 2025 (63.1%) than H=7 (64.3%). H=3 also starved rare classes (group-head effective sample size 26%; Malignant lymphomas LP head 14.2%, below the 25% guard). H=10 (ESS 72%) and H=20 (90%) were safe.

## Don't retry unless…
You have a larger and more recent holdout, and then only with a moderate half-life (about 7-10 years), never H=3.

## Where it went
Never promoted. The `--recency-half-life` flag, temporal-split tooling and scratch checkpoints were removed with the old tree or pruned on 2026-09-14. The `legacy-temporal` split (56,905 train / 1,408 test) survives in `ml/output/splits/`.
