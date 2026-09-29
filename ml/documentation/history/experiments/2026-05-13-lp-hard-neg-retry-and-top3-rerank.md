# LP hard-negative retry, threshold re-sweeps and top-3 group rerank

**Date:** 2026-05-13 · **Verdict:** reverted

## Hypothesis
The 16% Slight bucket was the biggest accessible lever on the concat-3 stack. Retrain the LPs with milder hard negatives (fraction 0.3), re-sweep per-LP thresholds (F1, then F0.5), and let stage 3 choose among the top three groups instead of only the top one.

## Setup
- Baseline: promoted four-stage `ml/` (concat-3, per-section backbone, 2304-dim gate, per-LP thresholds, tail gate K=2 gap 0.08), eval half of 5,833 cases, 4,414 prediction rows.
- Four experiments on top of it, each restored afterwards.

## Result
| Experiment | Good | Slight | CO | FP | FN | G+S | vs 62.1% |
|---|---|---|---|---|---|---|---|
| Baseline | 46.1% | 16.0% | 14.7% | 2.3% | 20.8% | 62.1% | n/a |
| LP hard-neg 0.3, F1 sweep | 46.4% | 14.9% | 15.2% | 2.2% | 21.4% | 61.3% | -0.8 |
| LP hard-neg 0.3, F0.5 sweep | 46.4% | 14.7% | 15.0% | 2.1% | 21.8% | 61.1% | -1.0 |
| Top-3 rerank, K=3, gap 1.0 | 44.9% | 16.1% | 19.5% | 2.3% | 17.1% | 61.0% | -1.1 |
| Top-3 rerank, K=3, gap 0.08 | 46.0% | 16.1% | 15.8% | 2.3% | 19.8% | 62.1% | 0.0 |

## Why
- Hard negatives improved within-group discrimination slightly (Slight -1.1 pp), but re-swept thresholds that were individually F1-optimal shifted cross-group routing: CO +0.5 pp and FN +0.6 pp ate the gain. A precision-weighted sweep made FN worse. This is the [QW1 story](2026-05-10-lp-hard-negatives-qw1-qw2.md) again at smaller size, now on the concat-3 backbone.
- The rerank could not attack CO as implemented: rank 0 was exempt from the LP filter ("never abstain on a high-confidence stage-2 case"), so opening to K=3 only appended tail predictions. The aggressive variant added 316 rows, of which 275 were CO and 86 Good.

## Don't retry unless…
For LP hard negatives: only with an architecture change (softmax over in-group labels, or backbone hard negatives, which also failed). For top-3 rerank: only after dropping the rank-0 exemption and re-ranking by `(lp_score - lp_threshold) x group_prob` across candidates.

## Where it went
Baseline restored. The `--beta` (F-beta) option added to the sweep script was removed with it when `calibrate.py` replaced the sweep tools. Two evaluator fixes made during this session (2304-dim dispatch in the gate evaluator; excluding an alias key in the group evaluator) were kept in the old tree and are gone with it.
