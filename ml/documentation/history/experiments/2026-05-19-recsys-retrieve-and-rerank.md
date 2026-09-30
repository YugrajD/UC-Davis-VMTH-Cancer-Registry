# Retrieve-and-rerank (ml-RecSys) replacing group and per-group LP stages

**Date:** 2026-05-19 · **Verdict:** reverted (folded after missing its exit criterion)

## Hypothesis
Replace `GroupClassifier -> per-group LabelPresence` with per-section max-cosine retrieval of the top K labels, then one global RankClassifier reranks them. Same stage 1 gate and stage 4 keyword correction. Run in an experimental sibling tree, since deleted.

## Setup
- Reused from `ml/`: backbone, splits, annotation and the 2304-dim gate (F1 0.92, AUC 0.98).
- Rank head: 25 epochs, batch 512, 10 hard and 10 random negatives per positive, recall weight 0.5 (validation F1 0.79).
- Baseline: `ml/` at G+S 62.1% (Good 46.1, Slight 16.0, CO 14.7, FP 2.3, FN 20.8; eval half).

## Result
| Config (eval half) | Good | G+S strict | CO | FN | Retrieval recall |
|---|---|---|---|---|---|
| ml/ baseline | 46.1% | 62.1% | 14.7% | 20.8% | n/a |
| K=20, threshold 0.995 | 35.7% | 47.4% | 15.8% | 35.0% | 0.748 |
| K=50, threshold 0.997 | 37.0% | 48.7% | 16.0% | 33.5% | 0.865 |

Slight is not recorded separately in the source (Slight = G+S - Good = 11.7 and 11.7 pp respectively, derived). Case-based lenient G+S was 61.9% (K=20) and 63.2% (K=50).

Calibration-fix attempts at K=50: dropping the weighted sampler with a boosted `pos_weight` gave peak F1 0.525 at t=0.99 against 0.532 with it (flat; the cliff is structural). A BPR pairwise loss reached pair accuracy 0.97-0.98 but strict G+S collapsed 48.7% to 34.7% and case-based 63.2% to 52.2%.

## Why
1. Retrieval ceiling: 25% of true positives never entered rerank at K=20 (recall 0.865 at K=50).
2. Calibration cliff: BCE with hard negatives pushes logits to extremes, so F1 peaks at threshold 0.995+.
3. BPR ranks within a case well but has no cross-case calibration, so absolute thresholds fail.
4. The pre-set exit criterion ("still more than 5 pp behind after one more variant") was hit at 13.4 pp behind.

Some of the loss was an evaluation artefact: the "Unidentified Group" bucket (585 rows, 13.2% of the eval set) scored as CO instead of FN.

## Don't retry unless…
You have a fix for both retrieval recall and the cross-case calibration cliff. The old notes suggested K=30/50 (done) and a focal loss (not tried).

## Where it went
Nothing merged into `ml/`. Flags `--rank-no-sampler` and `--rank-loss bce|bpr` lived only in the sibling tree, which was deleted on 2026-09-14.
