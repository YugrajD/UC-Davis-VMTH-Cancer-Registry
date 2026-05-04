# Ideas to Try — 3-Stage Pipeline Improvement Plan

Structured improvement plan for maximising Good+Slightly off on the 3-stage pipeline
(CasePresenceClassifier → GroupClassifier → KW correction).

**Baseline (Phase 25, per-label evaluation, 2026-05-02, test set):**
G+S = 51.8% | CO = 19.3% | FP = 4.7% | FN = 24.1% | Total = 8,744 rows

**Current best (Phase 26, 2026-05-04, gate=0.5, group-t=0.85, argmax fallback, subtype KW, GroupCLF F1=0.4335):**
G+S = 54.6% | CO = 22.3% | FP = 5.0% | FN = 18.2% | Total = 9,127 rows

Run `run_evaluation.py` with `--test-cases ml/output/splits/test_cases.txt` after each tier
to record the marginal change before moving on.

---

## Tier 4 — Backbone Adaptation (high effort)

**Attempt only after Tiers 1–3 are exhausted. Tiers 1–3 are complete — see ideas-accepted.md.**

### 4a — Round 3 backbone adaptation (hard-negative mining from CO bank)

**Status:** Not started

**Problem:** GroupClassifier CO% (19.3%) reflects cases where the backbone embedding space
does not separate adjacent groups well enough. A third round of InfoNCE + hard-negative
margin loss, using CO cases from the current pipeline as hard negatives, would tighten the
clusters where GroupClassifier currently fails.

**Approach:**
1. Collect CO predictions from current best pipeline into `hard_neg_pairs.csv`
2. Run `--mode adapt-backbone` with `--hard-neg-csv` pointing to those pairs
3. Cold-start the GroupClassifier on the new embedding space (delete cache first)
4. Retrain CasePresenceClassifier to match the new backbone

**Risk:** Medium — requires full cold start; GroupClassifier must be retrained from scratch.
Back up all current checkpoints before starting.

---

## Evaluation Reference

Standard test-set evaluation command after any change:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
  --test-cases ml/output/splits/test_cases.txt \
  --out-dir ml/output/evaluation/contrastive_test \
  --label "<tier and description>"
```

Results append to `ml/output/evaluation/contrastive_test/evaluation_history.csv`.

**Baseline to beat:** G+S = 54.6% (Phase 26, 2026-05-04, per-label evaluation).
