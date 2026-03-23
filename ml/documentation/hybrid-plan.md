# Hybrid Binary + KNN Group Selector Plan

## Context

The current production pipeline (Phase 16 binary classifier) achieves **41.9% Good+Slight** against keyword ground truth, and **37.8%** against the stricter LLM ground truth. The remaining error is split roughly:

| Error type | Rate (LLM GT) | Meaning |
|---|---|---|
| Completely off | 30.1% | Cancer detected, wrong group |
| False positive | 30.3% | No cancer, but classifier fired |
| False negative | 1.8% | Cancer missed entirely |

Two approaches were explored to improve group accuracy:

- **GroupClassifier MLP** — overfits at current data volume (~150 cases/group); macro F1 ≈ 0.09
- **KNN group selector** — no training, but no negatives in reference set; 53% FP at threshold=0.1

Neither works alone. The hybrid idea: use the **binary classifier to detect cancer and score labels**, and the **KNN to constrain which group those labels can come from**.

---

## Proposed Architecture

```
Per-column embeddings (2304-dim)
          │
          ├──► Binary Classifier ──► (N, M) presence score matrix
          │                           (845 labels scored per case)
          │
          └──► KNN Group Selector ──► (N, G) group vote fractions
                                       (top-K confirmed neighbours vote)
                           │
                           ▼
              For each case, restrict label candidates
              to groups with vote fraction ≥ threshold
                           │
                           ▼
              Pick highest binary score within those groups
                           │
                           ▼
              Final prediction (term + group + code)
```

**Why this might work:**
- Binary classifier provides a well-calibrated presence signal (trained against negatives)
- KNN redirects CO cases where binary picked the right label strength but wrong group
- Cases where KNN gives no group above threshold → Uncategorized (FP reduction)
- Non-cancer cases that the binary fires on might not cluster near any confirmed-cancer group in embedding space

**Why it might not:**
- KNN FP rate is high (53%); non-cancer cases still find cancer neighbours
- The 30% FP floor may be largely inherited from the binary gate
- Net improvement depends on how much CO↓ outweighs new KNN-induced FPs

---

## Implementation

### What already exists
- `ml/model/knn_group_selector.py` — `KnnGroupSelector.predict_proba(col_emb_concat)` returns `(N, G)`
- `ml/model/checkpoints/knn_group_selector.npz` — built from LLM predictions, k=10
- `ml/production/petbert_pipeline/pipeline.py` — `col_emb_concat` is now built unconditionally
- `ml/production/petbert_pipeline/categorization.py` — `run_categorization_group()` accepts `(N, G)` group_probs

### Changes needed

**1. New categorization function: `run_categorization_hybrid()`** in `categorization.py`

Accepts both `score_matrix` (N, M from binary) and `group_probs` (N, G from KNN). For each case:
- If no group vote ≥ threshold → Uncategorized
- Else mask the score matrix to only labels within voted groups, then pick argmax

**2. Pipeline wiring** in `pipeline.py`

Add a new branch:
```python
if config.presence_classifier_path and config.knn_group_selector_path:
    # Hybrid mode: binary scores + KNN group gating
    ...
    categorization = run_categorization_hybrid(...)
```

**3. New CLI flag** (optional): `--hybrid` or just detect both flags being set.

No changes to `ScanConfig` needed — both paths already exist as fields.

---

## Evaluation Results (2026-03-23) ❌

All runs against LLM ground truth. Baseline: binary-only Phase 16 c2 = **37.8% Good+Slight**.

| Run | Config | Good+Slight | CO% | FP% | FN% |
|---|---|---|---|---|---|
| Baseline | Binary only | **37.8%** | 30.1% | 30.3% | 1.8% |
| KNN only | threshold=0.1 | 5.6% | — | — | 25.8% |
| Hybrid | threshold=0.1 | 5.6% | 37.6% | 53.0% | 3.8% |
| Hybrid | threshold=0.2 | 7.5% | 31.9% | 47.7% | 13.0% |
| Hybrid | threshold=0.3 | ~6.1%* | ~28% | ~39% | 26.6% |

*measured against keyword GT

**Outcome: approach abandoned.** See training-log-binary.md for full root-cause analysis.

---

## Threshold Intuition

With k=10:
- threshold=0.1 → ≥1 neighbour in group → very permissive, high FP
- threshold=0.2 → ≥2 neighbours → moderate
- threshold=0.3 → ≥3 neighbours → conservative, risk of FN for rare groups

Starting point: **threshold=0.2**, which balances group recall and precision given the reference set size (~150 confirmed cases/group).

---

## Risks & Fallback

- If hybrid FP ≥ binary FP: the KNN is making things worse for non-cancer cases; abandon approach
- If hybrid CO improves but FP worsens: try using KNN only as a tie-breaker (when binary top-2 scores are close)
- If no improvement: accept Phase 16 binary as production ceiling; revisit when database grows past ~15k cases

---

## File Index

| File | Role |
|---|---|
| `ml/model/knn_group_selector.py` | KNN selector model |
| `ml/model/checkpoints/knn_group_selector.npz` | Pre-built reference (LLM GT, k=10, 2304-dim) |
| `ml/scripts/build_knn_selector.py` | Rebuild reference from updated LLM predictions |
| `ml/production/petbert_pipeline/categorization.py` | Add `run_categorization_hybrid()` here |
| `ml/production/petbert_pipeline/pipeline.py` | Wire hybrid branch |
| `ml/output/evaluation/evaluation_history.csv` | All evaluation runs logged here |
