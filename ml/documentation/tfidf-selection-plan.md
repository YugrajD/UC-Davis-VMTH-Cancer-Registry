# TF-IDF Multi-Column Text Selection — Implementation Plan

## Motivation

The current pipeline uses a fallback chain: it picks the first non-empty column from a
priority list (HISTOPATHOLOGICAL SUMMARY → FINAL COMMENT → COMMENT → …) and embeds only
that one column. This discards diagnostic signal present in secondary columns.

### Sanity-check findings (58,313 cases)

| Column | Fill rate | Median tokens | Value |
|---|---|---|---|
| HISTOPATHOLOGICAL SUMMARY | 97.1% | 229 | High — the diagnosis |
| COMMENT | 66.4% | 100 | Medium-high — pathologist notes |
| FINAL COMMENT | 31.5% | 137 | High — final conclusions |
| GROSS DESCRIPTION | 98.9% | 128 | Low — macroscopic appearance, not diagnosis |
| CLINICAL ABSTRACT | 99.4% | 96 | Low — patient history / signalment |

- 13.6% of cases hit the 512-token ceiling (truncated today)
- 73.3% of cases: HIST + FINAL COMMENT + COMMENT already fits within 512 tokens
- 26.7% of cases: combined overflows (need TF-IDF compression)
- GROSS DESCRIPTION and CLINICAL ABSTRACT are excluded — they add noise, not signal

### Goal

Replace the fallback chain with a text selector that:
1. Concatenates HIST + FINAL COMMENT + COMMENT (with section markers)
2. If combined ≤ 512 tokens → returns as-is
3. If combined > 512 tokens → applies TF-IDF sentence scoring to fit within the budget

This must be applied identically in both inference and contrastive training
so that the backbone embedding space is aligned with production inputs.

---

## Files to Create

### `ml/production/petbert_pipeline/text_selector.py` (new)

Core module. Single responsibility: given a dict of `{col_name: text}` for one case,
return a single selected string within a token budget.

**`TextSelector` class**

Source columns (defined inside this module, not in `model/constants.py`):
```python
_SOURCE_COLS = (
    "HISTOPATHOLOGICAL SUMMARY",
    "FINAL COMMENT",
    "COMMENT",
)
```

Methods:
- `fit(texts: list[str]) -> None` — fits `TfidfVectorizer(max_features=20000, sublinear_tf=True)` on corpus
- `save(path: str) -> None` — serializes with `joblib.dump`
- `load(path: str) -> None` — deserializes with `joblib.load`; raises `FileNotFoundError` with a clear message pointing to the fitting script if missing
- `select(col_texts: dict[str, str], max_tokens: int) -> str`

**`select()` logic:**
1. Pull source columns from `col_texts` in priority order; skip empty (< 10 chars)
2. Build combined: `"[HISTOPATHOLOGICAL SUMMARY] {text} [FINAL COMMENT] {text} [COMMENT] {text}"`
3. Estimate tokens: `len(combined) // 4`
4. If `<= max_tokens`: return combined as-is
5. If `> max_tokens`:
   - Split into sentences (paragraph-first, then sentence fallback — see below)
   - Score each sentence: TF-IDF transform, score = L1 norm of the vector
   - Greedy selection: sort by descending score, add sentences while budget allows
   - Restore original sentence order before joining

**Sentence splitting strategy:**
- Primary: split on `\n` — always safe boundaries in these reports
- For paragraphs > ~200 chars: split on `. ` but protect a hardcoded abbreviation safelist
  (`frozenset` of ~15 veterinary abbreviations: `"No."`, `"e.g."`, `"H.P.F."`, `"vs."`, etc.)
- Token budget uses `chars // 4` throughout — no tokenizer needed at selection time

**Module-level singleton:**
```python
_selector: TextSelector | None = None

def get_selector(vectorizer_path: str) -> TextSelector:
    global _selector
    if _selector is None:
        _selector = TextSelector()
        _selector.load(vectorizer_path)
    return _selector
```
Avoids re-loading the vectorizer from disk on every row.

---

### `ml/training/contrastive/fit_text_selector.py` (new)

Thin fitting script. Reads `report.csv`, builds combined 3-column text per case,
fits and saves the vectorizer.

```
--reports-csv  default=config.REPORTS_CSV
--out          default=config.TFIDF_VECTORIZER_PATH
```

Lives in `training/contrastive/` because it is a data preparation step that must run
before `build_contrastive_dataset.py`.

---

## Files to Modify

### `ml/config.py`

Add one constant in the training intermediates block:
```python
TFIDF_VECTORIZER_PATH = "ml/output/training/tfidf_selector.joblib"
```

### `ml/production/petbert_pipeline/types.py`

- **Remove:** `fallback_chain: tuple[str, ...] | None`
- **Add:** `tfidf_vectorizer_path: str` (required — no default; CLI sets it from `config`)
- Remove the `from model.constants import FALLBACK_CHAIN` import (now unused here)

### `ml/production/petbert_pipeline/cli.py`

- **Remove:** the arg wiring for `fallback_chain`
- **Add:**
  ```python
  parser.add_argument(
      "--tfidf-vectorizer",
      default=config.TFIDF_VECTORIZER_PATH,
  )
  ```
- **In `build_config`:** remove `fallback_chain=...`, add `tfidf_vectorizer_path=args.tfidf_vectorizer`
- `text_cols` and `col_weights` args are unchanged — they control the multi-column path
  which remains available for experiments

### `ml/production/petbert_pipeline/pipeline.py`

Replace the entire `if config.fallback_chain:` block (current lines ~79–97) with:

```python
from .text_selector import get_selector  # move to top-of-file imports

selector = get_selector(config.tfidf_vectorizer_path)
cols = ["tfidf_selected"]
selected_texts: list[str] = []
for i in range(n):
    row_col_texts = {
        col: clean_text(dataframe.iloc[i].get(col, ""))
        for col in ("HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "COMMENT")
    }
    selected_texts.append(selector.select(row_col_texts, max_tokens=512))
col_texts = {"tfidf_selected": selected_texts}
texts = selected_texts
```

The `else` branch (multi-column embedding path) is retained unchanged.

### `ml/training/contrastive/build_contrastive_dataset.py`

Two functions need updating: `build_contrastive_pairs` and `build_hard_neg_pairs`.
Both currently use the fallback-chain pattern (lines ~86–91 and the equivalent in
`build_hard_neg_pairs`).

**In each function:**
- Remove `fallback_chain: tuple[str, ...] = FALLBACK_CHAIN` parameter
- Add `tfidf_vectorizer_path: str = config.TFIDF_VECTORIZER_PATH` parameter
- Remove the `valid_chain` pre-loop and inner fallback selection
- Replace with:
  ```python
  selector = get_selector(tfidf_vectorizer_path)
  # inside the CSV row loop:
  col_texts = {col: row.get(col, "").strip()
               for col in ("HISTOPATHOLOGICAL SUMMARY", "FINAL COMMENT", "COMMENT")}
  report_text = selector.select(col_texts, max_tokens=512)
  ```
- Remove `from model.constants import FALLBACK_CHAIN` import
- Add `from production.petbert_pipeline.text_selector import get_selector`

### `ml/training/binary/run_cycle.py`

In `_make_scan_config`:
- Remove `fallback_chain=...` kwarg
- Add `tfidf_vectorizer_path=config.TFIDF_VECTORIZER_PATH`

### `ml/training/group/build_training_data.py`

Line 55 — change cache key:
```python
# Before:
all_embeddings: np.ndarray = cache["col_fallback_selected"]
# After:
all_embeddings: np.ndarray = cache["col_tfidf_selected"]
```

---

## Execution Sequence

Run these in order after all code changes are in place:

```bash
# 1. Fit the TF-IDF vectorizer on the full report corpus
ml/.venv/Scripts/python.exe ml/training/contrastive/fit_text_selector.py

# 2. Clear stale artifacts
rm -f ml/output/training/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f ml/output/checkpoints/contrastive/presence_classifier_current.pt
rm -f ml/output/checkpoints/contrastive/presence_classifier_best.pt
rm -f ml/output/checkpoints/contrastive/case_presence_classifier.pt

# 3. Build new contrastive training pairs (uses tfidf_selected text)
ml/.venv/Scripts/python.exe ml/training/contrastive/build_contrastive_dataset.py \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv

# 4. Adapt backbone on new text
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode adapt-backbone \
  --model ml/output/checkpoints/contrastive \
  --epochs 2 --lr 1e-5 --temperature 0.07 --batch-size 32 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv \
  --skip-pair-build

# 5. Rebuild embedding cache with new backbone and new text selection
ml/.venv/Scripts/python.exe ml/scripts/run_production.py --local-only

# 6. Retrain CasePresenceClassifier (gate)
#    (see training-guide.md for build_case_presence_dataset.py + train_case_presence.py)

# 7. Retrain GroupClassifier
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-groups \
  --epochs 50 --lr 5e-5 \
  --max-class-weight 50 --weight-decay 1e-3 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv

# 8. Run presence classifier cycles
ml/.venv/Scripts/python.exe ml/scripts/run_training.py \
  --mode train-classifier \
  --model ml/output/checkpoints/contrastive \
  --co-neg-per-case 5 --fp-neg-per-case 10 \
  --embedding-min-sim 0.05 --epochs 25 \
  --recall-weight 0.25 --hidden-dim 512 \
  --device xpu --local-only \
  --train-cases ml/output/splits/train_cases.txt \
  --annotation-csv ml/output/annotation/llm/llm_annotation.csv
```

---

## Architectural Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Cache key rename (`col_fallback_selected` → `col_tfidf_selected`) | Group training crashes with `KeyError` if cache rebuilt before `build_training_data.py` is updated | Update `build_training_data.py` before rebuilding cache; covered in execution sequence |
| `char/4` token estimate causes occasional slight overflow | PetBERT truncates to 512 anyway | Acceptable trade-off — selection quality is unaffected |
| `build_hard_neg_pairs` is a second call site easily missed | Old fallback chain used for hard-neg pairs | Explicitly covered in the plan; test with a dry run before committing |
| TF-IDF vectorizer goes stale if `report.csv` grows significantly | IDF weights shift, subtly affecting selection | Re-fit vectorizer whenever the report CSV is substantially updated |
| HIST sentences dominate TF-IDF selection for extreme overflow cases | Secondary columns squeezed out for very long HIST reports | For the 3.6% of extreme cases (> 1024 combined tokens), some secondary content will be dropped — acceptable |
