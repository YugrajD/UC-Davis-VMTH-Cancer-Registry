# Production Pipeline

Implementation-based description of what `ml/scripts/run_production.py` does today.

This is the authoritative source for current production inference behavior. Older
architectural experiments are preserved in the training logs and idea docs, not here.

The intended production path is a three-stage sequential pipeline where each stage has
one distinct responsibility:

```text
report.csv
  -> PetBERT embedding (cached or fresh)
  -> CasePresenceClassifier gate   — filters non-cancer cases        (reduces FP)
  -> GroupClassifier               — assigns cancer to ICD group      (reduces CO)
  -> KW correction                 — picks best term within group     (converts Slight → Good)
  -> (term, group, code) predictions + debug artifacts
```

The binary `PresenceClassifier` + `group-keyword` mode (earlier production path) is still
supported via `--presence-classifier` but is superseded by the three-stage design above.

## Flow Chart

```mermaid
flowchart TD
    subgraph IN3["Input"]
        A["report.csv<br/>Report text columns"]
        C["labels.csv<br/>Taxonomy labels"]
    end

    subgraph P3["Process"]
        B["Embed each report column with adapted PetBERT"]
        D["Embed each label with adapted PetBERT"]
        E["Concatenate report-column embeddings"]
        F["Prepare label embeddings"]
        G["PresenceClassifier scores each case-label pair"]
        H["Rank labels by score"]
        I["Select top predictions"]
        J["Project label index to term, group, ICD code"]
    end

    subgraph OUT3["Output"]
        K["predictions.csv"]
    end

    A --> B
    C --> D
    B --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```

## Entry Point And Auto-Selection

`ml/scripts/run_production.py` is the production launcher.

Before it calls the pipeline, it:

1. Looks for the best saved classifier checkpoint.
2. Prefers `ml/output/checkpoints/contrastive/presence_classifier_best.pt`.
3. Falls back to `ml/output/checkpoints/binary/presence_classifier_best.pt`.
4. Uses the matching embedding model for that checkpoint:
   - `ml/output/checkpoints/contrastive/` if the chosen classifier is contrastive-backed
   - `SAVSNET/PetBERT` otherwise
5. Sets the default embedding cache to `ml/output/training/embedding_cache.npz`.
6. Writes production outputs to `ml/output/production/{contrastive|binary}/`.

This means the code treats the contrastive-backed PresenceClassifier stack as the
default production path whenever that checkpoint exists.

## Input Format

The pipeline reads `ml/data/report.csv` with one row per case.

Important columns:

| Column | Role |
|---|---|
| `case_id` | Unique case identifier |
| `HISTOPATHOLOGICAL SUMMARY` | Microscopic pathology findings — primary diagnostic source |
| `FINAL COMMENT` | Pathologist's diagnostic conclusion |
| `COMMENT` | Pathologist notes |
| `ANCILLARY TESTS` | IHC, stains, PCR, and related tests (not used in TF-IDF path) |
| `GROSS DESCRIPTION` | Macroscopic specimen description (excluded — adds noise, not signal) |
| `CLINICAL ABSTRACT` | Referring clinician history (excluded — adds noise, not signal) |

Production uses TF-IDF-based text selection: it concatenates HISTOPATHOLOGICAL SUMMARY +
FINAL COMMENT + COMMENT with section markers, then compresses to a 512-token budget if
needed using TF-IDF sentence scoring. This replaces the old fallback-chain approach (single
column) as the default input path. See `production/petbert_pipeline/text_selector.py`.

## Step-by-Step Runtime Flow

The main implementation lives in `ml/production/petbert_pipeline/pipeline.py`.

### 1. Load and clean report data

The pipeline reads `ml/data/report.csv` using `latin-1`, strips BOM artifacts from
column names, and normalizes missing values to empty strings.

### 1b. TF-IDF text selection (default production path)

When `--text-cols` is empty (the production default), `TextSelector` builds a single
combined string per case from HISTOPATHOLOGICAL SUMMARY + FINAL COMMENT + COMMENT:

```text
[HISTOPATHOLOGICAL SUMMARY] ... [FINAL COMMENT] ... [COMMENT] ...
```

If the combined text fits within 512 tokens (≈2048 chars), it is used as-is.
If it overflows, the selector scores each sentence by its TF-IDF L1 norm and greedily
picks the highest-scoring sentences that fit within the budget. The order of selected
sentences is preserved in the output.

The fitted vectorizer must exist at `ml/output/training/tfidf_selector.joblib` before
the first run. Build it with `fit_text_selector.py`.

### 2. Reuse embedding cache when possible

If `ml/output/training/embedding_cache.npz` is valid for the current:

- report CSV
- labels CSV
- model name
- selected text columns

then the pipeline skips re-embedding and reuses:

- per-column report embeddings
- per-column content masks
- mean case embeddings
- token counts
- label embeddings

This is what keeps repeated production and training-cycle runs fast.

### 3. Otherwise embed each report column separately

On a cache miss, the pipeline loads PetBERT and embeds each selected report column
independently.

Important details:

- Each column gets its own token budget.
- Mean pooling over non-padding tokens produces one 768-d embedding per column.
- Empty cells are tracked separately with boolean masks.

### 4. Build a mean report embedding for analysis outputs

After per-column embedding, the pipeline averages the non-empty column embeddings into a
single 768-d mean embedding per case.

That mean embedding is used for:

- PCA visualization
- nearest-neighbor outputs
- the saved embeddings NPZ
- some non-default group-based paths

It is not the main tensor used by the default production classifier.

### 5. Embed every ICD label with the same base model

The taxonomy is loaded from `ml/ICD_labels/labels.csv`.

Each label is converted to display text and embedded through the same PetBERT base model,
producing a label embedding matrix aligned with the report embedding space.

### 6. Concatenate report columns for classifier scoring

For classifier inference, the pipeline concatenates the per-column report embeddings into
one wide vector per case, zeroing out empty columns first.

This `col_emb_concat` tensor is what the `PresenceClassifier` consumes.

### 7. Score all labels with the PresenceClassifier

In the default production path:

1. The pipeline loads the selected `PresenceClassifier` checkpoint.
2. It scores every `(case, label)` pair.
3. The result is an `(N, M)` score matrix over all taxonomy labels.

This is the key production decision point. The live production path is
classifier-driven scoring on top of PetBERT embeddings.

### 8. Apply group-keyword term correction

If production is run with `--categorization-mode group-keyword`, the code uses a
two-stage post-processing strategy on top of the score matrix:

1. Mean-center label scores and choose the top Stage 1 label.
2. Use that label's ICD group as the predicted group.
3. Infer an ICD behavior digit from report text using behavior keywords.
4. Restrict candidates within the predicted group to matching behavior codes when possible.
5. Pick the best term in that filtered pool using the raw classifier scores.

This stage is designed to improve term selection inside the already-predicted group.
It changes Good vs Slightly Off behavior without changing the core Stage 1
predict-vs-Uncategorized decision.

## Output Files

The production pipeline writes:

| File | Purpose |
|---|---|
| `petbert_predictions.csv` | Ranked predictions per case |
| `petbert_column_scores.csv` | Per-column debug breakdown |
| `petbert_provenance.csv` | Per-case traceability and merged report text |
| `petbert_similarity_scores.csv` | Full label-score matrix dump |
| `petbert_visualization.csv` | PCA coordinates per case |
| `petbert_embeddings.npz` | Saved mean embeddings and related arrays |
| `petbert_summary.json` | Run metadata and aggregate counts |

Optional neighbor output:

- `petbert_neighbors.csv` when `--task neighbors` or `--task both` is used

These files are written under `ml/output/production/{contrastive|binary}/` when launched
through `run_production.py`.

## Current CLI Behaviors That Matter

The production CLI still supports multiple modes, but the important current behaviors are:

- `run_production.py` auto-selects the best available classifier checkpoint.
- `--presence-classifier` can explicitly override that checkpoint choice.
- `--embedding-cache` reuses `ml/output/training/embedding_cache.npz` when provided.
- `--task neighbors` or `--task both` adds nearest-neighbor output alongside categorization.
- `--local-only` keeps model loading offline when the files are already cached locally.

## Three-Stage Pipeline (Intended Production Path)

Run after training `CasePresenceClassifier` and `GroupClassifier`:

```bash
ml/.venv/Scripts/python.exe ml/scripts/run_production.py \
  --case-presence-classifier ml/output/checkpoints/contrastive/case_presence_classifier.pt \
  --case-presence-threshold 0.5 \
  --group-classifier ml/output/checkpoints/group/group_classifier_best.pt \
  --group-classifier-threshold 0.90 \
  --embedding-cache ml/output/training/embedding_cache.npz \
  --device xpu --local-only
```

**Stage 1 — CasePresenceClassifier gate:**
Takes the mean report embedding (768-dim) and outputs a cancer probability. Cases below
`--case-presence-threshold` are predicted Uncategorized without reaching the GroupClassifier.
Trained with `recall_weight=0.7` so it errs toward passing uncertain cases rather than
missing cancer. Train with `--mode train-case-presence`.

**Stage 2 — GroupClassifier:**
For cases that passed the gate, predicts which of 42 cancer groups the case belongs to
(sigmoid per group, threshold applied). Cases where no group clears the threshold become
Uncategorized.

**Stage 3 — KW correction:**
Within each predicted group, ICD-O behavior keyword matching narrows candidates to the
matching behavior digit, then cosine similarity selects the best specific term.

## What Is Not The Intended Production Path

The following code paths exist as alternatives or fallbacks:

- `--presence-classifier` (without `--case-presence-classifier`)
  Uses the label-level PresenceClassifier score matrix (N × M) as a gate. Superseded by
  `--case-presence-classifier` which is simpler and purpose-trained.
- `--finetuned-model-path`
  Uses a sequence-classification checkpoint to predict groups directly. WIP, blocked.

Older experimental and deprecated paths are preserved in the training logs and idea docs,
not in this file.

## Source Of Truth

If this file and an older architecture doc disagree, trust the implementation in:

- `ml/scripts/run_production.py`
- `ml/config.py`
- `ml/production/petbert_pipeline/pipeline.py`
- `ml/production/petbert_pipeline/categorization.py`
- `ml/production/petbert_pipeline/embedding.py`
