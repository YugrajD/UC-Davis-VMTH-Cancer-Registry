# Model Training — Architectural Decisions

Why the production stack looks the way it does. For "how to run training" see [training-guide.md](training-guide.md); for the architecture and threshold mechanics of each head see [classifiers.md](classifiers.md).

## Why 4 stages

Each stage takes one job:

1. **CasePresenceClassifier (gate)** — binary cancer / non-cancer. Filters non-cancer reports out before they reach a multi-class classifier that would otherwise emit something. Reduces FP.
2. **GroupClassifier** — picks the ICD group. With 25 groups competing explicitly in a sigmoid BCE loss, wrong-group assignments are penalized directly during training rather than emerging implicitly from independent label scores. Reduces CO.
3. **LabelPresenceClassifier (per group)** — picks the specific term within the predicted group. Each head only sees labels from one group, so the within-group decision boundary is much sharper than a global "is this label present?" classifier sees. Converts Slight → Good.
4. **Keyword correction** — narrows by ICD-O behavior digit (benign / malignant / metastatic / in situ) and by group-specific subtype regex, after the learned models picked candidates. Pure-Python post-filter, no training.

Files: `ml/production/petbert_pipeline/stages/{case_presence_classifier,group_classifier,label_presence_classifier,keyword_correction}.py`. The per-case dispatcher is `stages/__init__.py::categorize_per_case`.

## Why concat-3 text representation

A pathology report is structurally divided into sections (HISTOPATHOLOGICAL SUMMARY, FINAL COMMENT + COMMENT, ANCILLARY TESTS). Concatenating them as one string and tokenizing forces a 512-token budget across the whole report, truncating the long sections and giving the model no way to weight one section more than another.

Concat-3 embeds each section independently (each gets its own 512-token budget), then concatenates the three 768-dim section vectors into a single 2304-dim case representation. The case-level vector preserves per-section information that downstream heads can learn to weight.

Implementation: `production/petbert_pipeline/pipeline.py::CONCAT_3_SECTIONS` defines the three section groupings; `pipeline.py::run_scan` embeds each section through PetBERT, then writes the 2304-dim concat to the cache under key `concat_3` (alongside the per-section views and a 768-dim masked-mean for cosine fallbacks).

## Why per-section contrastive backbone

The base PetBERT model is pre-trained on UK veterinary EHRs with masked-LM. Its weights don't pull report embeddings toward their correct label embeddings — that geometry has to be added supervised.

The contrastive adaptation in `ml/training/contrastive/train_contrastive.py` runs InfoNCE on `(report_section_text, label_text)` pairs. Critically, the pairs are built **per section** by `build_contrastive_dataset.py`: each annotated case produces three pairs (one per section), each pairing that section's text with the matched label text. This aligns the backbone with the same per-section view that concat-3 inference consumes, instead of training the backbone on whole-report text and then asking it to encode sections at inference.

## Why per-group LabelPresenceClassifiers (one per group)

A single global label-presence classifier scoring every (case, label) pair across all ~850 labels has to learn to separate "Squamous cell carcinoma" from "Adenocarcinoma" using the same parameters that separate "Hemangioma" from "Hemangiosarcoma". Different groups have different discriminating features (histologic subtype vs anatomic site vs behavior code), and forcing one MLP to fit all of them dilutes the signal.

Splitting into one head per group is small (a few KB per `.pt`) but lets each head specialize on the in-group decision. The shared section-pair architecture (`n_cols=3, col_pair_mode=True, col_combine="learned"`) gives each section its own `[section_emb | label_emb]` pair that the shared MLP scores, then a learned `Linear(3 → 1)` weights the three section logits — so the model can learn, for example, that FINAL COMMENT matters more than ANCILLARY TESTS for term selection.

Implementation: `ml/model/label_presence_classifier.py`; training driver in `ml/training/label_presence/train.py`; per-group dataset builder in `ml/training/label_presence/build_training_pairs.py`. The Uncommon group is one shared head trained against the union of all merged groups.

## Why per-LP thresholds and tail gate

A global threshold of 0.5 is wrong for most LPs because the score distributions vary by group: some LPs are sharp (high-confidence positives near 1.0) and benefit from a higher threshold (cuts false positives without losing many true positives); others are flatter and need a lower threshold to keep recall.

`ml/scripts/sweep_lp_thresholds.py` splits the per-(case, label) LP evaluation rows 50/50 by case-ID hash, picks the F-beta-maximising threshold per LP on the sweep half, and reports unbiased metrics on the eval half. The output JSON (`lp_thresholds.json`) is auto-loaded by `run_production.py`; LPs missing from the map use `--label-presence-threshold` (default 0.5). The `--beta` flag lets you trade precision against recall (`<1` reduces CO; `>1` reduces FN).

The Stage-2 tail gate (`--tail-max-predictions`, `--tail-max-group-prob-gap`) caps how many groups a case can be assigned to and drops tail groups whose probability is far below the top group's. Calibrate with `ml/scripts/sweep_tail_gate.py`. Without the gate, multi-label cases produce one prediction per group above threshold, and when the GroupClassifier is wrong on the tail predictions those become CO rows. Defaults (`K=2, gap=0.08`) were chosen 2026-05-11 by sweeping against the held-out test set.
