# Report mapping (bronze)

The PetBERT 4-stage pipeline that maps pathology report text to a Vet-ICD-O-canine-1 label,
independent of any diagnosis text. Package: `ml/report_mapping/` (`sections.py`, `model/`,
`training/`, `inference/`). Entry points: `scripts/train.py`, `scripts/calibrate.py`,
`scripts/predict.py`.

## Sections

`sections.py` defines the one concat-3 section spec (`SECTION_SPEC_VERSION`, bump on any change):

```
__sec_0__ = HISTOPATHOLOGICAL SUMMARY
__sec_1__ = FINAL COMMENT + "\n" + COMMENT
__sec_2__ = ANCILLARY TESTS
```

`build_section_frame` adds these three columns to a raw report frame; `section_texts` returns
per-section cleaned text lists; `merged_texts`/`merge_report_columns` produce the `[__sec_N__] text`
merged string that keyword correction and the Lipoma rescue read. This spec is defined exactly
once — legacy had it in two places that could drift.

## Model

- `model/backbone.py` — loads any HuggingFace checkpoint dir or name (`AutoModelForMaskedLM`, base
  transformer only) and mean-pools attended tokens into a 768-dim embedding per section.
  `max_length` is 512 at inference, 256 at training — both load-bearing, kept distinct.
- `model/heads.py` — the three trainable heads (`CasePresenceClassifier`, `GroupClassifier`,
  `LabelPresenceClassifier`), state_dict-compatible with the pre-rewrite `.pt` checkpoints:

  | Head | Input | Output | Architecture |
  |---|---|---|---|
  | CasePresence (gate) | 2304-dim concat-3 | cancer probability | `Linear(2304→512)→ReLU→Dropout(0.3)→Linear(512→1)→sigmoid` |
  | Group | 2304-dim concat-3 (section-masked) | per-group sigmoid | `Linear(2304→512)→ReLU→Dropout(0.1)→Linear(512→25)→sigmoid` |
  | LabelPresence (per group) | 2304-dim case + 768-dim label | in-group score | 3 section pairs `[sec_emb\|label_emb]` (1536-dim) → shared `Linear(1536→512)→ReLU→Dropout(0.3)→Linear(512→1)`, combined by a learned `Linear(3→1)` (`n_cols=3, col_pair_mode=True, col_combine="learned"`) |

- `model/generation.py` — loads/saves a **generation** directory (`current/`, `candidate/`), which
  *is* the cloud bundle layout:

  ```
  petbert/                                  HF backbone checkpoint dir
  labels/labels.csv                         Vet-ICD-O taxonomy snapshot
  checkpoints/case_presence_classifier.pt
  checkpoints/group_classifier_best.pt
  checkpoints/label_presence/<safe_group>.pt
  checkpoints/label_presence/lp_thresholds.json
  checkpoints/thresholds.json               gate / group / tail / LP-fallback
  checkpoints/uncommon_groups.txt
  manifest.json
  ```

  `load_generation` verifies the directory's manifest (file sha256s), its **embedding
  fingerprint** (`{backbone_sha256, section_spec_version, max_length}`) against the current code,
  and `calibration.status == "calibrated"` (unless `allow_uncalibrated=True`, used only by
  `calibrate.py` itself) — a mismatch raises `GenerationError` rather than loading stale
  classifiers silently. This is the code-level fix for CLAUDE.md's "Embedding & Classifier
  Versioning" rule.

## Inference

`inference/embedding_cache.py` — content-hash keyed (`sha256(report bytes + labels bytes +
embedding fingerprint)`), stored under `output/report_mapping/embedding_cache/<key>.npz`, one file
per distinct (report, labels, backbone) combination. Legacy keyed its cache on CSV mtime ±1s, which
a 2026-08 incident showed to be fragile (a mtime touch with no content change looked like a miss).
Never bundled into a generation directory.

`inference/stages.py` — `categorize_cases` dispatches gate → group (+ tail gate) → per-group
label-presence → keyword correction, behaviour-equivalent to the pre-rewrite pipeline:

1. **Gate** (`run_case_presence`) — cases below `thresholds["case_presence_gate"]` don't reach
   Stage 2+; their group probabilities are zeroed.
2. **Group** (`run_group`) — per-group sigmoid; groups ≥ `thresholds["group"]` advance, ranked by
   probability descending. Argmax fallback (always on) picks the top group when none clears
   threshold, for any gate-passed case. A **tail gate** then drops all but the top
   `thresholds["tail_max_predictions"]` groups, and any group more than
   `thresholds["tail_max_group_prob_gap"]` below the top group's probability.
3. **Per-group label presence (Stage 3a)** — for each surviving group, `score_within_group` scores
   every label in that group; labels ≥ the group's per-LP threshold (`lp_thresholds.json`, fallback
   `thresholds["label_presence_fallback"]`) survive, with an argmax fallback within the group.
4. **Keyword correction (Stage 3b)** — `inference/keyword_correction.py`: behavior-digit filter
   (`taxonomy/behavior.py`) then group-specific subtype filter (`taxonomy/subtype.py`, 7 groups),
   narrowing the Stage 3a pool.
5. **Lipoma rescue** — appends `Lipoma, NOS` when the report matches fatty-tissue vocabulary, no
   `liposarcoma` mention, and Lipomatous group probability ≥ 0.5, even if it didn't clear the group
   threshold. Unconditional, no flag.

**Fixed nondeterminism.** The merged "Uncommon" bucket (groups below the uncommon-merge threshold)
used to draw its label order from iterating a `frozenset`, so its order — and hence which
same-scoring label won top-1 — depended on `PYTHONHASHSEED`. This rewrite builds the Uncommon pool
from `sorted(uncommon_groups)` and ranks survivors within it by LP confidence (ties by term), so
the top-1 pick is always the best-scoring survivor. Named groups' pools were never hash-dependent
and are left exactly alone.

`inference/predict.py` — `predict_frame` (in-memory, for ml-worker: embeds fresh with the
generation's own backbone, never through the on-disk cache) and `run_predict` (the CLI path: reads
`config.REPORT_CSV`, gets-or-builds the content-hash cache, runs every stage, writes
`case_id, diagnosis_index, predicted_term, predicted_group, predicted_code, case_presence_prob,
confidence, group_prob, method, generation_id`). `--embed-only` stops after populating the cache
(allowed on an uncalibrated candidate). Dropped from legacy: similarity/visualization/provenance/
neighbors/embeddings-npz debug outputs.

## Training

`training/recipe.py` is the single source of production hyperparameters (verified against the old
training-guide.md commands and the old trainers' own defaults) and seeding (`seed_all` — python,
numpy, torch, CUDA — a deliberate improvement over legacy, which seeded inconsistently or not at
all per head). `tests/test_recipe.py` pins these values.

| Recipe | Key hyperparameters |
|---|---|
| `BACKBONE` | epochs=3, batch_size=32, lr=2e-5, temperature=0.07, max_length=256, weight_decay=0.01, warmup_frac=0.06 |
| `GATE` | epochs=20, recall_weight=0.7, pos_weight=1.0, dropout=0.3, lr=1e-3, weight_decay=1e-4 |
| `GROUP` | epochs=300, lr=5e-5, dropout=0.1, weight_decay=1e-3, max_class_weight=50, val_frac=0.2, uncommon_threshold=200, excluded_groups=("Neoplasms, NOS",) |
| `LABEL_PRESENCE` | epochs=25, negs_per_pos=5, recall_weight=0.5, dropout=0.3, weight_decay=1e-4, n_cols=3, col_pair_mode=True, col_combine="learned" |

Every recipe's `seed` defaults to `PRODUCTION_SEED = 42`. `LP_PAIR_SAMPLING_SEED` (also 42) is
always used for label-presence negative sampling, independent of the run's `--seed` — this is why
the L3 three-seed retrain's negative pairs are identical across seeds.

Trainers (`case_presence.py`, `group.py`, `label_presence.py`, `backbone.py`) are faithful ports of
the pre-rewrite trainers, each exposing `train` (the split-driven CLI path) and `train_on_case_ids`
(the reusable core `oof.py`'s k-fold out-of-fold runner also calls). `labels.py` builds every
stage's targets from any labels table (`case_id, matched_term, matched_group, matched_code`) and is
the one place that filters to the split's train partition and runs the train-only leakage guards.
`embeddings.py` gets-or-builds the same content-hash cache `predict.py` uses, so heads-only training
never re-embeds when the backbone hasn't changed.

`scripts/train.py --stage {backbone,case-presence,group,label-presence,heads,oof}` writes into
`candidate/` (or `--out current` / a literal directory), copies in a backbone (default: the current
generation's `petbert/`) for heads-only stages, and writes a placeholder manifest with
`calibration.status: "pending"` — `calibrate.py` fills in real thresholds afterward. `--stage oof`
is a diagnostic (k-fold out-of-fold gate/group predictions on train cases) and writes no manifest.

### Calibration

`training/calibrate.py` fits every threshold on one split partition (`calibration`, from cached
embeddings — never re-embeds):

1. **Per-LP thresholds** — for each label-presence head, F1 at every grid point over the (case,
   label) pairs of that head's in-scope, partition-annotated cases; probabilities rounded to 4 dp
   first (legacy wrote them to CSV at that precision and swept the re-read values); the lowest
   threshold with the strictly highest F1 wins. **Grid is 0.05 steps, `0.05` to `0.95`**
   (`calibrate.LP_GRID`) — this reproduces all 25 production values; a 0.01 grid (an earlier,
   incorrect training-guide.md claim) reproduces only 12.
2. **Gate, group and tail**, jointly, over a small fixed grid (gate/group ∈
   `{0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90}`, tail `(K, gap)` ∈ `{(1,1.00), (2,0.02), (2,0.05), (2,0.08), (2,0.10),
   (3,0.10), (5,1.00)}`), with the LP thresholds from step 1 fixed. Objective: per-code G+S share
   of the `evaluation.verdicts` table against the labels table on the partition; ties go to the
   higher `good` share, then the first point in grid order.

Every calibration run checks `guards.check_all` and that no calibration case is in the generation's
own training split. Output: `checkpoints/thresholds.json`,
`checkpoints/label_presence/lp_thresholds.json`, and a `calibration_diagnostics.json` (gate P/R/F1,
group top-k accuracy, per-LP P/R) beside them; the manifest's `calibration` block is rewritten last
(so a partial write never leaves loadable-but-wrong thresholds in place).

## Generations, manifests, fingerprints

A **generation** (`current/` = production, `candidate/` = being trained/evaluated) is identified by
a `generation_id` (`gen-<UTC timestamp>`, e.g. `gen-20260927T003905Z`, minted fresh on every
training write — also the cloud's `source_version`) and a manifest recording:

- `parents` — `silver_id`, `gold_train_snapshot`, `gold_train_codes`, `split_id` — what it trained on.
- `recipe` — the resolved hyperparameters and seed.
- `embedding_fingerprint` — `{backbone_sha256, section_spec_version, max_length}`.
- `calibration` — `status` (`pending`/`calibrated`), `partition`, `objective`, `values`.
- `status` — `candidate` / `current` / `archived` (set by `generations.promote`).
- `files` — a sha256 per file under the directory (`generations/manifest.py`), so any drift is
  caught by `verify_manifest` before loading.

`--model` (on `predict.py`/`train.py`) accepts any HuggingFace checkpoint dir or HF name; the
production default is always the current generation's own `petbert/`.

## Why this design

Carried forward from the pre-rewrite architecture (see [icd-mapping-strategy.md](icd-mapping-strategy.md)
for how bronze fits into gold/silver/bronze):

**Why 4 stages, not one classifier.** Each stage takes one job: the gate filters non-cancer
reports out before they reach a multi-class classifier that would otherwise emit something (cuts
false positives); the group classifier picks among ~25 groups with them competing explicitly in one
sigmoid BCE loss, so a wrong-group assignment is penalized directly during training rather than
emerging implicitly from independent per-label scores (cuts completely-off predictions); the
per-group label-presence head picks the specific term only within the predicted group, so its
decision boundary is much sharper than a single global "is this label present?" classifier's would
be (converts slightly-off into good); keyword correction narrows by ICD-O behavior digit and
group-specific subtype regex after the learned stages, a pure-Python post-filter with no training.

**Why concat-3.** A pathology report is structurally divided into sections (histopathological
summary, final comment + comment, ancillary tests). Concatenating them into one string and
tokenizing forces a single 512-token budget across the whole report, truncating long sections and
giving the model no way to weight one section over another. Concat-3 embeds each section
independently (each gets its own token budget) and concatenates the three 768-dim section vectors
into one 2304-dim case representation, preserving per-section information downstream heads can
learn to weight.

**Why a per-section contrastive backbone.** The base PetBERT model is pretrained on UK veterinary
EHRs with masked-LM; its weights don't by themselves pull report embeddings toward their correct
label embeddings — that geometry has to be added with supervision. The contrastive adaptation runs
InfoNCE on `(report_section_text, label_text)` pairs built **per section**, matching exactly the
per-section view concat-3 inference consumes, rather than training on whole-report text and then
asking the model to encode sections at inference. This alignment was the single largest lever
found: **G+S 13.1% → 24.0%** (+10.9pp) over a whole-report-trained backbone on the same concat-3
representation.

**Why per-group label-presence heads.** A single global label-presence classifier scoring every
(case, label) pair across ~850 labels has to separate "Squamous cell carcinoma" from
"Adenocarcinoma" using the same parameters that separate "Hemangioma" from "Hemangiosarcoma".
Splitting into one head per group lets each specialize on its own in-group decision; the shared
section-pair architecture (`n_cols=3, col_pair_mode=True, col_combine="learned"`) still lets each
head learn, per group, which section matters most for term selection.

**Why the case-presence gate's threshold and the tail gate.** Score distributions vary by group and
by stage: a single global threshold trades recall for precision differently in different places.
Per-LP thresholds and the Stage-2 tail gate (cap groups per case; drop tail groups far below the top
group) were both fitted by sweeping against held-out data and are now refit by `calibrate.py` on
the calibration partition every retrain, rather than hand-tuned once and left alone.

## The 62.1% caveat

The historically published **G+S 62.1%** (4,414 rows) does not reproduce from the files on disk
today — the case-presence, group and some label-presence heads were retrained and `annotation.csv`
was rewritten since that number was measured. The frozen, reproducible reference (regenerated
2026-09-25 with the pre-rewrite inference code, current legacy checkpoints and embedding cache,
`PYTHONHASHSEED=0`) is:

- **md5 eval half of the legacy split**: G+S 61.8%, exact-match 61.76% on 4,456 per-code rows
  (Good 45.8, Slight 16.0, CO 15.3, FP 2.6, FN 20.4).
- **Full legacy test split**: G+S 61.8%, exact-match 61.81% on 8,916 rows.

Treat 61.76% (not 62.1%) as the production baseline going forward. See
[ml-rewrite-plan.md](ml-rewrite-plan.md), "Findings that shape the rewrite", for the full
reconciliation and the L1–L3 parity results against it.

