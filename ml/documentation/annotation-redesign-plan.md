# Annotation Pipeline Redesign — Gold/Silver Bootstrap Plan

**Status:** Approved 2026-06-17. **Phase 0 tooling delivered & verified** (2026-06-17), then revised to an **Excel review surface in 200-case batches** per user feedback. Batch 1 is ready: `ml/output/annotation/gold_eval_review.xlsx` (200 cases / all 51 groups / 16 self-consistency duplicates / 842 rows) + ledger `gold_eval_batch1_cases.txt`. Next real-world step: a veterinary professional fills in the workbook, then `consistency` → `ingest` → `check-split` → `run_evaluation.py` against the gold store for the first honest accuracy number. Phases 1–3 to be re-brainstormed after that.

## Motivation

The current annotation pipeline (3-tier LLM cascade: exact → fuzzy → LLM, + ensemble
cleanup) maps the short clinical `diagnosis` field to a Vet-ICD-O-canine-1 `(term, group,
code)` triple. Its output (`annotation.csv`) is the **sole ground-truth supervision** for
every downstream classifier (CasePresence, Group, LabelPresence) **and** for evaluation.

The problem: that ground truth is itself machine-produced and noisy (~7% on confirmed
positives, ~25–30% on weaker matches, per the 2026-05-09 audit). Because training labels
and evaluation labels are the *same* silver file, **every G+S number is measured against
possibly-wrong labels** — including production's 62.1%. The accuracy of the pipeline is
currently *unmeasurable*.

Goal: build a **trustworthy gold/silver corpus** via model-assisted, human-in-the-loop
annotation — a model suggests, a veterinary professional confirms — growing silver → gold
over rounds. Inspired by Boguslav et al. 2026 (PLOS Digital Health,
`10.1371/journal.pdig.0001147`), which fine-tunes a transformer to code diagnoses and
recommends (as future work) a suggest-and-confirm deployment.

## Hard constraint that shapes the phasing

You **cannot** train an 845-class diagnosis coder on a few hundred gold rows (Boguslav had
246k *human-coded* examples). So the fine-tuned coder is a **later** deliverable. The
**immediate** win is measurement + tooling using the suggester we already have (the
cascade). Build the ruler before the thing it measures.

## Key facts established during research (2026-06-17)

- **No gold set exists.** Closest: 6 hardcoded `KNOWN_ERRORS` + a 90-row audit sample
  (seed 42) in `ml/annotation/llm_pipeline/audit.py`. A gold store is a *new* artifact.
- **Plugging a gold eval set is nearly free.** `ml/evaluation/evaluate.py` already takes
  `--expectation-csv` (defaults to `config.ANNOTATION_CSV`) + `--test-cases`. A gold eval
  file is a pure path substitution, *no code change* — but its case IDs **must be disjoint
  from `train_cases.txt`**, and there is no guard enforcing that today.
- **The cascade does not surface top-k.** Tier-3 builds a candidate list internally but
  returns only the single winner; the `confidence` column is a rule-based scalar, not a
  calibrated probability. Suggest-and-confirm needs top-k added (Phase 2).
- **The two pipelines are genuinely separate.** `ml/data/diagnoses.csv`
  (`case_id, diagnosis_number, diagnosis`) feeds annotation; `ml/data/report.csv` feeds the
  registry classifier (concat-3). They join only on `case_id`. The annotation coder stays
  in the diagnosis-text domain; do not cross the wires.
- **Coverage is thin:** only 135/845 terms ever matched; 83 terms have exactly 1 match.
  Rare-class stratification is mandatory when sampling.
- **annotation.csv schema:** `case_id, diagnosis_number, diagnosis, matched_term,
  matched_group, matched_code, matched_keyword, method, confidence`. No tier/gold column.
  One row per diagnosis row (multi-diagnosis cases → multiple rows). Evaluation is
  **per-case** (predicted term vs the case's annotated term set), so gold must be collected
  at the **case** level — confirm all diagnosis rows of a sampled case.
- **Cleanlab supports multi-label**; run at the 52-group level first (sparsity). Max-entropy
  uncertainty sampling is cheap from sigmoid outputs. Single-annotator quality is measured
  by intra-annotator κ via 5–10% duplicate items (target κ > 0.7).
- **Boguslav specifics:** weighted F1 76.9 / exact-match 52.2%, precision > recall, 25% of
  data → ~10% F1 loss; used SNOMED-CT (7,739 codes), not ICD-O; suggest-and-confirm was
  *future work, not demonstrated*.

## Locked decisions

1. **Review surface = CSV-based MVP** (KISS; single annotator; zero infra). Adopt
   Argilla (free, local) or Prodigy ($490) only if the CSV proves too slow (Phase 2 fork).
2. **Gold-eval source = stratified-random sample drawn from `test_cases.txt`** (stratified
   by group to force rare-term coverage). This "spends" held-out cases to buy an honest
   accuracy number — accepted.
3. **Execute Phase 0 only first**, then re-brainstorm Phases 1–3 once the real noise number
   is known. Do not build the active-learning loop before measuring how bad the problem is.

## Gold/silver schema (new artifact)

A new file `gold_annotation.csv` — the existing 9 annotation columns **plus**:
- `tier` ∈ {gold, silver, bronze} — gold = human-confirmed; silver = model-produced
  unconfirmed; bronze = low-confidence single-tier match.
- `verified_by` — annotator identity.
- `verified_date` — confirmation date.
- `provenance` / `round` — which suggester + active-learning round produced the suggestion.

`annotation.csv` stays the production silver corpus, untouched, until we explicitly promote.
Two gold pools, kept disjoint:
- **gold-eval** — sampled from `test_cases.txt` → unbiased measurement.
- **gold-train** — sampled from `train_cases.txt` (active learning, Phase 2) → corrections
  usable for retraining.

---

## Phase 0 — Measurement + review surface *(no model; the immediate win)* — EXECUTING

**What & why:** Define the `gold_annotation.csv` schema. Carve a stratified-random
**gold-EVAL** sample (~300–500 cases) from `test_cases.txt`, stratified by group to force
rare-term coverage, with 5–10% duplicate cases for intra-annotator self-consistency.
Generate a **review CSV** (diagnosis text + `case_id`/report context + the cascade's
*existing* prediction joined from `annotation.csv` — no need to re-run the cascade) and an
ingest script that reads confirmations back into the gold store. Then re-run `evaluate.py`
with `--expectation-csv gold_annotation.csv` to get the **first honest accuracy number** for
the cascade.

**How to test:** Round-trip a ~10-row sample end-to-end with a mock confirmation; compute
intra-annotator Cohen's κ on the duplicates (target > 0.7); assert gold case IDs ∩
`train_cases.txt` = ∅; produce the cascade's true Good/Slight on gold eval (shown separately
per the "show G and S separately" rule).

**Risk/rollback:** Purely additive new files; **no change** to `annotation.csv` or any
training input. Rollback = delete the new files. (No backbone/classifier change, so the
archive-before-retrain rule does not apply here.)

**Models:** Implement **Sonnet** · Verify **Sonnet** · Run **Haiku** · Evaluate **Opus**.

**Note:** the actual gold-labeling by the professional is a *human* step that happens after
the tooling lands — Phase 0 delivers the review CSV + verified plumbing, not the gold labels
themselves.

### Phase 0 — Tool usage

Entry point: `ml/scripts/run_gold_annotation.py`  
Config constants: `config.GOLD_ANNOTATION_CSV`, `config.GOLD_EVAL_REVIEW_XLSX`  
Package: `ml/annotation/gold/` (`sample`, `ingest`, `consistency`, `check_split`, `xlsx_io`)

**Review surface = Excel workbook** (locked decision 1). `sample` emits a `.xlsx` with four
sheets: *Review* (one row per diagnosis; dropdowns for `verdict` / `confirmed_group` /
`confirmed_term`; the cascade suggestion prefilled grey; HIST SUMMARY + FINAL COMMENT for
context; blank-verdict cells highlighted salmon, `wrong` rows highlight amber), *Instructions*,
*Taxonomy* (full Group/Term/Code reference), and a hidden *Lists* sheet driving the dropdowns
(52 groups, 844 terms). The `dup_pass` column is hidden so repeats aren't obvious. The ICD-O
code is derived automatically from the chosen group+term at ingest — the professional only
picks `verdict` and, for `wrong`, the corrected group/term.

**Batching = 200 cases per sitting** (locked decision 2). `sample` defaults to `--n-cases 200`
and writes a per-batch case-ID ledger; later batches pass earlier ledgers to `--exclude-cases`
so cases never repeat across batches.

**Step 1 — Generate batch 1** (before the professional sits down):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py sample --n-cases 200 --batch 1
# Writes: ml/output/annotation/gold_eval_review.xlsx
#         ml/output/annotation/gold_eval_batch1_cases.txt  (the gold-eval case IDs)
# Later: --batch 2 --exclude-cases ml/output/annotation/gold_eval_batch1_cases.txt
```
Hand the `.xlsx` to the professional. They set `verdict` (correct / wrong / no_cancer /
uncertain) per row and, for `wrong`, pick `confirmed_group` + `confirmed_term`.

**Step 2 — Measure self-consistency** (after the filled workbook returns):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py consistency
# Cohen's kappa on dup_pass=1 vs =2 for duplicate diagnosis rows. Target kappa > 0.7.
```

**Step 3 — Ingest into the gold store**:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py ingest --verified-by "Dr. Smith"
# Reads the filled .xlsx; derives matched_code from group+term via labels.csv.
# Writes: ml/output/annotation/gold_annotation.csv  (excludes dup_pass=2 rows).
```

**Step 4 — Guard the split**:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py check-split
# PASS if all gold case IDs are in test_cases.txt and none are in train_cases.txt.
```

**Step 5 — Evaluate inference against the gold store** (pure path substitution; no code change
— the gold case-ID ledger from Step 1 is the `--test-cases` filter):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_evaluation.py \
    --annotation-csv ml/output/annotation/gold_annotation.csv \
    --test-cases ml/output/annotation/gold_eval_batch1_cases.txt
```

---

## Phase 1 — Noise triage to prioritize re-annotation *(cheap, high-leverage)*

**What & why:** Run **cleanlab (multi-label, group level)** on out-of-sample probabilities
to flag the noisiest ~10–20% of silver rows → a ranked "re-annotate me" queue. Start with
the existing GroupClassifier's probs (approximate — it conflates annotation noise with model
error, but is a cheap triage prioritizer); graduate to a quick diagnosis-domain linear head
if needed.

**How to test:** Confirm the flags surface the 6 known errors; measure flag precision
against the Phase-0 gold sample.

**Risk/rollback:** Read-only analysis; no production change.

**Models:** Implement **Sonnet** · Run **Haiku** · Evaluate **Opus**.

---

## Phase 2 — Suggest-and-confirm + active-learning loop

**What & why:** Add a **ranked top-k suggester** for diagnosis text (extend Tier-3 to return
its candidate list, or a PetBERT-base cosine retriever over taxonomy terms; temperature-
scale the scores). Active-learning queue = max-entropy uncertainty + cleanlab flags + a
**rare-group quota**. Iterate rounds; confirmations grow **gold-train** (from
`train_cases.txt`, disjoint from gold-eval). Adopt Argilla/Prodigy only if the review CSV is
too slow.

**How to test:** Per round, track annotation throughput, gold growth, and the cascade's
accuracy-on-gold-eval as the corpus grows.

**Risk/rollback:** Additive; `annotation.csv` stays the production silver corpus until we
explicitly promote.

**Models:** Implement **Sonnet** · Verify **Sonnet** · Run **Haiku** · Evaluate **Opus**.

---

## Phase 3 — Train the diagnosis-text coder (the Boguslav model) *(later)*

**What & why:** Once gold-train is sufficient (or via distill-from-silver + gold
correction), fine-tune a **diagnosis-text multi-label coder** — bake-off PetBERT base vs
ModernBERT vs a clinical model (the paper found general models competitive), **hierarchical
group→term** for cold start, temperature scaling + per-class thresholds. This coder becomes
the new suggester (replacing cascade Tiers 2–3) and eventually the silver generator.

**How to test:** Weighted F1 / exact-match on the gold eval set vs the cascade baseline
(reference: Boguslav 76.9 F1 on a *larger* label space).

**Risk/rollback:** **Archive the current annotation generation** per the *Embedding &
Classifier Versioning* rule before any promotion; regenerate silver only after the coder
beats the cascade on gold.

**Models:** Implement **Sonnet** (bake-off design may warrant **Opus**) · Run **Haiku** ·
Evaluate **Opus**.

---

## Open questions to resolve before Phase 2/3

- **Cold start:** 845 classes vs a few hundred gold — hierarchical (group→term) head, or
  coder-on-silver with gold-as-eval?
- **Single annotator:** no inter-rater reliability — duplicate items measure self-consistency
  and set the noise floor.
- **Behavior codes (/0, /2, /3):** annotation side uses a simplistic 3-digit regex
  (`_detect_behavior` in `pipeline.py`); a known failure mode the coder must improve on.
