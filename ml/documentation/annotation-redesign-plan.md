# Annotation Pipeline Redesign — Gold/Silver Bootstrap Plan

**Status:** Approved 2026-06-17. **Phase 0 tooling delivered & verified** (2026-06-17); revised to 200-case batches, then revised again on 2026-08-05 to a **plain-CSV review surface** (the Excel workbook and the self-consistency duplicates were both dropped — see "Review surface" below), then re-cut on 2026-08-12 as a **row-level Tier-3 audit** now that `decision_stage` exists (see "Row-level Tier-3 audit" below). Batch 1 is ready: `ml/output/annotation/tier3_audit_review.csv` (200 rows across 198 cases, 100% Tier-2/Tier-3 decisions, zero padding) + the `tier3_audit_instructions.md` and `tier3_audit_taxonomy.csv` sidecars + ledger `tier3_audit_batch1_cases.txt`. Next real-world step: a veterinary professional fills in the CSV, then `ingest` → `check-split`, and the per-stratum rates say where the Tier-3 cascade is actually broken. **This batch is a diagnostic, not the accuracy number** — the per-case gold-eval batch that feeds `run_evaluation.py` comes after the problems it surfaces are fixed. Phases 1–3 to be re-brainstormed after that.

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
2. **First batch = row-level Tier-3 audit drawn from `test_cases.txt`** (revised 2026-08-12;
   supersedes the original group-stratified per-case draw — see "Row-level Tier-3 audit"
   below). Rows are drawn by quota from the Tier-2/Tier-3 strata only, with `sample_weight`
   recorded so per-stratum rates extrapolate to the Tier-3 population. Fix what it surfaces
   *before* spending held-out cases on the per-case accuracy batch — measuring accuracy
   ahead of fixing known bugs wastes the sample. The per-case gold-eval draw (stratified by
   group for rare-term coverage) remains the plan for that later batch.
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
rare-term coverage. Generate a **review CSV** (diagnosis text + `case_id` + the cascade's
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
Config constants: `config.GOLD_ANNOTATION_CSV`, `config.TIER3_AUDIT_REVIEW_CSV`,
`config.TIER3_AUDIT_INSTRUCTIONS_MD`, `config.TIER3_AUDIT_TAXONOMY_CSV`  
Package: `ml/annotation/gold/` (`sample`, `ingest`, `check_split`, `csv_io`)

**Review surface = plain CSV** (revised 2026-08-05; supersedes the earlier Excel-workbook
decision). `sample` emits three side-by-side files: the **review CSV** (one row per diagnosis
— `case_id` / `diagnosis`, the cascade suggestion in `cascade_matched_*`, the `decision_stage`
that produced it, the `sample_*` bookkeeping columns, and blank
`verdict` / `confirmed_term` / `confirmed_group` / `notes` columns to fill in),
**`tier3_audit_instructions.md`**, and **`tier3_audit_taxonomy.csv`** (the full Group/Term/Code
reference to copy corrections from). The ICD-O code is derived automatically from the chosen
group+term at ingest — the professional only sets `verdict` and, for `wrong`, the corrected
group/term.

**Diagnosis text only — no report sections** (decided 2026-08-05). The review CSV briefly
carried `HISTOPATHOLOGICAL SUMMARY` + `FINAL COMMENT` as reviewer context; they were removed
because the cascade never sees them. `_run_matching_pass` reads exactly one column
(`config.text_col`, the diagnosis) and the LLM prompt embeds only that string — even the
anatomic-site hint is extracted from the diagnosis itself. A reviewer holding the full report
would mark rows `wrong` on evidence the cascade structurally cannot access, producing errors
no cascade change could ever fix and inflating the measured error rate with unactionable
noise. Phase 0's number is the cascade's accuracy *at its own input*, so the reviewer gets
the cascade's own input and nothing more. `sample` no longer opens `report.csv` at all, and
`--reports-csv` is gone.

**Known bias this introduces downstream:** Step 4 below scores the *production inference*
pipeline against this gold store, and that pipeline **does** read HIST + FC+C + ANCILLARY
(concat_3). Diagnosis-only gold is therefore a conservative ruler for inference — the model
can read the HIST summary, produce a label better than the diagnosis line alone supports, and
be scored wrong for it. Correct for Phase 0's stated purpose; read the inference number with
that floor in mind.

A CSV has no dropdowns, so **typo-catching moved to `ingest`**: a `wrong` row whose
`confirmed_term` is absent from the taxonomy, or whose term/group pair doesn't exist, is a
hard validation error that aborts the whole ingest with a per-row report. Nothing is written
to the gold store unless every row resolves.

**Self-consistency duplicates were dropped** with the same revision. Hiding the `dup_pass`
column was what made the repeats unnoticeable, and a flat CSV cannot hide a column — a
visible flag would have turned Cohen's kappa into an optimistic ceiling rather than a real
measurement. Phase 0 therefore no longer reports an annotator noise floor, and the
`consistency` subcommand is gone. If the noise floor is wanted later, re-introduce it as a
separately-issued second batch of already-reviewed cases rather than as in-sheet repeats.

**Batching = 200 rows per sitting** (locked decision 2). `sample` defaults to `--n-rows 200`
and writes a per-batch case-ID ledger; later batches pass earlier ledgers to `--exclude-cases`.
Exclusion is per *case*, not per row: showing a reviewer a second row of a case they already
judged invites them to recall the earlier verdict instead of reading the new diagnosis.

### Row-level Tier-3 audit (revised 2026-08-12)

`annotation.csv` now carries `decision_stage`, which exposed how badly a case-level random
draw spends the review budget. The original batch 1 (675 rows / 200 cases) was 73.8%
`no_signal` and 21.3% `tier1_exact` — rows the cascade was never in doubt about. Only
**20 rows (3.0%)** were rows the LLM actually decided. Meanwhile **7,975 corpus rows**
silently became non-cancer (4,162 `tier3_llm` "No Match" + 3,813 `tier3_no_candidates`) with
nothing measuring whether they are real negatives or silent false negatives.

Re-stratifying the *case* draw by stage was tried first and rejected: because evaluation is
per-case, every selected case must emit all its diagnosis rows, and cases picked for their one
interesting row drag their `no_signal` siblings along. Measured, that gave 917 rows to review
for 177 useful ones (19.3%) — it raised burden 36% rather than lowering it.

`sample` therefore selects **rows, not cases**, from the Tier-2/Tier-3 pool only, splitting the
LLM tier by outcome because an accepted answer, a decline and a hedge are three different
questions (`_ROW_QUOTAS` in `annotation/gold/sample.py`):

| stratum | question asked | pool (test) | share |
|---|---|---:|---:|
| `tier3_llm_no_match` | the model refused to label it — **is there a cancer it missed?** | 849 | 25% |
| `tier3_no_candidates` | no shortlist was built, so the model was never asked — **recall hole** | 722 | 25% |
| `tier3_llm_answered` | the model picked a term — is it right? | 274 | 25% |
| `tier3_llm_uncertain` | the model called it hedged — is it genuinely unclassifiable? | 151 | 12.5% |
| `tier2_fuzzy` | partial-overlap match — same clinical entity? | 87 | 12.5% |

Result at `--n-rows 200`: **200 rows reviewed, 200 of them useful, 0% padding**, across 198
cases. Against the per-case alternative that is an **81% cut in reviewing for identical
Tier-3 content**.

**What this gives up.** A row-level sample cannot be scored by `evaluate.py`, which compares
predictions against a case's *whole* annotated term set — the un-sampled rows of a
partially-covered case would read as false positives. So this batch does **not** produce
Phase 0's "first honest accuracy number"; it produces a Tier-3 diagnostic. That is the
deliberate trade: fix what the audit surfaces first, and spend held-out cases on a per-case
accuracy batch afterwards, rather than measuring accuracy before fixing known bugs. The
`run_gold_annotation.py` docstring repeats this warning, and ingest stamps
`provenance=tier3_audit` so these rows stay distinguishable from any later per-case batch.

**Weighting.** The audit deliberately over-samples small strata, so raw rates off the store
are not Tier-3 population rates. Each row carries `sample_stratum` and `sample_weight`
(N_h/n_h); compute per-stratum rates and weight them. Verified: Σ(weight × rows) reconstructs
every stratum population exactly (849 / 722 / 274 / 151 / 87).


**Step 1 — Generate batch 1** (before the professional sits down):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py sample --n-rows 200 --batch 1
# Writes: ml/output/annotation/tier3_audit_review.csv        (the sheet to fill in)
#         ml/output/annotation/tier3_audit_instructions.md   (how to fill it in)
#         ml/output/annotation/tier3_audit_taxonomy.csv      (valid Group/Term/Code)
#         ml/output/annotation/tier3_audit_batch1_cases.txt  (cases touched by this batch)
# Later: --batch 2 --exclude-cases ml/output/annotation/tier3_audit_batch1_cases.txt
```
Hand all three reviewer files to the professional. They set `verdict` (correct / wrong /
no_cancer / uncertain) per row and, for `wrong`, copy `confirmed_group` + `confirmed_term`
from the taxonomy sidecar. Spelling matters — there are no dropdowns.

**Step 2 — Ingest into the gold store**:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py ingest --verified-by "Dr. Smith"
# Reads the filled .csv; derives matched_code from group+term via labels.csv.
# Aborts with a per-row report if any verdict is blank/unknown or any confirmed term
# is not in the taxonomy. Writes: ml/output/annotation/gold_annotation.csv
```

**Step 3 — Guard the split**:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_gold_annotation.py check-split
# PASS if all gold case IDs are in test_cases.txt and none are in train_cases.txt.
```

**Step 4 — Read the per-stratum rates.** There is deliberately no `run_evaluation.py` step
here: this batch is row-level, and `evaluate.py` scores per case (predicted terms vs the
case's *whole* term set), so the un-sampled rows of a partially-covered case would read as
false positives. Compute the rate within each `sample_stratum` and weight by `sample_weight`
to reach the Tier-3 population. The questions this answers:

- `tier3_llm_no_match` + `tier3_no_candidates` → **how many of the 7,975 silently-dropped
  corpus rows are real cancers?** This is the false-negative reservoir.
- `tier3_llm_answered` → precision of the LLM tier when it does commit.
- `tier3_llm_uncertain` → whether hedged wording is genuinely unclassifiable.
- `tier2_fuzzy` → whether token-overlap matching conflates distinct entities.

Fix what this surfaces, *then* cut the per-case gold-eval batch for the accuracy number.

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
- **Single annotator:** no inter-rater reliability, and as of 2026-08-05 no intra-annotator
  measurement either (the in-sheet duplicates were dropped with the move to a CSV review
  surface). The gold set currently has **no measured noise floor** — decide whether to
  re-establish one via a separately-issued re-review batch.
- **Behavior codes (/0, /2, /3):** annotation side uses a simplistic 3-digit regex
  (`_detect_behavior` in `pipeline.py`); a known failure mode the coder must improve on.
