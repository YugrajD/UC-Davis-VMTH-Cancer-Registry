# Manual audit (gold)

The specialist's review: the row-level Tier-3 audit (diagnostic, never gold), the case-level
eval-batch draw (gold-eval), the case-level gold store, and the cause pass. Package:
`ml/manual_audit/` (`sheets.py`, `terms.py`, `tier3_audit.py`, `eval_batch.py`, `gold.py`,
`cause_pass.py`). Entry point: `scripts/audit.py`.

See [icd-mapping-strategy.md](icd-mapping-strategy.md) for how gold fits into gold/silver/bronze,
and "Measuring accuracy" for why the audit store and the gold store are kept separate.

## Two stores, never confused

- **Audit store** (`config.AUDIT_STORE_CSV`) — row-level: one judgement per (case_id,
  diagnosis_number, batch) from the Tier-3 audit. A row-level judgement about one diagnosis line is
  not a case's complete code set, so it **never** counts as gold and never enters the corrected
  annotations.
- **Gold store** (`config.GOLD_STORE_CSV`) — case-level: one row per (case_id, code), or exactly one
  `NO_CANCER` row. Every row carries a mandatory `origin` ∈ `{eval_batch, review_queue,
  random_slice}` (`generations.guards.GOLD_ORIGINS`) — the only thing that later separates
  gold-eval (measures) from gold-train (trains).

**Gold is collected as taxonomy terms, not codes.** 186 of the taxonomy's 534 distinct codes map to
more than one term (always within the same group), so a code-only row is ambiguous and a blank term
would make `evaluation.verdicts.score` treat a real cancer case as non-cancer. `manual_audit/terms.py`
resolves a reviewer-typed term (case-insensitive; accepts `Group: Term` for the one ambiguous term,
"Papillary adenocarcinoma") to its code and group. A code-only row is accepted only for
`review_queue` (manual case-file corrections that already know the precise code) and only when that
code maps to exactly one term.

**Origin integrity.** A case keeps whatever origin it was first ingested under; re-ingesting under a
different origin is refused. Re-ingesting a case replaces *all* of that case's rows.

## Sheets

`sheets.py` is the shared read/write layer every audit tool uses: reviewer-facing sheets are
written `utf-8-sig` (Excel/LibreOffice safe); sidecar key/reference tables are plain CSV round
trips. **No sheet writer here can carry a prediction column** — `write_case_sheet` (used by
`eval_batch.py`) takes only case IDs and blank fill-in column names, so there is no argument slot
for a value to occupy. This is a **CSV-based review surface** (no Excel workbook, no external tool)
— the locked decision from the original bootstrap plan (see "Locked decisions" below).

The Tier-3 audit sheet is the one exception: it shows the cascade's own prediction, by its
row-level design (below).

## Row-level Tier-3 audit

Answers a different question than the eval batch: *when the diagnosis cascade reached Tier 2/3, was
it right, and are its declines silent false negatives?* Scored per row — writes to the audit store,
never the gold store.

**Row selection is stratified by outcome, not case**, because a case-level draw pulls in that
case's other, uninteresting `no_signal`/`tier1_exact` rows too (measured: 917 rows reviewed for 177
useful ones, an 81% burden increase over row-level for identical content). `_ROW_QUOTAS`:

| stratum | question asked | share |
|---|---|---:|
| `tier3_llm_no_match` | the model refused — is there a cancer it missed? | 25% |
| `tier3_no_candidates` | no shortlist was built, so the model was never asked — recall hole | 25% |
| `tier3_llm_answered` | the model picked a term — is it right? | 25% |
| `tier3_llm_uncertain` | the model hedged — genuinely unclassifiable? | 12.5% |
| `tier2_fuzzy` | partial-overlap match — same clinical entity? | 12.5% |

**Reviewer surface is three columns**: `Clinical Diagnosis`, `Predicted Match` (`(none)` when the
cascade found no cancer), `Actual Diagnosis` (the only fill-in). Everything else (case identity, the
cascade's full answer, sampling bookkeeping) lives in a separate key CSV that never goes to the
reviewer. The verdict is **derived**, not stated:

| `Actual Diagnosis` | `Predicted Match` | verdict |
|---|---|---|
| blank | a term | `correct` |
| blank | `(none)` | `no_cancer` |
| `unclear` (or similar) | either | `uncertain` |
| a taxonomy term | either | `wrong` |
| anything else | either | **abort**, per-row report |

**Known tradeoff: blank means agree.** A skipped row is indistinguishable from an agreed one — if
the reviewer stops partway, the remaining rows read as confirmations. Acceptable for a short pilot;
a caution for a full batch.

**Corrections are exact taxonomy terms**, resolved automatically to group/code — free text was
rejected because diagnosis wording contains a taxonomy term ~0% of the time on Tier-2/Tier-3 rows.

```
ml/.venv/Scripts/python.exe ml/scripts/audit.py tier3-sample --silver-id silver-0-legacy --test-cases-txt PATH --batch 1 --n-rows 200
ml/.venv/Scripts/python.exe ml/scripts/audit.py tier3-pilot --review-csv PATH --key-csv PATH --n-rows 30
ml/.venv/Scripts/python.exe ml/scripts/audit.py tier3-ingest --review-csv PATH --key-csv PATH --batch 1 --reviewer "Dr. Smith"
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py audit-rates
```

`tier3-pilot` splits a review CSV into a stratified pilot (proportional allocation, largest-remainder
rounding, a per-stratum floor) plus a remainder, so a short first pass can confirm the instructions
land correctly before the reviewer spends a full sitting on the rest — `--n-rows 30` is a
comprehension check, not a measurement (its `sample_weight` is proportionally short; `rates` warns
on a partial batch). `evaluate.py audit-rates` (`evaluation/audit_rates.py`) reports, per stratum,
both the raw sample rate and the weighted rate (Wilson interval on the Kish effective n, since the
audit deliberately over-samples small strata) — the false-negative reservoir estimate, declined-LLM
missed-cancer rate, `tier2_fuzzy` error rate, and more.

## Case-level eval batch (gold-eval)

Draws cases from a split's **test** partition — minus the 198 Tier-3 audit cases and any case
already drawn or already gold — stratified by the case's silver group (plus a `no_cancer` stratum),
and writes a case sheet the specialist reviews **in the registry app** (not blind — the app already
shows the case's predicted code(s); `review_mode=app_non_blind` records this, and every gold-eval
report states that accuracy may be optimistic from anchoring).

**Allocation is code-targeted**: a "big" stratum (≥1% of own-group codes) is drawn up to
`ceil(target_codes_per_group / codes_per_case)` cases (default target 30 codes/group); a stratum
that never reaches that share gets a fixed `rare_stratum_n` (default 2); `no_cancer` is fixed at
`no_cancer_target` (default 100). A series' `N_h`/targets are fixed from the **first** batch's frame
and never recomputed as gold accumulates — only the per-batch sampling pool shrinks (by cases
already ledgered or already gold, regardless of origin). Batches are issued in shares
(`--fraction`) of what's still remaining per stratum, not of the whole target.

```
ml/.venv/Scripts/python.exe ml/scripts/audit.py eval-batch --batch-id eval-batch-1 --silver-id silver-0-legacy --fraction 0.5
ml/.venv/Scripts/python.exe ml/scripts/audit.py ingest-sheet --sheet PATH --batch-id eval-batch-1 --reviewer "Dr. Smith"
```

Fill-in columns per case: `term_1..term_5` (rare case: >5 codes, list the rest in `notes`),
`no_cancer`, `notes`. `pooled_weights()` (`N_h / Σn_h` across the whole series) is the correct weight
once a series is complete — never a single batch's own local `sample_weight`.

## Gold store

```
ml/.venv/Scripts/python.exe ml/scripts/audit.py ingest-gold --rows-csv PATH --origin eval_batch --reviewer "Dr. Smith"
```

`gold.py` exposes:

- `gold_eval(split_id)` — origin ∈ `{eval_batch, random_slice}`, restricted to the split's test
  partition (a `random_slice` case outside every historical partition — a fresh upload — also
  counts). Never trained on.
- `gold_train(split_id)` — origin `== review_queue`, restricted to the split's **train** partition.
  A review-queue row for a calibration/test case is excluded from both pools — it measures nothing
  and trains nothing until a future use is defined.
- `gold_snapshot_hash(split_id)` — a stable sha256 of the current gold-train rows, stamped into a
  report-mapping generation's manifest (`parents.gold_train_snapshot`).

An `eval_batch`-origin row must already be in `config.EVAL_BATCH_LEDGER_CSV` (refuses a case no
batch ever drew); a `random_slice` row must carry `slice_rate ∈ (0, 1]`.

## Cause pass

On misses only (from `evaluation.gold_eval`'s miss table: `case_id, gold_code, method,
source_version`), the specialist answers one question per row: does the method's own input (the
diagnosis line for silver, the report text for bronze) actually support the gold code? "Yes" is a
method error, fixable in that method; "no" is an input gap.

```
ml/.venv/Scripts/python.exe ml/scripts/audit.py cause-sheet --verdicts-csv PATH --out-csv PATH
ml/.venv/Scripts/python.exe ml/scripts/audit.py ingest-cause --filled-csv PATH --reviewer "Dr. Smith"
```

Joined by `(case_id, gold_code, method)` regardless of `source_version` — the question is about the
method's input, which doesn't change between model versions.

## Locked decisions carried from the annotation-redesign plan

The 2026-06-17 gold/silver bootstrap plan (archived at
[archive/annotation-redesign-plan.md](archive/annotation-redesign-plan.md)) locked three decisions.
All three are implemented in the current code, described above:

1. **Review surface = CSV-based MVP** (KISS, single annotator, zero infra) — `manual_audit/sheets.py`
   writes plain CSVs; no Excel workbook or third-party annotation tool was ever adopted.
2. **First batch = row-level Tier-3 audit** drawn from the test split, quota-stratified by outcome —
   `tier3_audit.py`'s `_ROW_QUOTAS`, unchanged from the original tool so the already-issued sheets
   ingest byte-for-byte the same way.
3. **Execute the Tier-3 audit before the per-case gold-eval batch** — reflected in
   [icd-mapping-strategy.md](icd-mapping-strategy.md)'s roadmap item 1.1 ("Tier-3 audit... Pilot →
   remainder → fix the cascade") preceding item 1.2 ("First evaluation batch").

Other still-relevant details carried into the code and this doc: diagnosis-text-only review (no
report sections — the cascade never sees them, so showing more would inflate the measured error rate
with unactionable noise); the self-consistency duplicate columns from an earlier revision were
dropped (a flat CSV can't hide a `dup_pass` column, so a visible repeat would bias, not measure,
annotator noise — there is currently no measured single-annotator noise floor); and the three defects
found during the 2026-08-26 handoff round-trip (verdict not persisted, ingest not cumulative,
reviewer casing not canonicalized) are all fixed in `tier3_audit.py`/`gold.py` as described above.
