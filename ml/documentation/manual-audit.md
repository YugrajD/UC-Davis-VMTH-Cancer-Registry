# Manual audit (gold)

The specialist's review: the Diagnosis-Mapping audit (formerly the Tier-3 audit), the
Report-Mapping audit, the case-level eval-batch draw (gold-eval), the review queue, the case-level
gold store, and the cause pass. Every case awaiting review goes out on one universal audit list for
the dashboard. Package: `ml/manual_audit/` (`sheets.py`, `terms.py`, `diagnosis_mapping_audit.py`,
`report_mapping_audit.py`, `eval_batch.py`, `audit_list.py`, `gold.py`, `cause_pass.py`). Entry
points: `scripts/audit.py`, and `scripts/handoff.py export-audit-list` for the list.

See [icd-mapping-strategy.md](icd-mapping-strategy.md) for how gold fits into gold/silver/bronze,
and "Measuring accuracy" for why the audit store and the gold store are kept separate.

## Two stores, never confused

- **Audit store** (`config.AUDIT_STORE_CSV`) — row-level: one judgement per (case_id,
  diagnosis_number, batch). It holds the 27 rows of Diagnosis-Mapping audit batch 1's pilot sheet,
  reviewed line by line before the audit moved to the dashboard (2026-09-27). A row-level judgement
  about one diagnosis line is not a case's complete code set, so it **never** counts as gold and
  never enters the corrected annotations. Nothing writes to it any more.
- **Gold store** (`config.GOLD_STORE_CSV`) — case-level: one row per (case_id, code), or exactly one
  `NO_CANCER` row. Every row carries an `origin` (`generations.guards.GOLD_ORIGINS`) — the only
  thing that later separates gold-eval (measures) from gold-train (trains):

| origin | comes from | counts as |
|---|---|---|
| `eval_batch` | an eval batch (test partition) | gold-eval |
| `random_slice` | a random slice of uploads | gold-eval |
| `review_queue` | the review queue | gold-train, on train cases only |
| `report_mapping_audit` | the Report-Mapping audit (train cases) | gold-train |
| `diagnosis_mapping_audit` | the Diagnosis-Mapping audit (calibration ∪ test cases) | neither: it scores the diagnosis mapping on a sample skewed to its hardest rows |

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

`sheets.py` is the shared read/write layer: reviewer-facing sheets are written `utf-8-sig`
(Excel/LibreOffice safe); sidecar key/reference tables are plain CSV round trips. **No sheet writer
here can carry a prediction column** — `write_case_sheet` (used by `eval_batch.py`) takes only case
IDs and blank fill-in column names, so there is no argument slot for a value to occupy. The
Diagnosis-Mapping audit keeps its key CSV here too; it no longer writes a reviewer sheet.

## Diagnosis-Mapping audit

Answers a different question than the eval batch: *when the diagnosis cascade reached Tier 2/3, was
it right, and are its declines silent false negatives?*

**Rows are drawn stratified by outcome**, from the eval side (calibration ∪ test) of a split,
skipping cases in earlier batches. `_ROW_QUOTAS`:

| stratum | question asked | share |
|---|---|---:|
| `tier3_llm_no_match` | the model refused — is there a cancer it missed? | 25% |
| `tier3_no_candidates` | no shortlist was built, so the model was never asked — recall hole | 25% |
| `tier3_llm_answered` | the model picked a term — is it right? | 25% |
| `tier3_llm_uncertain` | the model hedged — genuinely unclassifiable? | 12.5% |
| `tier2_fuzzy` | partial-overlap match — same clinical entity? | 12.5% |

Each batch writes two files under `config.DIAGNOSIS_MAPPING_AUDIT_DIR`:

- `diagnosis_mapping_audit_batch<N>_key.csv` — the sampled rows, the cascade's answer and the
  sampling weight. Stays local: it holds the diagnosis text.
- `diagnosis_mapping_audit_batch<N>.txt` — the batch's case IDs, one per line.

The specialist reviews each **whole case** on the dashboard (it goes onto the universal audit list),
and the review comes back as case gold with origin `diagnosis_mapping_audit`. Batch 1's case list is
also what the eval batch excludes from its frame.

```
ml/.venv/Scripts/python.exe ml/scripts/audit.py dm-sample --silver-id silver-0-legacy --batch 2
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py audit-rates
```

**Batch 1** (198 cases) was issued as row-level review sheets before this redesign. Its 30-row pilot
came back filled in with codes rather than the sheet's own format; 27 rows were converted and
ingested into the audit store (2026-09-27, reviewer MSK), and 3 free-text rows were left out. Its
other 171 cases (`pending_case_ids()`: a key row not in the audit store, and no gold) go on the
audit list. The issued sheets are kept in `diagnosis_mapping_audit/batch1_tier3_sheets/`.
`evaluate.py audit-rates` (`evaluation/audit_rates.py`) reports the audit store's per-stratum rates,
raw and weighted (Wilson interval on the Kish effective n); on the 27 pilot rows they are a
comprehension check, not a measurement.

## Report-Mapping audit

Asks: *which training labels is the report model right to disagree with?* A mix of rule-based and
random sampling, all from the train partition:

- **Contradicted labels** (rule-based): train cases whose out-of-fold gate probability confidently
  contradicts their label — labelled cancer with p < 0.2, or labelled no cancer with p > 0.8. Split
  evenly between the two, strongest contradiction first, so each batch takes the next-strongest.
- **Random**: a seeded uniform draw from every other train case with an OOF score — the baseline
  rate of wrong training labels the contradicted cases are compared against.

The OOF scores come from `train.py --stage oof` (5-fold, heads only, saved to
`config.OOF_DIR/case_presence_oof_<labels>_<split>.csv`). Both halves skip cases in earlier batches
or already gold. Each batch writes, under `config.REPORT_MAPPING_AUDIT_DIR`,
`report_mapping_audit_batch<N>.csv` (case_id, reason, target, oof_prob) and
`report_mapping_audit_batch<N>.txt` (case IDs). Reviews come back as gold with origin
`report_mapping_audit`, which is gold-train: it replaces the case's silver label in the corrected
annotations.

```
ml/.venv/Scripts/python.exe ml/scripts/train.py --stage oof --oof-stage case-presence --labels silver-0-legacy --split three-way-v1 --device cuda
ml/.venv/Scripts/python.exe ml/scripts/audit.py rm-sample --oof-csv ml/output/report_mapping/oof/case_presence_oof_silver-0-legacy_three-way-v1.csv --batch 1
```

Defaults: 100 contradicted + 100 random per batch.

## Universal audit list

One worklist of every case awaiting review, for the backend to show on the dashboard:
`audit_list_<list_id>.txt` in the handoff outbox, one case_id per line, with a sha256 sidecar. Order:
eval batch, Diagnosis-Mapping audit, Report-Mapping audit, review queue (in its own priority order).
A case listed by two sources appears once, under the first. Cases with gold are left off, so each
new list is the current full worklist.

The backend receives only case IDs. The gold origin each case must come back under is kept locally
in `config.AUDIT_LIST_LEDGER_CSV` (case_id, origin, list_id; first list wins, so a case never changes
origin). When the backend's gold export leaves `origin` blank or absent, `handoff.py import-gold`
fills it from that ledger; it refuses a blank origin for a case never listed, and a given origin that
differs from the listed one.

```
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-audit-list --list-id 2026-09-27
ml/.venv/Scripts/python.exe ml/scripts/handoff.py import-gold --csv PATH --export-id ID --reviewer "Dr. Smith"
```

## Case-level eval batch (gold-eval)

Draws cases from a split's **test** partition — minus the Diagnosis-Mapping audit's batch-1 cases
and any case already drawn or already gold — stratified by the case's silver group (plus a
`no_cancer` stratum). The specialist reviews in the registry app (not blind — the app already shows
the case's predicted code(s); `review_mode=app_non_blind` records this, and every gold-eval report
states that accuracy may be optimistic from anchoring). Drawn cases go on the universal audit list;
a case sheet is also written for a CSV review.

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
- `gold_train(split_id)` — origin ∈ `{review_queue, report_mapping_audit}`, restricted to the
  split's **train** partition. Such a row for a calibration/test case is excluded from both pools.
- `gold_snapshot_hash(split_id)` — a stable sha256 of the current gold-train rows, stamped into a
  report-mapping generation's manifest (`parents.gold_train_snapshot`).

An `eval_batch`-origin row must already be in `config.EVAL_BATCH_LEDGER_CSV` (refuses a case no
batch ever drew); a `random_slice` row must carry `slice_rate ∈ (0, 1]`; every origin except
`review_queue` must resolve a term, not a bare code.

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

## Decisions carried from the annotation-redesign plan

The 2026-06-17 gold/silver bootstrap plan (archived at
[archive/annotation-redesign-plan.md](archive/annotation-redesign-plan.md)) locked three decisions:

1. **Review surface = CSV-based MVP** — superseded on 2026-09-27: review happens on the dashboard,
   fed by the universal audit list. The eval batch and cause pass still write CSV sheets.
2. **First batch = row-level Tier-3 audit** drawn from the test split, quota-stratified by outcome —
   kept as the Diagnosis-Mapping audit's `_ROW_QUOTAS`; from 2026-09-27 the sampled cases are
   reviewed whole, as gold, rather than line by line.
3. **Execute the Tier-3 audit before the per-case gold-eval batch** — reflected in
   [icd-mapping-strategy.md](icd-mapping-strategy.md)'s roadmap item 1.1 preceding item 1.2
   ("First evaluation batch").

Diagnosis-text-only review (no report sections) applied to the row-level sheets; a whole-case
dashboard review sees the full record.
