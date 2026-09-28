# Change request for the backend: combined codes and the review worklist (audit list)

**From:** ML. **To:** the backend developer.

Two files from ML complete the picture:

- **`combined_codes_<run>.csv`** — the best code set ML currently has for every case, combining
  gold (specialist reviews), silver (the diagnosis-text mapping) and bronze (the report model). The
  backend loads it as the registry's code of record.
- **`audit_list_<list_id>.txt`** — the cases ML needs a specialist to review, as a worklist on the
  dashboard. The reviews come back to ML as a gold export, and the next combined-codes file carries
  them.

**The Audit Worklist replaces the Review Queue.** There is one place to review: a case-level
worklist on the dashboard. It replaces the Review Queue's per-row confirm/correct/reject, the
backend's own review gate at ingest, and the review sheets (the Tier-3 audit and eval-batch CSVs).

```
pending_diagnoses ──► ML ──► combined_codes + review_queue ──► registry codes
                       ▲  └─► audit_list ──► dashboard worklist ──┐
                       └──────────── gold export ◄────────────────┘
```

## 1. Combined codes: the registry's code of record

### What ML sends

`combined_codes_<run>.csv`, plus a sidecar `combined_codes_<run>.csv.manifest.json`
(`{"kind": "combined_codes", "schema_version": 1, "sha256": ..., "written_at": ...}`). UTF-8,
header row, no report or diagnosis text:

```
case_id,code,term,group,code_source,source_version,source_confidence,review_status
CASE-0123,9740.2/1,Cutaneous mast cell tumor grade Patnaik II,Mast cell neoplasms,diagnosis,silver-0-legacy,tier1_exact,auto_accepted
CASE-0123,8810/3,"Fibrosarcoma, NOS",Fibromatous neoplasms,diagnosis,silver-0-legacy,tier1_exact,auto_accepted
CASE-0456,NO_CANCER,,,diagnosis,silver-0-legacy,no_signal,auto_accepted
CASE-0789,8810/3,"Fibrosarcoma, NOS",Fibromatous neoplasms,report,gen-0-legacy,0.74,auto_accepted
CASE-0999,8050/3,Papillary adenocarcinoma,"Epithelial neoplasms, NOS",manual,eval_batch:2026-10-01-1,,confirmed
```

- **One row per code**; a case with several cancers has several rows. A non-cancer case has exactly
  one row with `code` = `NO_CANCER` and empty `term`/`group`.
- **`code_source`** — where the case's codes came from, one value per case:

  | `code_source` | Means | `source_version` | `source_confidence` |
  |---|---|---|---|
  | `manual` | a specialist reviewed the case (gold) | `<origin>:<gold export_id>` | empty |
  | `diagnosis` | the diagnosis-text mapping (silver) | the silver generation | the mapping stage that decided it (`tier1_exact`, `tier2_fuzzy`, `tier3_llm`, `no_signal`) — **text, not a number** |
  | `report` | the report model (bronze), only for a case with no diagnosis text | the model generation | the model's confidence, `0`–`1` |

  Gold beats silver beats bronze: a reviewed case is always `manual`, and the report model never
  overrides the diagnosis text.
- **`review_status`**, one value per case: `confirmed` (a specialist reviewed it), `auto_accepted`
  (ML is confident), or `queued` (ML's best guess, but it needs a review).
- **Each file is complete**: every case ML has coded, not only what changed since the last one.

### What the backend needs to build

1. **Load the file as the registry's codes.** Check it against the sidecar's `sha256` first. For
   every case in the file, replace all of that case's code rows with the file's rows. The registry's
   counts, maps and exports read these codes.
2. **Map the status onto the dashboard's:** `confirmed` and `auto_accepted` → confirmed, `queued` →
   pending. Keep `code_source`, `source_version` and ML's own `review_status` alongside, so the
   dashboard can tell a specialist's code from a machine's.
3. **Store `source_confidence` as text.** Show it as a confidence only when `code_source` is
   `report`; for `diagnosis` it names the mapping stage.
4. **Record no cancer at the case level**, not as a code row: a `NO_CANCER` case is coded and
   cancer-free, which is different from a case with no codes yet.
5. **Mark cases awaiting review with `review_queue_<run>.csv`** (sent with every combined-codes
   file, same `<run>`; only its `case_id` column matters here). A case on the review queue that has
   no row in the combined codes has no code yet: show it as *awaiting review*, with no codes, and
   leave it out of the registry's counts. A case in neither file hasn't reached ML yet; leave it as
   it is.
6. **Retire the backend's own review gate at ingest** (`REVIEW_AUTO_ACCEPT_CONFIDENCE`/`MARGIN`) for
   codes that come from ML: `review_status` already carries ML's decision, made with the same
   thresholds (0.23 confidence, 0.15 margin).
7. **Retire the Review Queue** (per-row confirm/correct/reject); the Audit Worklist (section 2)
   replaces it. The dashboard may show a reviewer's answer straight away, but it is not the code of
   record until it comes back from ML as `code_source=manual` in the next combined-codes file.

### When ML sends it

A new `combined_codes_<run>` + `review_queue_<run>` pair after every pending-diagnoses import and
every gold import. A newer `<run>` replaces the older one.

**Current file: `combined_codes_2026-09-27.csv`**, 59,763 rows across 54,103 cases:

| `code_source` | `review_status` | Cancer cases | No-cancer cases |
|---|---|---:|---:|
| `diagnosis` | `auto_accepted` | 25,260 | 28,749 |
| `report` | `auto_accepted` | 23 | 0 |
| `report` | `queued` | 5 | 66 |

`review_queue_2026-09-27.csv` holds 4,281 cases: 71 of them are the `queued` cases above, and the
other 4,210 have no code yet (awaiting review). No case is `manual` yet: no gold has come back.

## 2. The audit list: the specialist's worklist

### What ML sends

`audit_list_<list_id>.txt`, plus a sidecar `audit_list_<list_id>.txt.manifest.json`
(`{"kind": "audit_list", "schema_version": 1, "sha256": ..., "written_at": ...}`):

```
CASE-0123
CASE-0456
...
```

- One `case_id` per line, UTF-8, `\n` line endings, no header, no other columns.
- **In review order.** Work from the top: the first cases matter most.
- **Each list is the complete current worklist, not an increment.** A newer list replaces the older
  one entirely. Cases already reviewed (already gold on ML's side) are left off.
- Check the file against the sidecar's `sha256` before loading it.

**Current list: `audit_list_2026-09-27-2.txt`, 688 cases.** In order:

| Cases | What they are |
|---:|---|
| 417 | Evaluation batch 1: a stratified sample of test cases that measures how accurate the coding is |
| 171 | Diagnosis-Mapping audit: cases where the diagnosis-text mapping reached its fuzzy or LLM stage |
| 100 | Report-Mapping audit: cases where the report model confidently disagrees with the diagnosis-text code |

If `audit_list_2026-09-27.txt` (4,937 cases) reached you, discard it; `-2` replaces it. The large
review queue is left off the worklist for now and will come back, smaller, in a later list.

### What the backend needs to build

1. **Load the list as the specialist's worklist**, in file order, replacing the previous list. A
   case that drops off a newer list no longer needs review.
2. **Review screen: the full picture.** Each case opens with everything the reviewer needs to make
   the best judgement:
   - the patient's demographics;
   - the report text;
   - the clinical diagnosis line(s), if the case has any;
   - the case's combined codes (section 1): each code with its taxonomy term and group, and where it
     came from (`code_source`: an earlier review, the diagnosis text, or the report model).

   The reviewer then either:
   - **approves** the combined codes, confirming they are the case's **complete** set: every
     reportable cancer in the case, or none (`NO_CANCER`). Approving is a statement about the whole
     case, not only that the codes shown are plausible; or
   - **corrects** them: adds, removes or replaces codes, each a term chosen from the taxonomy
     (`ml/taxonomy/labels.csv`) with a picker so nothing is typed, or marks the case as having no
     reportable cancer.

   Rules for the screen:
   - Never codes and no cancer together on one case.
   - **One case at a time; no bulk approve.** Approving means the reviewer opened the record and
     checked it.
   - A case with no combined code yet (awaiting review) opens with nothing to approve; the reviewer
     codes it from scratch.
   - The review can be edited until it is exported.

   Showing the codes makes the review non-blind. The ML engineer has accepted this: accuracy
   measured on the evaluation-batch cases may read somewhat optimistic, because reviewers tend to
   accept what they are shown, and ML labels every accuracy report accordingly. No bulk approve keeps
   that effect small.
3. **Record who reviewed each case.**
4. **Export the reviews as `gold_<export_id>.csv`** (UTF-8, header row):

   ```
   case_id,term
   CASE-0123,Mast cell neoplasms: Cutaneous mast cell tumor grade Patnaik II
   CASE-0123,"Fibromatous neoplasms: Fibrosarcoma, NOS"
   CASE-0789,"Epithelial neoplasms, NOS: Papillary adenocarcinoma"
   CASE-0456,NO_CANCER
   ```

   - Approved and corrected cases export the same way: the case's final code set.
   - One row per code, or exactly one `NO_CANCER` row for a no-cancer case.
   - Write each term as **`Group: Term`**, exactly as in `labels.csv`. One taxonomy term,
     "Papillary adenocarcinoma", exists in two groups, and the group prefix makes every row
     unambiguous.
   - Standard CSV quoting: many groups and terms contain commas, so quote those fields.
   - **Leave out `origin`** for cases that came from an audit list; ML fills it in from its own
     records. Only a case reviewed *outside* the list needs an `origin` column value (for example
     `random_slice` for the random slice of new uploads, with the slice rate given to ML alongside
     the file).
   - **One file per reviewer**, and tell ML the reviewer's name with it. `export_id` must be new
     each time (a date plus a counter is fine).
   - Either export everything reviewed so far or only what's new since the last export: re-sending a
     case replaces its earlier review on ML's side.
5. **Send the file to ML** the same way the pending-diagnoses exports travel today.
6. **Before switching off the Review Queue, send ML the case IDs of every case with a row that was
   corrected or rejected there.** Those corrections are per-row, so they aren't gold, and the first
   combined-codes load would overwrite them. ML puts those cases at the top of the next audit list,
   so they are reviewed again as whole cases and nothing is lost.

ML refuses the whole file, and imports nothing from it, if any row: has a term that isn't in the
taxonomy; mixes `NO_CANCER` with a code on one case; repeats a code for a case; carries an `origin`
different from the list the case came from; or leaves out `origin` for a case that was never on a
list. The error names the case IDs.

## 3. What stays the same

- `pending_diagnoses_<export>.csv` (cloud → ML) and `silver_codes_<silver_id>.csv` (ML → cloud) are
  unchanged.
- `review_queue_<run>.csv` still comes with every combined-codes file, but only to mark cases
  awaiting review (section 1). It is not a worklist, and there is no separate Review Queue on the
  dashboard any more; its cases join the audit list when ML adds them to it.
- The worker bundle is a separate request: [ml-worker-change-request.md](ml-worker-change-request.md).

## 4. What ML does on its side (for reference)

```
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py combine --silver <silver_id> --split <split_id> --predictions <predictions.csv>
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py queue   --silver <silver_id> --split <split_id> --predictions <predictions.csv>
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-coding --run-id <run>
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-audit-list --list-id <id> [--no-review-queue]
ml/.venv/Scripts/python.exe ml/scripts/handoff.py import-gold --csv gold_<export_id>.csv --export-id <export_id> --reviewer "<name>"
```

The combination rule is in [coding.md](coding.md). ML keeps the origin of every listed case in
`audit_list_ledger.csv`. Imported reviews become gold: evaluation-batch cases measure accuracy,
Report-Mapping audit cases correct training labels, and Diagnosis-Mapping audit cases score the
diagnosis-text mapping. See [manual-audit.md](manual-audit.md).
