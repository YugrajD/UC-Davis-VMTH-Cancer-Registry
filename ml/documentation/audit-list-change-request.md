# Change request for the backend: the dashboard review worklist (audit list)

**From:** ML. **To:** the backend developer.

The specialist now reviews every case ML needs a human answer on in one place: a worklist on the
dashboard. ML sends the worklist as a list of case IDs; the backend shows it, records each review as
the case's full code set, and sends the reviews back as a gold export. This replaces the review sheets
(the Tier-3 audit and eval-batch CSVs) and the review queue as the thing the specialist works from.

## 1. What ML sends: the audit list

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
review queue is left off for now and will come back, smaller, in a later list.

## 2. What the backend needs to build

1. **Load the list as the specialist's worklist**, in file order, replacing the previous list. A
   case that drops off a newer list no longer needs review.
2. **Review screen.** The specialist opens each case with its full record (diagnosis text, report,
   and the currently predicted codes; seeing the predictions is fine) and records one of:
   - **the case's complete set of cancer codes** — every reportable cancer in the case, not only the
     one on a particular diagnosis line. Each is a term chosen from the taxonomy
     (`ml/taxonomy/labels.csv`), ideally from a picker so nothing is typed; or
   - **no reportable cancer.**

   Never both for the same case. The review can be edited until it is exported.
3. **Record who reviewed each case.**
4. **Export the reviews as `gold_<export_id>.csv`** (UTF-8, header row):

   ```
   case_id,term
   CASE-0123,Mast cell neoplasms: Cutaneous mast cell tumor grade Patnaik II
   CASE-0123,"Fibromatous neoplasms: Fibrosarcoma, NOS"
   CASE-0789,"Epithelial neoplasms, NOS: Papillary adenocarcinoma"
   CASE-0456,NO_CANCER
   ```

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

ML refuses the whole file, and imports nothing from it, if any row: has a term that isn't in the
taxonomy; mixes `NO_CANCER` with a code on one case; repeats a code for a case; carries an `origin`
different from the list the case came from; or leaves out `origin` for a case that was never on a
list. The error names the case IDs.

## 3. What stays the same

- `pending_diagnoses_<export>.csv` (cloud → ML) and `silver_codes_<silver_id>.csv` /
  `combined_codes_<run>.csv` (ML → cloud) are unchanged.
- `review_queue_<run>.csv` is still produced by `handoff.py export-coding`, but it is no longer the
  specialist's worklist; the audit list is.
- The worker bundle is a separate request: [ml-worker-change-request.md](ml-worker-change-request.md).

## 4. What ML does on its side (for reference)

```
ml/.venv/Scripts/python.exe ml/scripts/handoff.py export-audit-list --list-id <id> [--no-review-queue]
ml/.venv/Scripts/python.exe ml/scripts/handoff.py import-gold --csv gold_<export_id>.csv --export-id <export_id> --reviewer "<name>"
```

ML keeps the origin of every listed case in `audit_list_ledger.csv`. Imported reviews become gold:
evaluation-batch cases measure accuracy, Report-Mapping audit cases correct training labels, and
Diagnosis-Mapping audit cases score the diagnosis-text mapping. See [manual-audit.md](manual-audit.md).
