# Change request for the backend: the dashboard review worklist (audit list)

**From:** ML. **To:** the backend developer.

## What ML sends

`audit_list_<list_id>.txt` (plus `audit_list_<list_id>.txt.manifest.json`, kind `audit_list`, schema
version 1, with the file's sha256), from `handoff.py export-audit-list`:

```
CASE-0123
CASE-0456
...
```

- One `case_id` per line, UTF-8, `\n` line endings, no header, no other columns.
- **In review order**: the first cases matter most (the evaluation sample, then the two audits, then
  the review queue by priority).
- **Each list is the complete current worklist**, not an increment. Cases that already have gold are
  left off, so a newer list replaces the older one.

## What the backend needs to do

1. **Show the list on the dashboard** as the specialist's review worklist, in file order. The
   specialist opens each case (full record, predicted codes) and records the case's true code set,
   or that it has no reportable cancer.
2. **Export the reviews as `gold_<export_id>.csv`**, as today (`case_id`, one row per code with the
   taxonomy `term`, or one `NO_CANCER` row). The `origin` column may now be left blank or omitted for
   cases that came from an audit list: ML fills it in from its own records. A case reviewed outside
   the list still needs its `origin`, and a given `origin` must match the one ML listed the case
   under, or ML refuses the whole import.
