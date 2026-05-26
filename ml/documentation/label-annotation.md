# Label Annotation

Maps free-text pathology diagnoses to Vet-ICD-O-canine-1 labels. Produces `ml/output/annotation/llm_annotation.csv` (raw cascade) plus a cleaned `llm_annotation_cleaned.csv` in the same directory; the cleaned file is then promoted by the user to the canonical training-supervision path `ml/output/annotation/annotation.csv` (`config.ANNOTATION_CSV`), which every training and evaluation script reads by default. Annotation does not run at inference time — production sees only report text, not the structured diagnosis field.

Entry point: `ml/scripts/run_annotation.py`.

## Purpose

The database stores free-text diagnoses written by pathologists (e.g. `"Hemangiosarcoma, NOS"`, `"Mast cell tumor, grade II"`). To train the classifiers we need each diagnosis mapped to a standardized `(term, group, code)` triple from the Vet-ICD-O taxonomy. The annotation pipeline does that mapping. Cases with no match are treated as non-cancer negatives.

Inputs:
- `ml/data/diagnoses.csv` — columns: `case_id`, `diagnosis_number`, `diagnosis` (free text).
- `ml/ICD_labels/labels.csv` — the Vet-ICD-O-canine-1 taxonomy.

Outputs (under `ml/output/annotation/`):
- `llm_annotation.csv` — raw cascade results.
- `llm_annotation_cleaned.csv` — after the ensemble cleanup pass. Promote to `annotation.csv` (the canonical training input) when ready: `Copy-Item ml/output/annotation/llm_annotation_cleaned.csv ml/output/annotation/annotation.csv`.
- `cleanup_diff.csv`, `cleanup_summary.json` — cleanup audit.
- `llm_summary.{json,md}` — coverage stats.

## The 3-tier cascade

`ml/annotation/llm_pipeline/pipeline.py::run_llm_scan`. Each diagnosis is normalized (lowercase, expand abbreviations, mask negations) and then tried against three matchers in order. The first matcher that succeeds wins.

**Pre-pass: normalization.** Lowercase; collapse hyphens/underscores/slashes to spaces; strip commas/parentheses/semicolons/colons; expand abbreviations (`GIST → gastrointestinal stromal tumor`, `HSA → hemangiosarcoma`, `MCT → mast cell tumor`, `DLBCL → diffuse large b cell lymphoma`, etc.); replace `neoplasia → neoplasm`. A negation masker blanks out phrases like `no evidence of`, `negative for`, `rule out`, `not consistent with` and the following ~6 tokens, plus `non-X` compounds.

**Tier 1 — Exact match.** Reuses a longest-first regex index built from the taxonomy. For each label, both the full normalized term and its qualifier-stripped core form are added (and 2–3 word permutations). Match returns `method=Exact, confidence=1.0`.

**Tier 2 — Fuzzy token overlap.** For each label, score = fraction of the label's core tokens present in the normalized diagnosis. Match when ≥85%. If the diagnosis contains an explicit behavior modifier (`benign` / `malignant` / `metastatic` / `in situ`), Tier 2 first restricts candidates to that behavior digit and falls back to all candidates if nothing qualifies. Returns `method=Fuzzy, confidence` ∈ `[0.85, 1.0]`.

**Tier 3 — LLM resolution.** Only runs if the masked diagnosis still contains a cancer-signal token (`-oma`/`-emia` suffix, or `tumor`/`leukemia`/`neoplasm`/`cancer`/`malignant`/`carcinoid`/etc.). A group token index picks the highest-scoring group; up to 30 candidate terms from that group plus any detected anatomic-site keywords (`cutaneous`, `nodal`, `oral`, `mammary`, …) are passed to a local LM Studio LLM. The prompt asks for the exact text of the best candidate, with hard rules: negation → `no match`, hedging → `uncertain`, nothing fits → `no match`. Returns `method=LLM, confidence=1.0` for exact candidate match (0.9 for difflib near-match), or `No Match` / `Uncertain` otherwise.

## Ensemble cleanup pass

`ml/annotation/llm_pipeline/cleanup.py::run_cleanup`. Runs by default after the cascade. For every confirmed positive row (Exact / Fuzzy / LLM), the row is sent to two diverse local LLMs which each return one of: `CORRECT`, `WRONG_should_be:<term>`, `WRONG_no_cancer`, `UNCERTAIN`.

| Both models say | Action |
|---|---|
| `CORRECT` | Keep original match |
| `WRONG_no_cancer` | Demote to `No Match` |
| `WRONG_should_be:<X>` with same X | Replace match with X |
| Anything else (disagreement) | Optional 3rd-model tiebreaker, else demote to `Uncertain` |

Default verifier pair: `google/gemma-4-31b` and `qwen/qwen3.6-27b` (both calibrated, architecturally diverse). Override with `--cleanup-models a,b` and optionally `--cleanup-tiebreaker c`. Pass `--skip-cleanup` to stop after the cascade. The cleanup can also be re-run without redoing the cascade via `python ml/annotation/llm_pipeline/run_annotation_cleanup.py`.

## LM Studio setup

The annotation pipeline talks to a local OpenAI-compatible HTTP server (LM Studio or Ollama). Settings come from `ml/annotation/llm_pipeline/.env`:

```ini
LLM_HOST=127.0.0.1
API_PORT=1234
LLM_MODEL=google/gemma-4-e4b
```

`LLM_MODEL` is the Tier-3 default — `gemma-4-e4b` was selected after a 6-model bake-off (2026-05-09) for best calibration among the fast models; larger models like `medgemma-27b` and `llama-3.3-70b` had higher raw match rates but fabricated subtypes and are not recommended. Override at runtime with `--model`.

The cleanup pass's verifier models are independent of `LLM_MODEL`; use `--cleanup-models`.

`--list-models` prints the models the configured server currently exposes.

## Known limitations

- **Metastasis maps to primary or generic.** Diagnoses like `"LYMPH NODE: METASTASIS (SEE COMMENT)"` typically resolve to `Neoplasm, metastatic`. The LLM occasionally chooses this even when a primary type appears in the text.
- **Hedged language sometimes leaks through.** Parenthetical hedges (`"(SUSPECT METASTASIS)"`) are occasionally matched rather than flagged `Uncertain`.
- **Group identification can mis-scope candidates.** If the Tier-3 group token index picks the wrong group, the correct term never enters the LLM's candidate list.
- **Speed.** Tier 3 takes ~1–2 s per LLM call. A full 188k-row corpus runs ~30–60 minutes plus the cleanup pass.
- **No behavior-code disambiguation at Tier 1.** Tier 1 uses regex match only; if the diagnosis lacks an explicit modifier and the taxonomy term has none either, behavior code is whatever the matched label carries.

## Run commands

Full run (cascade + cleanup):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py
```

Skip cleanup (cascade only — faster, less accurate):
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --skip-cleanup
```

Quick test on the first 100 rows:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --max-rows 100
```

Use a specific Tier-3 model:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --model qwen/qwen3.6-27b
```

Re-run cleanup only against an existing `llm_annotation.csv`:
```bash
ml/.venv/Scripts/python.exe ml/annotation/llm_pipeline/run_annotation_cleanup.py
```

List models available on the configured LM Studio server:
```bash
ml/.venv/Scripts/python.exe ml/scripts/run_annotation.py --list-models
```

## Output schema

`llm_annotation.csv` (one row per input diagnosis row):

| Column | Description |
|---|---|
| `case_id` | Case identifier |
| `diagnosis_number` | Diagnosis ordering within the report (if present) |
| `diagnosis` | Original diagnosis text |
| `matched_term` | Taxonomy term (empty if No Match / Uncertain) |
| `matched_group` | Taxonomy group for the matched term |
| `matched_code` | Vet-ICD-O-canine-1 morphology code |
| `matched_keyword` | The keyword/token string that triggered the match |
| `method` | `Exact` / `Fuzzy` / `LLM` / `Uncertain` / `No Match` |
| `confidence` | 1.0 (Exact/LLM exact); 0.9 (LLM difflib near-match); 0.85–1.0 (Fuzzy); 0.0 otherwise |

`llm_annotation_cleaned.csv` has the same schema with rows rewritten per the cleanup vote.

`llm_summary.{json,md}` reports row totals, per-tier call statistics, case-level coverage, taxonomy coverage, term/group distributions, and imbalance counts.

## Code paths

| File | Role |
|---|---|
| `ml/annotation/llm_pipeline/pipeline.py` | Cascade (tiers 1–3), normalization, negation masking, prompt builder, summary writer |
| `ml/annotation/llm_pipeline/cleanup.py` | Ensemble verification pass |
| `ml/annotation/llm_pipeline/client.py` | OpenAI-compatible HTTP client |
| `ml/annotation/llm_pipeline/cli.py` | CLI orchestrating cascade + cleanup |
| `ml/annotation/llm_pipeline/audit.py` | 90-row stratified noise-audit harness |
| `ml/annotation/llm_pipeline/run_annotation_cleanup.py` | Re-run cleanup without redoing cascade |
| `ml/annotation/llm_pipeline/compare_llm_models.py` | N-model bake-off on a Tier-3 sample |
| `ml/annotation/llm_pipeline/.env` | LM Studio connection settings |
| `ml/ICD_labels/taxonomy.py` | Loads `labels.csv` |
| `ml/ICD_labels/labels.csv` | Vet-ICD-O-canine-1 taxonomy (845 terms, 52 groups) |
