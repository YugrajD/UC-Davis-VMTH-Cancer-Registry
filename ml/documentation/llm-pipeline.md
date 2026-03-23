# LLM Pipeline — How It Works

This document describes the **LLM pipeline** — the system that maps individual
diagnosis field text to standardized Vet-ICD-O-canine-1 labels using a combination
of keyword matching, a locally-hosted Ollama large language model, and an optional
Claude API reasoning fallback.

> **Role in the three-pipeline architecture:**
> - **LLM pipeline** (this document) — evaluation only: `diagnosis text → cancer label`
>   Lives in `evaluation/llm_pipeline/`. Produces ground-truth labels with higher
>   coverage than the keyword pipeline alone. Does not run in production.
> - **Keyword pipeline** (`evaluation/keyword_pipeline/`) — lighter, faster, no LLM required.
>   See [keyword-pipeline.md](keyword-pipeline.md).
> - **PetBERT pipeline** (`production/petbert_pipeline/`) — production system: `report text → cancer label`
>   See [petbert-pipeline.md](petbert-pipeline.md).

The LLM pipeline is a four-tier cascade (five with the optional Claude tier). Tiers 1 and 2
are rule-based and fast; Tier 3 calls a local Ollama LLM only when there is a clear cancer
signal in the text; Tier 4 optionally calls the Claude API for free-form reasoning on cases
Tier 3 could not resolve. This keeps LLM calls to ~15% of rows, making a full 43k-row run feasible.

---

## Why a Separate LLM Pipeline?

The keyword pipeline covers ~19% of diagnosis rows. The remaining ~81% are either
genuinely non-neoplastic or contain cancer terminology that doesn't appear verbatim
in the taxonomy. The LLM pipeline adds ~1–2 pp more matches on top of the keyword
pipeline's Tier 1, and handles:

- Abbreviations (`GIST`, `HSA`, `SCC`) not in the keyword index
- Metastasis phrasing (`LYMPH NODE: METASTASIS (SEE COMMENT)`)
- Novel phrasing not covered by the keyword index
- Subtype disambiguation (e.g. `B CELL LYMPHOMA, LARGE CELL` → `B-cell lymphoma, NOS`)

Cases not matched by any tier are treated as non-cancer (negative training examples).

---

## How It Works (High-Level)

1. Load `diagnoses.csv` (one diagnosis row per line; one report can have multiple rows).
2. Load the Vet-ICD-O taxonomy from `labels.csv` (~846 unique terms across 52 groups).
3. Build three indexes: **keyword index**, **oma/emia index**, **group index**.
4. For each diagnosis row, attempt matching in cascade:
   - **Tier 1 — Exact match** — keyword patterns (same as keyword pipeline, plus abbreviation expansion)
   - **Tier 2 — Fuzzy match** — token overlap on core terms (≥85% threshold)
   - **Tier 3 — LLM** — only if a cancer signal term is present; narrows candidates by group, then asks the model
   - **Tier 4 — Claude** *(opt-in via `--use-claude`)* — presents the full taxonomy to Claude for free-form reasoning on cases Tier 3 could not resolve
   - **No match** — row is treated as a non-cancer negative training example
5. Write `llm_predictions.csv`, `llm_summary.json`, and `llm_summary.md`.

---

## Pipeline Flow (Tier by Tier)

The entry point is `run_llm_scan()` in `ml/evaluation/llm_pipeline/pipeline.py`.

### Pre-pass: Normalization + Abbreviation Expansion

All text passes through `_normalize_llm()` before any tier:

1. Apply keyword pipeline's `_normalize()` (lowercase, strip punctuation, collapse whitespace, `neoplasia → neoplasm` synonyms).
2. Expand `metastasis → metastatic neoplasm` (for Tier 1 keyword matching).
3. Expand abbreviations via `_ABBREVIATIONS`:

| Abbreviation / Synonym | Expansion |
|---|---|
| `GIST` | gastrointestinal stromal tumor |
| `HSA` | hemangiosarcoma |
| `OSA` | osteosarcoma |
| `HCC` | hepatocellular carcinoma |
| `SCC` | squamous cell carcinoma |
| `MCT` | mast cell tumor |
| `TVT` | transmissible venereal tumor |
| `DLBCL` | diffuse large b cell lymphoma |
| `PNET` | primitive neuroectodermal tumor |
| `CPNET` | central primitive neuroectodermal tumor |
| `angiosarcoma` | hemangiosarcoma |
| `plasma cell tumor` | plasmacytoma |
| `perivascular wall tumor` | canine perivascular wall tumor |

### Tier 1: Exact Match

Reuses the keyword pipeline's `_build_keyword_index()` directly.
Scans longest-first; first match wins. Returns method `Exact`, confidence `1.0`.

Abbreviation expansion means diagnoses like `"SPLEEN: HSA"` match `Hemangiosarcoma, NOS`
without needing a separate keyword entry.

### Tier 2: Fuzzy Match

For each taxonomy label, compute the **token overlap** between the label's core term
(qualifier words stripped) and the diagnosis tokens. If ≥85% of the core tokens appear
in the diagnosis, it's a fuzzy match. Best score wins.

```
label: "Cutaneous epitheliotropic lymphoma"
core:  "cutaneous epitheliotropic lymphoma"   (no qualifiers stripped)
diagnosis: "SKIN: CUTANEOUS T-CELL LYMPHOMA WITH DIFFUSE INFILTRATE"
overlap: 2/3 = 0.67  →  below 0.85 threshold, no match
```

Short core terms (single token) are skipped to avoid false positives.
Returns method `Fuzzy`, confidence = overlap score (0.85–1.0).

### Tier 3: Signal Fallback + LLM

Only triggered if the normalized diagnosis contains a **cancer signal term**:
`-oma`/`-emia` suffix words (via the oma index) or explicit signal terms:
`tumor`, `tumour`, `leukemia`, `neoplasm`, `cancer`, `malignant`, `malignancy`,
`metastatic`, `carcinoid`, `mycosis fungoides`, `refractory anemia`,
`acanthomatous`, `fibromatosis`, and others.

**Candidate selection:**
A group-level keyword index maps diagnosis text to a taxonomy group (e.g. `"lymphoma"` → `Malignant lymphomas, NOS or diffuse`).
Candidates are all terms in that group (up to 30). If no group is identified, candidates come from oma/emia suffix lookup.

**LLM prompt:**
```
You are a veterinary oncology classifier. Map the diagnosis below to the best
matching ICD term.

Diagnosis: "..."

Candidate ICD terms:
1. ...
2. ...

Rules:
- Reply with ONLY the exact text of the best matching candidate.
- If the diagnosis is negated (e.g. "no evidence of", "rule out", "negative for") → reply: no match
- If the diagnosis is uncertain (e.g. "suspect", "presumed", "versus", "likely") → reply: uncertain
- If no candidate fits → reply: no match
```

**Response parsing:**
- `"no match"` → `No Match`
- `"uncertain"` → `Uncertain` (excluded from confirmed matches)
- Exact candidate text → `LLM` match, confidence `1.0`
- Near-match (difflib ≥0.80) → `LLM` match, confidence `0.9`
- Anything else → `No Match`

### Tier 4: Claude API Reasoning Fallback *(opt-in)*

Only triggered when all of Tiers 1–3 have failed **and** `--use-claude` is passed.
Calls `claude_classify()` in `client_claude.py` via the Anthropic SDK.

Unlike Tier 3, which narrows to a group-specific candidate list, Tier 4 presents the
**full taxonomy** (all ~846 terms, grouped by cancer category) so Claude can reason
across the entire label space. This is more expensive but allows it to catch cases where
Tier 3's group identification went wrong.

**Prompt structure:**

```
You are a veterinary oncology classifier. Map the diagnosis to the best Vet-ICD-O term.

Diagnosis: "..."

Full Vet-ICD-O Taxonomy (grouped by category):
  Soft tissue tumors: Hemangiosarcoma, NOS | Lipoma, NOS | ...
  Malignant lymphomas: B-cell lymphoma, NOS | ...
  ...

Rules:
- Reply with ONLY the exact term text.
- If the diagnosis is negated or uncertain → reply: no match
- If genuinely no term fits → reply: no match
```

**Response parsing:** same rules as Tier 3 — exact match, difflib ≥0.80 near-match,
or `No Match`/`Uncertain`. Returns method `Claude`, confidence `1.0`.

**Configuration:**
- Model: `CLAUDE_MODEL` env var in `.env` (default: `claude-haiku-4-5-20251001`)
- API key: `ANTHROPIC_API_KEY` environment variable (must be set; not in `.env`)
- Timeout: `--claude-timeout` CLI flag (default: 30s)

**Expected call volume:** Tier 4 is only reached by rows that failed Tiers 1–3 *and* still
had a cancer signal. In practice this is a small subset — on a 43k-row run with gemma2:27b,
Claude returned `No Match` or `Uncertain` for all calls, consistent with those cases being
genuinely ambiguous or not in the taxonomy.

### No Match

If all tiers fail (or Tier 3 fails and `--use-claude` is not set), the row is recorded
as `No Match` and treated as a non-cancer negative training example.

---

## Input Format

Same as the keyword pipeline — `ml/data/diagnoses.csv`:

| Column | Role |
|--------|------|
| `case_id` | Links diagnosis rows back to the original report |
| `diagnosis_number` | Ordering within the report (optional, passed through) |
| `diagnosis` | The free-text diagnosis string — the field that is matched |

---

## Configuration

The LLM pipeline connects to a locally-hosted **Ollama** server over Tailscale.
Connection settings live in `ml/evaluation/llm_pipeline/.env`:

```ini
TAILSCALE_IP=<your tailscale IP>
API_PORT=11434
OLLAMA_MODEL=gemma2:27b        # current recommended model
CLAUDE_MODEL=claude-haiku-4-5-20251001  # optional; used only with --use-claude
```

The `--model` CLI flag overrides `OLLAMA_MODEL` at runtime.

**Claude API key** must be set as `ANTHROPIC_API_KEY` in the shell environment (not stored
in `.env`). The `anthropic` package reads it automatically.

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `ml/data/diagnoses.csv` | Input diagnoses CSV path |
| `--id-col` | `case_id` | Case ID column name |
| `--diag-num-col` | `diagnosis_number` | Diagnosis number column name |
| `--text-col` | `diagnosis` | Column containing the diagnosis text |
| `--labels-csv` | `ml/labels/labels.csv` | Path to Vet-ICD-O taxonomy CSV |
| `--out-dir` | `ml/output/evaluation/llm_pipeline` | Output directory |
| `--max-rows` | all | Cap on input rows (for testing) |
| `--llm-timeout` | 60 | Seconds to wait per LLM call |
| `--model` | `.env` value | Ollama model name (overrides `.env`) |
| `--list-models` | — | Print available Ollama models and exit |
| `--compare-models` | — | Run all available models on `--max-rows` rows and print a comparison table |
| `--use-claude` | off | Enable Tier 4: call Claude API for cases Tier 3 could not match |
| `--claude-timeout` | 30 | Seconds to wait per Claude API call |

---

## Example Commands

**Standard full run** (from repo root):
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline
```

**Quick test on first 100 rows:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline --max-rows 100
```

**Use a specific model:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline --model llama3.3:70b
```

**List available models:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline --list-models
```

**Compare all available models on 500 rows:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline --compare-models --max-rows 500
```

**Full run with Claude Tier 4 fallback:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline --use-claude
```

**Test Claude fallback on first 100 rows:**
```bash
PYTHONPATH=ml ml/.venv/Scripts/python.exe -m evaluation.llm_pipeline --max-rows 100 --use-claude --claude-timeout 45
```

---

## Output Files

### `llm_predictions.csv`

One row per input diagnosis row.

| Column | Description |
|--------|-------------|
| `case_id` | Case identifier |
| `diagnosis_number` | Diagnosis ordering within the report (if present in input) |
| `diagnosis` | Original diagnosis text |
| `matched_term` | Taxonomy term matched (empty if No Match or Uncertain) |
| `matched_group` | Taxonomy group for the matched term |
| `matched_code` | Vet-ICD-O-canine-1 morphology code |
| `matched_keyword` | The keyword/token string that triggered the match |
| `method` | `Exact`, `Fuzzy`, `LLM`, `Claude`, `Uncertain`, or `No Match` |
| `confidence` | Match confidence: 1.0 (Exact/LLM), 0.85–1.0 (Fuzzy), 0.0 (No Match) |

### `llm_summary.json`

Aggregate run statistics:

| Field | Description |
|-------|-------------|
| `csv_path` | Input file path |
| `total_rows` | Total diagnosis rows processed |
| `matched_rows` | Rows with a confirmed ICD match (excludes Uncertain) |
| `uncertain_rows` | Rows marked Uncertain by the LLM |
| `match_rate_pct` | Percentage of rows with a confirmed match |
| `method_counts` | Counts per method (Exact, Fuzzy, LLM, Claude, Uncertain, No Match) |
| `tier_stats` | Per-tier call counts (see below) |
| `total_cases` | Unique case IDs in the input |
| `cases_with_confirmed_match` | Cases with ≥1 confirmed ICD match |
| `cases_with_uncertain` | Cases with ≥1 Uncertain diagnosis |
| `case_match_rate_pct` | Percentage of cases with a confirmed match |
| `unique_terms_matched` | Distinct taxonomy terms matched |
| `unique_groups_matched` | Distinct taxonomy groups matched |
| `imbalance` | top/bottom term counts, single-match term counts |
| `term_distribution` | Full count per matched term |
| `group_distribution` | Count + % for all 52 taxonomy groups (including zeros) |
| `top_matched_terms` | Top 20 terms by match count |
| `top_matched_groups` | Top 10 groups by match count |

The `tier_stats` object breaks down what happened at each tier:

| Field | Description |
|-------|-------------|
| `claude_enabled` | Whether `--use-claude` was passed |
| `signal_rows` | Rows where a cancer signal was detected (Tier 3 eligible) |
| `tier3_calls` | Actual Ollama API calls made (signal present and candidates found) |
| `tier3_matched` | Ollama returned a confirmed match |
| `tier3_uncertain` | Ollama returned "uncertain" |
| `tier3_no_match` | Ollama returned no match or errored |
| `tier4_calls` | Actual Claude API calls made |
| `tier4_matched` | Claude returned a confirmed match |
| `tier4_uncertain` | Claude returned "uncertain" |
| `tier4_no_match` | Claude returned no match or errored |

Note: `signal_rows - tier3_calls` = rows where signal was detected but no candidates could be
found (group identification failed and no oma/emia suffix matched the index).

### `llm_summary.md`

Human-readable version of `llm_summary.json` — tables for overview, cases, method breakdown,
tier statistics, taxonomy coverage, imbalance, full group distribution (all 52), and top 20 terms.

---

## Current Coverage (as of 2026-03-22)

From the most recent run over 42,973 diagnosis rows using `gemma2:27b` (without `--use-claude`;
Claude was called on the remaining signal cases but returned No Match for all, consistent
with those cases being genuinely ambiguous):

| Method | Count | % of total |
|--------|-------|-----------|
| `Exact` | 7,315 | 17.0% |
| `LLM` | 517 | 1.2% |
| `Fuzzy` | 86 | 0.2% |
| `Uncertain` | 78 | 0.2% |
| `No Match` | 34,977 | 81.4% |
| **Total matched** | **7,918** | **18.4%** |

**Case-level:** 5,614 of 12,486 cases (45.0%) have ≥1 confirmed ICD match.

The ~81% no-match rate reflects:
1. Many submissions are genuinely non-neoplastic (inflammatory, degenerative, normal findings).
2. The LLM conservatively rejects hedged language (`suspect`, `versus`, `presumed`) as Uncertain.
3. Each report produces multiple diagnosis rows; non-cancer secondary findings contribute to no-match rows.

---

## Comparison with Keyword Pipeline

| | Keyword | LLM (gemma2:27b) |
|---|---|---|
| Matched rows | 8,238 (19.2%) | 7,918 (18.4%) |
| Cases with match | 5,864 (47.0%) | 5,614 (45.0%) |
| Unique terms matched | 70 | 69 |
| LLM call overhead | none | ~517 calls / 43k rows |

The keyword pipeline matches more rows overall because it includes Pyogenic granuloma
(~298 rows) and is more permissive with hedged language. The LLM pipeline adds
~106 cases the keyword pipeline misses (primarily metastasis phrasing and rare subtypes)
but rejects ~426 keyword matches as Uncertain or No Match.

**The LLM pipeline is a replacement for the keyword pipeline, not a complement.**
The keyword pipeline has no negation handling — phrases like `"NO EVIDENCE OF NEOPLASIA"`
or `"RULE OUT LYMPHOMA"` are matched as cancer positives. The LLM pipeline's Tier 3
explicitly filters negated and hedged language. Taking a union of both pipelines would
reintroduce exactly the false positives the LLM pipeline is designed to reject, since
Tier 1 of the LLM pipeline already reuses the same keyword index.

Use the LLM pipeline output (`llm_predictions.csv`) as the sole ground-truth source
when running a full pipeline scan. The keyword pipeline remains available as a fast,
no-LLM fallback for quick testing or when the Ollama server is unavailable.

---

## Known Limitations

**Metastasis maps to primary or generic.**
Diagnoses like `"LYMPH NODE: METASTASIS (SEE COMMENT)"` — where no primary tumor type
is mentioned — are mapped to `Neoplasm, metastatic`. For cases where the primary type
appears in text (`"METASTATIC MAST CELLS"`), the LLM should return the primary tumor
type but occasionally still returns `Neoplasm, metastatic`.

**Hedged language sometimes leaks through.**
Despite `"suspect"`, `"presumed"`, and `"versus"` being in the uncertain-phrase list,
some parenthetical hedges (e.g. `"(SUSPECT METASTASIS)"`) are occasionally matched
rather than flagged as Uncertain.

**Group identification can mis-scope candidates.**
If `_identify_group()` assigns a diagnosis to the wrong group, all candidates will be
from that group and the correct term won't be offered to the LLM. This is uncommon
but can occur for diagnoses that mention multiple tissue types.

**Speed.**
At ~1–2s per LLM call and ~6k Tier 3 rows, a full 43k-row run takes 1.5–3 hours
depending on the model and server. Tier 1/2 rows are fast (no LLM call).

---

## Code Location

| File | Role |
|------|------|
| `ml/evaluation/llm_pipeline/pipeline.py` | Core logic: tiers 1–4, prompt builders, summary writer, `run_llm_scan` |
| `ml/evaluation/llm_pipeline/client.py` | Ollama HTTP client: `chat()`, `list_models()` |
| `ml/evaluation/llm_pipeline/client_claude.py` | Claude API client: `claude_classify()` (Tier 4) |
| `ml/evaluation/llm_pipeline/cli.py` | CLI argument parsing, `--list-models`, `--compare-models`, `--use-claude` |
| `ml/evaluation/llm_pipeline/__main__.py` | `python -m evaluation.llm_pipeline` entry point |
| `ml/evaluation/llm_pipeline/.env` | Connection settings (`TAILSCALE_IP`, `API_PORT`, `OLLAMA_MODEL`, `CLAUDE_MODEL`) |
| `ml/labels/taxonomy.py` | Vet-ICD-O taxonomy CSV parser |
| `ml/labels/labels.csv` | Vet-ICD-O-canine-1 taxonomy (~846 terms, 52 groups) |
| `ml/output/evaluation/llm_predictions.csv` | Output: per-row match results |
| `ml/output/evaluation/llm_summary.json` | Output: aggregate run statistics |
| `ml/output/evaluation/llm_summary.md` | Output: human-readable summary |
