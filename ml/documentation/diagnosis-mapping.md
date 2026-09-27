# Diagnosis mapping (silver)

Maps the clinic's free-text `diagnosis` field to a Vet-ICD-O-canine-1 `(term, group, code)` triple.
Package: `ml/diagnosis_mapping/` (`keyword_tiers.py`, `llm_tier.py`, `llm_client.py`, `cleanup.py`,
`silver.py`, `stats.py`). Entry point: `scripts/map_diagnoses.py`.

This is one of the three coding methods in [icd-mapping-strategy.md](icd-mapping-strategy.md)
("silver") — it never reads report text (that's [report-mapping.md](report-mapping.md), "bronze"),
only the diagnosis line.

## The cascade

`match_diagnosis` (in `silver.py`) runs each diagnosis through, in order, the first tier that
matches:

1. **Normalize + mask negation** (`keyword_tiers.py`) — lowercase, collapse hyphens/underscores/
   slashes to spaces, strip punctuation, expand abbreviations (`GIST`, `HSA`, `MCT`, `DLBCL`, ...),
   `neoplasia → neoplasm`; a negation masker blanks `no evidence of`, `negative for`, `rule out`,
   `not consistent with` and the following tokens, plus `non-X` compounds.
2. **Tier 1 — exact match** (`tier1_exact`) — a longest-first regex index built from the taxonomy
   (full normalized term + qualifier-stripped core form + 2–3 word permutations). `method=Exact,
   confidence=1.0`.
3. **Tier 2 — fuzzy token overlap** (`tier2_fuzzy`) — score = fraction of a label's core tokens
   present in the diagnosis; match at ≥85%. An explicit behavior modifier (`benign`/`malignant`/
   `metastatic`/`in situ`) first restricts candidates to that behavior digit. `method=Fuzzy,
   confidence ∈ [0.85, 1.0]`.
4. **Tier 3 — local LLM** (`llm_tier.py`) — only reached when `has_signal` finds a cancer-signal
   token (`-oma`/`-emia` suffix, or `tumor`/`leukemia`/`neoplasm`/`cancer`/`malignant`/`carcinoid`/
   etc.) and neither Tier 1 nor Tier 2 matched. A group token index (or an `-oma`/`-emia` suffix
   index as fallback) picks the likely group; up to 30 candidate terms from that group, plus
   detected anatomic-site keywords, go to a local OpenAI-compatible LLM (LM Studio/Ollama, via
   `llm_client.py` — local only, diagnosis text never leaves the machine). `method=LLM` on a match
   (`confidence=1.0` exact, `0.9` difflib near-match), else `No Match` (declined) or `Uncertain`
   (hedged).

`decision_stage` records *which gate* produced the row, since `method="No Match"` is otherwise
ambiguous (declined vs never asked):

| `decision_stage` | Meaning |
|---|---|
| `no_signal` | No cancer vocabulary at all; Tier 3 never reached. |
| `tier1_exact` | Keyword index matched. |
| `tier2_fuzzy` | Token-overlap matched. |
| `tier3_llm` | The LLM was called — matched, hedged, or declined. |
| `tier3_no_candidates` | Cancer signal present, but the candidate build came back empty; the LLM was never asked. |

## Ensemble cleanup

`cleanup.py::clean` — optional, runs by default. For every confirmed positive (Exact/Fuzzy/LLM),
sends the row to two diverse local LLMs (default `google/gemma-4-31b` + `qwen/qwen3.6-27b`, selected
from a 6-model bake-off for best calibration and architectural diversity), each returning `CORRECT`,
`WRONG_should_be:<term>`, `WRONG_no_cancer`, or `UNCERTAIN`:

| Both models say | Action |
|---|---|
| `CORRECT` | Keep original match |
| `WRONG_no_cancer` | Demote to `No Match` |
| `WRONG_should_be:<X>` (same X) | Replace match with X |
| Anything else (disagreement) | Optional tiebreaker model, else demote to `Uncertain` |

Cleanup never touches `decision_stage`, only `method` — so it can produce four
`(decision_stage, method)` pairs the cascade alone never would (`tier1_exact`/"No Match",
`tier1_exact`/"Uncertain", `tier2_fuzzy`/"No Match", `tier2_fuzzy`/"Uncertain"). See
[coding.md](coding.md) for how every pair maps to decisive/vague.

LM Studio connection settings come from `diagnosis_mapping/.env` (gitignored):

```ini
LLM_HOST=127.0.0.1
API_PORT=1234
LLM_MODEL=google/gemma-4-e4b
```

`LLM_MODEL` is the Tier-3 default (a 6-model bake-off winner for calibration among fast models);
override with `--model`. The cleanup pass's verifier models are independent, set with
`--cleanup-models`.

## Silver generations

A silver generation is a versioned, **immutable** output: `output/silver/<silver_id>/annotation.csv`
+ `manifest.json` (cascade constants sha256, LLM model, cleanup on/off, input CSV sha256,
`decision_stage` counts). `silver.py::run` refuses to write into an existing `<silver_id>/`;
`load_silver` verifies the manifest before reading.

```
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py run --id silver-1
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py run --id silver-1-no-llm --no-llm
ml/.venv/Scripts/python.exe ml/scripts/map_diagnoses.py stats --silver silver-0-legacy
```

- `run --id <new>` runs the cascade (and cleanup, unless `--skip-cleanup`) over
  `config.DIAGNOSES_CSV` and writes a new generation.
- `--no-llm` records every Tier-3-eligible row as a declined LLM match. That generation is refused
  by `load_silver` (and by the coding rule / Tier-3 sampler) unless `allow_no_llm=True` is passed
  explicitly — it must never be adopted as an authoritative silver source.
- `silver-0-legacy` is the pre-rewrite cleaned `annotation.csv`, imported once before cutover: rows
  unchanged, only a `silver_generation` column added. This is the silver generation every
  pre-cutover parity comparison used.
- `stats` (replaces `run_data_analysis.py`) writes coverage statistics
  (`config.DIAGNOSIS_MAPPING_STATS_DIR/<silver_id>/`): a combined report, per-analysis CSVs, and PNG
  plots (skip with `--no-plots`).

`ANNOTATION_COLUMNS`: `case_id, diagnosis_number, diagnosis, matched_term, matched_group,
matched_code, matched_keyword, method, confidence, decision_stage`.

## Known limitations

Carried over from the pre-rewrite cascade (unchanged by this rewrite):

- Metastasis diagnoses sometimes resolve to `Neoplasm, metastatic` even when a primary type appears
  in the text.
- Hedged parenthetical language (`"(SUSPECT METASTASIS)"`) occasionally matches rather than being
  flagged `Uncertain`.
- If the Tier-3 group token index picks the wrong group, the correct term never enters the LLM's
  candidate list (this is the `tier3_no_candidates` failure mode the Tier-3 audit in
  [manual-audit.md](manual-audit.md) measures).
- Tier 3 takes ~1–2s per LLM call; a full corpus run is tens of minutes plus the cleanup pass.
- No behavior-code disambiguation at Tier 1 (regex match only).
