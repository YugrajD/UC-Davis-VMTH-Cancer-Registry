# Coding

Applies the adoption rule (gold > silver > bronze) to every case, builds the report mapping's
training labels, and routes what's left to the specialist. Package: `ml/coding/` (`rule.py`,
`adopt.py`, `corrected.py`, `queue.py`). Entry point: `scripts/code_cases.py`.

See [icd-mapping-strategy.md](icd-mapping-strategy.md), "Coding a case", for the rule this
implements.

## The vagueness table (`rule.py`)

Whether a diagnosis-mapping row is **decisive** or **vague**, read from `(decision_stage, method)`:

| `decision_stage` / `method` | Outcome |
|---|---|
| `no_signal` / "No Match" | Decisive: no cancer vocabulary |
| `tier1_exact` / "Exact" | Decisive |
| `tier2_fuzzy` / "Fuzzy" | Decisive (provisional — pending the Tier-3 audit's finding) |
| `tier3_llm` / "LLM" | Decisive: the LLM answered a code |
| `tier3_llm` / "No Match" | Decisive non-cancer (a declined LLM answer — no switch thrown) |
| `tier3_llm` / "Uncertain" | Vague: the LLM hedged |
| `tier3_no_candidates` / "No Match" | Vague: cancer vocabulary present, LLM never asked |

Cleanup ([diagnosis-mapping.md](diagnosis-mapping.md)) can rewrite a confirmed Exact/Fuzzy row's
`method` to "No Match" (decisive non-cancer) or "Uncertain" (vague) without touching
`decision_stage`, producing four more pairs the cascade alone never would — `tier1_exact`/tier2_fuzzy
crossed with No Match/Uncertain. These are in the table too. **Any other `(decision_stage, method)`
pair raises `UnknownDecisionError`** rather than defaulting either way — a cascade/cleanup change
this table hasn't caught up with must fail loudly.

**A case is vague if any of its diagnosis rows is vague** (`case_is_vague`) — the whole case is
queued, per icd-mapping-strategy.md ("gold is the case's complete code set").

## Adoption rule (`adopt.py`)

Precedence, per case:

1. **Any gold row (any origin)** → adopt gold. `code_source=manual`, `review_status=confirmed`.
2. **Else, diagnosis rows where every row is decisive** → adopt silver. `code_source=diagnosis`,
   `review_status=auto_accepted`, `source_confidence=decision_stage`.
3. **Else, diagnosis rows where any row is vague** → **no adopted row here** — queued instead
   (`coding.queue`) until gold resolves it.
4. **Else (no diagnosis rows at all)** → adopt bronze. `code_source=report`,
   `source_confidence=confidence`; `review_status` is `queued` if bronze is low-confidence, else
   `auto_accepted`. **Bronze never overrides silver** — it is only ever adopted for a case with zero
   diagnosis rows.

**Exception:** an `unidentified_cancer` bronze case (the gate passed, but no label resolved) gets
**no** adopted row at all — its score is always 0.0, so it is not a confident non-cancer call and
must not adopt `NO_CANCER`. It lands in the review queue instead.

A non-cancer adoption (silver `no_signal`/declined-LLM, or a gate-rejected bronze row) is written as
one `NO_CANCER` row — the same sentinel `manual_audit.gold` uses.

**The bronze review gate** (`bronze_case_is_low_confidence`) mirrors the backend's current gate: a
case queues if any row's method is the pipeline's low-confidence rejection
(`rejected_by_case_presence`), or any row's confidence is below `BRONZE_LOW_CONFIDENCE_THRESHOLD`
(0.23), or (rank-1 only) the top1–top2 confidence margin is below `BRONZE_LOW_MARGIN_THRESHOLD`
(0.15). These three numbers live together in `adopt.py` for easy recalibration once the random
slice gives a real basis — they are carried-over backend defaults, not yet validated against this
pipeline's own data.

```
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py adopt --silver silver-0-legacy --split three-way-v1 --predictions PATH
```

`ADOPTED_CODES_COLUMNS`: `case_id, code, term, group, code_source, source_version,
source_confidence, review_status`.

## Corrected annotations (`corrected.py`)

The report mapping's training labels — **train partition only**. Per case: gold-train (origin ==
`review_queue`, restricted to the split's train cases) replaces the case's silver rows entirely,
whether or not the case was actually vague; a vague case *without* gold-train is excluded outright
(its silver rows aren't usable training signal, and there's nothing to replace them with); a
decisive silver case with no gold-train keeps its silver rows unchanged. Gold-eval and the row-level
audit store are never read here — only `gold_train()`.

```
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py corrected --silver silver-0-legacy --split three-way-v1
```

Written to the fixed path `config.CORRECTED_ANNOTATIONS_CSV` (not a versioned generation directory
— it's a reproducible, regenerate-on-demand derived table; `generations.guards` reads it from that
exact path). The write is crash-safe: the guards (`check_labels_train_only`,
`check_eval_queue_gold_not_trained`, `check_corrected_sources`) run on the in-memory frame first, and
only on success does the module write to a temp file and `os.replace` it into place — a guard
failure never touches the target path.

Output columns: `case_id, matched_term, matched_group, matched_code, label_source (gold|silver),
silver_generation, gold_snapshot`.

## Review queue (`queue.py`)

Three reasons, skipping any case that already has gold (any origin):

- `vague_silver` — the case has diagnosis rows and at least one is vague.
- `low_conf_bronze` — no diagnosis rows at all, and bronze's own review gate flags it (also covers
  an `unidentified_cancer` bronze case, which `coding.adopt` refuses to adopt).
- `no_evidence` — no gold, no diagnosis row, no bronze prediction at all (e.g. an upload whose
  report never made it into `report.csv`). Always the lowest priority
  (`NO_EVIDENCE_PRIORITY = -1.0000`, below any real bronze probability) — there's no signal to rank
  it by, and it must never crowd out a case bronze or silver actually ran on.

```
ml/.venv/Scripts/python.exe ml/scripts/code_cases.py queue --silver silver-0-legacy --split three-way-v1 --predictions PATH
```

**Priority** is bronze's case-presence probability, descending (bronze runs on every case, so this
applies to `vague_silver` rows too). **Ordering**: every test-partition item sorts after every
train/calibration item, priority-descending within each block — a case under review becomes gold,
and `eval_batch.generate_batch` excludes any case with existing gold from its draw pool, so draining
train/calibration first delays depleting the test-partition sampling pool as long as possible.

**Coverage invariant**: every case_id in `split.train ∪ split.calibration ∪ split.test ∪ silver ∪
bronze` ends up either adopted (`coding.adopt`) or queued (here) — never neither.

`REVIEW_QUEUE_COLUMNS`: `case_id, reason, priority, partition, silver_generation, bronze_generation`.
