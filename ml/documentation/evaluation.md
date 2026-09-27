# Evaluation

Scores predictions against expectations, at every level from a raw verdict table up to the four
gold-eval results with confidence intervals. Package: `ml/evaluation/` (`verdicts.py`,
`intervals.py`, `silver_eval.py`, `gold_eval.py`, `audit_rates.py`). Entry point:
`scripts/evaluate.py silver|gold|audit-rates`.

## Verdicts (`verdicts.py`)

Exact port of the pre-rewrite per-code scorer. Every predicted label gets one verdict:

| Verdict | Meaning |
|---|---|
| `good` | Predicted term exactly matches a verified label for the case |
| `slightly_off` | Correct group, wrong specific term |
| `completely_off` | Neither term nor group matches any verified label |
| `false_positive` | The case has no verified labels; the model predicted cancer anyway |
| `false_negative` | The case has verified labels; the model predicted `Non-Cancer`/`Uncategorized`, or has no prediction row at all |
| `true_negative` | Correctly predicted non-cancer for a non-cancer case (excluded from the verdict table and from metrics) |

`GOOD_PLUS_SLIGHT` (G+S) is the exact per-code row share of `good ∪ slightly_off` — not
`round(good) + round(slight)`.

## Confidence intervals (`intervals.py`)

- `wilson` — Wilson score interval for a binomial proportion.
- `kish_n_eff` — Kish effective sample size of a set of weights (accounts for unequal weighting
  inflating variance).
- `weighted_proportion` — a stratified weighted proportion with a Wilson interval on the Kish
  effective n. Treats rows as independent; for per-code metrics (where a case's codes are
  correlated) use the bootstrap instead.
- `bootstrap` / `paired_bootstrap` — a **stratified case-cluster bootstrap**: cases (not individual
  rows) are resampled with replacement within each stratum, so every row of a case moves together.
  95% percentile intervals. A replicate that draws only zero-row cases (e.g. all true negatives) is
  skipped (undefined metric).

## Silver-eval — the cheap ruler (`silver_eval.py`)

Scores bronze predictions against a labels table (a silver generation, or the corrected
annotations), restricted to one partition of a split. Both tables are filtered to the partition's
cases first; a case with no label rows counts as non-cancer.

```
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py silver --predictions PATH --labels silver-0-legacy --split three-way-v1 --partition test
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py silver --predictions PATH --labels silver-0-legacy --split legacy-80-20 --partition test --half eval
```

- `--half {eval,sweep}` keeps only the md5 half of the partition (`generations.splits.in_sweep_half`)
  — how the frozen parity reference is scored.
- The uncommon-groups list comes from the report-mapping generation that produced the predictions
  (matched against the predictions' `generation_id` — refuses a mismatch).
- Per-group/per-term breakdowns key on the **expected** group/term (a row counts toward every group
  its case expects; false positives, with no expectation, are left out) — legacy keyed on the
  predicted group and silently dropped false negatives from that table.
- Appends one line per run to `config.SILVER_EVAL_HISTORY_CSV`.

## Gold-eval — the four results (`gold_eval.py`)

Per [icd-mapping-strategy.md](icd-mapping-strategy.md), "Measuring accuracy". Reads only
`manual_audit.gold.gold_eval(split_id)` (origin `eval_batch`/`random_slice`, on the split's test
side) — gold-train and review-queue gold never enter. `generations.guards.check_all` runs first.

```
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py gold --predictions PATH --silver silver-0-legacy --split three-way-v1
```

**Weights.** Every gold-eval case carries one weight; every verdict row of that case takes it:
- `eval_batch` cases: `eval_batch.pooled_weights` (N_h / Σn_h across the whole batch series) —
  computed only over cases that already have gold, so an un-reviewed drawn case doesn't dilute its
  stratum's weight.
- `random_slice` cases: `1 / slice_rate`, in their own `random_slice:<period>` stratum.

**The four results, always reported together, on the same cases:**

1. **Silver vs gold, by decision_stage** — silver's diagnosis rows are recast as predictions (a
   `stage` per row: its `decision_stage`, with `tier3_llm` split into answered/declined/uncertain,
   and a vague-outcome row prefixed `vague:`), then scored against gold like any prediction.
2. **Bronze vs gold, by confidence band** — one band per case from its top prediction: `non_cancer`
   (gate-rejected), else fixed 0.2-wide bands on `confidence` (`0.0-0.2` ... `0.8-1.0`), or
   `no_prediction`.
3. **Bronze vs silver, pre-gold-correction** — bronze scored against gold and against silver on the
   same cases, plus the paired (vs-silver − vs-gold) difference with a paired case-cluster
   bootstrap. Tests whether the cheap silver-eval ruler tracks the real one.
4. **Who was right on disagreements**, by `decision_stage × bronze band` — per case where silver's
   term set and bronze's differ, whichever side matches gold is right; `neither` otherwise.

Every cell reports case/code counts, weighted verdict shares, and for `good`/G+S both a Wilson
interval (Kish n) and a stratified case-cluster bootstrap interval. `tn_cases` counts correct
abstentions separately, since the verdict table itself excludes them.

**Misses and the cause pass.** `misses()` returns `case_id, gold_code, method, source_version` for
every gold code a method didn't predict exactly — the input to
[manual-audit.md](manual-audit.md#cause-pass)'s cause sheet. `cause_split()` joins the cause store
back in (by `case_id, gold_code, method`, regardless of `source_version` — the question is about the
method's input, which doesn't change between model versions) to report method-error vs input-gap
shares.

**Representativeness** — three pass/fail criteria (icd-mapping-strategy.md's proposed thresholds):
overall `good` CI half-width ≤ 5 points; every group at ≥1% of adopted codes has ≥30 gold codes
(reports which majors are unreachable from the test partition alone); at least one random-slice
case exists. Until all three pass, gold-eval numbers are indicative and promotion is a human call.

Every gold-eval result carries a `NON_BLIND_NOTE` whenever any case in it was reviewed
`app_non_blind` — see [manual-audit.md](manual-audit.md).

## Tier-3 audit rates (`audit_rates.py`)

Per-stratum verdict rates for the row-level Tier-3 audit (`config.AUDIT_STORE_CSV`), with Wilson/Kish
CIs — both the raw sample rate and the weighted (population) rate, since the audit deliberately
over-samples small strata. Deliberately *not* wired into `verdicts.score` — a row-level sample would
read the un-sampled rows of a partially-covered case as false positives.

```
ml/.venv/Scripts/python.exe ml/scripts/evaluate.py audit-rates
```
