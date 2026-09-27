"""Gold-eval: the four results, weighted, with CIs; the cause-pass split; representativeness.

Per icd-mapping-strategy.md, "Measuring accuracy". Only gold-eval is read, via
``manual_audit.gold.gold_eval(split_id)`` (origin eval_batch / random_slice, on
the split's test side); gold-train and review-queue gold never enter.
``generations.guards.check_all`` runs first and refuses on any violation.

**Weights.** Every case carries one weight, and every verdict row of a case
takes its case's weight.
- ``eval_batch`` cases: ``eval_batch.pooled_weights`` (N_h / Σn_h across the
  whole series), computed over the ledger rows of the cases that already have
  gold-eval. A drawn case not yet reviewed therefore does not dilute its
  stratum's weight; the count of such cases is reported.
- ``random_slice`` cases: ``1 / slice_rate`` (slice_rate is the sampled share of
  an upload period, e.g. 0.05), in their own stratum ``random_slice:<period>``.
The stratum is the ledger stratum (or the random-slice period); it drives the
Kish/Wilson intervals and the stratified case-cluster bootstrap.

**Cells.** Every table row reports case and code counts, weighted verdict shares,
and for ``good`` and G+S both a Wilson interval on the Kish effective n
(``intervals.weighted_proportion``, per stratum) and a stratified case-cluster
bootstrap interval (``intervals.bootstrap``). ``tn_cases`` counts correct
abstentions (method said no cancer, gold agrees): the verdict table excludes
them, so a no-code stage would otherwise show only its misses.

**Silver in prediction form (result 1).** Each silver row gets a stage label:
its ``decision_stage``, with ``tier3_llm`` split into ``tier3_llm`` (answered),
``tier3_llm_declined`` (No Match) and ``tier3_llm_uncertain``; a row the
coding rule (``coding.rule.row_outcome``) calls vague is prefixed ``vague:``,
so vague rows are reported as their own stages with what silver *would have
said*. Coded rows become predicted codes (a repeated term within a case counts
once, at its first row). A case with no coded row becomes one ``Non-Cancer``
prediction, like a gate-rejected bronze case. Rows with no code in a case that
has codes are dropped (a non-cancer diagnosis line beside a cancer one is not a
miss). Missed gold codes and ``Non-Cancer`` rows take the case's stage: the
label of its vague rows if any, else of its coded rows, else of all its rows;
``mixed`` when those disagree. Gold-eval cases with no silver rows (no
diagnosis) are not scored for silver. Silver is scored without the uncommon-
group leniency, which belongs to the report-mapping generation; bronze uses it.

**Bands (result 2).** One band per case, from its top-ranked prediction (the
review gate in ``coding.combine`` is also per case, on the top prediction):
``non_cancer`` when that prediction is Non-Cancer (gate-rejected), else fixed
bands of width 0.2 on ``confidence``: ``0.0-0.2`` ... ``0.8-1.0`` (lower bound
inclusive; 1.0 falls in the top band). Fixed rather than quantile bands, so a
band means the same across generations. ``no_prediction`` when a case has no
prediction row.

**Disagreements (result 4).** Per case with silver rows, silver's term set (as
above) vs bronze's (non-Non-Cancer predicted terms). When they differ, the side
whose term set equals gold's term set is right; ``neither`` otherwise.
Cells are the case stage × the case band.

**Misses and the cause pass.** A miss is a gold code whose term the method did
not predict exactly. ``misses`` returns the table the cause sheet is built from
(``manual_audit.cause_pass.build_misses_sheet``). The cause split joins the cause
store on (case_id, gold_code, method) regardless of ``source_version``: the
question is whether the method's *input* supports the code, and inputs do not
change between versions.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

import config
import io_utils
from coding import rule
from diagnosis_mapping.silver import load_silver
from evaluation import intervals, silver_eval, verdicts
from generations import guards
from generations.splits import load_split
from manual_audit import eval_batch, gold

BAND_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NON_CANCER_BAND = "non_cancer"
NO_PREDICTION_BAND = "no_prediction"
MIXED = "mixed"
ALL = "all"
VAGUE_PREFIX = "vague:"
NON_CANCER = "Non-Cancer"
NON_BLIND_NOTE = "eval batches reviewed non-blind (app); accuracy may be optimistic"

MAX_HALF_WIDTH = 0.05     # overall per-code accuracy CI half-width
MIN_GROUP_CODES = 30      # gold codes per major group
MIN_GROUP_SHARE = 0.01    # a group is major at >= 1% of combined codes

_TIER3_LLM_LABEL = {"LLM": "tier3_llm", "No Match": "tier3_llm_declined", "Uncertain": "tier3_llm_uncertain"}


class GoldEvalError(Exception):
    """Gold-eval could not run (no gold-eval rows, inconsistent ledger, ...)."""


# ---------------------------------------------------------------------------
# Labels: bands, stages
# ---------------------------------------------------------------------------


def confidence_band(confidence: float) -> str:
    for low, high in zip(BAND_EDGES, BAND_EDGES[1:]):
        if confidence < high:
            return f"{low:.1f}-{high:.1f}"
    return f"{BAND_EDGES[-2]:.1f}-{BAND_EDGES[-1]:.1f}"


def case_bands(predictions: pd.DataFrame, case_ids) -> pd.Series:
    """Band per case, from its top-ranked prediction row."""
    rows = predictions[predictions["case_id"].isin(case_ids)].copy()
    rows["rank"] = pd.to_numeric(rows["diagnosis_index"])
    top = rows.sort_values(["case_id", "rank"]).drop_duplicates("case_id").set_index("case_id")
    bands = pd.Series(NO_PREDICTION_BAND, index=pd.Index(sorted(case_ids), name="case_id"))
    for case_id, row in top.iterrows():
        bands[case_id] = (NON_CANCER_BAND if row["predicted_term"] in verdicts.NON_CANCER_PRED_TERMS
                          else confidence_band(float(row["confidence"])))
    return bands


def stage_label(decision_stage: str, method: str) -> str:
    outcome = rule.row_outcome(decision_stage, method)  # raises on a pair outside the vagueness table
    label = _TIER3_LLM_LABEL[method] if decision_stage == "tier3_llm" else decision_stage
    return VAGUE_PREFIX + label if outcome == rule.VAGUE else label


def _one_label(labels) -> str:
    distinct = set(labels)
    return distinct.pop() if len(distinct) == 1 else MIXED


def silver_as_predictions(silver: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(silver rows in prediction form with a ``stage`` column, stage per case). See the module docstring."""
    rows = silver.copy()
    rows["stage"] = [stage_label(s, m) for s, m in zip(rows["decision_stage"], rows["method"])]
    rows["coded"] = rows["matched_term"].str.strip() != ""
    rows["order"] = pd.to_numeric(rows["diagnosis_number"], errors="coerce")
    rows = rows.sort_values(["case_id", "order"], kind="stable")

    case_stage = {}
    for case_id, group in rows.groupby("case_id"):
        vague = group.loc[group["stage"].str.startswith(VAGUE_PREFIX), "stage"]
        coded = group.loc[group["coded"], "stage"]
        case_stage[case_id] = _one_label(vague if len(vague) else coded if len(coded) else group["stage"])
    case_stage = pd.Series(case_stage, dtype=object)

    coded = rows[rows["coded"]].drop_duplicates(["case_id", "matched_term"])
    uncoded_cases = sorted(set(rows["case_id"]) - set(coded["case_id"]))
    predictions = pd.concat([
        pd.DataFrame({
            "case_id": coded["case_id"], "predicted_term": coded["matched_term"],
            "predicted_group": coded["matched_group"], "predicted_code": coded["matched_code"],
            "stage": coded["stage"],
        }),
        pd.DataFrame({
            "case_id": uncoded_cases, "predicted_term": NON_CANCER, "predicted_group": NON_CANCER,
            "predicted_code": "", "stage": [case_stage[c] for c in uncoded_cases],
        }),
    ], ignore_index=True)
    return predictions, case_stage


# ---------------------------------------------------------------------------
# Weights and scoring
# ---------------------------------------------------------------------------


def case_weights(gold_rows: pd.DataFrame, ledger: pd.DataFrame, split_id: str) -> pd.DataFrame:
    """One row per gold-eval case (index case_id): ``origin, weight, stratum, review_mode``."""
    cases = gold_rows.drop_duplicates("case_id").set_index("case_id")[["origin", "upload_period", "slice_rate"]]
    batch_cases = cases.index[cases["origin"] == "eval_batch"]
    ledger = ledger.set_index("case_id")
    unledgered = sorted(set(batch_cases) - set(ledger.index))
    if unledgered:
        raise GoldEvalError(f"{len(unledgered)} eval_batch gold case(s) are not in the eval-batch ledger: "
                            f"{unledgered[:5]}")
    reviewed = ledger.loc[batch_cases]
    wrong_split = sorted(set(reviewed["split_id"]) - {split_id})
    if wrong_split:
        raise GoldEvalError(f"the eval-batch series was drawn on split(s) {wrong_split}, not {split_id!r}")
    pooled = eval_batch.pooled_weights(reviewed.reset_index())

    out = pd.DataFrame(index=cases.index)
    out["origin"] = cases["origin"]
    out["stratum"] = reviewed["stratum"].reindex(out.index).astype(object)
    out["weight"] = out["stratum"].map(pooled).astype(float)
    out["review_mode"] = reviewed["review_mode"].reindex(out.index).fillna("")
    sliced = cases["origin"] == "random_slice"
    rates = pd.to_numeric(cases.loc[sliced, "slice_rate"], errors="coerce")
    bad = sorted(rates.index[~((rates > 0) & (rates <= 1))])
    if bad:
        raise GoldEvalError(f"{len(bad)} random_slice case(s) lack a slice_rate in (0, 1]: {bad[:5]}")
    out.loc[sliced, "weight"] = 1.0 / rates
    out.loc[sliced, "stratum"] = "random_slice:" + cases.loc[sliced, "upload_period"]
    out.attrs["unreviewed_ledger_cases"] = len(set(ledger.index) - set(batch_cases))
    return out


def gold_expectations(gold_rows: pd.DataFrame) -> pd.DataFrame:
    """Gold rows as a labels table (``NO_CANCER`` rows carry an empty term, i.e. no cancer)."""
    return gold_rows.rename(columns={"term": "matched_term", "group": "matched_group", "code": "matched_code"})


def _weighted(table: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["weight"] = table["case_id"].map(cases["weight"]).astype(float)
    table["stratum"] = table["case_id"].map(cases["stratum"])
    return table


def score_bronze(gold_rows: pd.DataFrame, predictions: pd.DataFrame, uncommon_groups,
                 cases: pd.DataFrame) -> pd.DataFrame:
    """Bronze vs gold verdict table on the gold-eval cases, with ``weight`` and ``stratum`` columns."""
    table = verdicts.score(gold_expectations(gold_rows), predictions[predictions["case_id"].isin(cases.index)],
                           uncommon_groups)
    return _weighted(table, cases)


def _true_negatives(predictions: pd.DataFrame, expectations: pd.DataFrame) -> pd.DataFrame:
    """Non-Cancer prediction rows of cases whose expectations hold no term."""
    with_terms = set(expectations.loc[expectations["matched_term"].str.strip() != "", "case_id"])
    abstained = predictions["predicted_term"].isin(verdicts.NON_CANCER_PRED_TERMS)
    return predictions[abstained & ~predictions["case_id"].isin(with_terms)]


# ---------------------------------------------------------------------------
# Table cells
# ---------------------------------------------------------------------------


def cell_metrics(rows: pd.DataFrame, cases: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Weighted verdict shares plus Kish/Wilson and case-cluster bootstrap CIs for good and G+S."""
    out = {"cases": rows["case_id"].nunique(), "codes": len(rows)}
    for verdict in verdicts.VERDICTS:
        out[verdict] = verdicts.share(rows, (verdict,), "weight")
    for name, hits in (("good", verdicts.GOOD), ("gs", verdicts.GOOD_PLUS_SLIGHT)):
        if rows.empty:
            out.update({name: float("nan"), f"{name}_lo": float("nan"), f"{name}_hi": float("nan"),
                        f"{name}_boot_lo": float("nan"), f"{name}_boot_hi": float("nan")})
            continue
        est, low, high = intervals.weighted_proportion(rows["verdict"].isin(hits), rows["weight"], rows["stratum"])
        metric = partial(verdicts.share, verdicts=hits, weight_col="weight")
        strata = cases.loc[sorted(set(rows["case_id"])), "stratum"]
        _, boot_low, boot_high = intervals.bootstrap(rows, metric, strata, n_boot, seed)
        out.update({name: est, f"{name}_lo": low, f"{name}_hi": high,
                    f"{name}_boot_lo": boot_low, f"{name}_boot_hi": boot_high})
    return out


def _cell_table(table: pd.DataFrame, key: str, tn: pd.Series, cases: pd.DataFrame,
                n_boot: int, seed: int) -> pd.DataFrame:
    """One row per value of ``table[key]`` plus an ``all`` row. ``tn``: key value per true-negative case."""
    records = []
    for value in sorted(set(table[key]) | set(tn)) + [ALL]:
        rows = table if value == ALL else table[table[key] == value]
        records.append({key: value, **cell_metrics(rows, cases, n_boot, seed),
                        "tn_cases": int(len(tn) if value == ALL else (tn == value).sum())})
    return pd.DataFrame(records)


def _share_with_ci(hit: pd.Series, cases: pd.DataFrame) -> tuple[float, float, float]:
    """Weighted case-level share with a Kish/Wilson interval; ``hit`` is indexed by case_id."""
    if hit.empty:
        return float("nan"), float("nan"), float("nan")
    return intervals.weighted_proportion(hit.to_numpy(), cases.loc[hit.index, "weight"],
                                         cases.loc[hit.index, "stratum"])


# ---------------------------------------------------------------------------
# The four results
# ---------------------------------------------------------------------------


def silver_vs_gold(gold_rows, silver_predictions, case_stage, cases, n_boot, seed) -> pd.DataFrame:
    """Result 1: by stage label. Cases without silver rows are not scored."""
    silver_cases = cases.index.intersection(case_stage.index)
    expectations = gold_expectations(gold_rows[gold_rows["case_id"].isin(silver_cases)])
    predictions = silver_predictions[silver_predictions["case_id"].isin(silver_cases)]
    table = _weighted(verdicts.score(expectations, predictions, frozenset()), cases)
    table["stage"] = table["stage"].fillna("").where(table["stage"].fillna("") != "",
                                                     table["case_id"].map(case_stage))
    tn = _true_negatives(predictions, expectations)
    return _cell_table(table, "stage", tn.set_index("case_id")["stage"], cases, n_boot, seed)


def bronze_vs_gold(bronze_table, gold_rows, predictions, bands, cases, n_boot, seed) -> pd.DataFrame:
    """Result 2: by case confidence band."""
    table = bronze_table.assign(band=bronze_table["case_id"].map(bands))
    tn_cases = _true_negatives(predictions[predictions["case_id"].isin(cases.index)],
                               gold_expectations(gold_rows))["case_id"].unique()
    return _cell_table(table, "band", bands[list(tn_cases)], cases, n_boot, seed)


def bronze_vs_silver(bronze_table, silver, predictions, uncommon_groups, cases, n_boot, seed) -> pd.DataFrame:
    """Result 3: bronze scored against gold and against silver (as silver-eval does) on the same cases,
    plus the paired (vs silver − vs gold) difference with a paired case-cluster bootstrap."""
    vs_silver = _weighted(verdicts.score(silver[silver["case_id"].isin(cases.index)],
                                         predictions[predictions["case_id"].isin(cases.index)],
                                         uncommon_groups), cases)
    records = [{"ruler": "gold", **cell_metrics(bronze_table, cases, n_boot, seed)},
               {"ruler": "silver", **cell_metrics(vs_silver, cases, n_boot, seed)}]
    for name, hits in (("good", verdicts.GOOD), ("gs", verdicts.GOOD_PLUS_SLIGHT)):
        metric = partial(verdicts.share, verdicts=hits, weight_col="weight")
        diff, low, high = intervals.paired_bootstrap(vs_silver, bronze_table, metric, cases["stratum"], n_boot, seed)
        records.append({"ruler": f"silver - gold ({name})", name: diff, f"{name}_boot_lo": low,
                        f"{name}_boot_hi": high})
    return pd.DataFrame(records)


def _term_sets(frame: pd.DataFrame, term_col: str) -> dict[str, frozenset[str]]:
    terms = frame[(frame[term_col].str.strip() != "") & ~frame[term_col].isin(verdicts.NON_CANCER_PRED_TERMS)]
    return {case_id: frozenset(group[term_col]) for case_id, group in terms.groupby("case_id")}


def disagreements(gold_rows, silver_predictions, case_stage, predictions, bands, cases) -> pd.DataFrame:
    """Result 4: who was right when silver and bronze disagree, by case stage × band."""
    gold_terms = _term_sets(gold_rows, "term")
    silver_terms = _term_sets(silver_predictions, "predicted_term")
    bronze_terms = _term_sets(predictions, "predicted_term")
    empty = frozenset()
    rows = []
    for case_id in cases.index.intersection(case_stage.index):
        truth, s, b = gold_terms.get(case_id, empty), silver_terms.get(case_id, empty), bronze_terms.get(case_id, empty)
        rows.append({"case_id": case_id, "stage": case_stage[case_id], "band": bands[case_id],
                     "disagree": s != b, "silver_right": s == truth, "bronze_right": b == truth})
    frame = pd.DataFrame(rows, columns=["case_id", "stage", "band", "disagree", "silver_right", "bronze_right"])
    frame = frame.set_index("case_id")
    frame["neither_right"] = ~frame["silver_right"] & ~frame["bronze_right"]

    cells = sorted(set(zip(frame["stage"], frame["band"]))) + [(ALL, ALL)]
    records = []
    for stage, band in cells:
        in_cell = frame if stage == ALL else frame[(frame["stage"] == stage) & (frame["band"] == band)]
        split = in_cell[in_cell["disagree"]]
        record = {"stage": stage, "band": band, "cases": len(in_cell), "disagreements": len(split)}
        for who in ("silver_right", "bronze_right", "neither_right"):
            est, low, high = _share_with_ci(split[who].astype(float), cases)
            record.update({who: est, f"{who}_lo": low, f"{who}_hi": high})
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Misses, cause pass, representativeness
# ---------------------------------------------------------------------------


def misses(gold_rows: pd.DataFrame, method_predictions: dict[str, tuple[pd.DataFrame, str, set]]) -> pd.DataFrame:
    """``case_id, gold_code, method, source_version``: gold codes each method did not predict exactly.

    ``method_predictions``: method -> (prediction frame with ``predicted_term``, source_version,
    cases the method ran on).
    """
    coded = gold_rows[gold_rows["term"].str.strip() != ""]
    records = []
    for method, (predictions, source_version, ran_on) in method_predictions.items():
        predicted = _term_sets(predictions, "predicted_term")
        for row in coded[coded["case_id"].isin(ran_on)].itertuples():
            if row.term not in predicted.get(row.case_id, frozenset()):
                records.append({"case_id": row.case_id, "gold_code": row.code, "method": method,
                                "source_version": source_version})
    return pd.DataFrame(records, columns=["case_id", "gold_code", "method", "source_version"])


def cause_split(miss_rows: pd.DataFrame, cause_store: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Per method: misses, reviewed, method errors (input supports gold) vs input gaps, weighted share."""
    answers = cause_store.set_index(["case_id", "gold_code", "method"])["input_supports"]
    keys = pd.MultiIndex.from_frame(miss_rows[["case_id", "gold_code", "method"]])
    joined = miss_rows.assign(input_supports=answers.reindex(keys).fillna("").to_numpy())
    records = []
    for method in sorted(set(joined["method"])):
        rows = joined[(joined["method"] == method)]
        reviewed = rows[rows["input_supports"] != ""]
        record = {"method": method, "misses": len(rows), "reviewed": len(reviewed),
                  "method_error": int((reviewed["input_supports"] == "yes").sum()),
                  "input_gap": int((reviewed["input_supports"] == "no").sum())}
        if len(reviewed):
            est, low, high = intervals.weighted_proportion(
                (reviewed["input_supports"] == "yes").to_numpy(), reviewed["case_id"].map(cases["weight"]),
                reviewed["case_id"].map(cases["stratum"]))
        else:
            est = low = high = float("nan")
        record.update(method_error_share=est, method_error_lo=low, method_error_hi=high)
        records.append(record)
    return pd.DataFrame(records)


def code_counts_by_group(table: pd.DataFrame, code_col: str, group_col: str) -> pd.Series:
    """Distinct (case, code) pairs per group, ignoring rows without a group."""
    coded = table[table[group_col].str.strip() != ""].drop_duplicates(["case_id", code_col])
    return coded.groupby(group_col).size()


def representativeness(gold_rows: pd.DataFrame, half_width: float, group_codes: pd.Series,
                       test_codes: pd.Series) -> dict:
    """The strategy's three criteria, pass/fail each.

    ``group_codes``: codes per group in the combined codes (or silver), for the 1% share.
    ``test_codes``: silver codes per group in the split's test partition — a major group with
    fewer than ``MIN_GROUP_CODES`` there cannot reach the target from test alone.
    """
    shares = group_codes / group_codes.sum()
    major = shares[shares >= MIN_GROUP_SHARE].sort_values(ascending=False)
    gold_codes = code_counts_by_group(gold_rows[gold_rows["code"] != gold.NO_CANCER], "code", "group")
    groups = pd.DataFrame({
        "share": major,
        "gold_codes": gold_codes.reindex(major.index, fill_value=0).astype(int),
        "test_codes": test_codes.reindex(major.index, fill_value=0).astype(int),
    })
    groups["pass"] = groups["gold_codes"] >= MIN_GROUP_CODES
    slice_cases = gold_rows.loc[gold_rows["origin"] == "random_slice", "case_id"].nunique()
    criteria = {
        "ci_half_width": {"value": half_width, "threshold": MAX_HALF_WIDTH, "pass": bool(half_width <= MAX_HALF_WIDTH)},
        "major_groups": {"table": groups, "pass": bool(groups["pass"].all()),
                         "unreachable_from_test": sorted(groups.index[groups["test_codes"] < MIN_GROUP_CODES])},
        "random_slice": {"cases": int(slice_cases), "pass": slice_cases >= 1},
    }
    criteria["pass"] = all(c["pass"] for c in criteria.values())
    return criteria


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------


def evaluate(gold_rows: pd.DataFrame, ledger: pd.DataFrame, silver: pd.DataFrame, predictions: pd.DataFrame,
             uncommon_groups, *, split_id: str, silver_id: str, generation_id: str,
             cause_store: pd.DataFrame, group_codes: pd.Series, test_codes: pd.Series,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Every gold-eval table on already-loaded inputs (``gold_rows`` = ``gold.gold_eval(split_id)``)."""
    if gold_rows.empty:
        raise GoldEvalError(f"no gold-eval rows for split {split_id!r}; ingest an eval batch or a random slice first")
    cases = case_weights(gold_rows, ledger, split_id)
    silver = silver[silver["case_id"].isin(cases.index)]
    predictions = predictions[predictions["case_id"].isin(cases.index)]
    silver_predictions, case_stage = silver_as_predictions(silver)
    bands = case_bands(predictions, cases.index)
    bronze_table = score_bronze(gold_rows, predictions, uncommon_groups, cases)

    bronze_overall = cell_metrics(bronze_table, cases, n_boot, seed)
    miss_rows = misses(gold_rows, {
        "silver": (silver_predictions, silver_id, set(case_stage.index)),
        "bronze": (predictions, generation_id, set(cases.index)),
    })
    notes = [NON_BLIND_NOTE] if (cases["review_mode"] == eval_batch.REVIEW_MODE).any() else []
    return {
        "counts": {
            "cases": len(cases), "gold_codes": int((gold_rows["code"] != gold.NO_CANCER).sum()),
            "no_cancer_cases": int((gold_rows["code"] == gold.NO_CANCER).sum()),
            "eval_batch_cases": int((cases["origin"] == "eval_batch").sum()),
            "random_slice_cases": int((cases["origin"] == "random_slice").sum()),
            "cases_without_silver": len(cases.index.difference(case_stage.index)),
            "unreviewed_ledger_cases": cases.attrs["unreviewed_ledger_cases"],
        },
        "cases": cases.assign(band=bands, stage=case_stage.reindex(cases.index).fillna("")),
        "silver_vs_gold": silver_vs_gold(gold_rows, silver_predictions, case_stage, cases, n_boot, seed),
        "bronze_vs_gold": bronze_vs_gold(bronze_table, gold_rows, predictions, bands, cases, n_boot, seed),
        "bronze_vs_silver": bronze_vs_silver(bronze_table, silver, predictions, uncommon_groups, cases, n_boot, seed),
        "disagreements": disagreements(gold_rows, silver_predictions, case_stage, predictions, bands, cases),
        "misses": miss_rows,
        "cause_split": cause_split(miss_rows, cause_store, cases),
        "representativeness": representativeness(
            gold_rows, (bronze_overall["good_boot_hi"] - bronze_overall["good_boot_lo"]) / 2, group_codes, test_codes),
        "notes": notes,
    }


def _read_store(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    return io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def run(predictions_csv: str | Path, silver_id: str, split_id: str, *, generation: str = "current",
        n_boot: int = 1000, seed: int = 0) -> dict:
    """Guards, then load gold-eval, ledger, silver, predictions and stores from config, then ``evaluate``.

    Major groups are read from ``config.COMBINED_CODES_CSV`` when it exists, else from silver
    (``group_basis`` in the result says which).
    """
    guards.check_all(split_id)
    gold_rows = gold.gold_eval(split_id=split_id)
    if gold_rows.empty:
        raise GoldEvalError(f"no gold-eval rows for split {split_id!r} in {config.GOLD_STORE_CSV}; "
                            "ingest an eval batch or a random slice first")
    predictions = silver_eval.read_predictions(predictions_csv)
    generation_id, uncommon_groups = silver_eval.generation_uncommon_groups(generation, predictions)
    silver = load_silver(silver_id).drop(columns=silver_eval.TEXT_COLUMNS, errors="ignore")

    if config.COMBINED_CODES_CSV.is_file():
        combined = io_utils.read_csv(config.COMBINED_CODES_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
        group_basis, group_codes = "combined codes", code_counts_by_group(combined, "code", "group")
    else:
        group_basis, group_codes = f"silver {silver_id}", code_counts_by_group(silver, "matched_code", "matched_group")
    test_silver = silver[silver["case_id"].isin(load_split(split_id).test)]

    result = evaluate(
        gold_rows, _read_store(config.EVAL_BATCH_LEDGER_CSV, eval_batch.EVAL_BATCH_LEDGER_FIELDS), silver,
        predictions, uncommon_groups, split_id=split_id, silver_id=silver_id, generation_id=generation_id,
        cause_store=_read_store(config.CAUSE_STORE_CSV, ["case_id", "gold_code", "method", "input_supports"]),
        group_codes=group_codes, test_codes=code_counts_by_group(test_silver, "matched_code", "matched_group"),
        n_boot=n_boot, seed=seed,
    )
    result.update(group_basis=group_basis, generation_id=generation_id)
    return result
