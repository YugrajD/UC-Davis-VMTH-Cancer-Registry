"""Report-Mapping audit: training cases whose label the report model contradicts, plus a random baseline.

Input is the cross-fit CSV (``scripts/oof.py`` -> ``report_mapping.training.crossfit``, saved to
``config.OOF_DIR``): out-of-fold gate and term probabilities, from models that never saw the case.

Rule-based half: train cases the cross-fit model confidently contradicts, in three reasons —

- labelled cancer with gate p < ``CONTRADICTED_CANCER_MAX_PROB`` (strongest = lowest p);
- labelled no cancer with gate p > ``CONTRADICTED_NO_CANCER_MIN_PROB`` (strongest = highest p);
- labelled cancer, gate p >= ``CONTRADICTED_TERM_MIN_GATE_PROB``, and every labelled term's probability below
  ``CONTRADICTED_TERM_MAX_PROB`` (a term outside the fold model's vocabulary, i.e. an empty ``term_prob``, counts
  as below it; strongest = lowest best-term probability, cases with no scored term last).

The half is split evenly across the reasons that have candidates, one case at a time, so each batch takes the
next-strongest of each.

Random half: a seeded uniform draw from every other train case with an OOF score. It measures how often an
ordinary training label is wrong, the baseline the contradicted cases' error rate is compared against.

Both halves skip cases already in an earlier batch or already in the gold store.
``sample`` writes per batch, under ``config.REPORT_MAPPING_AUDIT_DIR``:

- ``report_mapping_audit_batch<N>.csv`` — case_id, reason, target, oof_prob (the gate probability), term_prob
  (the lowest probability among the case's labelled terms the fold model scored; empty for a no-cancer case or
  when none was scored). No text.
- ``report_mapping_audit_batch<N>.txt`` — the batch's case IDs, one per line.

The review comes back as case-level gold with origin ``report_mapping_audit``,
which is gold-train (``manual_audit.gold.gold_train``): every case here is a
train case, so its gold replaces its silver label in the corrected annotations.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

import config
import io_utils
from generations.splits import load_split
from manual_audit.gold import load_gold

CONTRADICTED_CANCER_MAX_PROB = 0.2
CONTRADICTED_NO_CANCER_MIN_PROB = 0.8
CONTRADICTED_TERM_MIN_GATE_PROB = 0.8
CONTRADICTED_TERM_MAX_PROB = 0.2
CONTRADICTED_CANCER = "contradicted_cancer_label"
CONTRADICTED_NO_CANCER = "contradicted_no_cancer_label"
CONTRADICTED_TERM = "contradicted_term_label"
RANDOM = "random"
LEDGER_COLS = ["case_id", "reason", "target", "oof_prob", "term_prob"]
OOF_COLS = ["case_id", "gate_target", "gate_prob", "term", "term_prob"]

_LEDGER_NAME = re.compile(r"^report_mapping_audit_batch(\d+)\.csv$")


class ReportMappingAuditError(Exception):
    """A Report-Mapping audit draw was refused."""


def ledger_path(batch: int, out_dir: str | Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else config.REPORT_MAPPING_AUDIT_DIR
    return out_dir / f"report_mapping_audit_batch{batch}.csv"


def case_list_path(batch: int, out_dir: str | Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else config.REPORT_MAPPING_AUDIT_DIR
    return out_dir / f"report_mapping_audit_batch{batch}.txt"


def _batch_ledgers(out_dir: str | Path | None) -> list[Path]:
    out_dir = Path(out_dir) if out_dir is not None else config.REPORT_MAPPING_AUDIT_DIR
    found = [(int(m.group(1)), p) for p in out_dir.glob("*.csv") if (m := _LEDGER_NAME.match(p.name))]
    return [p for _, p in sorted(found)]


def _read(path: Path) -> pd.DataFrame:
    return io_utils.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def _case_scores(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per case: target, oof_prob (gate), term_prob (lowest scored labelled term, as written in the CSV),
    and the floats ``gate_p`` and ``best_term_p`` (highest labelled-term probability, 0 if none was scored)."""
    scored = oof.assign(p=pd.to_numeric(oof["term_prob"], errors="coerce"))
    by_case = scored.groupby("case_id")
    cases = pd.DataFrame({"target": by_case["gate_target"].first(), "oof_prob": by_case["gate_prob"].first(),
                          "best_term_p": by_case["p"].max().fillna(0.0)})
    cases["gate_p"] = cases["oof_prob"].astype(float)
    lowest = scored.dropna(subset=["p"])
    lowest = lowest.loc[lowest.groupby("case_id")["p"].idxmin()].set_index("case_id")["term_prob"]
    cases["term_prob"] = lowest.reindex(cases.index).fillna("")
    return cases.reset_index()


def sample(
    oof_csv: str | Path,
    split_id: str,
    batch: int,
    n_contradicted: int = 100,
    n_random: int = 100,
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> dict:
    """Draw one batch from the cross-fit scores in ``oof_csv`` and write its ledger CSV and case-ID list."""
    ledger_csv, case_list = ledger_path(batch, out_dir), case_list_path(batch, out_dir)
    existing = [p for p in (ledger_csv, case_list) if p.is_file()]
    if existing:
        raise ReportMappingAuditError(
            f"batch {batch} already exists ({', '.join(str(p) for p in existing)}); pick a new batch number")
    oof = _read(Path(oof_csv))
    missing = [c for c in OOF_COLS if c not in oof.columns]
    if missing:
        raise ReportMappingAuditError(f"{oof_csv} lacks column(s) {missing}; it must be the cross-fit CSV from scripts/oof.py")
    not_train = set(oof["case_id"]) - load_split(split_id).train
    if not_train:
        raise ReportMappingAuditError(
            f"{len(not_train)} OOF case(s) are not in {split_id!r}'s train partition, e.g. {sorted(not_train)[:5]}; "
            f"the OOF scores must come from that split")
    taken = {c for path in _batch_ledgers(out_dir) for c in _read(path)["case_id"]} | set(load_gold()["case_id"])
    cases = _case_scores(oof)
    pool = cases[~cases["case_id"].isin(taken)]

    is_cancer = pool["target"] == "1"
    candidates = {
        CONTRADICTED_CANCER: pool[is_cancer & (pool["gate_p"] < CONTRADICTED_CANCER_MAX_PROB)].sort_values(
            ["gate_p", "case_id"]),
        CONTRADICTED_NO_CANCER: pool[~is_cancer & (pool["gate_p"] > CONTRADICTED_NO_CANCER_MIN_PROB)].sort_values(
            ["gate_p", "case_id"], ascending=[False, True]),
        # Cases whose terms the fold model could not score at all (empty term_prob) come after the scored ones.
        CONTRADICTED_TERM: pool[is_cancer & (pool["gate_p"] >= CONTRADICTED_TERM_MIN_GATE_PROB)
                                & (pool["best_term_p"] < CONTRADICTED_TERM_MAX_PROB)].assign(
            unscored=lambda d: d["term_prob"] == "").sort_values(
            ["unscored", "best_term_p", "gate_p", "case_id"], ascending=[True, True, False, True]),
    }
    # One case at a time per reason: an even split, and a reason that runs out hands its turns to the others.
    take = dict.fromkeys(candidates, 0)
    while sum(take.values()) < n_contradicted and any(take[r] < len(df) for r, df in candidates.items()):
        for reason, df in candidates.items():
            if take[reason] < len(df) and sum(take.values()) < n_contradicted:
                take[reason] += 1
    contradicted = pd.concat([df.head(take[reason]).assign(reason=reason) for reason, df in candidates.items()])

    rest = sorted(set(pool["case_id"]) - set(contradicted["case_id"]))
    random_ids = random.Random(seed).sample(rest, min(n_random, len(rest)))
    randoms = pool.set_index("case_id").loc[random_ids].reset_index().assign(reason=RANDOM)

    rows = pd.concat([contradicted, randoms])[LEDGER_COLS]
    ledger_csv.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(rows, ledger_csv)
    case_list.write_text("".join(f"{c}\n" for c in rows["case_id"]), encoding="utf-8", newline="\n")
    return {
        "ledger_csv": ledger_csv, "case_list": case_list, "n_cases": len(rows),
        "counts": rows["reason"].value_counts().to_dict(),
        "pools": {**{reason: len(df) for reason, df in candidates.items()}, RANDOM: len(rest)},
    }


def pending_case_ids(out_dir: str | Path | None = None, gold_csv: str | Path | None = None) -> list[str]:
    """Every sampled case without gold yet, in batch then ledger order."""
    gold_cases = set(load_gold(gold_csv)["case_id"])
    pending: dict[str, None] = {}
    for path in _batch_ledgers(out_dir):
        for case_id in _read(path)["case_id"]:
            if case_id not in gold_cases:
                pending[case_id] = None
    return list(pending)
