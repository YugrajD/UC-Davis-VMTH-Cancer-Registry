"""Report-Mapping audit: training cases whose label the report model contradicts, plus a random baseline.

Rule-based half: train cases whose out-of-fold gate probability (``train.py
--stage oof``, saved to ``config.OOF_DIR``) confidently contradicts their
label — labelled cancer with p < ``CONTRADICTED_CANCER_MAX_PROB``, or labelled
no cancer with p > ``CONTRADICTED_NO_CANCER_MIN_PROB``. Split evenly between the
two, strongest contradiction first, so each batch takes the next-strongest.

Random half: a seeded uniform draw from every other train case with an OOF
score. It measures how often an ordinary training label is wrong, the baseline
the contradicted cases' error rate is compared against.

Both halves skip cases already in an earlier batch or already in the gold store.
``sample`` writes per batch, under ``config.REPORT_MAPPING_AUDIT_DIR``:

- ``report_mapping_audit_batch<N>.csv`` — case_id, reason, target, oof_prob (no text).
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
CONTRADICTED_CANCER = "contradicted_cancer_label"
CONTRADICTED_NO_CANCER = "contradicted_no_cancer_label"
RANDOM = "random"
LEDGER_COLS = ["case_id", "reason", "target", "oof_prob"]

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


def sample(
    oof_csv: str | Path,
    split_id: str,
    batch: int,
    n_contradicted: int = 100,
    n_random: int = 100,
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> dict:
    """Draw one batch from the gate OOF scores in ``oof_csv`` and write its ledger CSV and case-ID list."""
    ledger_csv, case_list = ledger_path(batch, out_dir), case_list_path(batch, out_dir)
    existing = [p for p in (ledger_csv, case_list) if p.is_file()]
    if existing:
        raise ReportMappingAuditError(
            f"batch {batch} already exists ({', '.join(str(p) for p in existing)}); pick a new batch number")
    oof = _read(Path(oof_csv))
    not_train = set(oof["case_id"]) - load_split(split_id).train
    if not_train:
        raise ReportMappingAuditError(
            f"{len(not_train)} OOF case(s) are not in {split_id!r}'s train partition, e.g. {sorted(not_train)[:5]}; "
            f"the OOF scores must come from that split")
    taken = {c for path in _batch_ledgers(out_dir) for c in _read(path)["case_id"]} | set(load_gold()["case_id"])
    pool = oof[~oof["case_id"].isin(taken)].assign(p=lambda d: d["prob"].astype(float))

    cancer = pool[(pool["target"] == "1") & (pool["p"] < CONTRADICTED_CANCER_MAX_PROB)].sort_values(
        ["p", "case_id"])
    no_cancer = pool[(pool["target"] == "0") & (pool["p"] > CONTRADICTED_NO_CANCER_MIN_PROB)].sort_values(
        ["p", "case_id"], ascending=[False, True])
    # Half each; a side that runs short hands its remainder to the other.
    n_no_cancer = min(len(no_cancer), n_contradicted - min(len(cancer), n_contradicted // 2))
    n_cancer = min(len(cancer), n_contradicted - n_no_cancer)
    contradicted = pd.concat([cancer.head(n_cancer).assign(reason=CONTRADICTED_CANCER),
                              no_cancer.head(n_no_cancer).assign(reason=CONTRADICTED_NO_CANCER)])

    rest = sorted(set(pool["case_id"]) - set(contradicted["case_id"]))
    random_ids = random.Random(seed).sample(rest, min(n_random, len(rest)))
    randoms = pool.set_index("case_id").loc[random_ids].reset_index().assign(reason=RANDOM)

    rows = pd.concat([contradicted, randoms]).rename(columns={"prob": "oof_prob"})[LEDGER_COLS]
    ledger_csv.parent.mkdir(parents=True, exist_ok=True)
    io_utils.write_csv(rows, ledger_csv)
    case_list.write_text("".join(f"{c}\n" for c in rows["case_id"]), encoding="utf-8", newline="\n")
    return {
        "ledger_csv": ledger_csv, "case_list": case_list, "n_cases": len(rows),
        "counts": rows["reason"].value_counts().to_dict(),
        "pools": {CONTRADICTED_CANCER: len(cancer), CONTRADICTED_NO_CANCER: len(no_cancer), RANDOM: len(rest)},
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
