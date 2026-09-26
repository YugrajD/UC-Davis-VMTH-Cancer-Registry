"""The promotion rule, and applying it: archive the incumbent, swap the candidate in.

**Rule** (ml-rewrite-plan.md, Decisions). Primary metric: weighted per-code
exact accuracy (``good`` share); G+S is reported as secondary. Both the
challenger (``candidate/``) and the incumbent (``current/``) are scored on the
*current* gold-eval — the same cases, the incumbent re-scored every time — each
with its own generation's uncommon groups. Promote only if

(a) at least one retraining trigger is met (``generations.triggers``), and
(b) the lower 95% bound of the paired (challenger − incumbent) good share, from
    a stratified case-cluster paired bootstrap, is ≥ ``MARGIN`` (−2.0 pp).

``recommend`` computes this and changes nothing. ``apply`` carries it out: a
winning candidate is swapped in after the incumbent is archived; a losing one
is deleted.

**Swap safety.** The candidate is re-verified (manifest, embedding fingerprint,
``calibration.status == "calibrated"``) and the incumbent's manifest verified
immediately before anything moves. Both moves are same-filesystem
``os.rename`` calls: ``current → ARCHIVE_ROOT/YYYY-MM-DD_<desc>/`` then
``candidate → current``. If the second fails, the first is undone, so
``current/`` is never left half-moved or empty. The content-hash embedding
cache is not archived (its key includes the backbone fingerprint, so it cannot
load against the wrong generation).
"""

from __future__ import annotations

import os
import shutil
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd

import config
import io_utils
from evaluation import gold_eval, intervals, silver_eval, verdicts
from generations import guards, triggers
from generations.manifest import read_manifest, verify_manifest
from manual_audit import eval_batch, gold
from report_mapping.model.generation import verify_fingerprint

MARGIN = -0.02


class PromotionError(Exception):
    """Promotion could not be evaluated or applied."""


def check_candidate(directory: str | Path) -> dict:
    """Verify manifest + embedding fingerprint + calibration status; return the manifest."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PromotionError(f"no candidate generation at {directory}")
    verify_manifest(directory)
    manifest = verify_fingerprint(directory)
    status = (manifest.get("calibration") or {}).get("status")
    if status != "calibrated":
        raise PromotionError(f"{directory}: calibration.status is {status!r}; run calibrate.py first")
    return manifest


def _ledger() -> pd.DataFrame:
    if not config.EVAL_BATCH_LEDGER_CSV.is_file():
        return pd.DataFrame(columns=eval_batch.EVAL_BATCH_LEDGER_FIELDS)
    return io_utils.read_csv(config.EVAL_BATCH_LEDGER_CSV, encoding="utf-8", dtype=str, keep_default_na=False)


def compare(challenger: pd.DataFrame, incumbent: pd.DataFrame, cases: pd.DataFrame,
            n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired (challenger − incumbent) good and G+S on two gold-eval verdict tables
    (``gold_eval.score_bronze``) over the same ``cases`` (``gold_eval.case_weights``)."""
    out = {"cases": len(cases), "challenger_codes": len(challenger), "incumbent_codes": len(incumbent)}
    for name, hits in (("good", verdicts.GOOD), ("gs", verdicts.GOOD_PLUS_SLIGHT)):
        metric = partial(verdicts.share, verdicts=hits, weight_col="weight")
        diff, low, high = intervals.paired_bootstrap(challenger, incumbent, metric, cases["stratum"], n_boot, seed)
        out.update({f"challenger_{name}": metric(challenger), f"incumbent_{name}": metric(incumbent),
                    f"{name}_diff": diff, f"{name}_lo": low, f"{name}_hi": high})
    return out


def decide(comparison: dict, fired: list[triggers.Trigger]) -> bool:
    return bool(comparison["good_lo"] >= MARGIN and any(t.met for t in fired))


def recommend(challenger_predictions_csv: str | Path, incumbent_predictions_csv: str | Path, *,
              split_id: str | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Guards, candidate checks, both generations scored on gold-eval, triggers, the rule. Writes nothing."""
    split_id = split_id if split_id is not None else config.DEFAULT_SPLIT_ID
    guards.check_all(split_id)
    challenger_manifest = check_candidate(config.REPORT_MAPPING_CANDIDATE_DIR)
    verify_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    incumbent_manifest = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    if challenger_manifest["generation_id"] == incumbent_manifest["generation_id"]:
        raise PromotionError(f"candidate and current share generation_id {incumbent_manifest['generation_id']!r}; "
                             "their predictions cannot be told apart")

    gold_rows = gold.gold_eval(split_id=split_id)
    if gold_rows.empty:
        raise PromotionError(f"no gold-eval rows for split {split_id!r} in {config.GOLD_STORE_CSV}; "
                             "promotion needs gold-eval (ingest an eval batch or a random slice first)")
    cases = gold_eval.case_weights(gold_rows, _ledger(), split_id)

    tables = {}
    for generation, csv in (("candidate", challenger_predictions_csv), ("current", incumbent_predictions_csv)):
        predictions = silver_eval.read_predictions(csv)
        _, uncommon = silver_eval.generation_uncommon_groups(generation, predictions)
        unpredicted = sorted(set(cases.index) - set(predictions["case_id"]))
        if unpredicted:
            raise PromotionError(f"{generation} predictions miss {len(unpredicted)} gold-eval case(s): "
                                 f"{unpredicted[:5]}")
        tables[generation] = gold_eval.score_bronze(gold_rows, predictions, uncommon, cases)

    comparison = compare(tables["candidate"], tables["current"], cases, n_boot, seed)
    fired = [
        triggers.new_silver_lineage(incumbent_manifest, (challenger_manifest.get("parents") or {}).get("silver_id")),
        triggers.random_slice_drop(tables["current"], cases, gold_rows, n_boot, seed),
        triggers.gold_train_growth(incumbent_manifest, split_id),
    ]
    return {
        "split_id": split_id,
        "challenger_id": challenger_manifest["generation_id"],
        "incumbent_id": incumbent_manifest["generation_id"],
        "comparison": comparison,
        "triggers": fired,
        "margin": MARGIN,
        "promote": decide(comparison, fired),
    }


def apply(result: dict, description: str | None = None, today: date | None = None) -> dict:
    """Carry out ``result`` (from ``recommend``): archive + swap, or delete the losing candidate."""
    candidate, current = config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR
    if check_candidate(candidate)["generation_id"] != result["challenger_id"]:
        raise PromotionError("candidate/ changed since the recommendation; re-run it")
    if not result["promote"]:
        shutil.rmtree(candidate)
        return {"action": "discarded", "generation_id": result["challenger_id"]}

    verify_manifest(current)
    if read_manifest(current)["generation_id"] != result["incumbent_id"]:
        raise PromotionError("current/ changed since the recommendation; re-run it")
    name = f"{(today or date.today()).isoformat()}_{description or result['incumbent_id']}"
    archive = config.ARCHIVE_ROOT / name
    if archive.exists():
        raise PromotionError(f"{archive} already exists; pass another description")
    archive.parent.mkdir(parents=True, exist_ok=True)

    os.rename(current, archive)
    try:
        os.rename(candidate, current)
    except OSError:
        os.rename(archive, current)
        raise
    return {"action": "promoted", "generation_id": result["challenger_id"], "archive": archive}
