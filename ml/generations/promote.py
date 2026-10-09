"""The promotion rule, and applying it: archive the incumbent, swap the candidate in.

Two kinds of model change (restructure-plan.md, Decisions 1):

**Retraining** (same code, new weights) is judged on **gold only**. Both the challenger (``candidate/``) and the
incumbent (``current/``) are scored on the *current* gold-eval - the same cases, the incumbent re-scored every
time - against the one uncommon-groups ruler (``config.UNCOMMON_GROUPS_TXT``). Promote only if

(a) at least one retraining trigger is met (``generations.triggers``), and
(b) the lower 95% bound of the paired (challenger - incumbent) good share, from a stratified case-cluster paired
    bootstrap, is >= ``MARGIN`` (-2.0 pp).

There is no silver fallback: without gold-eval rows ``recommend`` refuses. It also refuses a candidate whose
``architecture`` block differs from current's - that is not a retraining; use ``adopt_architecture``.
``recommend`` computes the rule and changes nothing; ``trigger_status`` computes (a) alone, before any challenger
exists. ``apply`` carries it out: a winning candidate is swapped in after the incumbent is archived; a losing one is
deleted.

**Architectural change** (new code) is adopted by hand: ``adopt_architecture`` needs a reason, an existing file
under ``config.HISTORY_DIR`` (an ADR), records it in the manifest (``adoption``), archives ``current/`` whole and
swaps. No gold rule, no triggers.

``adopt_published`` is a third, separate thing: it swaps in a generation that was already promoted on another machine
(s3sync/models.py pulls it into ``candidate/`` first), without the rule.

**Swap safety.** The candidate is re-verified (``report_mapping.generation.verify_spec``: file hashes, architecture,
calibration bound to its weights) and the incumbent's manifest verified immediately before anything moves. Both moves
are same-filesystem ``os.rename`` calls: ``current -> ARCHIVE_ROOT/YYYY-MM-DD_<desc>/`` then ``candidate -> current``.
If the second fails, the first is undone, so ``current/`` is never left half-moved or empty.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path

import pandas as pd

import config
import io_utils
from evaluation import gold_eval, intervals, silver_eval, verdicts
from generations import guards, triggers
from generations.manifest import read_manifest, update_manifest, verify_manifest
from manual_audit import eval_batch, gold
from report_mapping import generation

MARGIN = -0.02


class PromotionError(Exception):
    """Promotion could not be evaluated or applied."""


def check_candidate(directory: str | Path) -> dict:
    """``generation.verify_spec`` (file hashes, architecture, calibrated and bound to its weights); return the
    manifest. A generation that fails the spec raises ``GenerationError``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PromotionError(f"no candidate generation at {directory}")
    return generation.verify_spec(directory)


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


def _score(generation_name: str, predictions_csv: str | Path, gold_rows: pd.DataFrame,
           cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One generation's gold-eval verdict table (the ruler's uncommon groups), and how many gold-eval cases
    have no prediction row. predict.py writes no row for a case with no report text, so such a case scores as
    "predicted nothing", as it does in silver-eval and gold-eval. The predictions must carry the generation's id."""
    predictions = silver_eval.read_predictions(predictions_csv)
    silver_eval.check_generation_id(generation_name, predictions)
    unpredicted = len(set(cases.index) - set(predictions["case_id"]))
    return gold_eval.score_bronze(gold_rows, predictions, silver_eval.uncommon_groups(), cases), unpredicted


def trigger_status(silver_id: str | None, incumbent_predictions_csv: str | Path, *, split_id: str | None = None,
                   n_boot: int = 1000, seed: int = 0) -> list[triggers.Trigger]:
    """The three triggers for a challenger trained on ``silver_id`` against ``current/``. Needs no
    challenger, so it can run before training. Without gold-eval the random slice cannot fire, and
    the incumbent predictions are not read."""
    split_id = split_id if split_id is not None else config.DEFAULT_SPLIT_ID
    incumbent_manifest = read_manifest(config.REPORT_MAPPING_CURRENT_DIR)
    gold_rows = gold.gold_eval(split_id=split_id)
    if gold_rows.empty:
        slice_drop = triggers.Trigger("random_slice_drop", False, {"slice_cases": 0})
    else:
        cases = gold_eval.case_weights(gold_rows, _ledger(), split_id)
        table, _ = _score("current", incumbent_predictions_csv, gold_rows, cases)
        slice_drop = triggers.random_slice_drop(table, cases, gold_rows, n_boot, seed)
    return [triggers.new_silver_lineage(incumbent_manifest, silver_id), slice_drop,
            triggers.gold_train_growth(incumbent_manifest, split_id)]


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
    if challenger_manifest["architecture"] != incumbent_manifest.get("architecture"):
        raise PromotionError(f"candidate architecture {challenger_manifest['architecture']} differs from current's "
                             f"{incumbent_manifest.get('architecture')}: that is an architectural change, not a "
                             "retraining. Adopt it with promote.py --adopt --reason <a file under "
                             f"{config.HISTORY_DIR}>")

    gold_rows = gold.gold_eval(split_id=split_id)
    if gold_rows.empty:
        raise PromotionError(f"no gold-eval rows for split {split_id!r} in {config.GOLD_STORE_CSV}; "
                             "promotion needs gold-eval (ingest an eval batch or a random slice first)")
    cases = gold_eval.case_weights(gold_rows, _ledger(), split_id)

    challenger_table, challenger_unpredicted = _score("candidate", challenger_predictions_csv, gold_rows, cases)
    incumbent_table, incumbent_unpredicted = _score("current", incumbent_predictions_csv, gold_rows, cases)
    comparison = {**compare(challenger_table, incumbent_table, cases, n_boot, seed),
                  "challenger_unpredicted": challenger_unpredicted, "incumbent_unpredicted": incumbent_unpredicted}
    fired = trigger_status((challenger_manifest.get("parents") or {}).get("silver_id"), incumbent_predictions_csv,
                           split_id=split_id, n_boot=n_boot, seed=seed)
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
    archive = archive_path(result["incumbent_id"], description, today)
    _swap_in(candidate, current, archive, {})
    return {"action": "promoted", "generation_id": result["challenger_id"], "archive": archive}


def archive_path(incumbent_id: str, description: str | None = None, today: date | None = None) -> Path:
    """Where ``apply`` / ``adopt_architecture`` / ``adopt_published`` archive the incumbent."""
    return config.ARCHIVE_ROOT / f"{(today or date.today()).isoformat()}_{description or incumbent_id}"


def adopt_published(description: str | None = None, today: date | None = None) -> dict:
    """Swap a verified ``candidate/`` in without the scoring rule: it was already promoted where it was
    published (s3sync/models.py pull_model). Archives the incumbent as ``apply`` does; a machine with no
    ``current/`` just takes it."""
    return _adopt(config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR, description, today, {})


def adopt_architecture(candidate_dir: str | Path, current_dir: str | Path, reason_path: str | Path,
                       description: str | None = None, today: date | None = None) -> dict:
    """Adopt a candidate built by new code. No gold rule, no triggers: ``reason_path`` must be an existing file
    under ``config.HISTORY_DIR`` (the ADR that explains the change); its repo-relative path is recorded in the
    manifest as ``adoption`` {reason, adopted_at}. ``current_dir`` is archived whole when it exists."""
    reason = Path(reason_path).resolve()
    if not reason.is_file() or not reason.is_relative_to(config.HISTORY_DIR.resolve()):
        raise PromotionError(f"the reason must be an existing file under {config.HISTORY_DIR}, got {reason_path}")
    current = Path(current_dir)
    if current.exists() and read_manifest(current).get("architecture") == check_candidate(candidate_dir)["architecture"]:
        raise PromotionError("candidate/ has the same architecture as current/: a retrain goes through the gold rule "
                             "(promote.py --apply), not --adopt")
    adoption ={"reason": reason.relative_to(config.REPO_ROOT.resolve()).as_posix(),
                "adopted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return _adopt(Path(candidate_dir), Path(current_dir), description, today, {"adoption": adoption})


def _adopt(candidate: Path, current: Path, description: str | None, today: date | None, fields: dict) -> dict:
    generation_id = check_candidate(candidate)["generation_id"]
    archive = None
    if current.exists():
        verify_manifest(current)
        archive = archive_path(read_manifest(current)["generation_id"], description, today)
    _swap_in(candidate, current, archive, fields)
    return {"action": "adopted", "generation_id": generation_id, "archive": archive}


def _swap_in(candidate: Path, current: Path, archive: Path | None, fields: dict) -> None:
    """Archive ``current`` into ``archive`` (None: there is no current/) and put ``candidate`` in its place."""
    if archive is None:
        current.parent.mkdir(parents=True, exist_ok=True)
        os.rename(candidate, current)
    else:
        if archive.exists():
            raise PromotionError(f"{archive} already exists; pass another description")
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.rename(current, archive)
        try:
            os.rename(candidate, current)
        except OSError:
            os.rename(archive, current)
            raise
        update_manifest(archive, {"status": "archived"})
    update_manifest(current, {"status": "current", **fields})
