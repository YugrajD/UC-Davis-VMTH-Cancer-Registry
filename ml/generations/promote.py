"""The promotion rule, and applying it: archive the incumbent, swap the candidate in.

**Rule** (ml-rewrite-plan.md, Decisions). Primary metric: weighted per-code
exact accuracy (``good`` share); G+S is reported as secondary. Both the
challenger (``candidate/``) and the incumbent (``current/``) are scored on the
*current* gold-eval — the same cases, the incumbent re-scored every time — each
with its own generation's uncommon groups. Promote only if

(a) at least one retraining trigger is met (``generations.triggers``), and
(b) the lower 95% bound of the paired (challenger − incumbent) good share, from
    a stratified case-cluster paired bootstrap, is ≥ ``MARGIN`` (−2.0 pp).

``recommend`` computes this and changes nothing; ``trigger_status`` computes (a)
alone, before any challenger exists. ``apply`` carries it out: a winning
candidate is swapped in after the incumbent is archived; a losing one is
deleted. ``adopt`` does the same swap without the rule, for a generation that
was already promoted on another machine (s3sync/models.py pulls it into
``candidate/`` first).

**Swap safety.** The candidate is re-verified (manifest, embedding fingerprint,
``calibration.status == "calibrated"``) and the incumbent's manifest verified
immediately before anything moves. Both moves are same-filesystem
``os.rename`` calls: ``current → ARCHIVE_ROOT/YYYY-MM-DD_<desc>/`` then
``candidate → current``. If the second fails, the first is undone, so
``current/`` is never left half-moved or empty. Only then are the manifest
statuses set (``archived`` / ``current``) and the embedding-cache entries the new
``current/`` cannot use moved into the archive. Their key includes the backbone
fingerprint, so a stale entry could never load against the wrong generation;
it is archived because CLAUDE.md archives a generation's embeddings with it.
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
from generations.manifest import read_manifest, sha256_file, update_manifest, verify_manifest
from manual_audit import eval_batch, gold
from report_mapping.inference import embedding_cache
from report_mapping.model.generation import compute_embedding_fingerprint, generation_paths, verify_fingerprint

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


def _score(generation: str, predictions_csv: str | Path, gold_rows: pd.DataFrame,
           cases: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One generation's gold-eval verdict table, with its own uncommon groups, and how many gold-eval
    cases have no prediction row. predict.py writes no row for a case whose report sections are all
    empty, so such a case scores as "predicted nothing", as it does in silver-eval and gold-eval."""
    predictions = silver_eval.read_predictions(predictions_csv)
    _, uncommon = silver_eval.generation_uncommon_groups(generation, predictions)
    unpredicted = len(set(cases.index) - set(predictions["case_id"]))
    return gold_eval.score_bronze(gold_rows, predictions, uncommon, cases), unpredicted


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
    moved = _archive_and_swap(archive)
    return {"action": "promoted", "generation_id": result["challenger_id"], "archive": archive,
            "archived_caches": len(moved)}


def archive_path(incumbent_id: str, description: str | None = None, today: date | None = None) -> Path:
    """Where ``apply`` / ``adopt`` archive the incumbent."""
    return config.ARCHIVE_ROOT / f"{(today or date.today()).isoformat()}_{description or incumbent_id}"


def adopt(description: str | None = None, today: date | None = None) -> dict:
    """Swap a verified ``candidate/`` in without the scoring rule: it was already promoted where it was
    published. Archives the incumbent as ``apply`` does; a machine with no ``current/`` just takes it."""
    candidate, current = config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR
    generation_id = check_candidate(candidate)["generation_id"]
    if not current.exists():
        current.parent.mkdir(parents=True, exist_ok=True)
        os.rename(candidate, current)
        update_manifest(current, {"status": "current"})
        return {"action": "adopted", "generation_id": generation_id, "archive": None, "archived_caches": 0}

    verify_manifest(current)
    archive = archive_path(read_manifest(current)["generation_id"], description, today)
    moved = _archive_and_swap(archive)
    return {"action": "adopted", "generation_id": generation_id, "archive": archive, "archived_caches": len(moved)}


def _archive_and_swap(archive: Path) -> list[str]:
    """Archive current/ into ``archive``, put candidate/ in its place; return the cache files archived too."""
    candidate, current = config.REPORT_MAPPING_CANDIDATE_DIR, config.REPORT_MAPPING_CURRENT_DIR
    if archive.exists():
        raise PromotionError(f"{archive} already exists; pass another description")
    archive.parent.mkdir(parents=True, exist_ok=True)

    os.rename(current, archive)
    try:
        os.rename(candidate, current)
    except OSError:
        os.rename(archive, current)
        raise
    update_manifest(current, {"status": "current"})
    moved = _archive_stale_caches(archive)
    # The archive's manifest lists the moved cache files too, so the archive still verifies.
    files = {**read_manifest(archive)["files"], **{rel: sha256_file(archive / rel) for rel in moved}}
    update_manifest(archive, {"status": "archived", "files": files})
    return moved


def _archive_stale_caches(archive: Path) -> list[str]:
    """Move every embedding-cache entry the new current/ cannot use into ``archive``; return their
    paths relative to it.

    CLAUDE.md archives a generation's embeddings with it. The cache is keyed by content, so the
    entry to keep is the one for the new current's backbone and today's report.csv. A heads-only
    promotion shares the incumbent's backbone, so its entry stays."""
    cached = sorted(config.EMBEDDING_CACHE_DIR.glob("*.npz"))
    if not cached:
        return []
    paths = generation_paths(config.REPORT_MAPPING_CURRENT_DIR)
    keep = embedding_cache.content_key(config.REPORT_CSV, paths.labels_csv,
                                       compute_embedding_fingerprint(paths.petbert_dir))
    moved = [f"embedding_cache/{path.name}" for path in cached if path.stem != keep]
    if moved:
        (archive / "embedding_cache").mkdir()
    for rel in moved:
        os.rename(config.EMBEDDING_CACHE_DIR / Path(rel).name, archive / rel)
    return moved
