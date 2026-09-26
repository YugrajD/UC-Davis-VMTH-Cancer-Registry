"""The frozen legacy reference pack at ``config.PARITY_REFERENCE_DIR``, and the file glue
the levels need (scoring a predictions CSV, loading embeddings).

LEGACY COMPARISON — deleted at cutover.

``freeze`` regenerates the legacy predictions with the OLD production code on
the current legacy checkpoints and embedding cache (``legacy_predict.py``,
read-only, on ``LEGACY_DEVICE``), keeps the 2026-05-29 production CSV beside
them for the record, and scores them with the OLD evaluator
(``config.LEGACY_EVALUATE_PY``). Both old programs run as subprocesses — in
process, the old tree's ``config`` / ``evaluation`` would collide with ours. It
refuses to write the manifest unless the scores match ``EXPECTED`` exactly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from generations.manifest import read_manifest, sha256_file, verify_manifest, write_manifest
from evaluation import verdicts
from evaluation.verdicts import VERDICTS
from generations.splits import MD5_HALF_RULE, in_sweep_half
from parity.levels import l2, summarize

EVAL_HALF_TXT = "eval_half_cases.txt"
SWEEP_HALF_TXT = "sweep_half_cases.txt"
PREDICTIONS_CSV = "legacy_predictions.csv"  # regenerated at freeze time
PREDICTIONS_2026_05_29_CSV = "legacy_predictions_2026-05-29.csv"  # the old production run, kept for the record
RERUN_DIR = "legacy_rerun"  # old pipeline's out_dir (predictions + summary json only)
UNCOMMON_GROUPS_TXT = "uncommon_groups.txt"
LP_THRESHOLDS_JSON = "lp_thresholds.json"
THRESHOLDS_JSON = "thresholds.json"
VERDICT_TABLE_CSV = "evaluation.csv"  # legacy evaluate() output names
SUMMARY_CSV = "evaluation_summary.csv"
# Scope -> sub-directory holding legacy evaluate()'s outputs for that case set.
SCOPE_DIRS = {"eval_half": "legacy_eval_half", "test": "legacy_test"}

# Inference thresholds in force when the legacy predictions were produced.
LEGACY_THRESHOLDS = {
    "case_presence_gate": 0.80,  # run_production.py set_defaults
    "group": 0.85,  # run_production.py set_defaults
    "tail_max_predictions": 2,  # production/petbert_pipeline/cli.py default
    "tail_max_group_prob_gap": 0.08,  # cli.py default
    "label_presence_fallback": 0.5,  # cli.py default; per-LP values in lp_thresholds.json
}

# CPU rather than the old "auto" (MPS here), and a fixed string-hash seed: the old
# stage 3 builds the Uncommon head's label list by iterating a frozenset
# (production/petbert_pipeline/stages/__init__.py:88-90), so ~300 rows change
# with PYTHONHASHSEED. Both pinned, a re-freeze reproduces byte for byte.
LEGACY_DEVICE = "cpu"
LEGACY_ENV = {"PYTHONHASHSEED": "0"}

# The regenerated legacy reference, scored by the old evaluator (first measured
# 2026-09-25; freeze re-checks it). The published 62.1% / 4,414 (classifiers.md)
# is the 2026-05-13 generation and is superseded: the checkpoints and
# annotation.csv have changed since.
EXPECTED = {
    "eval_half": {"total": 4456, "gs_pct": 61.8, "good_pct": 45.8, "slightly_off_pct": 16.0,
                  "completely_off_pct": 15.3, "false_positive_pct": 2.6, "false_negative_pct": 20.4},
    "test": {"total": 8916, "gs_pct": 61.8, "good_pct": 46.4, "slightly_off_pct": 15.4,
             "completely_off_pct": 15.0, "false_positive_pct": 2.6, "false_negative_pct": 20.6},
}

# Private text column of annotation.csv; never loaded (the scorer needs only
# case_id / matched_term / matched_group).
_TEXT_COLUMNS = {"diagnosis"}


def read_table(path: str | Path) -> pd.DataFrame:
    """A predictions / verdict-table CSV as strings, blanks kept as ""."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_ids(path: str | Path) -> list[str]:
    # strip() also drops the legacy files' CRLF line endings.
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{cid}\n" for cid in ids), encoding="utf-8")


def _rel(path: Path) -> str:
    """A legacy file's manifest key: its path relative to ml/ (e.g. "data/report.csv")."""
    return path.relative_to(config.ML_ROOT).as_posix()


# Legacy inputs read in place (not copied into the pack): hashed at freeze,
# re-verified by every level. labels.csv is here because legacy_predict bypasses
# its mtime check.
_VERIFIED_SOURCES = (
    config.LEGACY_ANNOTATION_CSV, config.LEGACY_TEST_CASES_TXT, config.REPORT_CSV,
    config.LEGACY_UNCOMMON_GROUPS_TXT, config.LEGACY_LP_THRESHOLDS_JSON, config.LEGACY_LABELS_CSV,
)


def _sha_map(paths) -> dict[str, str]:
    """sha256 of each file (directories expanded), keyed by ``_rel``."""
    files = []
    for path in paths:
        files += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return {_rel(p): sha256_file(p) for p in files}


def _legacy_overall(summary_csv: Path) -> dict:
    """The OVERALL row of a legacy evaluation_summary.csv, as numbers, plus G+S %.

    ``gs_pct`` is how the published 62.1 was formed: round(good %, 1) + round(slight %, 1).
    ``gs_pct_exact`` is the unrounded share the levels compare against; they can differ by ±0.1 pp.
    """
    row = read_table(summary_csv).set_index("scope").loc["OVERALL"]
    overall = {k: (int(v) if k in ("total", *VERDICTS) else float(v)) for k, v in row.items()}
    overall["gs_pct"] = round(overall["good_pct"] + overall["slightly_off_pct"], 1)
    overall["gs_pct_exact"] = 100.0 * (overall["good"] + overall["slightly_off"]) / overall["total"]
    return overall


def _run_legacy_evaluator(pack: Path, cases_txt: Path, out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(config.LEGACY_EVALUATE_PY),
         "--prediction-csv", str(pack / PREDICTIONS_CSV),
         "--expectation-csv", str(config.LEGACY_ANNOTATION_CSV),
         "--out-dir", str(out_dir),
         "--test-cases", str(cases_txt),
         "--uncommon-groups", str(pack / UNCOMMON_GROUPS_TXT)],
        cwd=config.ML_ROOT.parent, check=True, env=os.environ | LEGACY_ENV,
    )


def _regenerate_predictions(pack: Path) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("legacy_predict.py")),
         str(config.LEGACY_RUN_PRODUCTION_PY), str(pack / RERUN_DIR), LEGACY_DEVICE],
        cwd=config.ML_ROOT.parent, check=True, env=os.environ | LEGACY_ENV,
    )
    (pack / RERUN_DIR / "petbert_predictions.csv").rename(pack / PREDICTIONS_CSV)


def freeze() -> bool:
    """Build the reference pack. Returns True if the baseline reproduced and the manifest was written."""
    pack = config.PARITY_REFERENCE_DIR
    if (pack / "manifest.json").exists():
        raise FileExistsError(f"{pack} is already frozen; delete it by hand to re-freeze")
    if pack.exists():
        shutil.rmtree(pack)  # an unfrozen attempt; nothing in it is kept
    pack.mkdir(parents=True)

    test_ids = read_ids(config.LEGACY_TEST_CASES_TXT)
    eval_half = [cid for cid in test_ids if not in_sweep_half(cid)]
    sweep_half = [cid for cid in test_ids if in_sweep_half(cid)]
    _write_ids(pack / EVAL_HALF_TXT, eval_half)
    _write_ids(pack / SWEEP_HALF_TXT, sweep_half)
    shutil.copyfile(config.LEGACY_PREDICTIONS_CSV, pack / PREDICTIONS_2026_05_29_CSV)
    shutil.copyfile(config.LEGACY_UNCOMMON_GROUPS_TXT, pack / UNCOMMON_GROUPS_TXT)
    shutil.copyfile(config.LEGACY_LP_THRESHOLDS_JSON, pack / LP_THRESHOLDS_JSON)
    (pack / THRESHOLDS_JSON).write_text(json.dumps(LEGACY_THRESHOLDS, indent=2) + "\n", encoding="utf-8")

    cache_sha = sha256_file(config.LEGACY_EMBEDDING_CACHE_NPZ)
    _regenerate_predictions(pack)
    if sha256_file(config.LEGACY_EMBEDDING_CACHE_NPZ) != cache_sha:
        raise RuntimeError(f"{config.LEGACY_EMBEDDING_CACHE_NPZ} changed during regeneration")
    print("Regenerated vs 2026-05-29 predictions (drift report):")
    drift = l2(read_table(pack / PREDICTIONS_CSV), read_table(pack / PREDICTIONS_2026_05_29_CSV),
               reference_uncommon_groups())
    print("\n".join(drift.lines))

    scope_cases = {"eval_half": pack / EVAL_HALF_TXT, "test": config.LEGACY_TEST_CASES_TXT}
    results, reproduced = {}, True
    for scope, sub in SCOPE_DIRS.items():
        _run_legacy_evaluator(pack, scope_cases[scope], pack / sub)
        overall = _legacy_overall(pack / sub / SUMMARY_CSV)
        # The verdict table must agree with legacy's own summary, or later levels
        # (which summarise tables) would measure against a different number.
        table = summarize(read_table(pack / sub / VERDICT_TABLE_CSV))
        if any(table[k] != overall[k] for k in ("total", *VERDICTS)):
            print(f"  {scope}: verdict table counts disagree with legacy summary")
            reproduced = False
        results[scope] = overall
        for key, expected in EXPECTED[scope].items():
            ok = overall[key] == expected
            reproduced &= ok
            print(f"  {scope:<9} {key:<20} legacy {overall[key]:>7}  expected {expected:>7}  "
                  f"{'ok' if ok else 'DIFFERS'}")
    if not reproduced:
        print("Baseline did NOT reproduce — manifest not written; the pack is not frozen.")
        return False

    print("Hashing legacy checkpoints and embedding cache ...")
    write_manifest(pack, {
        "kind": "parity_reference",
        "md5_half_rule": MD5_HALF_RULE,
        "case_counts": {"test": len(test_ids), "eval_half": len(eval_half), "sweep_half": len(sweep_half)},
        "legacy_sources": _sha_map([config.LEGACY_PREDICTIONS_CSV, *_VERIFIED_SOURCES]),
        "predictions": {
            "reference": f"{PREDICTIONS_CSV}: regenerated by the old run_production.py (production defaults) "
                         "on the current legacy checkpoints + embedding cache, read-only",
            "device": LEGACY_DEVICE,
            "env": LEGACY_ENV,
            "record": f"{PREDICTIONS_2026_05_29_CSV}: the 2026-05-29 production run (CUDA), not used for scoring",
            "drift_vs_2026_05_29": drift.lines,
        },
        "thresholds": LEGACY_THRESHOLDS,
        "legacy_results": results,
        "legacy_checkpoint_sha256": _sha_map([
            config.LEGACY_CHECKPOINT_CONTRASTIVE_DIR, config.LEGACY_CHECKPOINT_CASE_PRESENCE_PT.parent,
            config.LEGACY_CHECKPOINT_GROUP_BEST_PT.parent, config.LEGACY_CHECKPOINT_LABEL_PRESENCE_DIR,
        ]),
        "legacy_embedding_cache_sha256": _sha_map([config.LEGACY_EMBEDDING_CACHE_NPZ]),
        "notes": "Scored by the legacy evaluator (ml/evaluation/evaluate.py). Verdict tables exclude true "
                 "negatives. LP thresholds were fitted on the sweep half; the gate and tail gate on all of test.",
    })
    return True


# ---------------------------------------------------------------------------
# Loading the pack (every level verifies it first)
# ---------------------------------------------------------------------------


def load_manifest() -> dict:
    """Verify the pack's files, and that every legacy input read in place
    (``_VERIFIED_SOURCES``) is unchanged since the freeze."""
    pack = config.PARITY_REFERENCE_DIR
    verify_manifest(pack)
    manifest = read_manifest(pack)
    for path in _VERIFIED_SOURCES:
        if sha256_file(path) != manifest["legacy_sources"][_rel(path)]:
            raise RuntimeError(f"{path} changed since the reference was frozen")
    return manifest


def scope_ids(scope: str) -> set[str]:
    if scope == "eval_half":
        return set(read_ids(config.PARITY_REFERENCE_DIR / EVAL_HALF_TXT))
    return set(read_ids(config.LEGACY_TEST_CASES_TXT))


def reference_uncommon_groups() -> frozenset[str]:
    return frozenset(read_ids(config.PARITY_REFERENCE_DIR / UNCOMMON_GROUPS_TXT))


def reference_predictions() -> pd.DataFrame:
    return read_table(config.PARITY_REFERENCE_DIR / PREDICTIONS_CSV)


def reference_verdicts(scope: str) -> pd.DataFrame:
    return read_table(config.PARITY_REFERENCE_DIR / SCOPE_DIRS[scope] / VERDICT_TABLE_CSV)


def reference_embeddings(manifest: dict):
    """The legacy embedding cache as (case_ids, concat-3 matrix), refused if it changed since the freeze."""
    rel = _rel(config.LEGACY_EMBEDDING_CACHE_NPZ)
    if sha256_file(config.LEGACY_EMBEDDING_CACHE_NPZ) != manifest["legacy_embedding_cache_sha256"][rel]:
        raise RuntimeError(f"{config.LEGACY_EMBEDDING_CACHE_NPZ} changed since the reference was frozen")
    return load_embeddings(config.LEGACY_EMBEDDING_CACHE_NPZ)


def score_scope(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Score predictions against the legacy silver on one case scope with the new scorer
    (certified identical to legacy by L1), using the frozen uncommon-groups list."""
    ids = scope_ids(scope)
    expectations = pd.read_csv(config.LEGACY_ANNOTATION_CSV, dtype=str, keep_default_na=False,
                               usecols=lambda c: c not in _TEXT_COLUMNS)
    return verdicts.score(expectations[expectations["case_id"].isin(ids)],
                          predictions[predictions["case_id"].isin(ids)], reference_uncommon_groups())


def load_embeddings(path: str | Path, key: str | None = None, ids_path: str | Path | None = None):
    """(case_ids, matrix) from an .npz (ids under "case_ids" or "ids") or an .npy plus an ids file.

    For .npz without ``key``, the matrix is "col_concat_3" (legacy cache), else
    "embeddings". Only the two needed arrays are read, so a file's text arrays
    are never loaded.
    """
    path = Path(path)
    if path.suffix == ".npy":
        if ids_path is None:
            raise ValueError(f"{path} is .npy: pass the case-id list file too")
        return np.array(read_ids(ids_path)), np.load(path)
    archive = np.load(path, allow_pickle=True)  # legacy ids are object arrays
    id_key = "case_ids" if "case_ids" in archive.files else "ids"
    if key is None:
        key = "col_concat_3" if "col_concat_3" in archive.files else "embeddings"
    return np.asarray(archive[id_key]).astype(str), archive[key]
