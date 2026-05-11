"""Run production inference: score all reports using the 4-stage pipeline.

No env PYTHONPATH needed — this script adds ml/ to sys.path automatically.

Pipeline: TF-IDF text selection → PetBERT (contrastive backbone) →
          CasePresenceClassifier gate → GroupClassifier →
          LabelPresenceClassifier (per-group) → ICD-O KW correction.

Usage:
  python ml/scripts/run_production.py --local-only
  python ml/scripts/run_production.py --max-rows 100 --local-only
  python ml/scripts/run_production.py --case-presence-threshold 0.3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from production.petbert_pipeline import build_config, build_parser, run_scan


def main() -> int:
    parser = build_parser()
    parser.set_defaults(
        model=config.CHECKPOINT_CONTRASTIVE_DIR,
        embedding_cache=config.EMBEDDING_CACHE_NPZ,
        group_classifier=f"{config.CHECKPOINT_GROUP_DIR}/group_classifier_best.pt",
        case_presence_classifier=config.CASE_PRESENCE_CLASSIFIER_PT,
        label_presence_classifier_dir=config.CHECKPOINT_LABEL_PRESENCE_DIR,
        label_presence_thresholds_json=config.LABEL_PRESENCE_THRESHOLDS_JSON,
        out_dir=f"{config.OUTPUT_PRODUCTION_DIR}/contrastive",
        local_only=True,
        text_cols="",  # empty = TF-IDF text selection
    )
    run_scan(build_config(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
