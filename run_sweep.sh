#!/usr/bin/env bash
set -e
PY=C:/Users/chris/Documents/GitHub/UC-Davis-VMTH-Cancer-Registry/ml/.venv/Scripts/python.exe
ROOT=C:/Users/chris/Documents/GitHub/UC-Davis-VMTH-Cancer-Registry
cd "$ROOT"

run() {
  local label="$1"; shift
  echo "============ $label ============"
  $PY ml-3-stage/scripts/run_production.py \
    --model ml-3-stage/output/checkpoints/contrastive \
    --device xpu --local-only "$@"
  $PY ml-3-stage/scripts/run_evaluation.py --stage all \
    --annotation-csv ml-3-stage/output/annotation/llm/llm_annotation.csv \
    --test-cases ml-3-stage/output/splits/test_cases.txt \
    --label "$label"
}

# Phase A: case-presence-threshold sweep
for cpt in 0.30 0.40 0.50 0.60 0.70; do
  run "sweep cpt=$cpt" --case-presence-threshold $cpt
done

# Phase B: presence-threshold sweep at best cpt (will pick from history afterward, default cpt=0.50)
for pt in 0.00 0.05 0.10 0.20; do
  run "sweep pt=$pt cpt=0.50" --case-presence-threshold 0.50 --presence-threshold $pt
done

# Phase C: max-predictions sweep
for mp in 2 3 4 5; do
  run "sweep mp=$mp cpt=0.50 pt=0.00" --case-presence-threshold 0.50 --presence-threshold 0.00 --presence-max-predictions $mp
done

echo "SWEEP DONE"
