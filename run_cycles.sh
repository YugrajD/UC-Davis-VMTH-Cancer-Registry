#!/usr/bin/env bash
set -e
PY=C:/Users/chris/Documents/GitHub/UC-Davis-VMTH-Cancer-Registry/ml/.venv/Scripts/python.exe
ROOT=C:/Users/chris/Documents/GitHub/UC-Davis-VMTH-Cancer-Registry
cd "$ROOT"

for i in 1 2 3 4 5; do
  echo "============ CYCLE $i: train ============"
  $PY ml-3-stage/scripts/run_training.py --mode train-presence \
    --annotation-csv ml-3-stage/output/annotation/llm/llm_annotation.csv \
    --train-cases ml-3-stage/output/splits/train_cases.txt \
    --model ml-3-stage/output/checkpoints/contrastive \
    --device xpu --local-only

  echo "============ CYCLE $i: production ============"
  $PY ml-3-stage/scripts/run_production.py \
    --model ml-3-stage/output/checkpoints/contrastive \
    --device xpu --local-only

  echo "============ CYCLE $i: evaluation ============"
  $PY ml-3-stage/scripts/run_evaluation.py --stage all \
    --annotation-csv ml-3-stage/output/annotation/llm/llm_annotation.csv \
    --test-cases ml-3-stage/output/splits/test_cases.txt \
    --label "cycle $i (gate+banks)"

  echo "============ CYCLE $i: update banks ============"
  $PY ml-3-stage/scripts/run_training.py --mode update-co-bank --update-fp-bank
done
echo "ALL CYCLES DONE"
