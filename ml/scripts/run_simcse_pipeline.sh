#!/bin/bash
set -e

VENV=".venv/bin/python"
TRAIN_CASES="ml/output/splits/train_cases.txt"
TEST_CASES="ml/output/splits/test_cases.txt"
BACKBONE="ml/output/checkpoints/contrastive"
SIMCSE_OUT="ml/output/checkpoints/simcse"
ANNO="ml/output/annotation/llm/llm_annotation.csv"
LOG="ml/output/simcse_pipeline.log"

echo "=== Stage 1: SimCSE training (3 epochs) ===" | tee "$LOG"
caffeinate -i $VENV -u ml/training/simcse/train_simcse.py \
  --model "$BACKBONE" \
  --out-dir "$SIMCSE_OUT" \
  --epochs 3 \
  --batch-size 8 \
  --device cpu \
  --local-only \
  --train-cases "$TRAIN_CASES" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Stage 2: kNN predictions (test set) ===" | tee -a "$LOG"
caffeinate -i $VENV -u ml/training/simcse/knn_predict.py \
  --model "$SIMCSE_OUT" \
  --out-csv "ml/output/production/simcse/knn_predictions.csv" \
  --k 5 \
  --device cpu \
  --local-only \
  --test-cases "$TEST_CASES" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Stage 3: Evaluate kNN predictions ===" | tee -a "$LOG"
$VENV ml/scripts/run_evaluation.py \
  --prediction-csv "ml/output/production/simcse/knn_predictions.csv" \
  --annotation-csv "$ANNO" \
  --out-dir "ml/output/evaluation/simcse" \
  --test-cases "$TEST_CASES" \
  --label "simcse-knn-k5" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
