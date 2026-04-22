#!/bin/bash
set -e

VENV=".venv/bin/python"
ANNO="ml/output/annotation/llm/llm_annotation.csv"
TRAIN_CASES="ml/output/splits/train_cases.txt"
BACKBONE="ml/output/checkpoints/contrastive"
LOG="ml/output/full_training_run.log"

echo "=== Stage 1: Adapt backbone (3 epochs) ===" | tee "$LOG"
caffeinate -i $VENV -u ml/scripts/run_training.py \
  --mode adapt-backbone \
  --device cpu \
  --epochs 3 \
  --batch-size 8 \
  --annotation-csv "$ANNO" \
  --train-cases "$TRAIN_CASES" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Stage 2: Cold-start cleanup ===" | tee -a "$LOG"
rm -f ml/output/training/embedding_cache.npz
rm -f ml/output/training/contrastive/evaluation_co_bank.csv
rm -f "$BACKBONE/presence_classifier_current.pt"
echo "Done." | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Stage 3: Classifier cycles (8x) ===" | tee -a "$LOG"
for i in 1 2 3 4 5 6 7 8; do
  echo "" | tee -a "$LOG"
  echo "--- Cycle $i ---" | tee -a "$LOG"
  caffeinate -i $VENV -u ml/scripts/run_training.py \
    --mode train-classifier \
    --label "c$i-mac-llm" \
    --device cpu \
    --model "$BACKBONE" \
    --local-only \
    --annotation-csv "$ANNO" \
    --train-cases "$TRAIN_CASES" \
    --co-neg-per-case 5 --fp-neg-per-case 10 \
    --embedding-min-sim 0.05 --epochs 25 \
    --recall-weight 0.25 --hidden-dim 512 \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
echo "Check: ml/output/evaluation/contrastive/evaluation_history.csv" | tee -a "$LOG"
