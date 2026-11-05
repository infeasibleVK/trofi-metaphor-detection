#!/usr/bin/env bash
# ==============================================================================
# One-Click Execution Script for Task 1: Classical SPV Baseline
# Project 11: TroFi Metaphor Detection (CS60075 Assignment II)
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================================="
echo " Starting Task 1: Classical Selectional-Preference Violation Pipeline"
echo " Repository root: $REPO_ROOT"
echo "=================================================================="

# Check virtual environment
if [ -d ".venv" ]; then
    echo "Found .venv. Activating virtual environment..."
    source .venv/bin/activate
fi

# 1. Run Preprocessing
echo ""
echo "[Step 1/2] Preprocessing raw TroFi dataset and generating splits..."
python code/task1/preprocess.py \
    --data_path data/TroFiExampleBase.txt \
    --output_dir data/ \
    --seed 42

# 2. Run Training, Validation Tuning, and Test Evaluation
echo ""
echo "[Step 2/2] Training SPV model, tuning threshold on Val, and evaluating on Test..."
python code/task1/evaluate_task1.py \
    --train_path data/trofi_train.csv \
    --val_path data/trofi_val.csv \
    --test_path data/trofi_test.csv \
    --output_scores data/task1/spv_scores.csv \
    --alpha 0.1 \
    --prior_type relation_specific \
    --profile_source all \
    --compare_configs

echo ""
echo "=================================================================="
echo " Task 1 Completed Successfully!"
echo " Primary outputs:"
echo "   - data/task1/spv_scores.csv"
echo "   - data/task1_scores.csv"
echo "   - data/task1_preds.csv"
echo "=================================================================="
