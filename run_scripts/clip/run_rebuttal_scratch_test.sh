#!/bin/bash
# Rebuttal Experiment B1: Test both scratch models
# Usage: bash run_scripts/clip/run_rebuttal_scratch_test.sh

set -e
export CUDA_VISIBLE_DEVICES=2
PYTHON="/home/weiwentao/miniconda3/envs/ms-gen/bin/python"
cd /home/weiwentao/workspace/ms-pred

# Determine which version directories contain the best checkpoints
BASELINE_DIR="results/rebuttal_scratch/clip_baseline_scratch/split_msg_rnd1"
PIFA_DIR="results/rebuttal_scratch/pifa_scratch/split_msg_rnd1"

# Find the latest version with best.ckpt for each
BASELINE_VERSION=$(ls -td ${BASELINE_DIR}/version_*/ 2>/dev/null | while read d; do
  if [ -f "$d/best.ckpt" ]; then echo "$d"; break; fi
done)
PIFA_VERSION=$(ls -td ${PIFA_DIR}/version_*/ 2>/dev/null | while read d; do
  if [ -f "$d/best.ckpt" ]; then echo "$d"; break; fi
done)

echo "Baseline version: $BASELINE_VERSION"
echo "PiFA version: $PIFA_VERSION"

DAG_FOLDER="/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/"

# Test function
run_test() {
    local CKPT_DIR=$1
    local LABEL_FILE=$2
    local DESC=$3

    echo "============================================"
    echo "Testing: $DESC"
    echo "Checkpoint: ${CKPT_DIR}/best.ckpt"
    echo "Labels: $LABEL_FILE"
    echo "============================================"

    $PYTHON src/ms_pred/clip/clip_predict_smi.py \
      --gpu \
      --num-workers 8 \
      --batch-size 1024 \
      --dataset-name msg \
      --subset-datasets test_only \
      --dataset-labels "$LABEL_FILE" \
      --checkpoint "${CKPT_DIR}/best.ckpt" \
      --save-dir "$CKPT_DIR" \
      --binned-out \
      --split-name split_msg.tsv \
      --magma-dag-folder "$DAG_FOLDER"
}

# Test Baseline on mass
run_test "$BASELINE_VERSION" "cands_df_test_mass_256_new_filtered_filteredagain.tsv" "CLIP-baseline (scratch) - mass"

# Test Baseline on formula
run_test "$BASELINE_VERSION" "cands_df_test_formula_256_new_filtered_filteredagain.tsv" "CLIP-baseline (scratch) - formula"

# Test PiFA on mass
run_test "$PIFA_VERSION" "cands_df_test_mass_256_new_filtered_filteredagain.tsv" "PiFA (scratch) - mass"

# Test PiFA on formula
run_test "$PIFA_VERSION" "cands_df_test_formula_256_new_filtered_filteredagain.tsv" "PiFA (scratch) - formula"

echo "============================================"
echo "All tests completed!"
echo "============================================"
