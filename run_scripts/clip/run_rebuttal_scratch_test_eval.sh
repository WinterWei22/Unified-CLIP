#!/bin/bash
# Rebuttal Experiment B1: Test and evaluate both scratch models
set -e
export CUDA_VISIBLE_DEVICES=2
PYTHON="/home/weiwentao/miniconda3/envs/ms-gen/bin/python"
cd /home/weiwentao/workspace/ms-pred

DAG_FOLDER="/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/"

# Find latest version directories with best.ckpt
BASELINE_DIR=$(ls -td results/rebuttal_scratch/clip_baseline_scratch/split_msg_rnd1/version_*/ 2>/dev/null | while read d; do
  if [ -f "$d/best.ckpt" ]; then echo "$d"; break; fi
done)
PIFA_DIR=$(ls -td results/rebuttal_scratch/pifa_scratch/split_msg_rnd1/version_*/ 2>/dev/null | while read d; do
  if [ -f "$d/best.ckpt" ]; then echo "$d"; break; fi
done)

echo "Baseline dir: $BASELINE_DIR"
echo "PiFA dir:     $PIFA_DIR"

run_predict_and_eval() {
    local DIR=$1
    local LABELS=$2
    local DESC=$3

    echo ""
    echo "============================================"
    echo "$DESC"
    echo "============================================"

    # Run prediction
    $PYTHON src/ms_pred/clip/clip_predict_smi.py \
      --gpu \
      --num-workers 8 \
      --batch-size 1024 \
      --dataset-name msg \
      --subset-datasets test_only \
      --dataset-labels "$LABELS" \
      --checkpoint "${DIR}/best.ckpt" \
      --save-dir "$DIR" \
      --binned-out \
      --split-name split_msg.tsv \
      --magma-dag-folder "$DAG_FOLDER"

    # Run evaluation
    $PYTHON run_scripts/clip/cal_retrieval_metrics_v6.py \
      --input "${DIR}/${LABELS}.pkl"
}

# Baseline - mass
run_predict_and_eval "$BASELINE_DIR" "cands_df_test_mass_256_new_filtered_filteredagain.tsv" "CLIP-baseline (scratch) - mass"

# Baseline - formula
run_predict_and_eval "$BASELINE_DIR" "cands_df_test_formula_256_new_filtered_filteredagain.tsv" "CLIP-baseline (scratch) - formula"

# PiFA - mass
run_predict_and_eval "$PIFA_DIR" "cands_df_test_mass_256_new_filtered_filteredagain.tsv" "PiFA (scratch) - mass"

# PiFA - formula
run_predict_and_eval "$PIFA_DIR" "cands_df_test_formula_256_new_filtered_filteredagain.tsv" "PiFA (scratch) - formula"

echo ""
echo "============================================"
echo "All tests and evaluations completed!"
echo "============================================"
