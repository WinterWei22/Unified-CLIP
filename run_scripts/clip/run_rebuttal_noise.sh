#!/bin/bash
# Rebuttal Experiment A1: MAGMa Noise Robustness Analysis
# noise_0.1 on GPU 0, noise_0.3 and noise_0.5 on GPU 1 (sequentially)
# Usage: conda run -n ms-gen bash run_scripts/clip/run_rebuttal_noise.sh

set -e
cd /home/weiwentao/workspace/ms-pred

RESULTS_BASE="results/rebuttal_noise"
PREDICT_SCRIPT="src/ms_pred/clip/clip_predict_smi.py"
METRICS_SCRIPT="run_scripts/clip/cal_retrieval_metrics_v3.py"
MAGMA_FOLDER="/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/"

MASS_LABELS="cands_df_test_mass_256_new_filtered_filteredagain.tsv"
FORMULA_LABELS="cands_df_test_formula_256_new_filtered_filteredagain.tsv"

echo "============================================"
echo "  Rebuttal Noise Robustness Experiment"
echo "============================================"

# ====== TRAINING: noise_0.1 on GPU 0 (background) ======
echo ">>> Training noise_ratio=0.1 on GPU 0 (background)..."
python launcher_scripts/run_from_config.py configs/clip/rebuttal_noise/train_noise_0.1.yaml &
PID_01=$!

# ====== TRAINING: noise_0.3 on GPU 1, then noise_0.5 on GPU 1 ======
echo ">>> Training noise_ratio=0.3 on GPU 1..."
python launcher_scripts/run_from_config.py configs/clip/rebuttal_noise/train_noise_0.3.yaml
echo ">>> Training noise_ratio=0.3 DONE."

echo ">>> Training noise_ratio=0.5 on GPU 1..."
python launcher_scripts/run_from_config.py configs/clip/rebuttal_noise/train_noise_0.5.yaml
echo ">>> Training noise_ratio=0.5 DONE."

# Wait for noise_0.1 to finish
echo ">>> Waiting for noise_ratio=0.1 to finish..."
wait $PID_01
echo ">>> Training noise_ratio=0.1 DONE."

# ====== TESTING ======
for ratio in 0.1 0.3 0.5; do
    MODEL_DIR="${RESULTS_BASE}/noise_${ratio}/version_0"
    CKPT="${MODEL_DIR}/best.ckpt"

    if [ ! -f "$CKPT" ]; then
        echo "WARNING: Checkpoint not found at ${CKPT}, trying to find best*.ckpt"
        CKPT=$(ls ${MODEL_DIR}/best*.ckpt 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            echo "ERROR: No checkpoint found for noise_ratio=${ratio}. Skipping."
            continue
        fi
    fi
    echo ""
    echo ">>> Testing noise_ratio=${ratio} with checkpoint: ${CKPT}"

    for benchmark in mass formula; do
        if [ "$benchmark" = "mass" ]; then
            LABELS=$MASS_LABELS
        else
            LABELS=$FORMULA_LABELS
        fi

        PRED_SAVE_DIR="${MODEL_DIR}/predict_${benchmark}"
        echo "  > Predicting ${benchmark} benchmark..."

        CUDA_VISIBLE_DEVICES=0 python ${PREDICT_SCRIPT} \
            --gpu \
            --num-workers 16 \
            --batch-size 32 \
            --dataset-name msg \
            --split-name split_msg.tsv \
            --subset-datasets test_only \
            --dataset-labels ${LABELS} \
            --checkpoint ${CKPT} \
            --save-dir ${PRED_SAVE_DIR} \
            --binned-out \
            --magma-dag-folder ${MAGMA_FOLDER}

        echo "  > Computing metrics for ${benchmark}..."
        PKL_FILE=$(ls ${PRED_SAVE_DIR}/*.pkl 2>/dev/null | head -1)
        if [ -n "$PKL_FILE" ]; then
            python ${METRICS_SCRIPT} --input "${PKL_FILE}"
        else
            echo "  WARNING: No pkl file found in ${PRED_SAVE_DIR}"
        fi
    done
done

echo ""
echo "============================================"
echo "  All experiments completed!"
echo "============================================"
