#!/bin/bash
cd /home/weiwentao/workspace/ms-pred
source activate dreams
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

CKPT="/tmp/best_unimol.ckpt"
OUTDIR="results_generation/clip_msg_unimol2/split_msg_unimol_rnd1/version_51/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/ms_pred/clip/clip_predict_smi_unimol.py \
  --gpu \
  --num-gpu 4 \
  --num-workers 4 \
  --batch-size 32 \
  --dataset-name msg \
  --dataset-labels cands_df_test_mass_256_new_filtered_filteredagain.tsv \
  --checkpoint-pth "$CKPT" \
  --binned-out \
  --unimol-cache-dir data/spec_datasets/msg/retrieval/unimol_features \
  --magma-dag-folder /data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/ \
  --save-dir "$OUTDIR"

echo "Prediction done. Running evaluation..."
python run_scripts/clip/cal_retrieval_metrics_v3.py \
  --input "${OUTDIR}/cands_df_test_mass_256_new_filtered_filteredagain.tsv.pkl"
