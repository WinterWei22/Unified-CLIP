#!/bin/bash
# Rebuttal Experiment B1: Decouple DreaMS pretraining contribution
# Train CLIP-baseline and PiFA both from scratch (no pretrained DreaMS weights)
# GPU: 2,3

set -e
export CUDA_VISIBLE_DEVICES=2,3
PYTHON="/home/weiwentao/miniconda3/envs/ms-gen/bin/python"
cd /home/weiwentao/workspace/ms-pred

BASELINE_DIR="results/rebuttal_scratch/clip_baseline_scratch/split_msg_rnd1"
PIFA_DIR="results/rebuttal_scratch/pifa_scratch/split_msg_rnd1"
DAG_FOLDER="/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/"
SPEC_CKPT="/home/weiwentao/workspace/DreaMS/dreams/models/pretrained/ssl_model.ckpt"
FRAG_PATH="/home/weiwentao/workspace/ms-pred/data/spec_datasets/msg/frags_index_map_observed.hdf5"

mkdir -p "$BASELINE_DIR" "$PIFA_DIR"

echo "============================================"
echo "Training CLIP-baseline (from scratch)"
echo "============================================"

$PYTHON src/ms_pred/clip/clip_train.py \
  --gpu \
  --num-gpu 2 \
  --seed 42 \
  --num-workers 8 \
  --batch-size 256 \
  --max-epochs 100 \
  --dataset-name msg \
  --split-name split_msg.tsv \
  --learning-rate 0.0001 \
  --lr-decay-rate 0.825 \
  --warm-up 1000 \
  --weight-decay 1e-5 \
  --patience 20 \
  --dropout 0.1 \
  --mpnn-type GGNN \
  --pe-embed-k 0 \
  --pool-op avg \
  --hidden-size 512 \
  --frag-set-layers 3 \
  --loss-fn cosine \
  --root-encode egt2d \
  --mabnet-layers 4 \
  --mabnet-heads 16 \
  --edge-update \
  --projection-dim 512 \
  --binned-targs \
  --embed-adduct \
  --embed-ce \
  --encode-forms \
  --grad-accumulate 1 \
  --add-hs \
  --dataset-labels labels.tsv \
  --spec-ckpt "$SPEC_CKPT" \
  --dreams \
  --no-pretrained-dreams \
  --magma-dag-folder "$DAG_FOLDER" \
  --save-dir "$BASELINE_DIR" \
  2>&1 | tee "$BASELINE_DIR/train.log"

echo "============================================"
echo "Training PiFA (from scratch)"
echo "============================================"

$PYTHON src/ms_pred/clip/clip_train.py \
  --gpu \
  --num-gpu 2 \
  --seed 1 \
  --num-workers 8 \
  --batch-size 256 \
  --max-epochs 100 \
  --dataset-name msg \
  --split-name split_msg.tsv \
  --learning-rate 0.0003 \
  --lr-decay-rate 0.825 \
  --warm-up 1000 \
  --weight-decay 1e-5 \
  --patience 20 \
  --dropout 0.1 \
  --mpnn-type GGNN \
  --pe-embed-k 0 \
  --pool-op avg \
  --hidden-size 512 \
  --frag-set-layers 3 \
  --loss-fn cosine \
  --root-encode egt2d \
  --mabnet-layers 4 \
  --mabnet-heads 16 \
  --edge-update \
  --projection-dim 512 \
  --binned-targs \
  --embed-adduct \
  --embed-ce \
  --encode-forms \
  --grad-accumulate 1 \
  --add-hs \
  --local-contra \
  --local-weight 0.5 \
  --local-threshold 0.5 \
  --frag-supervised \
  --frag-path "$FRAG_PATH" \
  --frag-supervised-weight 0.5 \
  --local-start-epochs 10 \
  --frags \
  --dataset-labels labels.tsv \
  --spec-ckpt "$SPEC_CKPT" \
  --dreams \
  --no-pretrained-dreams \
  --magma-dag-folder "$DAG_FOLDER" \
  --save-dir "$PIFA_DIR" \
  2>&1 | tee "$PIFA_DIR/train.log"

echo "============================================"
echo "Both training runs completed!"
echo "============================================"
