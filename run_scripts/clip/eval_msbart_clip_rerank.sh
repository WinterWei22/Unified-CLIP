#!/bin/bash
# Evaluate CLIP re-ranking of MS-BART generated molecules
# Usage: bash run_scripts/clip/eval_msbart_clip_rerank.sh
cd /home/weiwentao/workspace/ms-pred

# ── Option 1: Full inference (compute embeddings from scratch) ──
# CUDA_VISIBLE_DEVICES=1 python src/ms_pred/clip/eval_msbart_clip_rerank.py \
#     --gpu \
#     --batch-size 512 \
#     --num-workers 32 \
#     --top-k 10 \
#     --checkpoint-pth results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/best.ckpt \
#     --magma-dag-folder /data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/ \
#     --labels-tsv data/spec_datasets/msg/labels.tsv \
#     --clip-weight 1.0 \
#     --paper-jsonl /home/weiwentao/workspace/mol_gen/MS-BART/logs/results/msg_gt.jsonl \
#     --save-dir results/eval_msbart_clip_rerank_msg_gt/

# ── Option 2: Re-analyze saved results (no GPU needed, instant) ──
python src/ms_pred/clip/eval_msbart_clip_rerank.py \
    --from-saved results/eval_msbart_clip_rerank_msg_pred/eval_msbart_clip_rerank_details.json \
    --clip-weight 0.5 \
    --num-workers 64 \
    --top-k 10
