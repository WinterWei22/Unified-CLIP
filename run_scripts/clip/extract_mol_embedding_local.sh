#!/bin/bash

# Configuration
CHECKPOINT_PATH="/home/weiwentao/workspace/ms-pred/results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/best.ckpt"
SMILES_FILE="/home/weiwentao/workspace/ms-pred/data/ms-bart-smiles/train_smiles.txt"
OUTPUT_DIR="/home/weiwentao/workspace/ms-pred/data/ms-bart-smiles"
BATCH_SIZE=256
NUM_WORKERS=16

# Activate conda environment
source ~/miniconda3/bin/activate ms-gen

# Run the extraction script (without --gpu flag to use CPU)
cd /home/weiwentao/workspace/ms-pred

python run_scripts/clip/extract_mol_embedding.py \
    --checkpoint-pth "$CHECKPOINT_PATH" \
    --smiles-file "$SMILES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
