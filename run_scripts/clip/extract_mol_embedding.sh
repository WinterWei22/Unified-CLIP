#!/bin/bash

#SBATCH --job-name=extract_mol_embedding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=extract_mol_embedding_%j.log
#SBATCH --error=extract_mol_embedding_%j.err

# Configuration
CHECKPOINT_PATH="/home/weiwentao/workspace/ms-pred/results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/best.ckpt"
SMILES_FILE="/home/weiwentao/workspace/ms-pred/data/ms-bart-smiles/train_smiles.txt"
OUTPUT_DIR="/home/weiwentao/workspace/ms-pred/results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/mol_embeddings"
BATCH_SIZE=256
NUM_WORKERS=16

# Set CUDA device to 7
export CUDA_VISIBLE_DEVICES=7

# Run the extraction script
cd /home/weiwentao/workspace/ms-pred

python run_scripts/clip/extract_mol_embedding.py \
    --checkpoint-pth "$CHECKPOINT_PATH" \
    --smiles-file "$SMILES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --gpu
