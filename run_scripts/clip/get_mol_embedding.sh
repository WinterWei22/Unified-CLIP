#!/bin/bash

#SBATCH --job-name=get_mol_embedding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=get_mol_embedding_%j.log
#SBATCH --error=get_mol_embedding_%j.err

# Configuration
CHECKPOINT_PATH="/home/weiwentao/workspace/ms-pred/results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/best.ckpt"
MAGMA_DAG_FOLDER="/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/"
DATASET_NAME="msg"
DATASET_LABELS="labels.tsv"
OUTPUT_FILE="/home/weiwentao/workspace/ms-pred/results/clip_msg_continous_from_specvarse/split_msg_rnd1/version_1/mol_embedding.pkl"
BATCH_SIZE=256
NUM_WORKERS=16

# Set CUDA device to 0 (free GPU)
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/miniconda3/bin/activate ms-gen

# Run the extraction script
cd /home/weiwentao/workspace/ms-pred

python run_scripts/clip/get_mol_embedding.py \
    --checkpoint-pth "$CHECKPOINT_PATH" \
    --magma-dag-folder "$MAGMA_DAG_FOLDER" \
    --dataset-name "$DATASET_NAME" \
    --dataset-labels "$DATASET_LABELS" \
    --output-file "$OUTPUT_FILE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --gpu
