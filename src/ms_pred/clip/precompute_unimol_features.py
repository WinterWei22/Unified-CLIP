"""
Precompute UniMol2 features for all molecules in the dataset.
Saves features as individual .npz files for fast loading during training.

Usage:
    python src/ms_pred/clip/precompute_unimol_features.py \
        --dataset-name msg \
        --dataset-labels labels.tsv \
        --output-dir data/spec_datasets/msg/unimol_features \
        --num-workers 32
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import ms_pred.common as common
from ms_pred.clip.unimol_encoder import smiles_to_unimol_features


def process_one(row_tuple, output_dir, seed=42):
    """Process a single molecule: SMILES -> .npz"""
    spec_name, smiles = row_tuple
    out_path = os.path.join(output_dir, f"{spec_name}.npz")
    if os.path.exists(out_path):
        return spec_name, True

    try:
        feat, n_atoms = smiles_to_unimol_features(smiles, seed=seed)
        # Save all arrays
        np.savez_compressed(out_path, **{k: np.array(v) for k, v in feat.items()})
        return spec_name, True
    except Exception as e:
        return spec_name, f"FAILED: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="msg")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = common.get_data_dir(args.dataset_name)
    labels = data_dir / args.dataset_labels
    df = pd.read_csv(labels, sep='\t')

    if args.output_dir is None:
        output_dir = str(data_dir / "unimol_features")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Deduplicate by SMILES to avoid redundant computation
    unique_df = df.drop_duplicates(subset='smiles')
    print(f"Total specs: {len(df)}, unique SMILES: {len(unique_df)}")

    # But we save by spec_name for easy lookup
    row_tuples = list(zip(df['spec'].values, df['smiles'].values))

    # Check existing
    existing = set(os.listdir(output_dir))
    remaining = [(name, smi) for name, smi in row_tuples if f"{name}.npz" not in existing]
    print(f"Already cached: {len(row_tuples) - len(remaining)}, remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All features already cached!")
        return

    worker_fn = partial(process_one, output_dir=output_dir, seed=args.seed)

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_fn, remaining, chunksize=64),
                total=len(remaining), desc="Computing UniMol2 features"
            ))
    else:
        results = [worker_fn(r) for r in tqdm(remaining, desc="Computing UniMol2 features")]

    failed = [(name, msg) for name, msg in results if msg is not True]
    print(f"\nDone! Success: {len(results) - len(failed)}, Failed: {len(failed)}")
    if failed:
        for name, msg in failed[:20]:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
