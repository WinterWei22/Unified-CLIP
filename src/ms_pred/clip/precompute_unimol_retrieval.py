"""
Precompute UniMol2 features for retrieval candidate molecules.
Saves features keyed by SMILES hash to avoid redundant computation for shared candidates.

Usage:
    python src/ms_pred/clip/precompute_unimol_retrieval.py \
        --dataset-name msg \
        --dataset-labels cands_df_test_mass_256_new_filtered_filteredagain.tsv \
        --output-dir data/spec_datasets/msg/retrieval/unimol_features \
        --num-workers 32
"""
import argparse
import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import ms_pred.common as common
from ms_pred.clip.unimol_encoder import smiles_to_unimol_features


def smiles_hash(smiles):
    """Deterministic short hash for a SMILES string."""
    return hashlib.md5(smiles.encode()).hexdigest()


def process_one(row_tuple, output_dir, seed=42):
    """Process a single molecule: SMILES -> .npz"""
    smi_key, smiles = row_tuple
    out_path = os.path.join(output_dir, f"{smi_key}.npz")
    if os.path.exists(out_path):
        return smi_key, True

    try:
        feat, n_atoms = smiles_to_unimol_features(smiles, seed=seed)
        np.savez_compressed(out_path, **{k: np.array(v) for k, v in feat.items()})
        return smi_key, True
    except Exception as e:
        return smi_key, f"FAILED: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="msg")
    parser.add_argument("--dataset-labels", default="cands_df_test_mass_256_new_filtered_filteredagain.tsv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path("data/spec_datasets") / args.dataset_name / "retrieval"
    labels = data_dir / args.dataset_labels
    df = pd.read_csv(labels, sep='\t')

    if args.output_dir is None:
        output_dir = str(data_dir / "unimol_features")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Deduplicate by SMILES
    unique_smiles = df['smiles'].drop_duplicates().values
    print(f"Total rows: {len(df)}, unique SMILES: {len(unique_smiles)}")

    # Key by SMILES hash
    row_tuples = [(smiles_hash(smi), smi) for smi in unique_smiles]

    # Check existing
    existing = set(os.listdir(output_dir))
    remaining = [(key, smi) for key, smi in row_tuples if f"{key}.npz" not in existing]
    print(f"Already cached: {len(row_tuples) - len(remaining)}, remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All features already cached!")
        return

    worker_fn = partial(process_one, output_dir=output_dir, seed=args.seed)

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_fn, remaining, chunksize=64),
                total=len(remaining), desc="Computing UniMol2 retrieval features"
            ))
    else:
        results = [worker_fn(r) for r in tqdm(remaining, desc="Computing UniMol2 retrieval features")]

    failed = [(name, msg) for name, msg in results if msg is not True]
    print(f"\nDone! Success: {len(results) - len(failed)}, Failed: {len(failed)}")
    if failed:
        for name, msg in failed[:20]:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
