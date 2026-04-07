"""Verify two-stage prediction by comparing with direct per-pair computation."""
import pickle
import sys
import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import ms_pred.common as common
from ms_pred.clip.clip_model_unimol import CLIPModelUniMol
from ms_pred.clip.unimol_encoder import smiles_to_unimol_features, collate_unimol_features
from ms_pred.clip.clip_data_unimol import pad_tuples_to_tensor

CKPT = "/tmp/best_unimol.ckpt"
PKL = "/home/weiwentao/workspace/ms-pred/results_generation/clip_msg_unimol2/split_msg_unimol_rnd1/version_51/cands_df_test_mass_256_new_filtered_filteredagain.tsv.pkl"
LABELS = "data/spec_datasets/msg/retrieval/cands_df_test_mass_256_new_filtered_filteredagain.tsv"
MAGMA = "/data/weiwentao/ms-pred/results/dag_msg_train/split_msg_rnd1/preds_train_20_inten_corrected/"
CACHE_DIR = "data/spec_datasets/msg/retrieval/unimol_features"

N_CHECK = 30  # number of rows to verify


def smiles_hash(smiles):
    return hashlib.md5(smiles.encode()).hexdigest()


def main():
    os.chdir("/home/weiwentao/workspace/ms-pred")

    # Load saved results
    with open(PKL, "rb") as f:
        saved = pickle.load(f)
    print(f"Saved pkl keys: {list(saved.keys())}")
    print(f"Number of saved entries: {len(saved['spec_names'])}")

    saved_scores = saved["cosine_similarity"].flatten()
    saved_spec_names = saved["spec_names"]
    saved_smiles = saved["smiles"]

    # Load model
    model = CLIPModelUniMol.load_from_checkpoint(CKPT, map_location='cpu')
    model.eval()
    model = model.cuda()

    emb_ce = model.embed_ce
    print(f"embed_ce: {emb_ce}")

    # Load DataFrame
    df = pd.read_csv(LABELS, sep="\t")
    print(f"DataFrame shape: {df.shape}")

    # MAGMA files
    magma_folder = Path(MAGMA)
    all_jsons = list(magma_folder.glob("*.json"))
    name_to_json = {p.stem.replace("pred_", ""): p for p in all_jsons}

    # Verify first N rows
    print(f"\n{'='*80}")
    print(f"Verifying first {N_CHECK} rows...")
    print(f"{'='*80}")

    mismatches = 0
    for i in range(N_CHECK):
        row = df.iloc[i]
        spec_name = row['spec']
        smiles = row['smiles']
        ionization = row['ionization']
        ce = row.get('collision_energies', 0.0) if emb_ce else None

        # Check if saved entry matches DataFrame row
        if saved_spec_names[i] != str(spec_name):
            print(f"Row {i}: SPEC NAME MISMATCH: saved={saved_spec_names[i]}, df={spec_name}")
            continue
        if saved_smiles[i] != str(smiles):
            print(f"Row {i}: SMILES MISMATCH: saved={saved_smiles[i]}, df={smiles}")
            continue

        # --- Direct computation ---
        # Load molecule features
        cache_path = os.path.join(CACHE_DIR, f"{smiles_hash(smiles)}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            mol_feat = {k: data[k] for k in data.files}
        else:
            mol_feat, _ = smiles_to_unimol_features(smiles, seed=42)

        # Collate as single-item batch
        mol_feats_batched = collate_unimol_features([mol_feat])
        mol_feats_gpu = {k: v.cuda() for k, v in mol_feats_batched.items()}
        adduct = torch.FloatTensor([common.ion2onehot_pos[ionization]]).cuda()
        ces_tensor = torch.FloatTensor([ce]).cuda() if emb_ce and ce is not None else None

        # Load spectrum
        json_path = name_to_json.get(spec_name)
        if json_path is None:
            print(f"Row {i}: No magma json for {spec_name}")
            continue
        with open(json_path, "r") as f:
            tree = json.load(f)
        spec = tree["raw_spec"]
        prec_mz = tree['prec_mz']
        spec.insert(0, [prec_mz, 1.1])

        spec_tensor, spec_mask = pad_tuples_to_tensor([spec], max_len=64)
        spec_tensor = spec_tensor.cuda()
        spec_mask = spec_mask.cuda()
        spec_adduct = torch.FloatTensor([common.ion2onehot_pos[ionization]]).cuda()

        with torch.no_grad():
            mol_proj, _ = model.predict_smi(mol_feats_gpu, adduct, ces_tensor)
            spec_proj, _ = model.predict_spec(spec_tensor, spec_adduct, spec_mask)
            direct_score = (spec_proj * mol_proj).sum().item()

        diff = abs(direct_score - saved_scores[i])
        status = "OK" if diff < 0.01 else "MISMATCH"
        if diff >= 0.01:
            mismatches += 1
        print(f"Row {i:3d}: direct={direct_score:.6f}, saved={saved_scores[i]:.6f}, diff={diff:.6f} {status}"
              f"  spec={spec_name}, smi={smiles[:40]}...")

    print(f"\nTotal mismatches (diff>=0.01): {mismatches}/{N_CHECK}")

    # Also check: are mol embeddings in pkl the same as direct?
    print(f"\n{'='*80}")
    print("Checking saved mol embeddings directly...")
    print(f"{'='*80}")
    if "mol_emb" in saved:
        for i in range(min(5, N_CHECK)):
            row = df.iloc[i]
            smiles = row['smiles']
            ionization = row['ionization']
            ce = row.get('collision_energies', 0.0) if emb_ce else None

            cache_path = os.path.join(CACHE_DIR, f"{smiles_hash(smiles)}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                mol_feat = {k: data[k] for k in data.files}
            else:
                mol_feat, _ = smiles_to_unimol_features(smiles, seed=42)

            mol_feats_batched = collate_unimol_features([mol_feat])
            mol_feats_gpu = {k: v.cuda() for k, v in mol_feats_batched.items()}
            adduct = torch.FloatTensor([common.ion2onehot_pos[ionization]]).cuda()
            ces_tensor = torch.FloatTensor([ce]).cuda() if emb_ce and ce is not None else None

            with torch.no_grad():
                mol_proj, _ = model.predict_smi(mol_feats_gpu, adduct, ces_tensor)

            saved_mol = saved["mol_emb"][i]
            direct_mol = mol_proj.cpu().numpy().flatten()

            emb_diff = np.abs(saved_mol - direct_mol).max()
            emb_cos = np.dot(saved_mol, direct_mol) / (np.linalg.norm(saved_mol) * np.linalg.norm(direct_mol) + 1e-10)
            print(f"Row {i}: mol_emb max_diff={emb_diff:.6f}, cosine={emb_cos:.6f}")


if __name__ == "__main__":
    main()
