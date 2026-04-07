"""
Prediction script for CLIP model with UniMol2 molecular encoder.
Two-stage approach for efficiency:
  Stage 1: Encode unique molecules (776K) and unique spectra (17K)
  Stage 2: Look up embeddings for all 4.4M pairs and compute similarity

Usage:
    python src/ms_pred/clip/clip_predict_smi_unimol.py \
        --checkpoint-pth /path/to/best.ckpt \
        --dataset-name msg \
        --dataset-labels cands_df_test_mass_256_new_filtered_filteredagain.tsv \
        --gpu --batch-size 64 --num-workers 8
"""
import logging
import hashlib
from datetime import datetime
import yaml
import argparse
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import os

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

import ms_pred.common as common
from ms_pred.clip.clip_model_unimol import CLIPModelUniMol
from ms_pred.clip.unimol_encoder import smiles_to_unimol_features, collate_unimol_features
from ms_pred.clip.clip_data_unimol import pad_tuples_to_tensor


def smiles_hash(smiles):
    return hashlib.md5(smiles.encode()).hexdigest()


# ============================================================
# Stage 1a: Dataset for unique molecules (SMILES-only)
# ============================================================
class UniqueMolDataset(Dataset):
    """Dataset over unique SMILES for batch encoding."""

    def __init__(self, smiles_list, adduct_list, ce_list=None, mol_keys=None,
                 unimol_cache_dir=None, unimol_seed=42, unimol_max_atoms=128):
        self.smiles_list = smiles_list
        self.adduct_list = adduct_list
        self.ce_list = ce_list
        self.mol_keys = mol_keys
        self.unimol_cache_dir = unimol_cache_dir
        self.unimol_seed = unimol_seed
        self.unimol_max_atoms = unimol_max_atoms

    def __len__(self):
        return len(self.smiles_list)

    def _load_cached_features(self, smi_key):
        npz_path = os.path.join(self.unimol_cache_dir, f"{smi_key}.npz")
        data = np.load(npz_path)
        feat = {k: data[k] for k in data.files}
        n_atoms = int(feat['atom_mask'].sum())
        return feat, n_atoms

    def _featurize_smiles(self, smiles):
        if self.unimol_cache_dir:
            try:
                return self._load_cached_features(smiles_hash(smiles))
            except FileNotFoundError:
                pass
        feat, n_atoms = smiles_to_unimol_features(
            smiles, seed=self.unimol_seed, max_atoms=self.unimol_max_atoms
        )
        return feat, n_atoms

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        adduct = self.adduct_list[idx]
        ce = self.ce_list[idx] if self.ce_list is not None else 0.0
        mol_key = self.mol_keys[idx] if self.mol_keys is not None else smiles
        try:
            mol_feat, n_atoms = self._featurize_smiles(smiles)
        except Exception as e:
            logging.warning(f"Error featurizing {smiles}: {e}")
            return None
        return {
            "smiles": smiles,
            "mol_key": mol_key,
            "mol_feat": mol_feat,
            "adduct": adduct,
            "ce": ce,
        }

    @staticmethod
    def collate_fn(input_list):
        input_list = [j for j in input_list if j is not None]
        if not input_list:
            return None
        smiles_list = [j["smiles"] for j in input_list]
        mol_keys = [j["mol_key"] for j in input_list]
        mol_feats = collate_unimol_features([j["mol_feat"] for j in input_list])
        adducts = torch.FloatTensor([j["adduct"] for j in input_list])
        ces = torch.FloatTensor([j["ce"] for j in input_list])
        return {
            "smiles": smiles_list,
            "mol_keys": mol_keys,
            "mol_feats": mol_feats,
            "adducts": adducts,
            "ces": ces,
        }


# ============================================================
# Stage 1b: Dataset for unique spectra
# ============================================================
class UniqueSpecDataset(Dataset):
    """Dataset over unique spectra for batch encoding."""

    def __init__(self, spec_names, magma_map, adduct_map, ce_map=None, max_spec_len=64):
        self.spec_names = spec_names
        self.magma_map = magma_map
        self.adduct_map = adduct_map
        self.ce_map = ce_map
        self.max_spec_len = max_spec_len

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx):
        name = self.spec_names[idx]
        json_path = self.magma_map[name]
        with open(json_path, "r") as f:
            tree = json.load(f)
        spec = tree["raw_spec"]
        prec_mz = tree['prec_mz']
        spec.insert(0, [prec_mz, 1.1])
        adduct = self.adduct_map[name]
        ce = self.ce_map.get(name, 0.0) if self.ce_map else 0.0
        return {
            "name": name,
            "spec": spec,
            "adduct": adduct,
            "ce": ce,
            "max_spec_len": self.max_spec_len,
        }

    @staticmethod
    def collate_fn(input_list):
        names = [j["name"] for j in input_list]
        specs = [j["spec"] for j in input_list]
        max_spec_len = input_list[0]["max_spec_len"]
        batched_specs, specs_mask = pad_tuples_to_tensor(specs, max_len=max_spec_len)
        adducts = torch.FloatTensor([j["adduct"] for j in input_list])
        ces = torch.FloatTensor([j["ce"] for j in input_list])
        return {
            "names": names,
            "specs": batched_specs,
            "specs_mask": specs_mask,
            "adducts": adducts,
            "ces": ces,
        }


class MolEncoderWrapper(torch.nn.Module):
    """Wrapper for multi-GPU DataParallel molecule encoding."""

    def __init__(self, model):
        super().__init__()
        self.mol_encoder = model.mol_encoder
        self.mol_projection = model.mol_projection
        self.emb_adducts = model.emb_adducts
        self.embed_ce = model.embed_ce
        if self.emb_adducts:
            self.adduct_embedder = model.adduct_embedder
        if self.embed_ce:
            self.collision_embedder_denominators = model.collision_embedder_denominators
            self.collision_embed_merged = model.collision_embed_merged

    def forward(self, atom_feat, atom_mask, edge_feat, shortest_path,
                degree, pair_type, attn_bias, src_tokens, src_coord,
                adducts, ces):
        # Reconstruct mol_feats dict
        mol_feats = {
            'atom_feat': atom_feat,
            'atom_mask': atom_mask,
            'edge_feat': edge_feat,
            'shortest_path': shortest_path,
            'degree': degree,
            'pair_type': pair_type,
            'attn_bias': attn_bias,
            'src_tokens': src_tokens,
            'src_coord': src_coord,
        }
        atoms_emb, mol_emb, batch_indices = self.mol_encoder(mol_feats)
        if self.emb_adducts:
            embed_adducts = self.adduct_embedder[adducts.long()]
            mol_emb = torch.cat([mol_emb, embed_adducts], -1)
        if self.embed_ce and ces is not None:
            embed_collision = torch.cat(
                (torch.sin(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                 torch.cos(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                dim=1
            )
            embed_collision = torch.where(
                torch.isnan(embed_collision),
                self.collision_embed_merged.unsqueeze(0),
                embed_collision
            )
            mol_emb = torch.cat([mol_emb, embed_collision], -1)
        mol_proj = self.mol_projection(mol_emb)
        mol_proj = F.normalize(mol_proj, dim=1)
        return mol_proj


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument("--num-workers", default=8, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    parser.add_argument("--num-gpu", default=1, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_clip_unimol_pred/")
    parser.add_argument("--checkpoint-pth", default=None, type=str)
    parser.add_argument("--magma-dag-folder", type=str)
    parser.add_argument("--dataset-name", default="msg")
    parser.add_argument("--dataset-labels", default="cands_df_test_mass_256_new_filtered_filteredagain.tsv")
    parser.add_argument("--split-name", default="split_msg.tsv")
    parser.add_argument("--subset-datasets", default="none",
                        choices=["none", "train_only", "test_only", "debug_special"])
    parser.add_argument("--unimol-cache-dir", default=None, type=str,
                        help="Directory with precomputed .npz UniMol2 features for retrieval candidates")
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="clip_unimol_pred.log", debug=kwargs["debug"])
    pl.seed_everything(42)
    binned_out = kwargs["binned_out"]

    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name / "retrieval"
    labels = data_dir / kwargs["dataset_labels"]
    df = pd.read_csv(labels, sep="\t")

    if kwargs["debug"]:
        df = df[:1000]

    # Load model
    best_checkpoint = kwargs["checkpoint_pth"]
    logging.info(f"Loading model from {best_checkpoint}")
    model = CLIPModelUniMol.load_from_checkpoint(best_checkpoint, map_location='cpu')
    emb_ce = model.embed_ce

    num_gpu = kwargs.get("num_gpu", 1)
    model.eval()
    gpu = kwargs["gpu"]
    if gpu:
        model = model.cuda()
    device = torch.device("cuda") if gpu else torch.device("cpu")

    # MAGMA files for spectrum loading
    magma_dag_folder = Path(kwargs["magma_dag_folder"])
    all_json_pths = [Path(i) for i in magma_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    print(f"Found {len(name_to_json)} magma json files")

    # UniMol cache
    unimol_cache_dir = kwargs.get("unimol_cache_dir", None)

    # ============================================================
    # Build unique molecule and spectrum sets
    # ============================================================
    # Adduct/CE maps per spec
    name_to_adduct_vec = {}
    name_to_ce = {}
    for _, row in df.drop_duplicates(subset='spec').iterrows():
        name = row['spec']
        name_to_adduct_vec[name] = common.ion2onehot_pos[row['ionization']]
        if emb_ce:
            name_to_ce[name] = row.get('collision_energies', 0.0)

    # Unique (smiles, ionization) pairs for molecules
    # Each unique SMILES may appear with different adducts
    mol_key_cols = ['smiles', 'ionization']
    if emb_ce:
        mol_key_cols.append('collision_energies')
    unique_mol_df = df.drop_duplicates(subset=mol_key_cols).reset_index(drop=True)

    # Build mol key -> index mapping
    def make_mol_key(row):
        key = row['smiles'] + '||' + row['ionization']
        if emb_ce:
            key += '||' + str(row.get('collision_energies', 0.0))
        return key

    unique_mol_keys = [make_mol_key(row) for _, row in unique_mol_df.iterrows()]
    mol_key_to_idx = {k: i for i, k in enumerate(unique_mol_keys)}

    print(f"Total rows: {len(df)}, unique molecules (smiles+adduct): {len(unique_mol_df)}, "
          f"unique spectra: {df['spec'].nunique()}")

    # ============================================================
    # Stage 1a: Encode unique molecules
    # ============================================================
    print("Stage 1a: Encoding unique molecules...")
    unique_smiles = unique_mol_df['smiles'].tolist()
    unique_adducts = [common.ion2onehot_pos[ion] for ion in unique_mol_df['ionization']]
    unique_ces = None
    if emb_ce:
        unique_ces = unique_mol_df['collision_energies'].tolist()

    mol_dataset = UniqueMolDataset(
        unique_smiles, unique_adducts, unique_ces,
        mol_keys=unique_mol_keys,
        unimol_cache_dir=unimol_cache_dir,
    )
    # Scale batch size by number of GPUs
    mol_batch_size = kwargs["batch_size"] * num_gpu
    mol_loader = DataLoader(
        mol_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=UniqueMolDataset.collate_fn,
        shuffle=False,
        batch_size=mol_batch_size,
    )

    # Wrap mol encoder for multi-GPU
    if gpu and num_gpu > 1:
        mol_encoder_dp = torch.nn.DataParallel(
            MolEncoderWrapper(model),
            device_ids=list(range(num_gpu)),
        )
        mol_encoder_dp.eval()
        print(f"Using {num_gpu} GPUs for molecule encoding (batch_size={mol_batch_size})")
    else:
        mol_encoder_dp = None

    # Encode and track which mol_keys succeeded (collate skips None)
    all_mol_proj = []
    encoded_mol_keys = []
    with torch.no_grad():
        for batch in tqdm(mol_loader, desc="Encoding molecules"):
            if batch is None:
                continue
            if mol_encoder_dp is not None:
                mol_feats = batch["mol_feats"]
                mol_proj = mol_encoder_dp(
                    mol_feats['atom_feat'].cuda(),
                    mol_feats['atom_mask'].cuda(),
                    mol_feats['edge_feat'].cuda(),
                    mol_feats['shortest_path'].cuda(),
                    mol_feats['degree'].cuda(),
                    mol_feats['pair_type'].cuda(),
                    mol_feats['attn_bias'].cuda(),
                    mol_feats['src_tokens'].cuda(),
                    mol_feats['src_coord'].cuda(),
                    batch["adducts"].cuda(),
                    batch["ces"].cuda(),
                )
            else:
                mol_feats = {k: v.to(device) for k, v in batch["mol_feats"].items()}
                adducts = batch["adducts"].to(device)
                ces = batch["ces"].to(device) if emb_ce else None
                mol_proj, _ = model.predict_smi(mol_feats, adducts, ces)
            all_mol_proj.append(mol_proj.cpu())
            encoded_mol_keys.extend(batch["mol_keys"])

    all_mol_proj = torch.cat(all_mol_proj, dim=0).numpy()
    # Rebuild index from actually encoded mol_keys
    mol_key_to_idx = {k: i for i, k in enumerate(encoded_mol_keys)}
    print(f"Encoded {all_mol_proj.shape[0]} unique molecules -> {all_mol_proj.shape}")

    # ============================================================
    # Stage 1b: Encode unique spectra
    # ============================================================
    print("Stage 1b: Encoding unique spectra...")
    unique_specs = df['spec'].unique().tolist()
    # Filter to those that have magma json
    unique_specs = [s for s in unique_specs if s in name_to_json]
    spec_to_idx = {s: i for i, s in enumerate(unique_specs)}

    spec_dataset = UniqueSpecDataset(
        unique_specs, name_to_json, name_to_adduct_vec,
        ce_map=name_to_ce if emb_ce else None,
    )
    spec_loader = DataLoader(
        spec_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=UniqueSpecDataset.collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"] * 4,  # spectra are lightweight
    )

    all_spec_proj = []
    with torch.no_grad():
        for batch in tqdm(spec_loader, desc="Encoding spectra"):
            specs = batch["specs"].to(device)
            specs_mask = batch["specs_mask"].to(device)
            adducts = batch["adducts"].to(device)
            spec_global_emb, _ = model.predict_spec(specs, adducts, specs_mask)
            all_spec_proj.append(spec_global_emb.cpu())

    all_spec_proj = torch.cat(all_spec_proj, dim=0).numpy()  # (num_unique_spec, proj_dim)
    print(f"Encoded {all_spec_proj.shape[0]} unique spectra -> {all_spec_proj.shape}")

    # ============================================================
    # Stage 2: Look up embeddings and compute similarity
    # ============================================================
    print("Stage 2: Computing similarities for all pairs...")

    def cosine_similarity(a, b):
        dot_product = np.sum(a * b, axis=1)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        epsilon = 1e-10
        return (dot_product / (norm_a * norm_b + epsilon)).reshape(-1, 1)

    spec_names_ar = []
    flag_ar = []
    smiles_ar = []
    mol_embs = []
    spec_embs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building pairs"):
        spec_name = row['spec']
        smiles = row['smiles']
        flag = row.get('label', None)

        mol_key = make_mol_key(row)
        mol_idx = mol_key_to_idx.get(mol_key, None)
        spec_idx = spec_to_idx.get(spec_name, None)

        if mol_idx is None or spec_idx is None:
            continue

        spec_names_ar.append(str(spec_name))
        flag_ar.append(str(flag))
        smiles_ar.append(str(smiles))
        mol_embs.append(all_mol_proj[mol_idx])
        spec_embs.append(all_spec_proj[spec_idx])

    mol_emb_matrix = np.array(mol_embs)
    spec_emb_matrix = np.array(spec_embs)
    scores = cosine_similarity(spec_emb_matrix, mol_emb_matrix)

    print(f"Computed {len(scores)} similarities")

    if binned_out:
        output = {
            "spec_names": spec_names_ar,
            "flags": flag_ar,
            "mol_emb": mol_emb_matrix,
            "cosine_similarity": scores,
            "local_similarity": scores,  # same as global for now
            "smiles": smiles_ar,
        }
        out_file = Path(kwargs["save_dir"]) / f"{kwargs['dataset_labels']}.pkl"
        with open(out_file, "wb") as fp:
            pickle.dump(output, fp)
        print(f"{len(scores)} predictions saved to {out_file}")
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import time
    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
