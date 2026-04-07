"""
CLIPDataset variant that uses UniMol2 featurization for molecules
instead of DGL graph construction via MAGMA fragmentation.

Key differences from clip_data.py:
- Molecules are featurized as UniMol2 feature dicts (3D conformer + graph features)
- No DGL/PyG graph construction for molecules
- collate_fn uses UniMol2's padding utilities
- Still uses the same spectrum processing pipeline
"""

import torch
from torch.utils.data import Dataset

import logging
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from rdkit import Chem
from rdkit.Chem import AllChem

import ms_pred.common as common
from ms_pred.clip.unimol_encoder import smiles_to_unimol_features, collate_unimol_features

from torch.nn.utils.rnn import pad_sequence
import h5py


def pad_tuples_to_tensor(all_entries, max_len=64):
    """Pad variable-length spectrum peak lists to fixed tensor."""
    padded_tensors = []
    masks = []
    for entry in all_entries:
        entry_tensor = torch.FloatTensor(entry[:max_len])
        pad_len = max_len - entry_tensor.shape[0]
        if pad_len > 0:
            entry_tensor = torch.cat(
                [entry_tensor, torch.zeros(pad_len, entry_tensor.shape[1])], dim=0
            )
        mask = torch.ones(max_len)
        if pad_len > 0:
            mask[max_len - pad_len:] = 0
        padded_tensors.append(entry_tensor)
        masks.append(mask)

    return torch.stack(padded_tensors), torch.stack(masks)


def pad_2d_batch_with_mask(arrays):
    """Pad a list of 2D numpy arrays into a batched tensor with mask."""
    max_rows = max(a.shape[0] for a in arrays)
    max_cols = max(a.shape[1] for a in arrays)
    batch_size = len(arrays)

    padded = torch.full((batch_size, max_rows, max_cols), -1, dtype=torch.long)
    masks = torch.zeros(batch_size, max_rows, dtype=torch.bool)

    for i, a in enumerate(arrays):
        r, c = a.shape
        padded[i, :r, :c] = torch.from_numpy(a)
        masks[i, :r] = True

    return padded, masks


class CLIPDatasetUniMol(Dataset):
    """CLIP Dataset using UniMol2 for molecular featurization."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        magma_map: dict,
        max_spec_len: int = 64,
        decoys: bool = False,
        decoys_num: int = 5,
        decoys_df: pd.DataFrame = None,
        easy_decoys_ratio: float = 0.0,
        augment: bool = False,
        mz_shift_aug_max: float = 50.0,
        mz_shift_aug_p: float = 0.3,
        emb_ce: bool = False,
        frag_supervised: bool = False,
        frags_path: str = None,
        unimol_seed: int = 42,
        unimol_max_atoms: int = 128,
        unimol_cache_dir: str = None,
        **kwargs,
    ):
        self.df = df
        self.data_dir = data_dir
        self.magma_map = magma_map
        self.max_spec_len = max_spec_len
        self.decoys = decoys
        self.decoys_num = decoys_num
        self.decoys_df = decoys_df

        self.augment = augment
        self.mz_shift_aug_p = mz_shift_aug_p
        self.mz_shift_aug_max = mz_shift_aug_max

        self.frag_supervised = frag_supervised
        self.emb_ce = emb_ce

        self.unimol_seed = unimol_seed
        self.unimol_max_atoms = unimol_max_atoms

        # Feature cache directory (precomputed .npz files)
        self.unimol_cache_dir = unimol_cache_dir

        self.spec_names = self.df["spec"].values
        self.name_to_dict = self.df.set_index("spec").to_dict(orient="index")
        for i in self.name_to_dict:
            self.name_to_dict[i]["magma_file"] = self.magma_map[i]

        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)
        self.name_to_adducts = {
            i: common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        }

        if self.emb_ce:
            self.name_to_ce = dict(self.df[["spec", "collision_energies"]].values)

        self.spec_to_decoys = {}
        if self.decoys and self.decoys_df is not None:
            for k, g in self.decoys_df.groupby("spec"):
                self.spec_to_decoys[k] = g

        self.easy_decoys_ratio = easy_decoys_ratio
        if self.decoys and self.decoys_df is not None:
            self.easy_df = self.decoys_df[self.decoys_df["similarity"] > 0.05]

        if self.frag_supervised:
            self.frag_data = None
            self.frags_path = frags_path

    def read_tree(self, x):
        filename = self.name_to_dict[x]["magma_file"]
        with open(filename, "r") as f:
            tree = json.load(f)
        return tree

    def __len__(self):
        return len(self.df)

    def _load_cached_features(self, spec_name):
        """Load precomputed features from .npz cache."""
        npz_path = os.path.join(self.unimol_cache_dir, f"{spec_name}.npz")
        data = np.load(npz_path)
        feat = {k: data[k] for k in data.files}
        n_atoms = int(feat['atom_mask'].sum())
        return feat, n_atoms

    def _featurize_smiles(self, smiles, spec_name=None):
        """Convert SMILES to UniMol2 features, using cache if available."""
        if self.unimol_cache_dir and spec_name:
            try:
                return self._load_cached_features(spec_name)
            except FileNotFoundError:
                pass
        feat, n_atoms = smiles_to_unimol_features(
            smiles, seed=self.unimol_seed, max_atoms=self.unimol_max_atoms
        )
        return feat, n_atoms

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["smiles"]
        name = row["spec"]
        tree = self.read_tree(name)
        spec = tree["raw_spec"]
        prec_mz = tree['prec_mz']
        inchikey = row['inchikey']
        adduct = self.name_to_adducts[name]
        spec_scaled = [[sublist[0], sublist[1] * 100] for sublist in spec]

        spec.insert(0, [prec_mz, 1.1])
        spec_scaled.insert(0, [prec_mz, 1.1])

        if self.augment and self.mz_shift_aug_p > 0:
            if random.random() < self.mz_shift_aug_p:
                shift = (random.random() - 0.5) * 2 * self.mz_shift_aug_max
                for peak in spec:
                    peak[0] += shift

        ce = 0.0
        if self.emb_ce:
            ce = self.name_to_ce.get(name, float('nan'))

        # UniMol2 featurization instead of DGL graph
        mol_feat, n_atoms = self._featurize_smiles(smiles, spec_name=name)

        item = {
            "name": name,
            "spec": spec,
            "prec_mz": prec_mz,
            "spec_scaled": spec_scaled,
            "max_spec_len": self.max_spec_len,
            "adduct": adduct,
            "inchikey": inchikey,
            "ce": ce,
            "mol_feat": mol_feat,
            "n_atoms": n_atoms,
            "smiles": smiles,
        }

        # Fragment supervision
        frag_label = None
        if self.frag_supervised:
            if self.frag_data is None:
                self.frag_data = h5py.File(self.frags_path, 'r')
            if f'{name}' in self.frag_data:
                frag_label = self.frag_data[name][()]
            else:
                mol = Chem.MolFromSmiles(smiles)
                frag_label = np.full((2, mol.GetNumAtoms()), -1)
            item["frag_label"] = frag_label

        # Decoys
        if self.decoys:
            decoys_group = self.spec_to_decoys.get(name, pd.DataFrame())
            num_available = len(decoys_group)
            decoy_feats = []
            decoy_adducts = []
            decoy_sims = []

            if num_available > 0:
                if "similarity" in decoys_group.columns:
                    decoys_group = decoys_group.sort_values("similarity", ascending=False)

                hard_num = int(self.decoys_num * (1 - self.easy_decoys_ratio))
                easy_num = self.decoys_num - hard_num

                if len(decoys_group) >= hard_num:
                    sim_probs = torch.softmax(
                        torch.tensor(decoys_group["similarity"].values[:hard_num]), dim=0
                    )
                    selected_indices = torch.multinomial(sim_probs, hard_num, replacement=False)
                    selected_hard = decoys_group.iloc[selected_indices]
                else:
                    selected_hard = decoys_group.sample(hard_num, replace=True)

                easy_df_sample = self.easy_df.sample(easy_num, replace=True) if easy_num > 0 else pd.DataFrame()
                selected_decoys = pd.concat([selected_hard, easy_df_sample])

                for _, decoy_row in selected_decoys.iterrows():
                    try:
                        decoy_smiles = decoy_row["smiles"]
                        decoy_feat, _ = self._featurize_smiles(decoy_smiles)
                        decoy_ionization = decoy_row["ionization"]
                        decoy_adduct = common.ion2onehot_pos[decoy_ionization]
                        decoy_feats.append(decoy_feat)
                        decoy_adducts.append(decoy_adduct)
                        decoy_sims.append(decoy_row.get("similarity", 0.0))
                    except Exception as e:
                        logging.warning(f"Error processing decoy for {name}: {e}")
                        continue

            item["decoy_feats"] = decoy_feats
            item["decoy_adducts"] = decoy_adducts
            item["decoy_sims"] = decoy_sims

        return item

    @classmethod
    def get_collate_fn(cls):
        return CLIPDatasetUniMol.collate_fn

    @staticmethod
    def compute_batch_similarity(specs_list, bin_width=0.1):
        """Compute pairwise cosine similarity of binned spectra."""
        if not specs_list:
            return torch.empty(0, 0)

        max_mz = 0.0
        for spec in specs_list:
            if not spec:
                continue
            m_mz = max(p[0] for p in spec) if len(spec) > 0 else 0
            if m_mz > max_mz:
                max_mz = m_mz

        if max_mz == 0:
            bs = len(specs_list)
            return torch.eye(bs)

        n_bins = int(np.ceil(max_mz / bin_width)) + 1
        batch_size = len(specs_list)
        bin_matrix = np.zeros((batch_size, n_bins), dtype=np.float32)

        for i, spec in enumerate(specs_list):
            if not spec:
                continue
            spec_arr = np.array(spec, dtype=np.float32)
            mzs = spec_arr[:, 0]
            ints = spec_arr[:, 1]
            bin_indices = np.floor(mzs / bin_width).astype(int)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            np.add.at(bin_matrix[i], bin_indices, ints)

        norms = np.linalg.norm(bin_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        normalized_matrix = bin_matrix / norms
        sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        return torch.from_numpy(sim_matrix)

    @staticmethod
    def collate_fn(input_list):
        """Collate function for UniMol2-based CLIP dataset."""
        input_list = [j for j in input_list if j is not None]
        if not input_list:
            raise ValueError("All items in the batch are invalid and were skipped.")

        names = [j["name"] for j in input_list]
        specs = [j['spec'] for j in input_list]
        specs_scaled = [j['spec_scaled'] for j in input_list]
        max_spec_len = input_list[0]['max_spec_len']
        adducts = torch.FloatTensor([j["adduct"] for j in input_list])
        ces = torch.FloatTensor([j['ce'] for j in input_list])
        inchikeys = [j["inchikey"] for j in input_list]

        batch_size = len(input_list)
        same_molecule_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        if batch_size > 1:
            inchikey_list = [j["inchikey"] for j in input_list]
            for i in range(batch_size):
                for j in range(batch_size):
                    if inchikey_list[i] == inchikey_list[j]:
                        same_molecule_mask[i, j] = True

        spec_sim_matrix = CLIPDatasetUniMol.compute_batch_similarity(specs_scaled, bin_width=0.1)

        # Collate UniMol2 molecular features
        mol_feats = [j["mol_feat"] for j in input_list]
        batched_mol_feats = collate_unimol_features(mol_feats)

        # Pad spectra
        batched_specs, _ = pad_tuples_to_tensor(specs, max_len=max_spec_len)
        batched_specs_scaled, specs_mask = pad_tuples_to_tensor(specs_scaled, max_len=max_spec_len)

        # Fragment labels
        frag_labels = None
        frags_masks = None
        if input_list[0].get("frag_label", None) is not None:
            frag_label_list = [j["frag_label"] for j in input_list]
            frag_labels, frags_masks = pad_2d_batch_with_mask(frag_label_list)

        # Decoys
        decoys_feats_list = []
        decoys_adducts_list = []
        decoys_batch = []
        all_decoy_sims = []
        for batch_idx, item in enumerate(input_list):
            d_feats = item.get("decoy_feats", [])
            d_adducts = item.get("decoy_adducts", [])
            d_sims = item.get("decoy_sims", [])
            decoys_feats_list.extend(d_feats)
            decoys_adducts_list.extend(d_adducts)
            decoys_batch.extend([batch_idx] * len(d_feats))
            all_decoy_sims.extend(d_sims)

        batched_decoys_feats = None
        batched_decoys_adducts = None
        decoys_batch_tensor = None
        if len(decoys_feats_list) > 0:
            batched_decoys_feats = collate_unimol_features(decoys_feats_list)
            batched_decoys_adducts = torch.FloatTensor(decoys_adducts_list)
            decoys_batch_tensor = torch.LongTensor(decoys_batch)

        output = {
            "names": names,
            "inchikeys": inchikeys,
            "same_molecule_mask": same_molecule_mask,
            "mol_feats": batched_mol_feats,
            "decoys_feats": batched_decoys_feats,
            "decoys_adducts": batched_decoys_adducts,
            "decoys_batch": decoys_batch_tensor,
            "decoy_sims": torch.tensor(all_decoy_sims).float() if all_decoy_sims else None,
            "specs": batched_specs,
            "specs_scaled": batched_specs_scaled,
            "specs_mask": specs_mask,
            "adducts": adducts,
            "ces": ces,
            "frag_labels": frag_labels,
            "frag_masks": frags_masks,
            "spec_sim_matrix": spec_sim_matrix,
        }
        return output
