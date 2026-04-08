"""
UniMol2 Molecular Encoder wrapper for CLIP training.

Wraps UniMolV2Model to provide the same interface as Mol_Encoder:
  forward(batch_dict) -> (atoms_emb, mol_emb)
    atoms_emb: (total_atoms, hidden_dim) - atom-level embeddings
    mol_emb: (batch_size, hidden_dim) - molecule-level CLS embeddings
"""

import os
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
# Add Uni-Mol/unimol_tools to path (relative to repo root)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(_repo_root, 'Uni-Mol', 'unimol_tools'))

from unimol_tools.models.unimolv2 import UniMolV2Model
from unimol_tools.data.conformer import mol2unimolv2, inner_smi2coords, UniMolV2Feature
from unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords


# Model size -> embedding dimension mapping
UNIMOL2_EMBED_DIMS = {
    '84m': 768,
    '164m': 768,
    '310m': 1024,
    '570m': 1536,
    '1.1b': 1536,
}


def smiles_to_unimol_features(smiles, seed=42, max_atoms=128):
    """Convert a SMILES string to UniMol2 feature dict.

    Returns:
        feat: dict with keys: atom_feat, atom_mask, edge_feat, shortest_path,
              degree, pair_type, attn_bias, src_tokens, src_coord
        n_atoms: number of heavy atoms (excluding H)
    """
    mol = inner_smi2coords(smiles, seed=seed, mode='fast', remove_hs=True, return_mol=True)
    feat = mol2unimolv2(mol, max_atoms=max_atoms, remove_hs=True)
    n_atoms = int(feat['atom_mask'].sum())
    return feat, n_atoms


def collate_unimol_features(feat_list):
    """Collate a list of UniMol2 feature dicts into a batched dict.

    Args:
        feat_list: list of dicts from smiles_to_unimol_features

    Returns:
        batch: dict of batched tensors ready for UniMolV2Model.forward()
    """
    padding_idx = 0
    batch = {}

    for k in feat_list[0].keys():
        tensors = [torch.tensor(s[k]) for s in feat_list]
        if k == 'atom_feat':
            v = pad_coords(tensors, pad_idx=padding_idx, dim=8)
        elif k == 'atom_mask':
            v = pad_1d_tokens(tensors, pad_idx=padding_idx)
        elif k == 'edge_feat':
            v = pad_2d(tensors, pad_idx=padding_idx, dim=3)
        elif k == 'shortest_path':
            v = pad_2d(tensors, pad_idx=padding_idx)
        elif k == 'degree':
            v = pad_1d_tokens(tensors, pad_idx=padding_idx)
        elif k == 'pair_type':
            v = pad_2d(tensors, pad_idx=padding_idx, dim=2)
        elif k == 'attn_bias':
            v = pad_2d(tensors, pad_idx=padding_idx)
        elif k == 'src_tokens':
            v = pad_1d_tokens(tensors, pad_idx=padding_idx)
        elif k == 'src_coord':
            v = pad_coords(tensors, pad_idx=padding_idx)
        else:
            continue
        batch[k] = v

    return batch


class UniMolEncoder(nn.Module):
    """UniMol2 molecular encoder that outputs (atoms_emb, mol_emb).

    Compatible with the CLIP pipeline as a drop-in replacement for Mol_Encoder.
    """

    def __init__(self, model_size='84m', frozen=False, frozen_epochs=0):
        super().__init__()
        self.model_size = model_size
        self.embed_dim = UNIMOL2_EMBED_DIMS[model_size]
        self.frozen = frozen
        self.frozen_epochs = frozen_epochs

        # Initialize UniMolV2 (loads pretrained weights automatically)
        self.unimol = UniMolV2Model(output_dim=2, model_size=model_size)
        # Remove classification head - we don't need it
        del self.unimol.classification_head

        if frozen:
            for param in self.unimol.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters for fine-tuning."""
        self.frozen = False
        for param in self.unimol.parameters():
            param.requires_grad = True

    def forward(self, batch_dict):
        """Forward pass through UniMol2 encoder.

        Args:
            batch_dict: dict with keys from collate_unimol_features,
                        all tensors should be on the correct device

        Returns:
            atoms_emb: (total_atoms_in_batch, embed_dim) - flat atom embeddings
            mol_emb: (batch_size, embed_dim) - CLS token embeddings
            batch_indices: (total_atoms_in_batch,) - batch index for each atom
        """
        # Forward through the UniMol2 backbone (without classification head)
        src_tokens = batch_dict['src_tokens']
        atom_mask = batch_dict['atom_mask']
        src_coord = batch_dict['src_coord']

        n_mol, n_atom = batch_dict['atom_feat'].shape[:2]

        # Get token embeddings
        token_feat = self.unimol.embed_tokens(src_tokens)
        x = self.unimol.atom_feature(
            {'atom_feat': batch_dict['atom_feat'], 'degree': batch_dict['degree']},
            token_feat
        )

        dtype = self.unimol.dtype
        x = x.type(dtype)

        attn_mask = batch_dict['attn_bias'].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.unimol.args.encoder_attention_heads, 1, 1
        )
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.unimol.args.pair_embed_dim)
        attn_bias = self.unimol.edge_feature(
            {'shortest_path': batch_dict['shortest_path'], 'edge_feat': batch_dict['edge_feat']},
            attn_bias
        )
        attn_mask = attn_mask.type(dtype)

        atom_mask_cls = torch.cat(
            [torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype), atom_mask],
            dim=1,
        ).type(dtype)

        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        pos = src_coord
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        attn_bias_3d = self.unimol.se3_invariant_kernel(dist.detach(), batch_dict['pair_type'])
        new_attn_bias = attn_bias.clone()
        new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
        new_attn_bias = new_attn_bias.type(dtype)

        x, pair = self.unimol.encoder(
            x, new_attn_bias,
            atom_mask=atom_mask_cls,
            pair_mask=pair_mask,
            attn_mask=attn_mask,
        )

        # Extract CLS and atom representations
        cls_repr = x[:, 0, :]  # (batch_size, embed_dim)

        # Extract atom representations (skip CLS token at position 0)
        # atom_mask: (batch_size, n_atom), 1 for real atoms, 0 for padding
        atoms_list = []
        batch_indices_list = []

        for i in range(n_mol):
            mask_i = atom_mask[i].bool()  # (n_atom,)
            n_real = mask_i.sum().item()
            atom_repr_i = x[i, 1:n_atom+1, :][mask_i]  # (n_real, embed_dim)
            atoms_list.append(atom_repr_i)
            batch_indices_list.append(torch.full((n_real,), i, device=x.device, dtype=torch.long))

        atoms_emb = torch.cat(atoms_list, dim=0)  # (total_atoms, embed_dim)
        batch_indices = torch.cat(batch_indices_list, dim=0)  # (total_atoms,)

        return atoms_emb, cls_repr, batch_indices
