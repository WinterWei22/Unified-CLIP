import torch

import logging
from pathlib import Path
from typing import List
import json
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm
import dgl
from torch.utils.data.dataset import Dataset

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation

from .utils import dgl_to_pyg

from rdkit import Chem
from rdkit.Chem import Crippen

class TreeProcessor:
    """TreeProcessor.

    Hold key functionalities to read in a magma dag and proces it.

    """

    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
    ):
        """ """
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs

    def featurize_frag(
        self,
        frag: int,
        engine: fragmentation.FragmentEngine,
        add_random_walk: bool = False,
    ) -> False:
        """featurize_frag.

        Prev.  dgl_from_frag

        """

        num_atoms = engine.natoms
        atom_symbols = engine.atom_symbols
        # Need to find all kept atoms and all kept bonds between
        kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
        kept_bond_orders, kept_bonds = engine.get_present_edges(frag)

        # H count
        form = engine.formula_from_kept_inds(kept_atom_inds)    # counting each type of atoms

        # Need to re index the targets to match the new graph size
        num_kept = len(kept_atom_inds)
        new_inds = np.arange(num_kept)
        old_inds = kept_atom_inds

        old_to_new = np.zeros(num_atoms, dtype=int)
        old_to_new[old_inds] = new_inds

        # Keep new_to_old for autoregressive predictions
        new_to_old = np.zeros(num_kept, dtype=int)
        new_to_old[new_inds] = old_inds

        # Remap new bond inds
        new_bond_inds = np.empty((0, 2), dtype=int)
        if len(kept_bonds) > 0:
            new_bond_inds = old_to_new[np.vstack(kept_bonds)]

        if self.add_hs:
            h_adds = np.array(engine.atom_hs)[kept_atom_inds]
        else:
            h_adds = None

        # Make dgl graphs for new targets
        graph = self.dgl_featurize(
            np.array(atom_symbols)[kept_atom_inds],
            h_adds=h_adds,
            bond_inds=new_bond_inds,
            bond_types=np.array(kept_bond_orders),
        )

        if add_random_walk:
            self.add_pe_embed(graph)
        return {
            "graph": graph,
            "new_to_old": new_to_old,
            "old_to_new": old_to_new,
            "form": form,
        }

    def add_pe_embed(self, graph):
        pe_embeds = nn_utils.random_walk_pe(
            graph, k=self.pe_embed_k, eweight_name="e_ind"
        )
        graph.ndata["h"] = torch.cat((graph.ndata["h"], pe_embeds), -1).float()
        return graph

    def process_mol(self, inchi: str):
        engine = fragmentation.FragmentEngine(mol_str=inchi, mol_str_type="inchi")
        if self.root_encode in ["gnn","mabnet", "visnet", "egt2d", "egt3d", "egt"]:
            root_frag = engine.get_root_frag()
            root_graph_dict = self.featurize_frag(
                frag=root_frag,
                engine=engine,
            )
            root_repr = root_graph_dict["graph"]
            if self.pe_embed_k > 0:
                self.add_pe_embed(root_repr)
        elif self.root_encode == "fp":
            root_repr = common.get_morgan_fp_inchi(inchi)
        else:
            raise ValueError()
        return root_repr

    def dgl_featurize(
        self,
        atom_symbols: List[str],
        h_adds: np.ndarray,
        bond_inds: np.ndarray,
        bond_types: np.ndarray,
    ):
        """dgl_featurize.

        Args:
            atom_symbols (List[str]): node_types
            h_adds (np.ndarray): h_adds
            bond_inds (np.ndarray): bond_inds
            bond_types (np.ndarray)
        """
        node_types = [common.element_to_position[el] for el in atom_symbols]
        node_types = np.vstack(node_types)
        num_nodes = node_types.shape[0]

        src, dest = bond_inds[:, 0], bond_inds[:, 1]
        src_tens_, dest_tens_ = torch.from_numpy(src), torch.from_numpy(dest)
        bond_types = torch.from_numpy(bond_types)
        src_tens = torch.cat([src_tens_, dest_tens_])
        dest_tens = torch.cat([dest_tens_, src_tens_])
        bond_types = torch.cat([bond_types, bond_types])
        bond_featurizer = torch.eye(fragmentation.MAX_BONDS)

        bond_types_onehot = bond_featurizer[bond_types.long()]
        node_data = torch.from_numpy(node_types)

        # H data is defined, add that
        if h_adds is None:
            zero_vec = torch.zeros((node_data.shape[0], common.MAX_H))
            node_data = torch.hstack([node_data, zero_vec])
        else:
            h_featurizer = torch.eye(common.MAX_H)
            h_adds_vec = torch.from_numpy(h_adds)
            node_data = torch.hstack([node_data, h_featurizer[h_adds_vec]])

        g = dgl.graph(data=(src_tens, dest_tens), num_nodes=num_nodes)
        g.ndata["h"] = node_data.float()
        g.edata["e"] = bond_types_onehot.float()
        g.edata["e_ind"] = bond_types.long()
        return g

    def get_node_feats(self):
        return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H

    def create_masked_graph_for_nodes(self, graph: dgl.DGLGraph, mask_ratio: float = 0.15):
        """Create a masked graph for node (token) prediction."""
        masked_graph = copy.deepcopy(graph)
        num_nodes = masked_graph.num_nodes()
        mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_ratio)]
        
        # Original node features for labels (element one-hot part)
        original_feats = masked_graph.ndata["h"][:, :common.ELEMENT_DIM]
        node_labels = torch.argmax(original_feats, dim=-1)  # Assuming one-hot, get indices as labels
        
        # Mask: set to zero or a mask token (here, set element part to zero)
        masked_graph.ndata["h"][mask_indices, :common.ELEMENT_DIM] = 0
        
        # Mask for loss computation
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[mask_indices] = True
        
        return masked_graph, node_labels, mask

    def create_masked_graph_for_edges(self, graph: dgl.DGLGraph, mask_ratio: float = 0.15):
        """Create a masked graph for edge prediction."""
        masked_graph = copy.deepcopy(graph)
        num_edges = masked_graph.num_edges()
        if num_edges == 0:
            return masked_graph, None, None, None, None
        
        mask_indices = torch.randperm(num_edges)[:int(num_edges * mask_ratio)]
        
        # Original edge indices for labels
        original_edge_types = masked_graph.edata["e_ind"]
        edge_labels = original_edge_types.clone()
        
        # Mask edges: remove masked edges from the graph
        masked_graph = dgl.remove_edges(masked_graph, mask_indices)
        
        # For prediction, we might need to predict bond types for potential edges, but here simplify to predict types of masked edges
        # Assuming we predict the bond type of the masked edges, and use the original src/dst
        src, dst = graph.all_edges()
        masked_src = src[mask_indices]
        masked_dst = dst[mask_indices]
        masked_edge_labels = edge_labels[mask_indices]
        
        return masked_graph, masked_edge_labels, mask_indices, masked_src, masked_dst

class MolDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tree_processor: TreeProcessor,
        **kwargs,
    ):
        self.df = df
        self.tree_processor = tree_processor
        self.valid_indices = []

        # Pre-filter valid entries to determine dataset length
        for idx in tqdm(range(len(self.df)), desc="Filtering valid molecules"):
            row = self.df.iloc[idx]
            smiles = row["smiles"]
            try:
                # mol = Chem.MolFromSmiles(smiles)
                # if mol is None:
                #     continue
                # Just check if InChI can be generated; no heavy processing here
                # Chem.MolToInchi(mol)
                self.valid_indices.append(idx)
            except Exception:
                continue

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        smiles = row["smiles"]
        name = row.get("name", smiles)

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Convert to InChI
            inchi = Chem.MolToInchi(mol)

            # Generate molecular graph
            root_repr = self.tree_processor.process_mol(inchi)
            if not isinstance(root_repr, dgl.DGLGraph):
                raise ValueError(f"Non-graph representation for {name}")

            return {
                "name": name,
                "root_repr": root_repr,
            }

        except Exception as e:
            # Log the error and skip by returning None
            logging.warning(f"Error processing {name} at index {actual_idx}: {e}")
            return None

    def get_node_feats(self) -> int:
        """get_node_feats."""
        return self.tree_processor.get_node_feats()
    
    @classmethod
    def get_collate_fn(cls):
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(
        input_list,
    ):
        # Filter out None (skipped) items
        input_list = [j for j in input_list if j is not None]
        
        # If all items are skipped, raise an error or handle appropriately
        if not input_list:
            raise ValueError("All items in the batch are invalid and were skipped.")

        names = [j["name"] for j in input_list]
        root_reprs = [j["root_repr"] for j in input_list]
        
        if len(root_reprs) > 0:
            if isinstance(root_reprs[0], dgl.DGLGraph):
                batched_reprs = dgl.batch(root_reprs)
                # batched_reprs = dgl_to_pyg(batched_reprs)
            else:
                batched_reprs = torch.FloatTensor(np.stack(root_reprs))
        else:
            batched_reprs = None
        
        
        output = {
            "names": names,
            "root_reprs": dgl_to_pyg(batched_reprs),
        }
        return output


def _unroll_pad(input_list, key):
    if input_list[0][key] is not None:
        out = [torch.FloatTensor(i[key]) for i in input_list]
        out_padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
        return out_padded
    return None

def _stack_tensors(tensor_list, max_nodes=None):
    num = len(tensor_list)
    nodes_list = [t.shape[0] for t in tensor_list]
    num_nodes = max(nodes_list) if max_nodes is None else max_nodes

    result = torch.zeros((num, num_nodes, 3), dtype=tensor_list[0].dtype)

    for i, tensor in enumerate(tensor_list):
        nodes = tensor.shape[0]
        result[i, :nodes, :] = tensor
    
    return result

def _collate_root(input_list):
    root_reprs = [j["root_repr"] for j in input_list]
    if isinstance(root_reprs[0], dgl.DGLGraph):
        batched_reprs = dgl.batch(root_reprs)
    elif isinstance(root_reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    else:
        raise NotImplementedError()
    return batched_reprs

def _collate_root_poses(input_list):
    # root_reprs = [j["root_repr"] for j in input_list]
    # if isinstance(root_reprs[0], dgl.DGLGraph):
    #     batched_reprs = dgl.batch(root_reprs)
    # elif isinstance(root_reprs[0], np.ndarray):
    #     batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    # else:
    #     raise NotImplementedError()
    # return batched_reprs

    root_reprs = [j["root_repr"] for j in input_list]
    if hasattr(root_reprs[0], 'graph_data'):
        poses = [j['root_repr'].graph_data['pos'] for j in input_list]
        # poses_paded = _stack_tensors(poses)
        poses_cat = torch.cat(poses, dim=0)
    if isinstance(root_reprs[0], dgl.DGLGraph):
        batched_reprs = dgl.batch(root_reprs)
    elif isinstance(root_reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    else:
        raise NotImplementedError()
    batched_reprs.graph_data = {'pos': poses_cat}
    return batched_reprs


class CLIPDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tree_processor: TreeProcessor,
        magma_map: dict,
        max_spec_len: int = 128,
        **kwargs,
    ):
        self.df = df
        self.tree_processor = tree_processor
        self.magma_map = magma_map
        self.max_spec_len = max_spec_len
        # self.valid_indices = []

        # for idx in tqdm(range(len(self.df)), desc="Filtering valid molecules"):
        #     row = self.df.iloc[idx]
        #     smiles = row["smiles"]
        #     try:
        #         # mol = Chem.MolFromSmiles(smiles)
        #         # if mol is None:
        #         #     continue
        #         # Just check if InChI can be generated; no heavy processing here
        #         # Chem.MolToInchi(mol)
        #         self.valid_indices.append(idx)
        #     except Exception:
        #         continue
        self.spec_names = self.df["spec"].values
        self.name_to_dict = self.df.set_index("spec").to_dict(orient="index")
        for i in self.name_to_dict:
            self.name_to_dict[i]["magma_file"] = self.magma_map[i]
            
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)  
        self.name_to_adducts = {
            i: common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names   # (name, adduct_type)
        }
        
    def read_tree(self, magma_file):
        with open(magma_file, "r") as f:
            tree = json.load(f)
        return tree
    
    def __len__(self):
        return len(self.df)
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # actual_idx = self.valid_indices[idx]
        actual_idx = idx
        row = self.df.iloc[actual_idx]
        smiles = row["smiles"]
        name = row["spec"]
        filename = self.name_to_dict[name]["magma_file"]
        spec = self.read_tree(filename)['raw_spec']
        adduct = self.name_to_adducts[name]
        instrument = self.name_to_instrument[name]
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Convert to InChI
            inchi = Chem.MolToInchi(mol)

            # Generate molecular graph
            root_repr = self.tree_processor.process_mol(inchi)
            if not isinstance(root_repr, dgl.DGLGraph):
                raise ValueError(f"Non-graph representation for {name}")

            return {
                "name": name,
                "root_repr": root_repr,
                "spec": spec,
                "max_spec_len": self.max_spec_len,
                "adduct": adduct,
            }

        except Exception as e:
            # Log the error and skip by returning None
            logging.warning(f"Error processing {name} at index {actual_idx}: {e}")
            return None

    def get_node_feats(self) -> int:
        """get_node_feats."""
        return self.tree_processor.get_node_feats()
    
    @classmethod
    def get_collate_fn(cls):
        return CLIPDataset.collate_fn

    @staticmethod
    def collate_fn(
        input_list,
    ):
        # Filter out None (skipped) items
        input_list = [j for j in input_list if j is not None]
        
        # If all items are skipped, raise an error or handle appropriately
        if not input_list:
            raise ValueError("All items in the batch are invalid and were skipped.")

        names = [j["name"] for j in input_list]
        root_reprs = [j["root_repr"] for j in input_list]
        specs = [j['spec'] for j in input_list]
        max_spec_len = input_list[0]['max_spec_len']
        
        if len(root_reprs) > 0:
            if isinstance(root_reprs[0], dgl.DGLGraph):
                batched_reprs = dgl.batch(root_reprs)
                # batched_reprs = dgl_to_pyg(batched_reprs)
            else:
                batched_reprs = torch.FloatTensor(np.stack(root_reprs))
        else:
            batched_reprs = None
        
        batched_specs = pad_tuples_to_tensor(specs, max_len=max_spec_len)
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)
        
        output = {
            "names": names,
            "root_reprs": dgl_to_pyg(batched_reprs),
            "specs": batched_specs,
            "adducts": adducts,
        }
        return output


def _unroll_pad(input_list, key):
    if input_list[0][key] is not None:
        out = [torch.FloatTensor(i[key]) for i in input_list]
        out_padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
        return out_padded
    return None

def _stack_tensors(tensor_list, max_nodes=None):
    num = len(tensor_list)
    nodes_list = [t.shape[0] for t in tensor_list]
    num_nodes = max(nodes_list) if max_nodes is None else max_nodes

    result = torch.zeros((num, num_nodes, 3), dtype=tensor_list[0].dtype)

    for i, tensor in enumerate(tensor_list):
        nodes = tensor.shape[0]
        result[i, :nodes, :] = tensor
    
    return result

import torch

def pad_tuples_to_tensor(data, max_len=128, dtype=torch.float32, device=None):
    """
    data: List[List[Tuple[Number, Number]]], length n
    max_len: target length (default 128)
    dtype: output tensor dtype
    device: output tensor device (optional)

    Returns: a tensor of shape (n, max_len, 2)
    """
    n = len(data)
    out = torch.zeros((n, max_len, 2), dtype=dtype, device=device)

    for i, pairs in enumerate(data):
        # Normalize each element to a 2D list/tensor
        # Truncate or pad
        length = min(len(pairs), max_len)
        if length > 0:
            # Copy the first `length` pairs
            # Convert to a tensor of shape (length, 2)
            t = torch.tensor(pairs[:length], dtype=dtype, device=device)
            if t.ndim != 2 or t.shape[1] != 2:
                raise ValueError(f"Sample {i} is not composed of 2-tuples, got shape {t.shape}")
            out[i, :length, :] = t

    return out

def _collate_root(input_list):
    root_reprs = [j["root_repr"] for j in input_list]
    if isinstance(root_reprs[0], dgl.DGLGraph):
        batched_reprs = dgl.batch(root_reprs)
    elif isinstance(root_reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    else:
        raise NotImplementedError()
    return batched_reprs

def _collate_root_poses(input_list):
    # root_reprs = [j["root_repr"] for j in input_list]
    # if isinstance(root_reprs[0], dgl.DGLGraph):
    #     batched_reprs = dgl.batch(root_reprs)
    # elif isinstance(root_reprs[0], np.ndarray):
    #     batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    # else:
    #     raise NotImplementedError()
    # return batched_reprs

    root_reprs = [j["root_repr"] for j in input_list]
    if hasattr(root_reprs[0], 'graph_data'):
        poses = [j['root_repr'].graph_data['pos'] for j in input_list]
        # poses_paded = _stack_tensors(poses)
        poses_cat = torch.cat(poses, dim=0)
    if isinstance(root_reprs[0], dgl.DGLGraph):
        batched_reprs = dgl.batch(root_reprs)
    elif isinstance(root_reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(root_reprs)).float()
    else:
        raise NotImplementedError()
    batched_reprs.graph_data = {'pos': poses_cat}
    return batched_reprs