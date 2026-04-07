""" dag_data.py

Fragment dataset to build out model class

"""
import logging
from pathlib import Path
from typing import List
import json
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
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

class PretrainDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tree_processor: TreeProcessor,
        mask_node_ratio: float = 0.15,
        mask_edge_ratio: float = 0.15,
        **kwargs,
    ):
        self.df = df
        self.tree_processor = tree_processor
        self.mask_node_ratio = mask_node_ratio
        self.mask_edge_ratio = mask_edge_ratio
        self.entries = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            smiles = row["canonical_smiles"]
            name = row.get("name", smiles)

            try:
                # 尝试解析 SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logging.warning(f"Invalid SMILES: {smiles}")
                    continue

                # 转换为 InChI
                inchi = Chem.MolToInchi(mol)

                # 生成分子图
                root_repr = self.tree_processor.process_mol(inchi)
                if not isinstance(root_repr, dgl.DGLGraph):
                    logging.warning(f"Skipping non-graph representation for {name}")
                    continue

                # logD（这里直接取 cx_logp）
                logd = row["cx_logp"]

                # 节点掩码任务
                masked_node_graph, node_labels, node_mask = self.tree_processor.create_masked_graph_for_nodes(
                    root_repr, self.mask_node_ratio
                )

                # 边掩码任务
                masked_edge_graph, masked_edge_labels, _, masked_src, masked_dst = self.tree_processor.create_masked_graph_for_edges(
                    root_repr, self.mask_edge_ratio
                )

                self.entries.append({
                    "name": name,
                    "root_repr": root_repr,
                    "logd": logd,
                    "masked_node_graph": masked_node_graph,
                    "node_labels": node_labels,
                    "node_mask": node_mask,
                    "masked_edge_graph": masked_edge_graph,
                    "masked_edge_labels": masked_edge_labels,
                    "masked_src": masked_src,
                    "masked_dst": masked_dst,
                })

            except Exception as e:
                logging.warning(f"Skipping {name} due to error: {e}")
                continue

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def get_node_feats(self) -> int:
        """get_node_feats."""
        return self.tree_processor.get_node_feats()
    
    @classmethod
    def get_collate_fn(cls):
        return PretrainDataset.collate_fn

    @staticmethod
    def collate_fn(
        input_list,
    ):
        names = [j["name"] for j in input_list]
        root_reprs = [j["root_repr"] for j in input_list]
        logds = torch.FloatTensor([j["logd"] for j in input_list])
        
        if len(root_reprs) > 0:
            if isinstance(root_reprs[0], dgl.DGLGraph):
                batched_reprs = dgl.batch(root_reprs)
                # batched_reprs = dgl_to_pyg(batched_reprs)
            else:
                batched_reprs = torch.FloatTensor(np.stack(root_reprs))
        else:
            batched_reprs = None
        
        # Collate masked node graphs
        masked_node_graphs = [j["masked_node_graph"] for j in input_list]
        batched_masked_node_graphs = dgl.batch(masked_node_graphs) if masked_node_graphs else None
        
        node_labels_list = [j["node_labels"] for j in input_list]
        node_labels_padded = torch.nn.utils.rnn.pad_sequence(node_labels_list, batch_first=True, padding_value=-1)  # -1 as ignore index
        
        node_masks_list = [j["node_mask"] for j in input_list]
        node_masks_padded = torch.nn.utils.rnn.pad_sequence(node_masks_list, batch_first=True, padding_value=False)
        
        # Collate masked edge graphs
        masked_edge_graphs = [j["masked_edge_graph"] for j in input_list]
        batched_masked_edge_graphs = dgl.batch(masked_edge_graphs) if masked_edge_graphs else None
        
        masked_edge_labels_list = [j["masked_edge_labels"] for j in input_list if j["masked_edge_labels"] is not None]
        if masked_edge_labels_list:
            masked_edge_labels_padded = torch.nn.utils.rnn.pad_sequence(masked_edge_labels_list, batch_first=True, padding_value=-1)
        else:
            masked_edge_labels_padded = None
        
        masked_src_list = [j["masked_src"] for j in input_list if j["masked_src"] is not None]
        if masked_src_list:
            masked_src_padded = torch.nn.utils.rnn.pad_sequence(masked_src_list, batch_first=True, padding_value=-1)
        else:
            masked_src_padded = None
        
        masked_dst_list = [j["masked_dst"] for j in input_list if j["masked_dst"] is not None]
        if masked_dst_list:
            masked_dst_padded = torch.nn.utils.rnn.pad_sequence(masked_dst_list, batch_first=True, padding_value=-1)
        else:
            masked_dst_padded = None
        
        output = {
            "names": names,
            "root_reprs": dgl_to_pyg(batched_reprs),
            "logds": logds,
            "masked_node_graphs": dgl_to_pyg(batched_masked_node_graphs),
            "node_labels": node_labels_padded,
            "node_masks": node_masks_padded,
            "masked_edge_graphs": dgl_to_pyg(batched_masked_edge_graphs),
            "masked_edge_labels": masked_edge_labels_padded,
            "masked_src": masked_src_padded,
            "masked_dst": masked_dst_padded,
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