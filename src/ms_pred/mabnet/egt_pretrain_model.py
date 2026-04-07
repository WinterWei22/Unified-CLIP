"""DAG intensity prediction model."""
import numpy as np
import copy
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch_scatter as ts
import dgl.nn as dgl_nn
from ms_pred.mabnet import dag_pyg_data

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as run_magma


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GlobalAttention
from torch_geometric.data import Data
import pytorch_lightning as pl
import torch_scatter as ts
import copy

from ms_pred.mabnet.mabnet import mabnet
from ms_pred.mabnet.visnet import visnet_block
from ms_pred.mabnet.egt import egt_model
from torch_scatter import scatter

class PyGGNN(nn.Module):
    def __init__(self, hidden_size, num_layers, node_feats, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        in_feats = node_feats
        for _ in range(num_layers):
            self.layers.append(
                GCNConv(
                    in_channels=in_feats,
                    out_channels=hidden_size,
                )
            )
            in_feats = hidden_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.relu(h)
            h = self.dropout(h)
        return h

def aggregate_edge_features(data: Data):
    h = data.h.float()
    edge_index = data.edge_index
    edge_attr = data.e.float()
    num_nodes = h.shape[0]
    
    edge_to_node = torch.zeros(num_nodes, edge_attr.shape[1], device=h.device)
    edge_counts = torch.zeros(num_nodes, device=h.device)
    target_nodes = edge_index[1]
    edge_to_node.index_add_(0, target_nodes, edge_attr)
    edge_counts.index_add_(0, target_nodes, torch.ones_like(target_nodes, dtype=torch.float))
    edge_counts = edge_counts.clamp(min=1)  # 避免除零
    edge_to_node = edge_to_node / edge_counts.unsqueeze(-1)
    
    h = torch.cat([h, edge_to_node], dim=-1)
    return h

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch  # Assuming usage of PyG

# Assuming imports from previous code
# import egt_model, mabnet, visnet_block, nn_utils
# import ms_pred.common as common
# import ms_pred.magma.fragmentation as fragmentation

class Pretrain_EGT(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        root_encode: str = "gnn",
        mabnet_layers: int = 2,
        mabnet_heads: int = 4,
        many_body: bool = False,
        edge_update: bool = False,
        frag_layers: int = 8,
        frag_heads: int = 4,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_decay_rate: float = 0.96,
        warmup: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.root_encode = root_encode
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup

        if self.root_encode == "gnn":
            self.root_module = egt_model.EdgeEnhancedGraphTransformer2D(input_node_dim=52,
                                                                hidden_dim=hidden_size,
                                                                num_heads=frag_heads,
                                                                num_layers=frag_layers,
                                                                # edge_update=edge_update,
                                                                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                num_layers=1,
                use_residuals=True,
            )
        elif self.root_encode == "mabnet":
            self.root_module = mabnet.PhyMabNet(hidden_channels=hidden_size,
                                                num_heads=mabnet_heads,
                                                num_layers=mabnet_layers,
                                                many_body=many_body)
        elif self.root_encode == "visnet":
            self.root_module = visnet_block.ViSNetBlock(hidden_channels=hidden_size,
                                                        num_heads=mabnet_heads,
                                                        num_layers=mabnet_layers,)
        elif self.root_encode == "egt3d" or  self.root_encode == "egt":
            self.root_module = egt_model.EdgeEnhancedGraphTransformer3D(
                                                                    input_node_dim=26,
                                                                    hidden_dim=hidden_size,
                                                                    num_heads=mabnet_heads,
                                                                    num_layers=mabnet_layers,
                                                                    edge_update=edge_update,
                                                                    )
        elif self.root_encode == "egt2d":
            self.root_module = egt_model.EdgeEnhancedGraphTransformer2D(
                                                            input_node_dim=26,
                                                            hidden_dim=hidden_size,
                                                            num_heads=mabnet_heads,
                                                            num_layers=mabnet_layers,
                                                            edge_update=edge_update,
                                                            )
        else:
            raise ValueError("Invalid root_encode")

        # Pretraining heads
        self.logd_head = nn.Linear(self.hidden_size, 1)
        self.atom_head = nn.Linear(self.hidden_size, common.ELEMENT_DIM)
        self.edge_head = nn.Linear(2 * self.hidden_size, fragmentation.MAX_BONDS)

    def get_ptr(self, graph):
        device = self.device
        if hasattr(graph, 'ptr'):
            return graph.ptr.to(device)
        elif hasattr(graph, 'batch'):
            num_graphs = graph.batch.max().item() + 1
            ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
            for i in range(num_graphs):
                ptr[i + 1] = (graph.batch == i).sum()
            ptr = torch.cumsum(ptr, dim=0)
            return ptr
        else:
            num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
            return torch.tensor([0, num_nodes], dtype=torch.long, device=device)

    def forward(
        self,
        root_reprs: Batch,
        masked_node_graphs: Batch,
        masked_edge_graphs: Batch,
        masked_srcs: torch.Tensor,
        masked_dsts: torch.Tensor,
    ):
        # logD prediction using original graph
        _, graph_emb = self.root_module(root_reprs)
        logd_pred = self.logd_head(graph_emb)

        # Masked node prediction
        node_emb_n, _ = self.root_module(masked_node_graphs)
        atom_preds = self.atom_head(node_emb_n)

        # Masked edge prediction
        node_emb_e, _ = self.root_module(masked_edge_graphs)
        bs = root_reprs.batch.max().item() + 1 if hasattr(root_reprs, 'batch') else 1
        device = node_emb_e.device
        max_masked_edges = masked_srcs.size(1)
        ptr = self.get_ptr(masked_edge_graphs)
        offsets = ptr[:-1].view(-1, 1).repeat(1, max_masked_edges).to(device)
        valid_edges = (masked_srcs >= 0)
        src_global = (masked_srcs + offsets).long()
        src_global[~valid_edges] = 0  # Dummy index
        emb_src = node_emb_e[src_global]
        dst_global = (masked_dsts + offsets).long()
        dst_global[~valid_edges] = 0
        emb_dst = node_emb_e[dst_global]
        concat_emb = torch.cat([emb_src, emb_dst], dim=-1)
        edge_preds = self.edge_head(concat_emb)

        return {
            "logd_pred": logd_pred,
            "atom_preds": atom_preds,
            "edge_preds": edge_preds,
        }
    
    def _common_step(self, batch, name="train"):
        pred_obj = self.forward(
            batch["root_reprs"],
            batch["masked_node_graphs"],
            batch["masked_edge_graphs"],
            batch["masked_src"],
            batch["masked_dst"],
        )

        # logD loss
        loss_logd = F.mse_loss(pred_obj["logd_pred"].squeeze(), batch["logds"])

        # Edge loss
        edge_preds = pred_obj["edge_preds"]  # [bs, max_masked, classes]
        edge_labels = batch["masked_edge_labels"]  # [bs, max_masked]
        valid_edge = (edge_labels != -1)
        if valid_edge.any():
            loss_edge = F.cross_entropy(edge_preds[valid_edge], edge_labels[valid_edge])
        else:
            loss_edge = torch.tensor(0.0, device=self.device)

        # Node loss
        atom_preds = pred_obj["atom_preds"]  # [total_nodes, classes]
        bs = batch["root_reprs"].batch.max().item() + 1 if hasattr(batch["root_reprs"], 'batch') else 1
        node_masks_p = batch["node_masks"]  # [bs, max_nodes]
        node_labels_p = batch["node_labels"]  # [bs, max_nodes]
        ptr = self.get_ptr(batch["masked_node_graphs"])
        masked_indices = []
        masked_labels_list = []
        for i in range(bs):
            mask_i = node_masks_p[i]
            labels_i = node_labels_p[i]
            valid_len = (labels_i != -1).sum().item()
            mask_i = mask_i[:valid_len]
            masked_pos_i = torch.nonzero(mask_i).squeeze()
            if len(masked_pos_i.shape) == 0:
                continue
            offset_i = ptr[i]
            global_pos_i = masked_pos_i + offset_i
            masked_indices.append(global_pos_i)
            masked_lab_i = labels_i[masked_pos_i]
            masked_labels_list.append(masked_lab_i)
        if masked_indices:
            masked_indices = torch.cat(masked_indices).to(self.device)
            masked_labels = torch.cat(masked_labels_list).to(self.device)
            masked_atom_preds = atom_preds[masked_indices]
            loss_node = F.cross_entropy(masked_atom_preds, masked_labels)
        else:
            loss_node = torch.tensor(0.0, device=self.device)

        # Total loss (sum or weighted sum)
        loss = loss_logd + loss_node + loss_edge

        # Logging (optional)
        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_logd_loss", loss_logd)
        self.log(f"{name}_node_loss", loss_node)
        self.log(f"{name}_edge_loss", loss_edge)

        return loss

    def training_step(self, batch, batch_idx):
        """training_step."""
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        """validation_step."""
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        """test_step."""
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        """configure_optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
            "gradient_clip_val": 1.0,  # 设置梯度裁剪的阈值
            "gradient_clip_algorithm": "value",  # 可选："norm"（默认）或 "value"
        }
        return ret