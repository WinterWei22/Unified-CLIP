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

class IntenGNN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        gnn_layers: int = 2,
        mlp_layers: int = 0,
        set_layers: int = 2,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        dropout: float = 0,
        mpnn_type: str = "GCN",
        pool_op: str = "avg",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        pe_embed_k: int = 0,
        max_broken: int = run_magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        frag_set_layers: int = 0,
        loss_fn: str = "cosine",
        root_encode: str = "gnn",
        inject_early: bool = False,
        warmup: int = 1000,
        embed_adduct: bool = False,
        binned_targs: bool = True,
        encode_forms: bool = False,
        add_hs: bool = False,
        mabnet_layers: int = 2,
        mabnet_heads: int = 4,
        many_body: bool = False,
        **kwargs,
    ):
        """初始化 IntenGNN，使用 PyG 和 GCNConv 实现的图神经网络。

        参数:
            hidden_size (int): 隐藏层维度。
            gnn_layers (int, optional): GNN 层数。默认为 2。
            mlp_layers (int, optional): MLP 层数。默认为 0。
            set_layers (int, optional): 集合变换层数。默认为 2。
            learning_rate (float, optional): 学习率。默认为 7e-4。
            lr_decay_rate (float, optional): 学习率衰减率。默认为 1.0。
            weight_decay (float, optional): 权重衰减。默认为 0。
            dropout (float, optional): Dropout 比率。默认为 0。
            mpnn_type (str, optional): GNN 类型（如 GCN）。默认为 "GCN"。
            pool_op (str, optional): 池化操作（avg 或 attn）。默认为 "avg"。
            node_feats (int, optional): 节点特征维度。默认为 38。
            pe_embed_k (int, optional): 位置编码维度。默认为 0。
            max_broken (int, optional): 最大断裂键数。默认为 3。
            frag_set_layers (int, optional): 碎片集合变换层数。默认为 0。
            loss_fn (str, optional): 损失函数（mse 或 cosine）。默认为 "cosine"。
            root_encode (str, optional): 根编码方式（gnn 或 fp）。默认为 "gnn"。
            inject_early (bool, optional): 是否提前注入根嵌入。默认为 False。
            warmup (int, optional): 预热步数。默认为 1000。
            embed_adduct (bool, optional): 是否嵌入加合物。默认为 False。
            binned_targs (bool, optional): 是否使用分桶目标。默认为 True。
            encode_forms (bool, optional): 是否编码分子形式。默认为 False。
            add_hs (bool, optional): 是否添加氢原子。默认为 False。

        异常:
            ValueError: 如果 root_encode 或 binned_targs 无效。
            NotImplementedError: 如果 loss_fn 或 pool_op 不支持。
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.pool_op = pool_op
        self.inject_early = inject_early
        self.embed_adduct = embed_adduct
        self.binned_targs = binned_targs
        self.encode_forms = encode_forms
        self.add_hs = add_hs

        self.tree_processor = dag_pyg_data.TreeProcessor(
            root_encode=root_encode, pe_embed_k=pe_embed_k, add_hs=self.add_hs
        )
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = 18
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim * 2

        self.gnn_layers = gnn_layers
        self.set_layers = set_layers
        self.frag_set_layers = frag_set_layers
        self.mpnn_type = mpnn_type
        self.mlp_layers = mlp_layers
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = torch.nn.Parameter(torch.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        edge_feats = fragmentation.MAX_BONDS
        orig_node_feats = node_feats + edge_feats  # 节点特征 + 聚合边特征
        if self.inject_early:
            node_feats = orig_node_feats + self.hidden_size
        else:
            node_feats = orig_node_feats

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        self.gnn = PyGGNN(
            hidden_size=self.hidden_size,
            num_layers=self.gnn_layers,
            node_feats=node_feats + adduct_shift,
            dropout=self.dropout,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn
            if self.inject_early:
                self.root_module = PyGGNN(
                    hidden_size=self.hidden_size,
                    num_layers=self.gnn_layers,
                    node_feats=orig_node_feats + adduct_shift,
                    dropout=self.dropout,
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
        else:
            raise ValueError("Invalid root_encode")

        self.intermediate_out = nn_utils.MLPBlocks(
            input_size=self.hidden_size * 3 + self.max_broken + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.mlp_layers,
            use_residuals=True,
        )

        trans_layer = nn_utils.TransformerEncoderLayer(
            self.hidden_size,
            nhead=8,
            batch_first=True,
            norm_first=False,
            dim_feedforward=self.hidden_size * 4,
        )
        self.trans_layers = nn_utils.get_clones(trans_layer, self.frag_set_layers)

        self.loss_fn_name = loss_fn
        if loss_fn == "mse":
            raise NotImplementedError()
        elif loss_fn == "cosine":
            self.loss_fn = self.cos_loss
            self.cos_fn = nn.CosineSimilarity()
            self.output_activations = [nn.Sigmoid()]
        else:
            raise NotImplementedError()

        self.num_outputs = len(self.output_activations)
        self.output_size = run_magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"] * 2 + 1
        self.output_map = nn.Linear(
            self.hidden_size, self.num_outputs * self.output_size
        )
        self.isomer_attn_out = copy.deepcopy(self.output_map)

        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False

        if self.pool_op == "avg":
            self.pool = global_mean_pool
        elif self.pool_op == "attn":
            self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()

    def cos_loss(self, pred, targ):
        """余弦损失函数。

        参数:
            pred: 预测值。
            targ: 目标值。

        异常:
            ValueError: 如果 binned_targs 无效。

        返回:
            dict: 包含损失值的字典。
        """
        if not self.binned_targs:
            raise ValueError("Cosine loss requires binned targets")
        loss = 1 - self.cos_fn(pred, targ)
        loss = loss.mean()
        return {"loss": loss}

    def predict(
        self,
        graphs: Data,
        root_reprs: Data,
        ind_maps,
        num_frags,
        max_breaks,
        adducts,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
        binned_out: bool = False,
    ) -> dict:
        """预测强度谱。

        参数:
            graphs (Data): 碎片的 PyG Data 对象。
            root_reprs (Data): 根分子的 PyG Data 对象。
            ind_maps: 碎片的映射索引。
            num_frags: 每个样本的碎片数，(bs,)。
            max_breaks: 最大断裂键数，(bs, max_frags)。
            adducts: 加合物索引，(bs,)。
            max_add_hs: 最大添加氢数，(bs, max_frags)，可选。
            max_remove_hs: 最大移除氢数，(bs, max_frags)，可选。
            masses: 碎片质量，(bs, max_frags, 13)，可选。
            root_forms: 根分子形式，(bs, 18)，可选。
            frag_forms: 碎片分子形式，(bs, max_frags, 18)，可选。
            binned_out (bool, optional): 是否返回分桶输出。默认为 False。

        异常:
            NotImplementedError: 如果 loss_fn 不支持。

        返回:
            dict: 包含预测谱的字典，键为 "spec"。
        """
        out = self.forward(
            graphs=graphs,
            root_repr=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            broken=max_breaks,
            adducts=adducts,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
        )

        if self.loss_fn_name not in ["mse", "cosine"]:
            raise NotImplementedError()

        output = out["output"][:, :, 0, :]  # (bs, max_frags, output_size)
        output_binned = out["output_binned"][:, 0, :]  # (bs, 15000)
        out_preds_binned = [i.cpu().detach().numpy() for i in output_binned]
        out_preds = [
            pred[:num_frag, :].cpu().detach().numpy()
            for pred, num_frag in zip(output, num_frags)
        ]

        if binned_out:
            out_dict = {"spec": out_preds_binned}
        else:
            out_dict = {"spec": out_preds}
        return out_dict

    def forward(
        self,
        graphs: Data,
        root_repr: Data,
        ind_maps,
        num_frags,
        broken,
        adducts,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
        new_to_old=None,
    ):
        """IntenGNN 的前向传播，使用 PyG 和 GCNConv 实现。

        参数:
            graphs (Data): 碎片的 PyG Data 对象。
            root_repr (Data): 根分子的 PyG Data 对象。
            ind_maps: 碎片的映射索引。
            num_frags: 每个样本的碎片数，(bs,)。
            broken: 断裂键数量，(bs, max_frags)。
            adducts: 加合物索引，(bs,)。
            max_add_hs: 最大添加氢数，(bs, max_frags)，可选。
            max_remove_hs: 最大移除氢数，(bs, max_frags)，可选。
            masses: 碎片质量，(bs, max_frags, 13)，可选。
            root_forms: 根分子形式，(bs, 18)，可选。
            frag_forms: 碎片分子形式，(bs, max_frags, 18)，可选。
            new_to_old: 未使用，保留兼容性。

        返回:
            dict: 包含 output_binned 和 output 的字典。
        """
        device = num_frags.device
        embed_adducts = self.adduct_embedder[adducts.long()]  # (bs, 10)

        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            root_h = aggregate_edge_features(root_repr)  # (num_nodes, 38+4)
            if self.embed_adduct:
                root_batch_sizes = torch.bincount(root_repr.batch) if root_repr.batch is not None else torch.tensor([root_repr.num_nodes])
                embed_adducts_expand = embed_adducts.repeat_interleave(root_batch_sizes, 0)  # (num_nodes, 10)
                root_h = torch.cat([root_h, embed_adducts_expand], -1)  # (num_nodes, 42+10)
            root_embeddings = self.root_module(x=root_h, edge_index=root_repr.edge_index)  # (num_nodes, hidden_size)
            root_embeddings = self.pool(root_embeddings, root_repr.batch)  # (bs, hidden_size)
        elif self.root_encode == 'mabnet' or self.root_encode == 'visnet':
            root_batch_sizes = torch.bincount(root_repr.batch) if root_repr.batch is not None else torch.tensor([root_repr.num_nodes])
            embed_adducts_expand = embed_adducts.repeat_interleave(root_batch_sizes, 0)  # (num_nodes, 10)
            root_h = torch.cat([root_repr.h, embed_adducts_expand], -1)
            root_repr.h = root_h
            x, vec = self.root_module(root_repr)    # x: [nodes, dim], vec: [nodes, 8, 256]
            root_embeddings = scatter(x, root_repr.batch, dim=0, reduce="mean")
        else:
            pass

        ext_root = root_embeddings[ind_maps]  # (n_frags, hidden_size)
        graph_batch_sizes = torch.bincount(graphs.batch) if graphs.batch is not None else torch.tensor([graphs.num_nodes])
        ext_root_atoms = torch.repeat_interleave(ext_root, graph_batch_sizes, dim=0)  # (f_nodes, hidden_size)

        graphs_h = aggregate_edge_features(graphs)  # (f_nodes, 38+4)
        concat_list = [graphs_h]
        if self.inject_early:
            concat_list.append(ext_root_atoms)
        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]
            adducts_exp = torch.repeat_interleave(adducts_mapped, graph_batch_sizes, dim=0)
            concat_list.append(adducts_exp)  # (f_nodes, 42+10)

        graphs_h = torch.cat(concat_list, -1).float()  # (f_nodes, 42+10)
        frag_embeddings = self.gnn(x=graphs_h, edge_index=graphs.edge_index)  # (f_nodes, hidden_size)
        avg_frags = self.pool(frag_embeddings, graphs.batch)  # (n_frags, hidden_size)

        broken_arange = torch.arange(broken.shape[-1], device=device)  # (max_frags,)
        broken_mask = broken_arange[None, :] < num_frags[:, None]  # (bs, max_frags)
        broken = torch.clamp(broken[broken_mask], max=self.broken_clamp)  # (n_frags,)
        broken_onehots = self.broken_onehot[broken.long()]  # (n_frags, max_broken+1)

        mlp_cat_list = [ext_root, ext_root - avg_frags, avg_frags, broken_onehots]  # (n_frags, hidden_size*3+max_broken+1)
        hidden = torch.cat(mlp_cat_list, dim=1)  # (n_frags, hidden_size*3+max_broken+1)

        padded_hidden = nn_utils.pad_packed_tensor(hidden, num_frags, 0)  # (bs, max_frags, hidden_size*3+max_broken+1)
        if self.encode_forms:
            diffs = root_forms[:, None, :] - frag_forms  # (bs, max_frags, 18)
            form_encodings = self.embedder(frag_forms)  # (bs, max_frags, 144)
            diff_encodings = self.embedder(diffs)  # (bs, max_frags, 144)
            new_hidden = torch.cat([padded_hidden, form_encodings, diff_encodings], dim=-1)  # (bs, max_frags, hidden_size*3+max_broken+1+288)
            padded_hidden = new_hidden

        padded_hidden = self.intermediate_out(padded_hidden)  # (bs, max_frags, hidden_size)
        batch_size, max_frags, hidden_dim = padded_hidden.shape

        arange_frags = torch.arange(max_frags, device=device)  # (max_frags,)
        attn_mask = ~(arange_frags[None, :] < num_frags[:, None])  # (bs, max_frags)

        hidden = padded_hidden  # (bs, max_frags, hidden_size)
        for trans_layer in self.trans_layers:
            hidden, _ = trans_layer(hidden, src_key_padding_mask=attn_mask)  # (bs, max_frags, hidden_size)

        max_inten_shift = (self.output_size - 1) / 2
        max_break_ar = torch.arange(self.output_size, device=device)[None, None, :]  # (1, 1, output_size)
        max_breaks_ub = max_add_hs + max_inten_shift  # (bs, max_frags)
        max_breaks_lb = -max_remove_hs + max_inten_shift  # (bs, max_frags)

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]  # (bs, max_frags, output_size)
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]  # (bs, max_frags, output_size)
        valid_pos = torch.logical_and(ub_mask, lb_mask)  # (bs, max_frags, output_size)

        valid_pos = valid_pos[:, :, None, :].expand(batch_size, max_frags, self.num_outputs, self.output_size)  # (bs, max_frags, num_outputs, output_size)
        output = self.output_map(hidden)  # (bs, max_frags, num_outputs*output_size)
        attn_weights = self.isomer_attn_out(hidden)  # (bs, max_frags, num_outputs*output_size)

        output = output.reshape(batch_size, max_frags, self.num_outputs, -1)  # (bs, max_frags, num_outputs, output_size)
        attn_weights = attn_weights.reshape(batch_size, max_frags, self.num_outputs, -1)  # (bs, max_frags, num_outputs, output_size)
        attn_weights.masked_fill_(~valid_pos, -99999)  # (bs, max_frags, num_outputs, output_size)

        output = output.transpose(1, 2)  # (bs, num_outputs, max_frags, output_size)
        attn_weights = attn_weights.transpose(1, 2)  # (bs, num_outputs, max_frags, output_size)
        valid_pos_binned = valid_pos.transpose(1, 2)  # (bs, num_outputs, max_frags, output_size)

        inverse_indices = torch.bucketize(masses, self.inten_buckets, right=False)  # (bs, max_frags, 13)
        inverse_indices = inverse_indices[:, None, :, :].expand(attn_weights.shape)  # (bs, num_outputs, max_frags, output_size)

        attn_weights = attn_weights.reshape(batch_size, self.num_outputs, -1)  # (bs, num_outputs, max_frags*output_size)
        output = output.reshape(batch_size, self.num_outputs, -1)  # (bs, num_outputs, max_frags*output_size)
        inverse_indices = inverse_indices.reshape(batch_size, self.num_outputs, -1)  # (bs, num_outputs, max_frags*output_size)
        valid_pos_binned = valid_pos_binned.reshape(batch_size, self.num_outputs, -1)  # (bs, num_outputs, max_frags*output_size)

        pool_weights = ts.scatter_softmax(attn_weights, index=inverse_indices, dim=-1)  # (bs, num_outputs, max_frags*output_size)
        weighted_out = pool_weights * output  # (bs, num_outputs, max_frags*output_size)

        output_binned = ts.scatter_add(
            weighted_out,
            index=inverse_indices,
            dim=-1,
            dim_size=self.inten_buckets.shape[-1],
        )  # (bs, num_outputs, 15000)

        output = output.reshape(batch_size, max_frags, self.num_outputs, -1)  # (bs, max_frags, num_outputs, output_size)
        pool_weights_reshaped = pool_weights.reshape(batch_size, max_frags, self.num_outputs, -1)  # (bs, max_frags, num_outputs, output_size)
        inverse_indices_reshaped = inverse_indices.reshape(batch_size, max_frags, self.num_outputs, -1)  # (bs, max_frags, num_outputs, output_size)

        valid_pos_binned = ts.scatter_max(
            valid_pos_binned.long(),
            index=inverse_indices,
            dim_size=self.inten_buckets.shape[-1],
            dim=-1,
        )[0].bool()  # (bs, num_outputs, 15000)

        new_outputs_binned = []
        for output_ind, act in enumerate(self.output_activations):
            new_outputs_binned.append(
                act(output_binned[:, output_ind : output_ind + 1, :])
            )
        output_binned = torch.cat(new_outputs_binned, dim=-2)  # (bs, num_outputs, 15000)
        output_binned.masked_fill_(~valid_pos_binned, 0)

        inverse_indices_reshaped_temp = inverse_indices_reshaped.transpose(1, 2).reshape(batch_size, self.num_outputs, -1)  # (bs, num_outputs, max_frags*output_size)
        output_unbinned = torch.take_along_dim(output_binned, inverse_indices_reshaped_temp, dim=-1)  # (bs, num_outputs, max_frags*output_size)
        output_unbinned = output_unbinned.reshape(batch_size, max_frags, self.num_outputs, -1).transpose(1, 2)  # (bs, max_frags, num_outputs, output_size)
        output_unbinned_alpha = output_unbinned * pool_weights_reshaped  # (bs, max_frags, num_outputs, output_size)

        return {"output_binned": output_binned, "output": output_unbinned_alpha}

    def _common_step(self, batch, name="train"):
        pred_obj = self.forward(
            batch["frag_graphs"],
            batch["root_reprs"],
            batch["inds"],
            batch["num_frags"],
            broken=batch["broken_bonds"],
            adducts=batch["adducts"],
            max_remove_hs=batch["max_remove_hs"],
            max_add_hs=batch["max_add_hs"],
            masses=batch["masses"],
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
            # new_to_old=batch["new_to_old"]
        )
        pred_inten = pred_obj["output_binned"]
        pred_inten = pred_inten[:, 0, :]
        batch_size = len(batch["names"])

        loss = self.loss_fn(pred_inten, batch["inten_targs"])
        self.log(
            f"{name}_loss", loss["loss"].item(), batch_size=batch_size, on_epoch=True
        )

        for k, v in loss.items():
            if k != "loss":
                self.log(f"{name}_aux_{k}", v.item(), batch_size=batch_size)
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
