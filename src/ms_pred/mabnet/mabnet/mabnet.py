from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from ms_pred.mabnet.mabnet.utils import *

from ms_pred.mabnet.mabnet.many_body import F_dim, F_two_body, F_three_body, F_four_body
from ms_pred.mabnet.mabnet.many_body import triplets, quadruplets, CrossMultiHeadAttention, NonLinear
from ms_pred.mabnet.mabnet.decoder import Decoder


class PhyMabNet(nn.Module):

    def __init__(
        self,
        atom_index=None,
        atom_dim=48,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=4,
        num_layers=2,
        hidden_channels=256,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=100,
        cutoff=5.0,
        cutoff_pruning=1.6,
        max_num_neighbors=32,
        max_num_edges_save=32,
        use_padding=True,
        many_body=False,
    ):
        super(PhyMabNet, self).__init__()
        self.atom_index = atom_index
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.max_z = max_z
        self.cutoff = cutoff
        self.cutoff_pruning = cutoff_pruning
        self.max_num_neighbors = max_num_neighbors
        self.max_num_edges_save = max_num_edges_save
        self.use_padding = use_padding
        self.many_body = many_body

        # learnable parameters
        self.embedding = nn.Linear(in_features=atom_dim, out_features=hidden_channels)
        # self.embedding = nn.Embedding(max_z, hidden_channels)   # scalar embedding
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels).jittable()

        self.mp_layers = nn.ModuleList()
        mp_kwargs = dict(
            num_heads=num_heads, 
            hidden_channels=hidden_channels, 
            activation=activation, 
            attn_activation=attn_activation, 
            cutoff=cutoff, 
            cutoff_pruning=cutoff_pruning,
            max_num_edges_save=max_num_edges_save,
            vecnorm_type=vecnorm_type, 
            trainable_vecnorm=trainable_vecnorm,
            use_padding=use_padding,
        )

        for _ in range(num_layers - 1):
            layer = ManybodyMPLayer(last_layer=False, **mp_kwargs).jittable()
            self.mp_layers.append(layer)
        self.mp_layers.append(ManybodyMPLayer(last_layer=True, **mp_kwargs).jittable())

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)

        # physical constraints
        self.hidden_channels = hidden_channels
        if self.many_body:
            self.F_linear = nn.Sequential(NonLinear(F_dim, hidden_channels),  nn.LayerNorm(hidden_channels),)
            self.Fs_linear = nn.ModuleList([
                nn.Sequential(
                    NonLinear(F_dim, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                )
                for i in range(num_layers)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

        
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """
        Args:
            data (Data): Input data containing:
                - z (Tensor): Atomic numbers, shape [num_nodes].
                - pos (Tensor): Node positions (coordinates), shape [num_nodes, 3].
                - batch (Tensor): Batch indices for each node, shape [num_nodes].

        Returns:
            Tuple[Tensor, Tensor]: A tuple of:
                - x (Tensor): Updated node scalar features, shape [num_nodes, hidden_channels].
                - vec (Tensor): Updated node vector features, shape [num_nodes, ((lmax + 1)^2 - 1), hidden_channels].
        """

        x, pos, bond_edge_index, bond_edge_attr, batch = data.h, data.pos, data.edge_index, data.e, data.batch

        # physical constraints
        if self.many_body:
            tri_i, tri_j, tri_k, tri_kj, tri_ji, qua_i, qua_j, qua_k, qua_l, R_ij, theta_ijk, theta_jki, theta_kij, phi_ijkl, psi_ijkl = self.forward_with_physics(pos, batch, bond_edge_index)
        
        # Embedding Layers
        x = self.embedding(x)   # [node_nums, hidden_channels]
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)  # data.e 
        edge_attr = self.distance_expansion(edge_weight)  
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1) 
        edge_vec = self.sphere(edge_vec) 

        x = self.neighbor_embedding(x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device)    # vector embedding
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)   # [edge_nums, hidden_channels]

        if self.many_body:
            i, j = bond_edge_index
            xs = x
            F_list = []
            if i.shape[0] != 0:
                F_two = F_two_body(xs[:, :self.hidden_channels//4], R_ij, i, j)
                F_list.append(F.pad(F_two, pad=(0, 0, 0, batch.shape[0] - F_two.shape[-2])))
            if tri_i.shape[0] != 0:
                F_three = F_three_body(xs[:, self.hidden_channels//4:self.hidden_channels//2], theta_ijk, theta_jki, theta_kij,
                                       tri_i, tri_j, tri_k, tri_kj, tri_ji,
                                       R_ij, pos)
                F_list.append(F.pad(F_three, pad=(0, 0, 0, batch.shape[0] - F_three.shape[-2])))
            if qua_i.shape[0] != 0:
                F_four = F_four_body(xs[:, self.hidden_channels//2:self.hidden_channels], phi_ijkl, psi_ijkl, qua_i, qua_j, qua_k, qua_l)
                F_list.append(F.pad(F_four, pad=(0, 0, 0, batch.shape[0] - F_four.shape[-2])))
            node_F = F.pad(torch.cat(F_list, dim=-1), pad=(0, F_dim - sum([f.shape[-1] for f in F_list])))
            node_F = self.F_linear(node_F)
            x = x + node_F
        
        # MP Layers
        F_all = []
        for layer_i, attn in enumerate(self.mp_layers[:-1]):
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec, batch)

            if self.many_body:
                F_list = []
                if i.shape[0] != 0:
                    F_two = F_two_body(xs[:, :self.hidden_channels//4], R_ij, i, j)
                    F_list.append(F.pad(F_two, pad=(0, 0, 0, batch.shape[0] - F_two.shape[-2])))
                if tri_i.shape[0] != 0:
                    F_three = F_three_body(xs[:, self.hidden_channels//4:self.hidden_channels//2], theta_ijk, theta_jki, theta_kij,
                                           tri_i, tri_j, tri_k, tri_kj, tri_ji,
                                           R_ij, pos)
                    F_list.append(F.pad(F_three, pad=(0, 0, 0, batch.shape[0] - F_three.shape[-2])))
                if qua_i.shape[0] != 0:
                    F_four = F_four_body(xs[:, self.hidden_channels//2:self.hidden_channels], phi_ijkl, psi_ijkl, qua_i, qua_j, qua_k, qua_l)
                    F_list.append(F.pad(F_four, pad=(0, 0, 0, batch.shape[0] - F_four.shape[-2])))
                node_F = F.pad(torch.cat(F_list, dim=-1), pad=(0, F_dim - sum([f.shape[-1] for f in F_list])))
                F_all.append(node_F)
                emb_F = self.Fs_linear[layer_i](node_F)
                x = x + emb_F

            x = x + dx  # [node_nums, hidden_channels]
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec, batch)
        x = x + dx
        vec = vec + dvec
        
        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)
        return x, vec
    

    def forward_with_physics(self, pos, batch, edge_index) -> Tuple[Tensor, Tensor]:
        i, j = edge_index
        tri_i, tri_j, tri_k, tri_kj, tri_ji = triplets(edge_index.long(), num_nodes=batch.shape[0])
        qua_i, qua_j, qua_k, qua_l, qua_lk, qua_kj, qua_ji = quadruplets(edge_index.long(),
                                                                            num_nodes=batch.shape[0])
        R_ij = torch.norm(pos[i] - pos[j], p=2, dim=-1)

        tri_ij_vec = pos[tri_i] - pos[tri_j]
        tri_kj_vec = pos[tri_k] - pos[tri_j]
        tri_ik_vec = pos[tri_i] - pos[tri_k]

        theta_ijk = torch.acos(torch.clamp(
            torch.sum(tri_ij_vec * tri_kj_vec, dim=-1) /
            (torch.norm(tri_ij_vec, p=2, dim=-1) * torch.norm(tri_kj_vec, p=2, dim=-1)),
            min=-0.999999, max=0.999999))
        theta_jki = torch.acos(torch.clamp(
            torch.sum(-tri_kj_vec * tri_ik_vec, dim=-1) /
            (torch.norm(-tri_kj_vec, p=2, dim=-1) * torch.norm(tri_ik_vec, p=2, dim=-1)),
            min=-0.999999, max=0.999999))
        theta_kij = torch.acos(torch.clamp(
            torch.sum(-tri_ik_vec * -tri_ij_vec, dim=-1) /
            (torch.norm(-tri_ik_vec, p=2, dim=-1) * torch.norm(-tri_ij_vec, p=2, dim=-1)),
            min=-0.999999, max=0.999999))

        qua_ij_vec = pos[qua_i] - pos[qua_j]
        qua_lk_vec = pos[qua_l] - pos[qua_k]
        qua_kj_vec = pos[qua_k] - pos[qua_j]
        phi_ijkl = torch.acos(torch.clamp(
            torch.sum(qua_ij_vec * qua_lk_vec, dim=-1) /
            (torch.norm(qua_ij_vec, p=2, dim=-1) * torch.norm(qua_lk_vec, p=2, dim=-1)),
            min=-0.999999, max=0.999999))
        psi_vec_1 = torch.multiply(qua_kj_vec, qua_ij_vec)
        psi_vec_2 = torch.multiply(-qua_kj_vec, qua_lk_vec)
        psi_ijkl = torch.acos(torch.clamp(
            torch.sum(-psi_vec_1 * psi_vec_2, dim=-1) /
            (torch.norm(psi_vec_1, p=2, dim=-1) * torch.norm(psi_vec_2, p=2, dim=-1)),
            min=-0.999999, max=0.999999))
        return tri_i, tri_j, tri_k, tri_kj, tri_ji, qua_i, qua_j, qua_k, qua_l, R_ij, theta_ijk, theta_jki, theta_kij, phi_ijkl, psi_ijkl


class ManybodyMPLayer(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        cutoff_pruning,
        max_num_edges_save,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
        use_padding=True,
        is_bidirec=True,
    ):
        super(ManybodyMPLayer, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.cutoff_pruning = cutoff_pruning
        self.max_num_edges_save = calculate_max_edges(max_num_edges_save, is_bidirec)
        self.use_padding = use_padding

        # learnable parameters
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False) # 将vec embedding进行线性变换（投影）扩展维度
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.mabybody_attention_three_body = ManyBodyPadAttn(hidden_channels, num_heads)
        if self.max_num_edges_save > 0:
            self.manybody_attention_four_body = ManyBodyPadAttn(hidden_channels, num_heads)

        if self.max_num_edges_save > 0:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        
        self.reset_parameters()
        
    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, batch):
        """
        Args:
            x (Tensor): Scalar embedding of nodes, shape [batchsize * num_nodes, hidden_channels].
            vec (Tensor): Vector embedding of nodes, shape [batchsize * num_nodes, ((lmax + 1)^2 - 1), hidden_channels].
            edge_index (Tensor): Edge indices, shape [2, batchsize * num_edges].
            r_ij (Tensor): Edge distances, shape [batchsize * num_edges].
            f_ij (Tensor): Edge features, shape [batchsize * num_edges, hidden_channels].
            d_ij (Tensor): Edge directions, shape [batchsize * num_edges, ((lmax + 1)^2 - 1)].
            batch (Tensor): Batch indices for each node, shape [batchsize * num_nodes].

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: A tuple containing:
                - dx (Tensor): Updated scalar embedding, shape [batchsize * num_nodes, hidden_channels].
                - dvec (Tensor): Updated vector embedding, shape [batchsize * num_nodes, ((lmax + 1)^2 - 1), hidden_channels].
                - df_ij (Optional[Tensor]): Updated edge features, shape [batchsize * num_edges, hidden_channels], or None if last layer.
        """

        x = self.layernorm(x)   # scalar embedding
        vec = self.vec_layernorm(vec)   # vector embedding

        batch_size = batch.max().item() + 1
        num_nodes_per_sample = torch.bincount(batch)
        x_x, x_e, mask_three_body, mask_four_body = get_feats_with_padding(
            x,
            edge_index,
            r_ij,
            f_ij,
            self.cutoff_pruning,
            self.max_num_edges_save,
            self.hidden_channels,
            batch,
        )
        v = self.mabybody_attention_three_body(x_x, x_x, mask_three_body).mean(dim=2)    # 3-body    # [batchsize, num_nodes, num_nodes, hidden_channels] -> [batchsize, num_nodes, hidden_channels]
        if self.max_num_edges_save > 0:
            v_four_body = self.manybody_attention_four_body(x_x, x_e, mask_four_body).mean(dim=2)   # 4-body    # [batchsize, num_nodes, num_nodes, hidden_channels] -> [batchsize, num_nodes, hidden_channels]
            v = torch.cat([v, v_four_body], dim=-1)
            v = self.mlp(v)
        
        v_list = []
        for b in range(batch_size):
            num_nodes = num_nodes_per_sample[b].item()
            v_b = v[b, :num_nodes]
            v_list.append(v_b)
        v = torch.cat(v_list, dim=0)

        v = v.reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)   # [batchsize * num_edges, num_heads * head_dim] -> [batchsize * num_edges, num_heads, head_dim]

        # vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)    
        vec1, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_sum = vec1.sum(dim=1)

        # propagate_type: (v: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            v=v,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1) 
        dx = vec_sum * o2 + o3  
        dvec = vec3 * o1.unsqueeze(1) + vec_out

        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij) 
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, v_j, vec_j, dv, r_ij, d_ij):
        """
        Args:
            v_j (Tensor): Value embeddings of neighboring nodes, shape [batchsize * num_edges, num_heads, head_dim].
            vec_j (Tensor): Vector embeddings of neighboring nodes, shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels].
            dv (Tensor): Projected edge features, shape [batchsize * num_edges, num_heads, head_dim].
            r_ij (Tensor): Edge distances, shape [batchsize * num_edges].
            d_ij (Tensor): Edge directions, shape [batchsize * num_edges, ((lmax + 1)^2 - 1)].

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - v_j (Tensor): Updated value embeddings, shape [batchsize * num_edges, hidden_channels].
                - vec_j (Tensor): Updated vector embeddings, shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels].
        """
        
        v_j = v_j * self.cutoff(r_ij).unsqueeze(1).unsqueeze(2)
        v_j = v_j * dv   
        v_j = v_j.view(-1, self.hidden_channels)
        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)   
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2) 
        """
        v_j: shape [batchsize * num_edges, hidden_channels]
        vec_j: shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels]
        """
        return v_j, vec_j
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        """
        x: shape [batchsize * num_edges, hidden_channels]
        vec: shape [batchsize * num_edges, ((lmax + 1)^2 - 1), hidden_channels]
        """
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        """
        x: shape [batchsize * num_nodes, hidden_channels]
        vec: shape [batchsize * num_nodes, ((lmax + 1)^2 - 1), hidden_channels]
        """
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

