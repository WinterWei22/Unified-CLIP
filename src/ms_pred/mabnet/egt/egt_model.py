import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool

class EdgeEnhancedGraphTransformer3D(nn.Module):
    def __init__(self, input_node_dim=48, input_edge_dim=4, hidden_dim=128, num_layers=6, num_heads=8, dropout=0.1, edge_update=False):
        super(EdgeEnhancedGraphTransformer3D, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_update = edge_update
        
        # Node embedding layer
        self.node_embed = nn.Linear(input_node_dim, hidden_dim)
        
        # Edge embedding layer for bias
        self.edge_embed = nn.Linear(input_edge_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([self._build_transformer_layer() for _ in range(num_layers)])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _build_transformer_layer(self):
        layer_dict = nn.ModuleDict({
            'attn': MultiHeadAttentionWithEdgeBias3D(self.hidden_dim, self.num_heads, self.dropout),
            'ffn': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.Dropout(self.dropout)
            ),
            'norm1': nn.LayerNorm(self.hidden_dim),
            'norm2': nn.LayerNorm(self.hidden_dim)
        })
        if self.edge_update:
            layer_dict['edge_ffn'] = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.Dropout(self.dropout)
            )
            layer_dict['edge_norm'] = nn.LayerNorm(self.hidden_dim)
        return layer_dict

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.h, data.pos, data.edge_index, data.e, data.batch  # Assuming data.e is edge_attr
        
        # Embed nodes
        x = self.node_embed(x)  # [nodes, hidden_dim]
        
        # Embed edges for bias
        edge_emb = self.edge_embed(edge_attr)  # [edges, hidden_dim]
        
        for layer in self.layers:
            # Residual for attention
            res = x
            x = layer['norm1'](x)
            x = layer['attn'](x, edge_index, edge_emb, pos) + res  # Attention with edge and pos bias
            
            # Update edge embeddings if enabled
            if self.edge_update:
                res_edge = edge_emb
                edge_emb = layer['edge_norm'](edge_emb)
                row, col = edge_index
                src = x[row]  # [edges, hidden_dim]
                tgt = x[col]  # [edges, hidden_dim]
                concat = torch.cat([src, tgt, edge_emb], dim=-1)  # [edges, 3*hidden_dim]
                edge_emb = layer['edge_ffn'](concat) + res_edge
                
            # Residual for FFN
            res = x
            x = layer['norm2'](x)
            x = layer['ffn'](x) + res
        
        x = self.final_norm(x)
        
        # Node embeddings
        node_embeddings = x  # [nodes, hidden_dim]
        
        # Graph embeddings via global mean pooling
        graph_embeddings = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        return node_embeddings, graph_embeddings

class MultiHeadAttentionWithEdgeBias3D(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttentionWithEdgeBias3D, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # For edge bias projection (to multi-head)
        self.edge_bias_proj = nn.Linear(hidden_dim, num_heads)
        
        # Relative position embedding (for pos [nodes, 3])
        self.rel_pos_embed = nn.Linear(3, hidden_dim)  # Project relative distances to hidden_dim

    def forward(self, x, edge_index, edge_emb, pos):
        num_nodes = x.shape[0]
        
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)  # [nodes, heads, head_dim]
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Sparse attention: gather k and v based on edge_index
        row, col = edge_index
        k_gather = k[col]  # [edges, heads, head_dim]
        v_gather = v[col]  # [edges, heads, head_dim]
        q_gather = q[row]  # [edges, heads, head_dim]
        
        # Attention scores: QK^T / sqrt(d)
        attn_scores = (q_gather * k_gather).sum(dim=-1) / (self.head_dim ** 0.5)  # [edges, heads]
        
        # Add edge bias
        edge_bias = self.edge_bias_proj(edge_emb)  # [edges, heads]
        attn_scores = attn_scores + edge_bias  # [edges, heads]
        
        # Add relative position bias
        rel_pos = pos[row] - pos[col]  # [edges, 3] relative positions
        rel_pos_emb = self.rel_pos_embed(rel_pos)  # [edges, hidden_dim]
        rel_pos_bias = rel_pos_emb.view(-1, self.num_heads, self.head_dim).sum(dim=-1)  # [edges, heads]
        attn_scores = attn_scores + rel_pos_bias  # [edges, heads]
        
        # Per-node softmax
        max_scores = scatter(attn_scores, row, dim=0, dim_size=num_nodes, reduce='max')  # [nodes, heads]
        max_scores = max_scores[row]  # [edges, heads]
        exp_scores = (attn_scores - max_scores).exp()  # [edges, heads]
        sum_exp = scatter(exp_scores, row, dim=0, dim_size=num_nodes, reduce='add')  # [nodes, heads]
        sum_exp = sum_exp[row] + 1e-10  # [edges, heads]
        attn_probs = exp_scores / sum_exp  # [edges, heads]
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum: scatter back to nodes
        out = torch.zeros_like(v)  # [nodes, heads, head_dim]
        attn_probs = attn_probs.unsqueeze(-1)  # [edges, heads, 1]
        weighted_v = attn_probs * v_gather  # [edges, heads, head_dim]
        index = row.view(-1, 1, 1).expand_as(weighted_v)  # [edges, heads, head_dim]
        out = scatter(weighted_v, index, dim=0, dim_size=num_nodes, reduce='add')  # [nodes, heads, head_dim]
        
        out = out.view(num_nodes, self.hidden_dim)  # [nodes, hidden_dim]
        out = self.out_proj(out)
        return self.dropout(out)
    
class EdgeEnhancedGraphTransformer2D(nn.Module):
    def __init__(self, input_node_dim=48, input_edge_dim=4, hidden_dim=128, num_layers=6, num_heads=8, dropout=0.1, edge_update=False):
        super(EdgeEnhancedGraphTransformer2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_update = edge_update
        
        # Node embedding layer
        self.node_embed = nn.Linear(input_node_dim, hidden_dim)
        
        # Edge embedding layer for bias
        self.edge_embed = nn.Linear(input_edge_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([self._build_transformer_layer() for _ in range(num_layers)])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _build_transformer_layer(self):
        layer_dict = nn.ModuleDict({
            'attn': MultiHeadAttentionWithEdgeBias2D(self.hidden_dim, self.num_heads, self.dropout),
            'ffn': nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.Dropout(self.dropout)
            ),
            'norm1': nn.LayerNorm(self.hidden_dim),
            'norm2': nn.LayerNorm(self.hidden_dim)
        })
        if self.edge_update:
            layer_dict['edge_ffn'] = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.Dropout(self.dropout)
            )
            layer_dict['edge_norm'] = nn.LayerNorm(self.hidden_dim)
        return layer_dict

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.h, data.edge_index, data.e, data.batch  # Assuming data.e is edge_attr; ignoring pos
        
        # Embed nodes
        x = self.node_embed(x)  # [nodes, hidden_dim]
        
        # Embed edges for bias
        edge_emb = self.edge_embed(edge_attr)  # [edges, hidden_dim]
        
        for layer in self.layers:
            # Residual for attention
            res = x
            x = layer['norm1'](x)
            x = layer['attn'](x, edge_index, edge_emb) + res  # Attention with only edge bias (no pos)
            
            # Update edge embeddings if enabled
            if self.edge_update:
                res_edge = edge_emb
                edge_emb = layer['edge_norm'](edge_emb)
                row, col = edge_index
                src = x[row]  # [edges, hidden_dim]
                tgt = x[col]  # [edges, hidden_dim]
                concat = torch.cat([src, tgt, edge_emb], dim=-1)  # [edges, 3*hidden_dim]
                edge_emb = layer['edge_ffn'](concat) + res_edge
                
            # Residual for FFN
            res = x
            x = layer['norm2'](x)
            x = layer['ffn'](x) + res
        
        x = self.final_norm(x)
        
        # Node embeddings
        node_embeddings = x  # [nodes, hidden_dim]
        
        # Graph embeddings via global mean pooling
        graph_embeddings = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        return node_embeddings, graph_embeddings

class MultiHeadAttentionWithEdgeBias2D(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttentionWithEdgeBias2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # For edge bias projection (to multi-head)
        self.edge_bias_proj = nn.Linear(hidden_dim, num_heads)

    def forward(self, x, edge_index, edge_emb):
        num_nodes = x.shape[0]
        
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)  # [nodes, heads, head_dim]
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Sparse attention: gather k and v based on edge_index
        row, col = edge_index
        k_gather = k[col]  # [edges, heads, head_dim]
        v_gather = v[col]  # [edges, heads, head_dim]
        q_gather = q[row]  # [edges, heads, head_dim]
        
        # Attention scores: QK^T / sqrt(d)
        attn_scores = (q_gather * k_gather).sum(dim=-1) / (self.head_dim ** 0.5)  # [edges, heads]
        
        # Add edge bias
        edge_bias = self.edge_bias_proj(edge_emb)  # [edges, heads]
        attn_scores = attn_scores + edge_bias  # [edges, heads]
        
        # Per-node softmax
        max_scores = scatter(attn_scores, row, dim=0, dim_size=num_nodes, reduce='max')  # [nodes, heads]
        max_scores = max_scores[row]  # [edges, heads]
        exp_scores = (attn_scores - max_scores).exp()  # [edges, heads]
        sum_exp = scatter(exp_scores, row, dim=0, dim_size=num_nodes, reduce='add')  # [nodes, heads]
        sum_exp = sum_exp[row] + 1e-10  # [edges, heads]
        attn_probs = exp_scores / sum_exp  # [edges, heads]
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum: scatter back to nodes
        out = torch.zeros_like(v)  # [nodes, heads, head_dim]
        attn_probs = attn_probs.unsqueeze(-1)  # [edges, heads, 1]
        weighted_v = attn_probs * v_gather  # [edges, heads, head_dim]
        index = row.view(-1, 1, 1).expand_as(weighted_v)  # [edges, heads, head_dim]
        out = scatter(weighted_v, index, dim=0, dim_size=num_nodes, reduce='add')  # [nodes, heads, head_dim]
        
        out = out.view(num_nodes, self.hidden_dim)  # [nodes, hidden_dim]
        out = self.out_proj(out)
        return self.dropout(out)