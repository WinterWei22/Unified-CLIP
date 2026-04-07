import torch
import torch.nn as nn
import math

class MassSpecTransformer(nn.Module):
    def __init__(self,
                dim=768,
                nhead=12,
                num_layers=8,
                dim_feedforward=3072,
                dropout=0.1,
                mz_min=50.0,
                mz_max=2000.0,
                use_cls_token=True,
                use_seq_pe=True,  # 新增：是否使用顺序位置编码
                activation="gelu"):
        super().__init__()
        self.dim = dim
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.use_cls_token = use_cls_token
        self.use_seq_pe = use_seq_pe

        # 1. 输入投影：(m/z, intensity) → dim
        self.input_proj = nn.Linear(2, dim)

        # 2. div_term 用于动态位置编码（预计算以加速）
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        self.register_buffer('div_term', div_term)  # (dim//2,)

        # 3. CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            # norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (bs, len, 2)    float tensor, 最后一维为 [m/z, intensity]（已预归一化）
            mask: (bs, len)    bool tensor, True 表示 padding 位置需 mask
        Returns:
            spec_emb: (bs, dim)         整张谱的全局 embedding
            peak_emb: (bs, len, dim)    每个峰的 contextual embedding
        """
        bs, len_, _ = x.shape
        mz = x[:, :, 0]  # 在投影前提取原始 m/z

        # 输入投影
        x = self.input_proj(x)  # (bs, len, dim)

        # === m/z-based 连续位置编码（优化为动态计算） ===
        norm_mz = ((mz - self.mz_min) / (self.mz_max - self.mz_min)).clamp(0.0, 1.0)  # (bs, len)
        position = norm_mz.unsqueeze(-1)  # (bs, len, 1)
        div_term = self.div_term.unsqueeze(0).unsqueeze(0)  # (1, 1, dim//2)
        mz_pe = torch.zeros(bs, len_, self.dim, device=x.device)
        mz_pe[:, :, 0::2] = torch.sin(position * div_term)
        mz_pe[:, :, 1::2] = torch.cos(position * div_term)
        x = x + mz_pe

        # === 顺序位置编码（动态计算，无长度限制） ===
        if self.use_seq_pe:
            seq_position = torch.arange(0, len_, dtype=torch.float, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, len, 1)
            seq_pe = torch.zeros(1, len_, self.dim, device=x.device)
            seq_pe[:, :, 0::2] = torch.sin(seq_position * div_term)
            seq_pe[:, :, 1::2] = torch.cos(seq_position * div_term)
            x = x + seq_pe

        # === CLS token ===
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(bs, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (bs, 1+len, dim)
            if mask is not None:
                cls_mask = torch.zeros(bs, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # === Transformer ===
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x

        # # === 输出 ===
        # if self.use_cls_token:
        #     spec_emb = x[:, 0]
        #     peak_emb = x[:, 1:]
        # else:
        #     spec_emb = x.mean(dim=1)
        #     peak_emb = x

        # return spec_emb, peak_emb