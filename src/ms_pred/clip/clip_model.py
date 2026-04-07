import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ms_pred.mabnet.egt import egt_model
from ms_pred.DreaMS.dreams.models.dreams.dreams import DreaMS as DreaMSModel
from ms_pred.DreaMS.dreams.api import PreTrainedModel
from ms_pred.mabnet import egt_pretrain_model as pretrain_model
from ms_pred.clip.ms_model import MassSpecTransformer

import ms_pred.nn_utils as nn_utils
import ms_pred.common as common
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from ms_pred.mabnet import egt_pretrain_model as mol_pretrain_model
import math

class Mol_Encoder_with_adducts(nn.Module):
    def __init__(self, input_node_dim, hidden_size=128, mabnet_heads=8, mabnet_layers=6, edge_update=True, dropout=0.1, instrument=False):
        super().__init__()
        self.mol_encoder = egt_model.EdgeEnhancedGraphTransformer2D(
            input_node_dim=input_node_dim,
            hidden_dim=hidden_size,
            num_heads=mabnet_heads,
            num_layers=mabnet_layers,
            edge_update=edge_update,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
        adduct_types = len(common.ion2onehot_pos)
        onehot = torch.eye(adduct_types)
        
        self.adduct_embedder = nn.Parameter(onehot.float())
        self.adduct_embedder.requires_grad = False

    def forward(self, root_repr, adducts, ind_maps=None):
        embed_adducts = self.adduct_embedder[adducts.long()]  # (batch_size, adduct_dim)
        if ind_maps is None:
            root_batch_sizes = torch.bincount(root_repr.batch) if root_repr.batch is not None else torch.tensor([root_repr.num_nodes])
            embed_adducts_expand = embed_adducts.repeat_interleave(root_batch_sizes, 0)
        else:
            adducts_mapped = embed_adducts[ind_maps]
            root_batch_sizes = torch.bincount(root_repr.batch) if root_repr.batch is not None else torch.tensor([root_repr.num_nodes])
            embed_adducts_expand = adducts_mapped.repeat_interleave(
                root_batch_sizes, dim=0
            )
        root_h = torch.cat([root_repr.h, embed_adducts_expand], -1)
        root_repr.h = root_h
        x, mol_emb = self.mol_encoder(root_repr)  # (batch_size, hidden_size)
        return x, mol_emb
    
class Mol_Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_size=128, mabnet_heads=8, mabnet_layers=6, edge_update=True, dropout=0.1, instrument=False):
        super().__init__()
        self.mol_encoder = egt_model.EdgeEnhancedGraphTransformer2D(
            input_node_dim=input_node_dim,
            hidden_dim=hidden_size,
            num_heads=mabnet_heads,
            num_layers=mabnet_layers,
            edge_update=edge_update,
            dropout=dropout,
        )

    def forward(self, root_repr):
        x, mol_emb = self.mol_encoder(root_repr)  # (batch_size, hidden_size)
        return x, mol_emb

class CLIPModel(pl.LightningModule):
    def __init__(self, hidden_size=128, mabnet_heads=8, mabnet_layers=6, edge_update=True, projection_dim=128, temperature=0.07, lr=1e-3, spec_ckpt_path=None, mol_ckpt_path=None,
                lr_decay_rate=0.825, warmup=1000, dropout=0.1, weight_decay=0.0005, pe_embed_k=0, emb_adducts=False, inject_early=False, local_contra=False, decoys=False,
                triplet_margin=0.2, triplet_alpha=0.8, decoys_temp=0.05, frags=False, dreams=False, decoy_loss_weight=5.0, local_weight=0.5, local_threshold=0.8, pooling_strategy='cls',
                frozen_dreams=False, unfreeze_epoch=10, local_start_epoch=0, embed_ce=False, mol_ckpt=None, total_steps=0, frag_supervised=False, frag_supervised_weight=0.3, spec_sim_entropy=False,
                wo_global=False, use_pretrained_dreams=True):
        super().__init__()
        self.save_hyperparameters()
        self.wo_global = wo_global
        self.pe_embed_k = pe_embed_k
        self.emb_adducts = emb_adducts
        self.inject_early = inject_early

        self.local_contra = local_contra
        self.local_weight = local_weight
        self.local_threshold = local_threshold
        self.frags = frags
        
        self.decoys = decoys
        self.decoy_loss_weight = decoy_loss_weight
        
        self.triplet_margin = triplet_margin
        self.triplet_alpha = triplet_alpha
        self.decoys_temp = decoys_temp
        self.dreams = dreams
        self.pooling_strategy = pooling_strategy
        self.frozen_dreams = frozen_dreams
        self.unfreeze_epoch = unfreeze_epoch
        
        self.local_flag = local_contra
        self.local_start_epoch = local_start_epoch
        if self.local_start_epoch > 0:
            self.local_contra = False
        
        self.embed_ce = embed_ce
        
        self.spec_sim_entropy = spec_sim_entropy
        
        spec_output_dim = hidden_size  # Placeholder; replace with actual spec_encoder output dim
        mol_output_dim = hidden_size   # Placeholder; replace with actual mol_encoder output dim
        
        self.use_pretrained_dreams = use_pretrained_dreams
        if self.dreams and spec_ckpt_path:
            spec_output_dim = 1024
            self.spec_encoder = PreTrainedModel.from_ckpt(
                ckpt_path=spec_ckpt_path,
                ckpt_cls=DreaMSModel,
                n_highest_peaks=60
            ).model.eval()

            if not self.use_pretrained_dreams:
                # Reinitialize all weights randomly (same architecture, no pretrained weights)
                print(">>> Reinitializing DreaMS encoder with random weights (no pretraining) <<<")
                for name, param in self.spec_encoder.named_parameters():
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)
                self.spec_encoder.train()

            if self.frozen_dreams:
                for param in self.spec_encoder.parameters():
                    param.requires_grad = False
        else:
            self.spec_encoder = MassSpecTransformer(
                dim=hidden_size,
                nhead=4,
                num_layers=4,
                mz_min=0.0,
                mz_max=1000.0,
                use_cls_token=True,
                use_seq_pe=False
            )
                    
        self.mol_ckpt_path = mol_ckpt_path
        if self.mol_ckpt_path is not None:
            self.mol_encoder = Mol_Encoder(
                input_node_dim=26,
                hidden_size=hidden_size,
                mabnet_heads=mabnet_heads,
                mabnet_layers=mabnet_layers,
                edge_update=edge_update,
                dropout=dropout
            )
            pretrained_model = pretrain_model.Pretrain_EGT.load_from_checkpoint(mol_ckpt_path, strict=False)
            print(f'loaded mol encoder from: {mol_ckpt_path}')
            
            self.mol_encoder.load_state_dict(pretrained_model.root_module.state_dict(), strict=False)
        elif self.emb_adducts and self.inject_early:
            self.mol_encoder = Mol_Encoder_with_adducts(
            input_node_dim=26 + 10 + 14,
            hidden_size=hidden_size,
            mabnet_heads=mabnet_heads,
            mabnet_layers=mabnet_layers,
            edge_update=edge_update,
            dropout=dropout,  # Assuming the model accepts dropout; adjust if not
            )
        else:
            self.mol_encoder = Mol_Encoder(
                input_node_dim=26,
                hidden_size=hidden_size,
                mabnet_heads=mabnet_heads,
                mabnet_layers=mabnet_layers,
                edge_update=edge_update,
                dropout=dropout,  # Assuming the model accepts dropout; adjust if not
            )
        # Assume the output dimensions of encoders; adjust these based on actual output sizes

        
        if self.emb_adducts:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            # spec_output_dim += 10
            if not self.inject_early:
                mol_output_dim += adduct_types
            
        if self.embed_ce and not self.inject_early:
            pe_dim = common.COLLISION_PE_DIM
            pe_scalar = common.COLLISION_PE_SCALAR
            pe_power = 2 * torch.arange(pe_dim // 2) / pe_dim
            self.collision_embedder_denominators = nn.Parameter(torch.pow(pe_scalar, pe_power))
            self.collision_embedder_denominators.requires_grad = False
            collision_shift = pe_dim
            self.collision_embed_merged = nn.Parameter(torch.zeros(pe_dim))
            self.collision_embed_merged.requires_grad = False
            mol_output_dim += pe_dim
            
        # Projection heads to align dimensions with dropout
        self.spec_projection = nn.Sequential(
            nn.Linear(spec_output_dim, projection_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.mol_projection = nn.Sequential(
            nn.Linear(mol_output_dim, projection_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        
        self.atom_projection = nn.Sequential(
            nn.Linear(mol_output_dim, projection_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        if self.decoys:
            self.decoys_projection = nn.Sequential(
                nn.Linear(mol_output_dim, projection_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )
            
        if self.frags:
            self.frags_projection = nn.Sequential(
                nn.Linear(mol_output_dim, projection_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )

        
        
        self.temperature = temperature
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.frag_supervised = frag_supervised
        self.frag_supervised_weight = frag_supervised_weight

    def forward(self, specs, root_reprs, adducts,  decoys_reprs=None, decoys_adducts=None, frags_reprs=None, ind_maps=None, specs_mask=None, ces=None):
        # Assuming spec_encoder and mol_encoder handle their respective data formats
        if self.emb_adducts and self.inject_early:
            atoms_emb, mol_emb = self.mol_encoder(root_reprs, adducts)
            
            spec_emb = self.spec_encoder(specs) # Adjust if spec_encoder requires specific input unpacking
            spec_proj = self.spec_projection(spec_emb)
            if self.pooling_strategy == 'mean':
                spec_global_emb = (spec_emb * specs_mask.unsqueeze(-1)).sum(dim=1) / \
                    specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            elif self.pooling_strategy == 'cls':
                spec_global_emb = spec_proj[:, 0, :]
            else:
                raise("pooling strategy should be cls or mean.")
        else:
            spec_emb = self.spec_encoder(specs) # Adjust if spec_encoder requires specific input unpacking
            atoms_emb, mol_emb = self.mol_encoder(root_reprs)    # Adjust if mol_encoder requires specific input unpacking (e.g., graph data)
            if self.emb_adducts:
                embed_adducts = self.adduct_embedder[adducts.long()]  # (batch_size, adduct_dim)
                # if ind_maps is None:
                #     root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                #     embed_adducts_expand = embed_adducts.repeat_interleave(root_batch_sizes, 0)
                # else:
                #     adducts_mapped = embed_adducts[ind_maps]
                #     root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                #     embed_adducts_expand = adducts_mapped.repeat_interleave(
                #         root_batch_sizes, dim=0
                #     )
                # Project to common space
                # spec_global_emb = (spec_emb * specs_mask.unsqueeze(-1)).sum(dim=1) / \
                #     specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                # spec_global_emb = spec_emb[:, 0, :]
                # if self.embed_ce and ces is not None:
                #     spec_emb =  torch.cat([spec_emb, embed_adducts, ces], -1)
                # else:
                #     spec_emb =  torch.cat([spec_emb, embed_adducts], -1)
                    
                spec_proj = self.spec_projection(spec_emb)
                if self.pooling_strategy == 'mean':
                    spec_global_emb = (spec_emb * specs_mask.unsqueeze(-1)).sum(dim=1) / \
                        specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                elif self.pooling_strategy == 'cls':
                    spec_global_emb = spec_proj[:, 0, :]
                else:
                    raise("pooling strategy should be cls or mean.")
                
                
                # spec_global_emb = torch.cat([spec_global_emb, embed_adducts], -1)
                mol_emb = torch.cat([mol_emb, embed_adducts], -1)
                
                root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_batch_sizes, dim=0
                    )
                atoms_emb = torch.cat([atoms_emb, embed_adducts_expand], -1)
                
            if self.embed_ce:
                embed_collision = torch.cat(
                    (torch.sin(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                        torch.cos(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                    dim=1
                )
                embed_collision = torch.where(  # handle entries without collision energy (== nan)
                    torch.isnan(embed_collision), self.collision_embed_merged.unsqueeze(0), embed_collision
                )
                mol_emb = torch.cat([mol_emb, embed_collision], -1)
                
                root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                embed_collision_expand = embed_collision.repeat_interleave(
                        root_batch_sizes, dim=0
                    )
                atoms_emb = torch.cat([atoms_emb, embed_collision_expand], -1)
                
        mol_proj = self.mol_projection(mol_emb)
        atom_proj = self.atom_projection(atoms_emb)
        # mol_proj = mol_emb
        # spec_proj = spec_emb
        
        if self.decoys:
            if decoys_reprs is not None and decoys_adducts is not None:
                # !!! without complishing inds_map in decoys mol
                
                if self.emb_adducts and self.inject_early:
                    _, decoys_mol_emb = self.mol_encoder(decoys_reprs, decoys_adducts)
                else:
                    if self.emb_adducts:
                        decoys_embed_adducts = self.adduct_embedder[decoys_adducts.long()]
                        decoys_atoms_emb, decoys_mol_emb = self.mol_encoder(decoys_reprs) 
                        decoys_mol_emb = torch.cat([decoys_mol_emb, decoys_embed_adducts], -1)
                        
                    if self.embed_ce:
                        embed_collision = torch.cat(
                            (torch.sin(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                                torch.cos(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                            dim=1
                        )
                        embed_collision = torch.where(  # handle entries without collision energy (== nan)
                            torch.isnan(embed_collision), self.collision_embed_merged.unsqueeze(0), embed_collision
                        )
                        decoys_mol_emb = torch.cat([decoys_mol_emb, embed_collision], -1)
                        
                decoys_proj = self.decoys_projection(decoys_mol_emb)
            else:
                decoys_proj = None
        else:
            decoys_proj = None
        
        if self.frags:
            frags_reprs_h = frags_reprs.h
            if self.emb_adducts and self.inject_early:
                adducts_mapped = adducts[ind_maps]
                _, frags_emb = self.mol_encoder(frags_reprs, adducts_mapped)
            else:
                _, frags_emb = self.mol_encoder(frags_reprs)
                # Concatenate adduct/CE embeddings to match mol_output_dim (same as mol pathway)
                if self.emb_adducts:
                    frags_adducts = self.adduct_embedder[adducts[ind_maps].long()]
                    frags_emb = torch.cat([frags_emb, frags_adducts], -1)
                if self.embed_ce:
                    frags_ce = torch.cat(
                        (torch.sin(ces[ind_maps].unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                         torch.cos(ces[ind_maps].unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                        dim=1
                    )
                    frags_ce = torch.where(
                        torch.isnan(frags_ce), self.collision_embed_merged.unsqueeze(0), frags_ce
                    )
                    frags_emb = torch.cat([frags_emb, frags_ce], -1)
            frags_proj = self.frags_projection(frags_emb)
        else:
            frags_proj = atom_proj
        
        return spec_global_emb, mol_proj, frags_proj, spec_proj, decoys_proj
    

    def predict(self, specs, root_reprs, adducts, ces, specs_mask=None):
        # ... [省略原始代码]
        # 1. 获取特征
        spec_global, spec_seq = self.predict_spec(specs, adducts, specs_mask=specs_mask)
        # 注意：这里 atoms_flat 很有可能没有归一化，导致了数值异常
        mol_global, atoms_flat = self.predict_smi(root_reprs, adducts, ces)

        # 2. Global Similarity
        # 假设 mol_global 和 spec_global 已经 L2 归一化
        global_sim = torch.sum(spec_global * mol_global, dim=1)

        # 3. 准备局部特征 (Language Tokens 和 Visual Patches)
        if self.pooling_strategy == 'cls':
            # 质谱的 Token Embeddings (去掉 [CLS] token)
            spec_seq_local = spec_seq[:, 1:, :] 
            if specs_mask is not None:
                # 质谱的 Mask (去掉 [CLS] mask)
                specs_mask_local = specs_mask[:, 1:]
            else:
                # 如果没有 mask，默认为全 True
                specs_mask_local = torch.ones(spec_seq_local.shape[:2], device=self.device).bool()
        else:
            spec_seq_local = spec_seq
            specs_mask_local = specs_mask

        # 局部特征赋值
        # 对应 compute_local_loss 中的 l_token_embed
        l_token_embed = spec_seq_local
        # 对应 compute_local_loss 中的 v_patch_embed
        v_patch_embed = atoms_flat
        
        # 对应 compute_local_loss 中的 language_mask
        language_mask = specs_mask_local
        v_patch_embed, patch_mask = self.group_tensor_to_batch(v_patch_embed, root_reprs.batch)
        _, l2, _ = v_patch_embed.shape


        ## 4. Local Similarity Calculation (遵循 compute_local_loss 逻辑)

        # A. 原始 Similarity calculation (l_token_embed x v_patch_embed)
        # 对应: similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)
        similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)

        # B. Min-max normalization and Masking (得到 [0, 1] 范围的对齐分数)
        # 对应: min-max normalization 和 similarity.masked_fill(...)
        min_sim = similarity.min(dim=-1, keepdim=True)[0]
        max_sim = similarity.max(dim=-1, keepdim=True)[0]
        
        # 防止除以零
        similarity = (similarity - min_sim) / (max_sim - min_sim + 1e-8)
        
        # 应用语言 (spec/peak) mask 和 patch (atom) mask
        mask_peaks = language_mask.unsqueeze(2)
        similarity = similarity.masked_fill(mask_peaks == 0, float(0))
        mask_atoms = patch_mask.unsqueeze(1)
        similarity = similarity.masked_fill(mask_atoms == 0, float(0))
        
        # similarity = torch.where(similarity < self.local_threshold, torch.tensor(0.0, device=self.device), similarity)

        local_align_scores = similarity
        
        v_align_weights = local_align_scores / (local_align_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        v_patch_embed_masked = v_patch_embed * patch_mask.unsqueeze(-1).float() 
        
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed_masked)

        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)
        l_token_embed_normalized = F.normalize(l_token_embed, p=2, dim=-1)

        # 最终的局部相似性分数：取 l_token_embed_normalized 和 l_grouped_v_patch_embed 之间的平均点积
        # 它代表了 **spec 峰 token** 与其 **对齐的原子/碎片特征** 之间的相似度
        
        # 计算所有有效 token 的点积
        # (bs, l1, dim) -> (bs, l1)
        local_token_dot_product = torch.sum(l_grouped_v_patch_embed * l_token_embed_normalized, dim=-1)

        row_sum = (local_token_dot_product * language_mask).sum(dim=1)
        valid_count = language_mask.sum(dim=1)
        local_sim = torch.where(valid_count > 0, row_sum / valid_count, torch.zeros_like(row_sum))

        return global_sim, local_sim, mol_global, spec_global, local_align_scores

    def predict_smi(self, root_reprs, adducts, ces, ind_maps=None):
        if self.emb_adducts and self.inject_early:
            atoms_emb, mol_emb = self.mol_encoder(root_reprs, adducts)
        else:
            atoms_emb, mol_emb = self.mol_encoder(root_reprs)
            if self.emb_adducts or self.embed_ce:
                embed_adducts = self.adduct_embedder[adducts.long()]  # (batch_size, adduct_dim)
                if ind_maps is None:
                    root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                    embed_adducts_expand = embed_adducts.repeat_interleave(root_batch_sizes, 0)
                else:
                    adducts_mapped = embed_adducts[ind_maps]
                    root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                    embed_adducts_expand = adducts_mapped.repeat_interleave(
                        root_batch_sizes, dim=0
                    )
                mol_emb = torch.cat([mol_emb, embed_adducts], -1)
                
                root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_batch_sizes, dim=0
                    )
                # atoms_emb = torch.cat([atoms_emb, embed_adducts_expand], -1)
                
                if self.embed_ce:
                    embed_collision = torch.cat(
                        (torch.sin(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                            torch.cos(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                        dim=1
                    )
                    embed_collision = torch.where(  # handle entries without collision energy (== nan)
                        torch.isnan(embed_collision), self.collision_embed_merged.unsqueeze(0), embed_collision
                    )
                    mol_emb = torch.cat([mol_emb, embed_collision], -1)
                    
                    root_batch_sizes = torch.bincount(root_reprs.batch) if root_reprs.batch is not None else torch.tensor([root_reprs.num_nodes])
                    embed_collision_expand = embed_collision.repeat_interleave(
                            root_batch_sizes, dim=0
                        )
                    # atoms_emb = torch.cat([atoms_emb, embed_collision_expand], -1)
                
        mol_proj = self.mol_projection(mol_emb)
        atom_proj = atoms_emb
        # atom_proj = self.atom_projection(atoms_emb)
        mol_proj = F.normalize(mol_proj, dim=1)
        # atom_proj = F.normalize(atom_proj)
        
        return mol_proj, atom_proj
    
    def predict_spec(self, specs, adducts, specs_mask=None):
        if self.emb_adducts and self.inject_early:
            spec_emb = self.spec_encoder(specs)
            # spec_global_emb = (spec_emb * specs_mask.unsqueeze(-1)).sum(dim=1) / \
            #     specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            # spec_global_emb = spec_emb[:, 0, :]
        else:
            spec_emb = self.spec_encoder(specs)
            if self.emb_adducts:
                embed_adducts = self.adduct_embedder[adducts.long()]
                spec_global_emb = spec_emb[:, 0, :]
                # spec_global_emb = torch.cat([spec_global_emb, embed_adducts], -1)
                
        # spec_proj = self.spec_projection(spec_global_emb)
        spec_proj = self.spec_projection(spec_emb)
        if self.pooling_strategy == 'mean':
            # Mean Pooling on projected sequence
            spec_global_emb = (spec_proj * specs_mask.unsqueeze(-1)).sum(dim=1) / \
                                specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        elif self.pooling_strategy == 'cls':
            # CLS token is at index 0
            spec_global_emb = spec_proj[:, 0, :]
        spec_proj = F.normalize(spec_proj, dim=2)
        spec_global_emb = F.normalize(spec_global_emb, dim=1)
        
        return spec_global_emb, spec_proj
    
    def training_step(self, batch, batch_idx):
        spec_proj, mol_proj, frags_emb, spec_emb, decoys_proj = self.forward(
            batch['specs'], batch['root_reprs'], batch['adducts'],
            decoys_reprs=batch.get('decoys_reprs', None),
            decoys_adducts=batch.get('decoys_adducts', None),
            frags_reprs=batch.get('frag_graphs', None),
            ind_maps=batch.get('inds', None),
            specs_mask=batch.get('specs_mask', None),
            ces=batch.get("ces", None)
        )

        spec_proj = F.normalize(spec_proj, dim=1)   # (B, D)
        mol_proj  = F.normalize(mol_proj,  dim=1)   # (B, D)
        B = spec_proj.shape[0]
        
        if self.spec_sim_entropy:
            sim_matrix = batch["spec_sim_matrix"]
            target_temperature = self.temperature 
            targets = F.softmax(sim_matrix / target_temperature, dim=1)
            logits = torch.mm(spec_proj, mol_proj.t()) / self.temperature
            log_probs_s2m = F.log_softmax(logits, dim=1)
            loss_inbatch_s2m = -(targets * log_probs_s2m).sum(dim=1).mean()
            log_probs_m2s = F.log_softmax(logits.t(), dim=1)
            loss_inbatch_m2s = -(targets * log_probs_m2s).sum(dim=1).mean()
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2
        else:
            temperature = self.temperature

            # ====================== 1. 标准 in-batch contrastive loss ======================
            logits = torch.mm(spec_proj, mol_proj.t()) / temperature  # (B, B)

            if 'same_molecule_mask' in batch:
                mask = batch['same_molecule_mask'].clone()
                mask.fill_diagonal_(False)
                logits = logits.masked_fill(mask, float('-inf'))
                same_molecule = batch['same_molecule_mask'].sum().item() - B
                self.log("train/same_molecule", same_molecule)

            labels = torch.arange(B, device=self.device)
            loss_inbatch_s2m = F.cross_entropy(logits, labels)
            loss_inbatch_m2s = F.cross_entropy(logits.t(), labels)
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2

        # Apply global loss only if wo_global is False
        if self.wo_global:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = loss_global

        if self.decoys and decoys_proj is not None and 'decoy_sims' in batch:
            decoys_proj = F.normalize(decoys_proj, dim=1)                    # (N_total, D)
            decoys_batch = batch["decoys_batch"].long()                      # (N_total,)
            decoy_sims   = batch["decoy_sims"]                               # (N_total,)

            alpha = getattr(self.hparams, "hard_negative_alpha", 1.0)
            beta  = getattr(self.hparams, "decoy_loss_weight", 1.0)
            # (B, N_total)
            decoy_logits = torch.mm(spec_proj, decoys_proj.t()) / temperature

            # 只保留每个 spec 自己的 decoys
            row_idx = torch.arange(B, device=spec_proj.device)
            own_mask = (decoys_batch[None, :] == row_idx[:, None])           # (B, N_total)
            decoy_logits = decoy_logits.masked_fill(~own_mask, float('-inf'))

            # # Hard negative weighting
            # weight_bias = (alpha * decoy_sims).log1p()
            # decoy_logits = decoy_logits + weight_bias.unsqueeze(0)

            pos_logits = torch.diagonal(logits)  # (B,)
            decoy_loss_per_sample = torch.zeros(B, device=self.device)
            for i in range(B):
                own_decoy_logits = decoy_logits[i][own_mask[i]]
                if own_decoy_logits.numel() > 0:
                    # (1, 1 + n_decoys)
                    sample_logits = torch.cat([pos_logits[i].unsqueeze(0), own_decoy_logits])
                    sample_logits = sample_logits.unsqueeze(0)
                    sample_label = torch.zeros(1, dtype=torch.long, device=self.device)
                    decoy_loss_per_sample[i] = F.cross_entropy(sample_logits, sample_label)

            loss_decoy = decoy_loss_per_sample.sum()
            self.log("train/loss_decoy", loss_decoy, prog_bar=True)
            self.log("train/loss_decoy_weighted", beta * loss_decoy)

            with torch.no_grad():
                max_decoy = decoy_logits.max(dim=1).values
                has_decoy = own_mask.any(dim=1)
                if has_decoy.any():
                    self.log("train/max_decoy_logit", max_decoy[has_decoy].mean())
                    self.log("train/decoy_beats_pos_rate",
                            (max_decoy[has_decoy] > pos_logits[has_decoy]).float().mean())
                    self.log("train/frac_with_decoys", has_decoy.float().mean())

            loss += beta * loss_decoy
        else:
            loss_decoy = torch.tensor(0.0, device=self.device)

        # ====================== Final logging ======================
        self.log("train/loss_global", loss_global, prog_bar=True)
        self.log("train/spec2mol_global", loss_inbatch_s2m)
        self.log("train/mol2spec_global", loss_inbatch_m2s)

        # local contrastive
        if self.local_contra:
            if self.frags:
                frags_emb, frags_mask = self.group_tensor_to_batch(frags_emb, batch['inds'])
            else:
                frags_emb, frags_mask = self.group_tensor_to_batch(frags_emb, batch['root_reprs'].batch)
            frags_mask = frags_mask.to(self.device)
            if self.pooling_strategy == 'cls':
                spec_emb = spec_emb[:, 1:, :]
                specs_mask = batch['specs_mask'][:, 1:]
            frag_labels=None
            frag_masks=None
            if self.frag_supervised:
                frag_labels = batch["frag_labels"]
                frag_masks = batch["frag_masks"]
            loss_local, frag_supervised_loss = self.compute_local_loss(spec_emb, frags_emb,
                                                specs_mask, frags_mask, similarity_threshold=self.local_threshold, frag_labels=frag_labels, frag_masks=frag_masks)
            self.log('train/loss_local', loss_local, prog_bar=True)
            loss += loss_local  * self.local_weight
            if self.frag_supervised:
                self.log('train/loss_frag_supervised', frag_supervised_loss * self.frag_supervised_weight, prog_bar=True)
                loss += frag_supervised_loss * self.frag_supervised_weight

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spec_proj, mol_proj, frags_emb, spec_emb, decoys_proj = self.forward(
                batch['specs'], batch['root_reprs'], batch['adducts'],
                decoys_reprs=batch.get('decoys_reprs', None),
                decoys_adducts=batch.get('decoys_adducts', None),
                frags_reprs=batch.get('frag_graphs', None),
                ind_maps=batch.get('inds', None),
                specs_mask=batch.get('specs_mask', None),
                ces=batch.get("ces", None)
            )

        spec_proj = F.normalize(spec_proj, dim=1)   # (B, D)
        mol_proj  = F.normalize(mol_proj,  dim=1)   # (B, D)
        B = spec_proj.shape[0]
        
        if self.spec_sim_entropy:
            sim_matrix = batch["spec_sim_matrix"]
            target_temperature = self.temperature 
            targets = F.softmax(sim_matrix / target_temperature, dim=1)
            logits = torch.mm(spec_proj, mol_proj.t()) / self.temperature
            log_probs_s2m = F.log_softmax(logits, dim=1)
            loss_inbatch_s2m = -(targets * log_probs_s2m).sum(dim=1).mean()
            log_probs_m2s = F.log_softmax(logits.t(), dim=1)
            loss_inbatch_m2s = -(targets * log_probs_m2s).sum(dim=1).mean()
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2
        else:
            temperature = self.temperature
            # ====================== 1. 标准 in-batch contrastive loss ======================
            logits = torch.mm(spec_proj, mol_proj.t()) / temperature  # (B, B)

            if 'same_molecule_mask' in batch:
                mask = batch['same_molecule_mask'].clone()
                mask.fill_diagonal_(False)
                logits = logits.masked_fill(mask, float('-inf'))
                same_molecule = batch['same_molecule_mask'].sum().item() - B
                self.log("train/same_molecule", same_molecule)

            labels = torch.arange(B, device=self.device)
            loss_inbatch_s2m = F.cross_entropy(logits, labels)
            loss_inbatch_m2s = F.cross_entropy(logits.t(), labels)
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2

        # Apply global loss only if wo_global is False
        if self.wo_global:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = loss_global

        self.log("val/loss_global", loss_global, prog_bar=True)     # batch size ↑, negative samples ↑, so loss ↑
        self.log("val/normalized_loss", loss_global / torch.log(torch.tensor(B).float()), prog_bar=True)    # val normalized loss is not dependent on batch size
        
        if self.decoys and decoys_proj is not None and 'decoy_sims' in batch:
            decoys_proj = F.normalize(decoys_proj, dim=1)                    # (N_total, D)
            decoys_batch = batch["decoys_batch"].long()                      # (N_total,)
            # decoy_sims   = batch["decoy_sims"]                               # (N_total,)

            alpha = getattr(self.hparams, "hard_negative_alpha", 1.0)
            beta  = getattr(self.hparams, "decoy_loss_weight", 1.0)           # 新增：控制 decoy 强度

            # (B, N_total)
            decoy_logits = torch.mm(spec_proj, decoys_proj.t()) / temperature

            # 只保留每个 spec 自己的 decoys
            row_idx = torch.arange(B, device=spec_proj.device)
            own_mask = (decoys_batch[None, :] == row_idx[:, None])           # (B, N_total)
            decoy_logits = decoy_logits.masked_fill(~own_mask, float('-inf'))

            # Hard negative weighting
            # weight_bias = (alpha * decoy_sims).log1p()
            # decoy_logits = decoy_logits
            # decoy_logits = decoy_logits + weight_bias.unsqueeze(0)

            # 正样本 logit（来自 in-batch）
            pos_logits = torch.diagonal(logits)  # (B,)

            # 为每个 sample 单独计算：logits = [pos_logit, own_decoy1, own_decoy2, ...]
            decoy_loss_per_sample = torch.zeros(B, device=self.device)
            for i in range(B):
                own_decoy_logits = decoy_logits[i][own_mask[i]]
                if own_decoy_logits.numel() > 0:
                    # (1, 1 + n_decoys)
                    sample_logits = torch.cat([pos_logits[i].unsqueeze(0), own_decoy_logits])
                    sample_logits = sample_logits.unsqueeze(0)   # (1, 1+n)
                    sample_label = torch.zeros(1, dtype=torch.long, device=self.device)
                    decoy_loss_per_sample[i] = F.cross_entropy(sample_logits, sample_label)
                # else: remains 0

            loss_decoy = decoy_loss_per_sample.sum()
            # loss_decoy = decoy_loss_per_sample.mean()
            self.log("val/loss_decoy", loss_decoy, prog_bar=True)
            self.log("val/loss_decoy_weighted", beta * loss_decoy)

            # 监控指标
            with torch.no_grad():
                max_decoy = decoy_logits.max(dim=1).values
                has_decoy = own_mask.any(dim=1)
                if has_decoy.any():
                    self.log("val/max_decoy_logit", max_decoy[has_decoy].mean())
                    self.log("val/decoy_beats_pos_rate",
                            (max_decoy[has_decoy] > pos_logits[has_decoy]).float().mean())
                    self.log("val/frac_with_decoys", has_decoy.float().mean())

            # loss += beta * loss_decoy
        else:
            loss_decoy = torch.tensor(0.0, device=self.device)

        # local contrastive
        if self.local_contra:
            if self.frags:
                frags_emb, frags_mask = self.group_tensor_to_batch(frags_emb, batch['inds'])    # subgraph
            else:
                frags_emb, frags_mask = self.group_tensor_to_batch(frags_emb, batch['root_reprs'].batch)    # atoms
            frags_mask = frags_mask.to(self.device)
            if self.pooling_strategy == 'cls':
                spec_emb = spec_emb[:, 1:, :]
                specs_mask = batch['specs_mask'][:, 1:]
            frag_labels=None
            frag_masks=None
            if self.frag_supervised:
                frag_labels = batch["frag_labels"]
                frag_masks = batch["frag_masks"]
            loss_local, frag_supervised_loss = self.compute_local_loss(spec_emb, frags_emb,
                                                specs_mask, frags_mask, similarity_threshold=self.local_threshold, frag_labels=frag_labels, frag_masks=frag_masks)
            self.log('val/loss_local', loss_local * self.local_weight)
            if self.frag_supervised:
                self.log('val/loss_local', frag_supervised_loss * self.frag_supervised_weight)
                # loss += frag_supervised_loss * self.frag_supervised_weight
        if self.wo_global:
            loss = loss_local
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """configure_optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
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
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "value",
        }
        return ret
    
    # def configure_optimizers(self):
    #     """configure_optimizers."""
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(), 
    #         lr=self.lr, 
    #         weight_decay=self.weight_decay
    #     )        
    #     def lr_lambda(step):
    #         if step < self.warmup:
    #             return float(step) / float(max(1, self.warmup))
            
    #         progress = float(step - self.warmup) / float(max(1, self.total_steps - self.warmup))
    #         return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

    #     scheduler = {
    #         'scheduler': LambdaLR(optimizer, lr_lambda),
    #         'interval': 'step',
    #         'frequency': 1,
    #         'name': 'learning_rate'
    #     }
        
    #     ret = {
    #         "optimizer": optimizer,
    #         "lr_scheduler": scheduler,
    #         "gradient_clip_val": 1.0,
    #         "gradient_clip_algorithm": "value",
    #     }
    #     return ret
    
    def on_train_epoch_start(self):
        if self.current_epoch >= self.unfreeze_epoch:
            print(f"Unfreezing spec_encoder at epoch {self.current_epoch}")
            for param in self.spec_encoder.parameters():
                param.requires_grad = True
            self.spec_encoder.train()
        
        if self.current_epoch >= self.local_start_epoch and self.local_flag:
            self.local_contra = True
                
    # def group_tensor_to_batch(self, tensor1, batch):
    #     _, counts = torch.unique_consecutive(batch, return_counts=True)
    #     groups = torch.split(tensor1, counts.tolist())
    #     padded = pad_sequence(groups, batch_first=True, padding_value=0.0)
        
    #     # Create a 2D mask of shape (bs, max_len) with 1.0 for valid positions and 0.0 for padding
    #     mask_groups = [torch.ones(count) for count in counts.tolist()]
    #     padded_mask = pad_sequence(mask_groups, batch_first=True, padding_value=0.0)
        
    #     return padded, padded_mask
    
    def group_tensor_to_batch(self, tensor1, batch):
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        groups = torch.split(tensor1, counts.tolist())
        
        # 计算本地最大长度
        local_max_len = max(len(g) for g in groups) if groups else 0
        
        # 如果是分布式模式，all-gather 所有 GPU 的本地 max_len，并取全局 max
        # PyTorch 1.9 兼容的语法
        if dist.is_initialized() and dist.get_world_size() > 1:
            local_max_tensor = torch.tensor([local_max_len], dtype=torch.long, device=self.device)
            all_maxes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(dist.get_world_size())]  # 创建输出列表
            dist.all_gather(all_maxes, local_max_tensor)  # PyTorch 1.9 的 all_gather 语法
            global_max_len = max(m.item() for m in all_maxes)
        else:
            global_max_len = local_max_len
        
        # 填充到 global_max_len
        padded = pad_sequence(groups, batch_first=True, padding_value=0.0)
        if padded.shape[1] < global_max_len:
            pad_amount = global_max_len - padded.shape[1]
            padded = nn.functional.pad(padded, (0, 0, 0, pad_amount), value=0.0)  # 在序列维度 (dim=1) 填充
        
        # 创建掩码（1 for valid, 0 for padding）
        mask_groups = [torch.ones(len(g), device=self.device) for g in groups]  # 确保掩码在设备上
        padded_mask = pad_sequence(mask_groups, batch_first=True, padding_value=0.0)
        if padded_mask.shape[1] < global_max_len:
            pad_amount = global_max_len - padded_mask.shape[1]
            padded_mask = nn.functional.pad(padded_mask, (0, pad_amount), value=0.0)  # 掩码是 (bs, seq)，填充序列维度
        
        return padded, padded_mask

    def masked_pairwise_contrastive_loss(self, a, b, mask, inverse_temperature=1.0):
        batch_size, seq_len = a.shape[0], a.shape[1]
        expanded_mask = mask[:, None, :].float() * mask[:, :, None].float()
        mask_logits = (1.0 - expanded_mask).contiguous().view(batch_size * seq_len, -1)
        # mask_logits = torch.einsum('bnm->(bn)m', 1.0 - mask, n=seq_len)
        # labels = torch.einsum('ns->(bn)s', torch.eye(a.shape[1]), b=batch_size)
        eye_matrix = torch.eye(seq_len)  # (l, l)
        labels = eye_matrix.unsqueeze(0).repeat(batch_size, 1, 1)  # (bs, l, l)
        labels = labels.reshape(batch_size * seq_len, -1).to(self.device)  # (bs * l, l)，-1 

        logits = torch.einsum('bmd,bnd->bnm', a, b) * inverse_temperature   # (bs, l, l)
        # logits = torch.einsum('bnm->(bn)m', logits)
        logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
        # pred = logits - mask_logits * float('-inf')
        mask_bool = (mask_logits == 1).bool()
        preds = logits.masked_fill(mask_bool, float('1e-9'))
        labels_indices = labels.argmax(dim=-1)
        
        valid_mask = mask.reshape(-1) == True
        loss = F.cross_entropy(preds, labels_indices, reduction='none')
        loss = loss[valid_mask].sum() / valid_mask.sum().clamp(min=1e-8).float()
        # loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1e-8)  # Avoid division by zero
        return loss

    def compute_frag_loss(self, similarity, frag_labels):
        """
        利用碎片监督信息来约束相似度矩阵。
        """
        # ==================== Fix Start ====================
        # 获取预测矩阵的原子维度 (Global Max, e.g., 69)
        bs, num_peaks, num_atoms_pred = similarity.shape
        # 获取标签矩阵的原子维度 (Local Max, e.g., 67)
        _, num_frags, num_atoms_target = frag_labels.shape

        # 如果预测维度的原子数 > 标签维度的原子数 (DDP 场景常见)
        if num_atoms_pred > num_atoms_target:
            diff = num_atoms_pred - num_atoms_target
            # 在最后一个维度 (Atoms) 的右侧进行 padding
            # 使用 -1 进行填充 (保持与 dataset padding_value 一致)
            frag_labels = F.pad(frag_labels, (0, diff), value=-1)
        elif num_atoms_pred < num_atoms_target:
            diff = num_atoms_target - num_atoms_pred
            # 如果预测维度的原子数 < 标签维度的原子数 (fragment graph atoms < molecule atoms)
            similarity = F.pad(similarity, (0, diff), value=0)
        # ==================== Fix End ====================

        # 1. 处理 frag_labels (Ground Truth)
        # 将 -1 和 0 都视为 0，只关注 1 (存在的原子)
        target_mask = (frag_labels == 1).float()  # (bs, F, A)
        
        # ... (后续代码保持不变) ...
        
        # 创建有效碎片掩码
        valid_frag_mask = (frag_labels == 1).sum(dim=-1) > 0 
        
        target_dist = target_mask / (target_mask.sum(dim=-1, keepdim=True) + 1e-8)

        # 2. 处理 Similarity (Prediction)
        pred_attn = similarity
        
        pred_dist = pred_attn / (pred_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. 计算对齐分数矩阵
        # (bs, P, A) @ (bs, F, A)^T -> (bs, P, F)
        align_matrix = torch.einsum('bpa,bfa->bpf', pred_dist, target_dist)

        # 4. 计算 Loss
        best_match_scores, _ = align_matrix.max(dim=1)  # (bs, F)
        nll_loss = -torch.log(best_match_scores.clamp(min=1e-8))
        
        loss = (nll_loss * valid_frag_mask.float()).sum() / valid_frag_mask.sum().clamp(min=1.0)
        
        return loss
    
    # def compute_local_loss(self, l_token_embed, v_patch_embed, language_mask, patch_mask, 
    #                         inverse_temperature=1.0, similarity_threshold=0.5, 
    #                         frag_labels=None, frag_masks=None):
    #     """
    #     l_token_embed: (BS, Atoms, D) - Mol (Language in classic CLIP)
    #     v_patch_embed: (BS, Peaks, D) - Spec (Image Patch in classic CLIP)
    #     """
    #     bs, num_atoms, dim = l_token_embed.shape
    #     _, num_peaks, _ = v_patch_embed.shape

    #     scale = 1.0 / (dim ** 0.5) 
        
    #     raw_sim = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)
        
    #     attn_logits = raw_sim * scale
        
    #     mask_peaks = language_mask.unsqueeze(2)
    #     attn_logits = attn_logits.masked_fill(mask_peaks == 0, float('-inf'))
    #     mask_atoms = patch_mask.unsqueeze(1)
    #     attn_logits = attn_logits.masked_fill(mask_atoms == 0, float('-inf'))
        
    #     attn_map = F.softmax(attn_logits, dim=1) # (BS, Atoms, Peaks)

    #     frag_loss = torch.tensor(0.0, device=self.device)
    #     if self.frag_supervised and frag_labels is not None:
    #         frag_loss = self.compute_frag_loss(attn_map, frag_labels)

    #     l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', attn_map, v_patch_embed)

    #     l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)
    #     l_token_embed = F.normalize(l_token_embed, p=2, dim=-1)

    #     temp = 20.0
        
    #     loss_vl_local = self.masked_pairwise_contrastive_loss(
    #         l_grouped_v_patch_embed, l_token_embed, language_mask, inverse_temperature=temp
    #     )
    #     loss_lv_local = self.masked_pairwise_contrastive_loss(
    #         l_token_embed, l_grouped_v_patch_embed, language_mask, inverse_temperature=temp
    #     )

    #     local_loss = (loss_vl_local + loss_lv_local) * 0.5

    #     return local_loss, frag_loss

    def compute_local_loss(self, l_token_embed, v_patch_embed, language_mask, patch_mask, inverse_temperature=5.0, similarity_threshold=0.3, frag_labels=None, frag_masks=None):
        # Ensure inputs have correct shapes: (bs, l1, dim) and (bs, l2, dim) for embeds, (bs, l1) and (bs, l2) for masks
        bs, l1, dim = l_token_embed.shape
        _, l2, _ = v_patch_embed.shape

        # Similarity calculation
        similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)

        # Min-max normalization
        min_sim = similarity.min(dim=-1, keepdim=True)[0]
        max_sim = similarity.max(dim=-1, keepdim=True)[0]
        similarity = (similarity - min_sim) / (max_sim - min_sim + 1e-8)
        mask_peaks = language_mask.unsqueeze(2)
        similarity = similarity.masked_fill(mask_peaks == 0, float(0))
        mask_atoms = patch_mask.unsqueeze(1)
        similarity = similarity.masked_fill(mask_atoms == 0, float(0))

        frag_loss = torch.tensor(0.0, device=self.device)
        if self.frag_supervised and (self.current_epoch < self.local_start_epoch + 5):
            frag_loss = self.compute_frag_loss(similarity, frag_labels)
        
        # Thresholding
        similarity = torch.where(similarity < similarity_threshold, torch.tensor(0.0, device=self.device), similarity)

        # Alignment weighting
        v_align_weights = similarity / (similarity.sum(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero
        """TO DO add random noise?"""
        
        # Apply patch_mask to v_patch_embed
        v_patch_embed_masked = v_patch_embed * patch_mask.unsqueeze(-1).float()  # Broadcast patch_mask to (bs, l2, dim)

        # Grouped v_patch_embed with alignment weights
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed_masked)

        # L2 normalization
        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)
        l_token_embed = F.normalize(l_token_embed, p=2, dim=-1)

        # joint_mask = language_mask.unsqueeze(1) * patch_mask.unsqueeze(2)
        # Contrastive loss with masks
        loss_vl_local = self.masked_pairwise_contrastive_loss(l_grouped_v_patch_embed, l_token_embed, language_mask, inverse_temperature=inverse_temperature)
        loss_lv_local = self.masked_pairwise_contrastive_loss(l_token_embed, l_grouped_v_patch_embed, language_mask, inverse_temperature=inverse_temperature)

        # Combined loss
        local_loss = (loss_vl_local + loss_lv_local) * 0.5

        return local_loss, frag_loss