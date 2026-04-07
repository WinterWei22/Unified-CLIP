"""
CLIP Model variant using UniMol2 as the molecular encoder.

Architecture:
- Spectrum Encoder: DreaMS (pretrained, optionally frozen)
- Molecule Encoder: UniMol2 (pretrained, optionally frozen)
- Projection heads: Same MLP structure as original CLIP model
- Loss: Same contrastive losses (global + local + decoy)

Key differences from clip_model.py:
- mol_encoder is UniMolEncoder (768-dim output for 84m model)
- forward() takes batch_dict (UniMol2 features) instead of PyG graph
- atom embeddings come from UniMol2's transformer, not EGT
- batch tracking uses UniMolEncoder's returned batch_indices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ms_pred.DreaMS.dreams.models.dreams.dreams import DreaMS as DreaMSModel
from ms_pred.DreaMS.dreams.api import PreTrainedModel
from ms_pred.clip.ms_model import MassSpecTransformer
from ms_pred.clip.unimol_encoder import UniMolEncoder, UNIMOL2_EMBED_DIMS

import ms_pred.nn_utils as nn_utils
import ms_pred.common as common
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import math


class CLIPModelUniMol(pl.LightningModule):
    def __init__(
        self,
        # UniMol2 params
        unimol_model_size='84m',
        frozen_unimol=False,
        unfreeze_unimol_epoch=0,
        # Projection params
        projection_dim=512,
        temperature=0.07,
        dropout=0.1,
        # Spectrum encoder params
        spec_ckpt_path=None,
        dreams=True,
        pooling_strategy='cls',
        frozen_dreams=False,
        unfreeze_epoch=10,
        # Training params
        lr=3e-4,
        lr_decay_rate=0.825,
        warmup=1000,
        weight_decay=1e-5,
        total_steps=0,
        # Feature flags
        emb_adducts=False,
        embed_ce=False,
        local_contra=False,
        local_weight=0.5,
        local_threshold=0.5,
        local_start_epoch=0,
        decoys=False,
        decoy_loss_weight=5.0,
        triplet_margin=0.2,
        triplet_alpha=0.8,
        decoys_temp=0.05,
        frag_supervised=False,
        frag_supervised_weight=0.3,
        spec_sim_entropy=False,
        wo_global=False,
        # Queue for extra contrastive negatives (MoCo-style)
        queue_size=0,
        # Unused but kept for compat
        hidden_size=512,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Embedding queue for extra contrastive negatives
        self.queue_size = queue_size
        if queue_size > 0:
            self.register_buffer("queue_spec", torch.zeros(queue_size, projection_dim))
            self.register_buffer("queue_mol", torch.zeros(queue_size, projection_dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.wo_global = wo_global
        self.emb_adducts = emb_adducts
        self.embed_ce = embed_ce
        self.local_contra = local_contra
        self.local_weight = local_weight
        self.local_threshold = local_threshold
        self.local_flag = local_contra
        self.local_start_epoch = local_start_epoch
        if self.local_start_epoch > 0:
            self.local_contra = False

        self.decoys = decoys
        self.decoy_loss_weight = decoy_loss_weight
        self.triplet_margin = triplet_margin
        self.triplet_alpha = triplet_alpha
        self.decoys_temp = decoys_temp
        self.dreams = dreams
        self.pooling_strategy = pooling_strategy
        self.frozen_dreams = frozen_dreams
        self.unfreeze_epoch = unfreeze_epoch
        self.spec_sim_entropy = spec_sim_entropy
        self.frag_supervised = frag_supervised
        self.frag_supervised_weight = frag_supervised_weight
        self.frozen_unimol = frozen_unimol
        self.unfreeze_unimol_epoch = unfreeze_unimol_epoch

        # ============= Spectrum Encoder =============
        if self.dreams and spec_ckpt_path:
            spec_output_dim = 1024
            # Load to CPU first to avoid OOM when multiple DDP processes
            # all try to load onto GPU 0 simultaneously
            ckpt = DreaMSModel.load_from_checkpoint(spec_ckpt_path, map_location='cpu')
            pretrained = PreTrainedModel(ckpt, n_highest_peaks=60)
            pretrained.model = PreTrainedModel.remove_unused_backbone_parameters(pretrained.model)
            self.spec_encoder = pretrained.model.eval()
            if self.frozen_dreams:
                for param in self.spec_encoder.parameters():
                    param.requires_grad = False
        else:
            spec_output_dim = hidden_size
            self.spec_encoder = MassSpecTransformer(
                dim=hidden_size, nhead=4, num_layers=4,
                mz_min=0.0, mz_max=1000.0,
                use_cls_token=True, use_seq_pe=False
            )

        # ============= Molecular Encoder (UniMol2) =============
        self.mol_encoder = UniMolEncoder(
            model_size=unimol_model_size,
            frozen=frozen_unimol,
            frozen_epochs=unfreeze_unimol_epoch,
        )
        mol_output_dim = self.mol_encoder.embed_dim  # 768 for 84m

        # ============= Adduct & CE embeddings =============
        if self.emb_adducts:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            mol_output_dim += adduct_types

        if self.embed_ce:
            pe_dim = common.COLLISION_PE_DIM
            pe_scalar = common.COLLISION_PE_SCALAR
            pe_power = 2 * torch.arange(pe_dim // 2) / pe_dim
            self.collision_embedder_denominators = nn.Parameter(torch.pow(pe_scalar, pe_power))
            self.collision_embedder_denominators.requires_grad = False
            self.collision_embed_merged = nn.Parameter(torch.zeros(pe_dim))
            self.collision_embed_merged.requires_grad = False
            mol_output_dim += pe_dim

        # ============= Projection Heads =============
        self.spec_projection = nn.Sequential(
            nn.Linear(spec_output_dim, projection_dim),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.mol_projection = nn.Sequential(
            nn.Linear(mol_output_dim, projection_dim),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        # atom_projection input is mol_output_dim (with adduct/ce concat)
        self.atom_projection = nn.Sequential(
            nn.Linear(mol_output_dim, projection_dim),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        if self.decoys:
            self.decoys_projection = nn.Sequential(
                nn.Linear(mol_output_dim, projection_dim),
                nn.Dropout(dropout), nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
                nn.Dropout(dropout), nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )

        self.temperature = temperature
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup
        self.dropout_rate = dropout
        self.weight_decay = weight_decay
        self.total_steps = total_steps

    def _move_mol_feats_to_device(self, mol_feats):
        """Move UniMol2 feature dict tensors to model device."""
        return {k: v.to(self.device) for k, v in mol_feats.items()}

    def _encode_mol(self, mol_feats):
        """Encode molecules using UniMol2.

        Returns:
            atoms_emb: (total_atoms, embed_dim)
            mol_emb: (batch_size, embed_dim)
            batch_indices: (total_atoms,) mapping atoms to batch index
        """
        mol_feats = self._move_mol_feats_to_device(mol_feats)
        atoms_emb, mol_emb, batch_indices = self.mol_encoder(mol_feats)
        return atoms_emb, mol_emb, batch_indices

    def forward(self, specs, mol_feats, adducts, decoys_feats=None,
                decoys_adducts=None, specs_mask=None, ces=None):
        # Encode spectrum
        spec_emb = self.spec_encoder(specs)
        spec_proj = self.spec_projection(spec_emb)
        if self.pooling_strategy == 'mean':
            spec_global_emb = (spec_proj * specs_mask.unsqueeze(-1)).sum(dim=1) / \
                specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        elif self.pooling_strategy == 'cls':
            spec_global_emb = spec_proj[:, 0, :]

        # Encode molecule with UniMol2
        atoms_emb, mol_emb, batch_indices = self._encode_mol(mol_feats)

        # Adduct embedding
        if self.emb_adducts:
            embed_adducts = self.adduct_embedder[adducts.long()]
            mol_emb = torch.cat([mol_emb, embed_adducts], -1)
            # Expand adducts for atom-level
            embed_adducts_expand = embed_adducts[batch_indices]
            atoms_emb = torch.cat([atoms_emb, embed_adducts_expand], -1)

        # Collision energy embedding
        if self.embed_ce and ces is not None:
            embed_collision = torch.cat(
                (torch.sin(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                 torch.cos(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                dim=1
            )
            embed_collision = torch.where(
                torch.isnan(embed_collision),
                self.collision_embed_merged.unsqueeze(0),
                embed_collision
            )
            mol_emb = torch.cat([mol_emb, embed_collision], -1)
            embed_collision_expand = embed_collision[batch_indices]
            atoms_emb = torch.cat([atoms_emb, embed_collision_expand], -1)

        # Project
        mol_proj = self.mol_projection(mol_emb)
        atom_proj = self.atom_projection(atoms_emb)

        # Decoys
        decoys_proj = None
        if self.decoys and decoys_feats is not None and decoys_adducts is not None:
            decoys_atoms_emb, decoys_mol_emb, _ = self._encode_mol(decoys_feats)
            if self.emb_adducts:
                decoys_embed_adducts = self.adduct_embedder[decoys_adducts.long()]
                decoys_mol_emb = torch.cat([decoys_mol_emb, decoys_embed_adducts], -1)
            if self.embed_ce and ces is not None:
                decoys_mol_emb = torch.cat([decoys_mol_emb, embed_collision], -1)
            decoys_proj = self.decoys_projection(decoys_mol_emb)

        return spec_global_emb, mol_proj, atom_proj, spec_proj, decoys_proj, batch_indices

    def predict_smi(self, mol_feats, adducts, ces):
        atoms_emb, mol_emb, batch_indices = self._encode_mol(mol_feats)
        if self.emb_adducts:
            embed_adducts = self.adduct_embedder[adducts.long()]
            mol_emb = torch.cat([mol_emb, embed_adducts], -1)
        if self.embed_ce and ces is not None:
            embed_collision = torch.cat(
                (torch.sin(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                 torch.cos(ces.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                dim=1
            )
            embed_collision = torch.where(
                torch.isnan(embed_collision),
                self.collision_embed_merged.unsqueeze(0),
                embed_collision
            )
            mol_emb = torch.cat([mol_emb, embed_collision], -1)
        mol_proj = self.mol_projection(mol_emb)
        mol_proj = F.normalize(mol_proj, dim=1)
        return mol_proj, atoms_emb

    def predict_spec(self, specs, adducts, specs_mask=None):
        spec_emb = self.spec_encoder(specs)
        spec_proj = self.spec_projection(spec_emb)
        if self.pooling_strategy == 'mean':
            spec_global_emb = (spec_proj * specs_mask.unsqueeze(-1)).sum(dim=1) / \
                specs_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        elif self.pooling_strategy == 'cls':
            spec_global_emb = spec_proj[:, 0, :]
        spec_proj = F.normalize(spec_proj, dim=2)
        spec_global_emb = F.normalize(spec_global_emb, dim=1)
        return spec_global_emb, spec_proj

    @staticmethod
    def _gather_with_grad(tensor):
        """Gather tensors across all DDP processes, keeping gradients for the local shard."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # Gather without gradients
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.detach())
        # Replace the local shard with the original (gradient-bearing) tensor
        gathered[rank] = tensor
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def _enqueue(self, spec_proj, mol_proj):
        """Add current batch embeddings to the queue."""
        if self.queue_size <= 0:
            return
        batch_size = spec_proj.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue_spec[ptr:] = spec_proj[:remaining].detach()
            self.queue_mol[ptr:] = mol_proj[:remaining].detach()
            self.queue_spec[:batch_size - remaining] = spec_proj[remaining:].detach()
            self.queue_mol[:batch_size - remaining] = mol_proj[remaining:].detach()
        else:
            self.queue_spec[ptr:ptr + batch_size] = spec_proj.detach()
            self.queue_mol[ptr:ptr + batch_size] = mol_proj.detach()
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def _gather_sim_matrix(self, sim_matrix):
        """Gather and assemble block-diagonal spec_sim_matrix across all GPUs.

        Each GPU has a local (B, B) similarity matrix. The global matrix is
        (B*world_size, B*world_size) with the local blocks on the diagonal and
        -inf elsewhere (so softmax assigns zero mass to cross-GPU entries).
        """
        world_size = dist.get_world_size()
        B = sim_matrix.shape[0]
        gathered = [torch.zeros_like(sim_matrix) for _ in range(world_size)]
        dist.all_gather(gathered, sim_matrix)
        global_B = B * world_size
        # Use -inf for cross-GPU blocks so softmax gives them zero probability
        global_sim = torch.full((global_B, global_B), float('-inf'), device=sim_matrix.device)
        for i, block in enumerate(gathered):
            global_sim[i * B:(i + 1) * B, i * B:(i + 1) * B] = block
        return global_sim

    def training_step(self, batch, batch_idx):
        spec_proj, mol_proj, atom_proj, spec_emb, decoys_proj, batch_indices = self.forward(
            batch['specs'], batch['mol_feats'], batch['adducts'],
            decoys_feats=batch.get('decoys_feats', None),
            decoys_adducts=batch.get('decoys_adducts', None),
            specs_mask=batch.get('specs_mask', None),
            ces=batch.get("ces", None)
        )

        spec_proj = F.normalize(spec_proj, dim=1)
        mol_proj = F.normalize(mol_proj, dim=1)

        # Gather embeddings across all GPUs for more contrastive negatives
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_spec_proj = self._gather_with_grad(spec_proj)
            all_mol_proj = self._gather_with_grad(mol_proj)
        else:
            all_spec_proj = spec_proj
            all_mol_proj = mol_proj
        B = all_spec_proj.shape[0]

        # Queue negatives (no gradients, from previous steps)
        has_queue = self.queue_size > 0 and self.queue_spec.norm(dim=1).sum() > 0
        if has_queue:
            q_spec = self.queue_spec.clone().detach()
            q_mol = self.queue_mol.clone().detach()

        # Global contrastive loss
        if self.spec_sim_entropy:
            sim_matrix = batch["spec_sim_matrix"].float()
            if dist.is_initialized() and dist.get_world_size() > 1:
                sim_matrix = self._gather_sim_matrix(sim_matrix)

            if has_queue:
                logits_s2m = torch.mm(all_spec_proj, torch.cat([all_mol_proj, q_mol], 0).t()) / self.temperature
                logits_m2s = torch.mm(all_mol_proj, torch.cat([all_spec_proj, q_spec], 0).t()) / self.temperature
                Q = q_mol.shape[0]
                sim_padded = F.pad(sim_matrix, (0, Q), value=float('-inf'))
                targets_s2m = F.softmax(sim_padded / self.temperature, dim=1)
                targets_m2s = F.softmax(F.pad(sim_matrix.t(), (0, Q), value=float('-inf')) / self.temperature, dim=1)
                log_probs_s2m = F.log_softmax(logits_s2m.float(), dim=1)
                loss_inbatch_s2m = -(targets_s2m * log_probs_s2m).sum(dim=1).mean()
                log_probs_m2s = F.log_softmax(logits_m2s.float(), dim=1)
                loss_inbatch_m2s = -(targets_m2s * log_probs_m2s).sum(dim=1).mean()
            else:
                targets = F.softmax(sim_matrix / self.temperature, dim=1)
                logits = torch.mm(all_spec_proj, all_mol_proj.t()) / self.temperature
                log_probs_s2m = F.log_softmax(logits.float(), dim=1)
                loss_inbatch_s2m = -(targets * log_probs_s2m).sum(dim=1).mean()
                log_probs_m2s = F.log_softmax(logits.t().float(), dim=1)
                loss_inbatch_m2s = -(targets * log_probs_m2s).sum(dim=1).mean()
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2
        else:
            temperature = self.temperature
            if has_queue:
                logits_s2m = torch.mm(all_spec_proj, torch.cat([all_mol_proj, q_mol], 0).t()) / temperature
                logits_m2s = torch.mm(all_mol_proj, torch.cat([all_spec_proj, q_spec], 0).t()) / temperature
                labels = torch.arange(B, device=self.device)
                loss_inbatch_s2m = F.cross_entropy(logits_s2m, labels)
                loss_inbatch_m2s = F.cross_entropy(logits_m2s, labels)
            else:
                logits = torch.mm(all_spec_proj, all_mol_proj.t()) / temperature
                if 'same_molecule_mask' in batch:
                    mask = batch['same_molecule_mask'].clone()
                    mask.fill_diagonal_(False)
                    logits = logits.masked_fill(mask, float('-inf'))
                labels = torch.arange(B, device=self.device)
                loss_inbatch_s2m = F.cross_entropy(logits, labels)
                loss_inbatch_m2s = F.cross_entropy(logits.t(), labels)
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2

        # Update queue with current batch embeddings
        if self.queue_size > 0:
            self._enqueue(all_spec_proj, all_mol_proj)

        if self.wo_global:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = loss_global

        # Decoy loss (computed on local batch only, since decoys are per-GPU)
        if self.decoys and decoys_proj is not None and 'decoy_sims' in batch and batch['decoy_sims'] is not None:
            decoys_proj = F.normalize(decoys_proj, dim=1)
            decoys_batch = batch["decoys_batch"].long()
            local_B = spec_proj.shape[0]
            beta = getattr(self.hparams, "decoy_loss_weight", 1.0)
            temperature = self.temperature
            decoy_logits = torch.mm(spec_proj, decoys_proj.t()) / temperature
            row_idx = torch.arange(local_B, device=spec_proj.device)
            own_mask = (decoys_batch[None, :] == row_idx[:, None])
            decoy_logits = decoy_logits.masked_fill(~own_mask, float('-inf'))
            # Positive logits: dot product of each local spec with its paired mol
            pos_logits = (spec_proj * mol_proj).sum(dim=1) / temperature
            decoy_loss_per_sample = torch.zeros(local_B, device=self.device)
            for i in range(local_B):
                own_decoy_logits = decoy_logits[i][own_mask[i]]
                if own_decoy_logits.numel() > 0:
                    sample_logits = torch.cat([pos_logits[i].unsqueeze(0), own_decoy_logits]).unsqueeze(0)
                    sample_label = torch.zeros(1, dtype=torch.long, device=self.device)
                    decoy_loss_per_sample[i] = F.cross_entropy(sample_logits, sample_label)
            loss_decoy = decoy_loss_per_sample.sum()
            self.log("train/loss_decoy", loss_decoy, prog_bar=True)
            loss += beta * loss_decoy

        self.log("train/loss_global", loss_global, prog_bar=True)
        self.log("train/spec2mol_global", loss_inbatch_s2m)
        self.log("train/mol2spec_global", loss_inbatch_m2s)
        self.log("train/contrastive_batch_size", float(B))

        # Local contrastive loss
        if self.local_contra:
            frags_emb, frags_mask = self.group_tensor_to_batch(atom_proj, batch_indices)
            frags_mask = frags_mask.to(self.device)
            if self.pooling_strategy == 'cls':
                local_spec_emb = spec_emb[:, 1:, :]
                local_specs_mask = batch['specs_mask'][:, 1:]
            else:
                local_spec_emb = spec_emb
                local_specs_mask = batch['specs_mask']

            frag_labels = batch.get("frag_labels", None)
            frag_masks = batch.get("frag_masks", None)
            loss_local, frag_supervised_loss = self.compute_local_loss(
                local_spec_emb, frags_emb,
                local_specs_mask, frags_mask,
                similarity_threshold=self.local_threshold,
                frag_labels=frag_labels, frag_masks=frag_masks
            )
            self.log('train/loss_local', loss_local, prog_bar=True)
            loss += loss_local * self.local_weight
            if self.frag_supervised:
                self.log('train/loss_frag_supervised', frag_supervised_loss * self.frag_supervised_weight, prog_bar=True)
                loss += frag_supervised_loss * self.frag_supervised_weight

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spec_proj, mol_proj, atom_proj, spec_emb, decoys_proj, batch_indices = self.forward(
            batch['specs'], batch['mol_feats'], batch['adducts'],
            decoys_feats=batch.get('decoys_feats', None),
            decoys_adducts=batch.get('decoys_adducts', None),
            specs_mask=batch.get('specs_mask', None),
            ces=batch.get("ces", None)
        )

        spec_proj = F.normalize(spec_proj, dim=1)
        mol_proj = F.normalize(mol_proj, dim=1)
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
            logits = torch.mm(spec_proj, mol_proj.t()) / temperature
            if 'same_molecule_mask' in batch:
                mask = batch['same_molecule_mask'].clone()
                mask.fill_diagonal_(False)
                logits = logits.masked_fill(mask, float('-inf'))
            labels = torch.arange(B, device=self.device)
            loss_inbatch_s2m = F.cross_entropy(logits, labels)
            loss_inbatch_m2s = F.cross_entropy(logits.t(), labels)
            loss_global = (loss_inbatch_s2m + loss_inbatch_m2s) / 2

        if self.wo_global:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = loss_global

        self.log("val/loss_global", loss_global, prog_bar=True)
        self.log("val/normalized_loss", loss_global / torch.log(torch.tensor(B).float()), prog_bar=True)

        # Local contrastive
        if self.local_contra:
            frags_emb, frags_mask = self.group_tensor_to_batch(atom_proj, batch_indices)
            frags_mask = frags_mask.to(self.device)
            if self.pooling_strategy == 'cls':
                local_spec_emb = spec_emb[:, 1:, :]
                local_specs_mask = batch['specs_mask'][:, 1:]
            else:
                local_spec_emb = spec_emb
                local_specs_mask = batch['specs_mask']

            frag_labels = batch.get("frag_labels", None)
            frag_masks = batch.get("frag_masks", None)
            loss_local, frag_supervised_loss = self.compute_local_loss(
                local_spec_emb, frags_emb,
                local_specs_mask, frags_mask,
                similarity_threshold=self.local_threshold,
                frag_labels=frag_labels, frag_masks=frag_masks
            )
            self.log('val/loss_local', loss_local * self.local_weight)

        if self.wo_global:
            loss = loss_local
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
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

    def on_train_epoch_start(self):
        # Unfreeze DreaMS
        if self.current_epoch >= self.unfreeze_epoch:
            for param in self.spec_encoder.parameters():
                param.requires_grad = True
            self.spec_encoder.train()

        # Unfreeze UniMol2
        if self.frozen_unimol and self.current_epoch >= self.unfreeze_unimol_epoch:
            print(f"Unfreezing UniMol2 encoder at epoch {self.current_epoch}")
            self.mol_encoder.unfreeze()

        # Enable local contrastive
        if self.current_epoch >= self.local_start_epoch and self.local_flag:
            self.local_contra = True

    def group_tensor_to_batch(self, tensor1, batch):
        """Group flat atom embeddings into padded batch tensor using batch indices."""
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        groups = torch.split(tensor1, counts.tolist())

        local_max_len = max(len(g) for g in groups) if groups else 0

        if dist.is_initialized() and dist.get_world_size() > 1:
            local_max_tensor = torch.tensor([local_max_len], dtype=torch.long, device=self.device)
            all_maxes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(dist.get_world_size())]
            dist.all_gather(all_maxes, local_max_tensor)
            global_max_len = max(m.item() for m in all_maxes)
        else:
            global_max_len = local_max_len

        padded = pad_sequence(groups, batch_first=True, padding_value=0.0)
        if padded.shape[1] < global_max_len:
            pad_amount = global_max_len - padded.shape[1]
            padded = nn.functional.pad(padded, (0, 0, 0, pad_amount), value=0.0)

        mask_groups = [torch.ones(len(g), device=self.device) for g in groups]
        padded_mask = pad_sequence(mask_groups, batch_first=True, padding_value=0.0)
        if padded_mask.shape[1] < global_max_len:
            pad_amount = global_max_len - padded_mask.shape[1]
            padded_mask = nn.functional.pad(padded_mask, (0, pad_amount), value=0.0)

        return padded, padded_mask

    def masked_pairwise_contrastive_loss(self, a, b, mask, inverse_temperature=1.0):
        batch_size, seq_len = a.shape[0], a.shape[1]
        expanded_mask = mask[:, None, :].float() * mask[:, :, None].float()
        mask_logits = (1.0 - expanded_mask).contiguous().view(batch_size * seq_len, -1)
        eye_matrix = torch.eye(seq_len)
        labels = eye_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        labels = labels.reshape(batch_size * seq_len, -1).to(self.device)
        logits = torch.einsum('bmd,bnd->bnm', a, b) * inverse_temperature
        logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
        mask_bool = (mask_logits == 1).bool()
        preds = logits.masked_fill(mask_bool, float('1e-9'))
        labels_indices = labels.argmax(dim=-1)
        valid_mask = mask.reshape(-1) == True
        loss = F.cross_entropy(preds, labels_indices, reduction='none')
        loss = loss[valid_mask].sum() / valid_mask.sum().clamp(min=1e-8).float()
        return loss

    def compute_frag_loss(self, similarity, frag_labels):
        bs, num_peaks, num_atoms_pred = similarity.shape
        _, num_frags, num_atoms_target = frag_labels.shape
        if num_atoms_pred > num_atoms_target:
            diff = num_atoms_pred - num_atoms_target
            frag_labels = F.pad(frag_labels, (0, diff), value=-1)

        target_mask = (frag_labels == 1).float()
        valid_frag_mask = (frag_labels == 1).sum(dim=-1) > 0
        target_dist = target_mask / (target_mask.sum(dim=-1, keepdim=True) + 1e-8)

        pred_dist = similarity / (similarity.sum(dim=-1, keepdim=True) + 1e-8)
        align_matrix = torch.einsum('bpa,bfa->bpf', pred_dist, target_dist)
        best_match_scores, _ = align_matrix.max(dim=1)
        nll_loss = -torch.log(best_match_scores.clamp(min=1e-8))
        loss = (nll_loss * valid_frag_mask.float()).sum() / valid_frag_mask.sum().clamp(min=1.0)
        return loss

    def compute_local_loss(self, l_token_embed, v_patch_embed, language_mask, patch_mask,
                           inverse_temperature=5.0, similarity_threshold=0.3,
                           frag_labels=None, frag_masks=None):
        bs, l1, dim = l_token_embed.shape
        _, l2, _ = v_patch_embed.shape

        similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)
        min_sim = similarity.min(dim=-1, keepdim=True)[0]
        max_sim = similarity.max(dim=-1, keepdim=True)[0]
        similarity = (similarity - min_sim) / (max_sim - min_sim + 1e-8)

        mask_peaks = language_mask.unsqueeze(2)
        similarity = similarity.masked_fill(mask_peaks == 0, float(0))
        mask_atoms = patch_mask.unsqueeze(1)
        similarity = similarity.masked_fill(mask_atoms == 0, float(0))

        frag_loss = 0.0
        if self.frag_supervised and (self.current_epoch < self.local_start_epoch + 5):
            frag_loss = self.compute_frag_loss(similarity, frag_labels)

        similarity = torch.where(
            similarity < similarity_threshold,
            torch.tensor(0.0, device=self.device),
            similarity
        )

        v_align_weights = similarity / (similarity.sum(dim=-1, keepdim=True) + 1e-8)
        v_patch_embed_masked = v_patch_embed * patch_mask.unsqueeze(-1).float()
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed_masked)
        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)
        l_token_embed = F.normalize(l_token_embed, p=2, dim=-1)

        loss_vl_local = self.masked_pairwise_contrastive_loss(
            l_grouped_v_patch_embed, l_token_embed, language_mask, inverse_temperature=inverse_temperature
        )
        loss_lv_local = self.masked_pairwise_contrastive_loss(
            l_token_embed, l_grouped_v_patch_embed, language_mask, inverse_temperature=inverse_temperature
        )
        local_loss = (loss_vl_local + loss_lv_local) * 0.5
        return local_loss, frag_loss
