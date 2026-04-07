import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ms_pred.mabnet.egt import egt_model
from ms_pred.DreaMS.dreams.models.dreams.dreams import DreaMS as DreaMSModel
from ms_pred.DreaMS.dreams.api import PreTrainedModel
from ms_pred.mabnet import egt_pretrain_model as pretrain_model
import ms_pred.nn_utils as nn_utils

class CLIPModel(pl.LightningModule):
    def __init__(self, hidden_size=128, mabnet_heads=8, mabnet_layers=6, edge_update=True, projection_dim=128, temperature=0.07, lr=1e-3, spec_ckpt_path=None, mol_ckpt_path=None, lr_decay_rate=0.825, warmup=1000, dropout=0.1, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()

        # Spectrum encoder
        self.spec_encoder = PreTrainedModel.from_ckpt(
            ckpt_path=spec_ckpt_path,
            ckpt_cls=DreaMSModel,
            n_highest_peaks=60
        ).model.train()

        # Molecule encoder (pass dropout if applicable; assuming it can take dropout param, otherwise apply in forward)
        self.mol_encoder = egt_model.EdgeEnhancedGraphTransformer2D(
            input_node_dim=55,
            hidden_dim=hidden_size,
            num_heads=mabnet_heads,
            num_layers=mabnet_layers,
            edge_update=edge_update,
            dropout=dropout,  # Assuming the model accepts dropout; adjust if not
        )
        if mol_ckpt_path:
            pretrained_model = pretrain_model.Pretrain_EGT.load_from_checkpoint(mol_ckpt_path, strict=False)
            self.mol_encoder.load_state_dict(pretrained_model.root_module.state_dict(), strict=False)

        # Assume the output dimensions of encoders; adjust these based on actual output sizes
        spec_output_dim = 1024  # Placeholder; replace with actual spec_encoder output dim
        mol_output_dim = hidden_size   # Placeholder; replace with actual mol_encoder output dim

        # Projection heads to align dimensions with dropout
        self.spec_projection = nn.Sequential(
            nn.Linear(spec_output_dim, projection_dim),
            nn.Dropout(dropout)
        )
        self.mol_projection = nn.Sequential(
            nn.Linear(mol_output_dim, projection_dim),
            nn.Dropout(dropout)
        )

        self.temperature = temperature
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup
        self.dropout = dropout
        self.weight_decay = weight_decay

    def forward(self, mol_data, spec_data):
        # Assuming spec_encoder and mol_encoder handle their respective data formats
        spec_emb = self.spec_encoder(spec_data['spectrum'])[:, 0, :] # Adjust if spec_encoder requires specific input unpacking
        _, mol_emb = self.mol_encoder(mol_data['root_reprs'])    # Adjust if mol_encoder requires specific input unpacking (e.g., graph data)

        # Project to common space
        spec_proj = self.spec_projection(spec_emb)
        mol_proj = self.mol_projection(mol_emb)

        return spec_proj, mol_proj

    def training_step(self, batch, batch_idx):
        mol_data = batch['mol_data']
        spec_data = batch['spec_data']

        spec_proj, mol_proj = self.forward(mol_data, spec_data)

        # Normalize embeddings
        spec_proj = F.normalize(spec_proj, dim=1)
        mol_proj = F.normalize(mol_proj, dim=1)

        # Compute similarity matrix (spec to mol)
        logits = torch.mm(spec_proj, mol_proj.t()) / self.temperature
        labels = torch.arange(len(logits)).to(self.device)
        loss_spec_to_mol = F.cross_entropy(logits, labels)

        # Compute similarity matrix (mol to spec)
        logits_t = logits.t()
        loss_mol_to_spec = F.cross_entropy(logits_t, labels)

        # Average the losses for bidirectional alignment
        loss = (loss_spec_to_mol + loss_mol_to_spec) / 2

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mol_data = batch['mol_data']
        spec_data = batch['spec_data']

        spec_proj, mol_proj = self.forward(mol_data, spec_data)

        # Normalize embeddings
        spec_proj = F.normalize(spec_proj, dim=1)
        mol_proj = F.normalize(mol_proj, dim=1)

        # Compute similarity matrix (spec to mol)
        logits = torch.mm(spec_proj, mol_proj.t()) / self.temperature
        labels = torch.arange(len(logits)).to(self.device)
        loss_spec_to_mol = F.cross_entropy(logits, labels)

        # Compute similarity matrix (mol to spec)
        logits_t = logits.t()
        loss_mol_to_spec = F.cross_entropy(logits_t, labels)

        # Average the losses
        loss = (loss_spec_to_mol + loss_mol_to_spec) / 2

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