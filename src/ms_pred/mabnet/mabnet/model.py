
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad
from torch_geometric.data import Data
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn

from ms_pred.mabnet.mabnet.mabnet import PhyMabNet
from ms_pred.mabnet.mabnet.uncertainty_network import UncertaintyNetwork
from ms_pred.mabnet.mabnet.output_modules import Scalar, EquivariantScalar



def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            rank_zero_warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    
    return model.to(device)


def create_model(args, mean=None, std=None):
    """
    Create a model based on the provided arguments.
    """
    if args["model"] == "PhyMabNet":
        is_equivariant = True
        model_args = dict(
            atom_index=args["atom_index"],  # 12
            lmax=args["lmax"],
            vecnorm_type=args["vecnorm_type"],
            trainable_vecnorm=args["trainable_vecnorm"],
            num_heads=args["num_heads"],
            num_layers=args["num_layers"],
            hidden_channels=args["embedding_dimension"],
            num_rbf=args["num_rbf"],
            rbf_type=args["rbf_type"],
            trainable_rbf=args["trainable_rbf"],
            activation=args["activation"],
            attn_activation=args["attn_activation"],
            max_z=args["max_z"],
            cutoff=args["cutoff"],
            cutoff_pruning=args["cutoff_pruning"],
            max_num_neighbors=args["max_num_neighbors"],
            max_num_edges_save=args["max_num_edges_save"],
            use_padding=args["use_padding"],
            many_body=args["many_body"],
        )
        
        representation_model = PhyMabNet(**model_args)
        uncertainty_model = UncertaintyNetwork(
            hidden_channels=args["embedding_dimension"],
            activation=args["activation"]) if args.get("evidential", False) else None  
        output_model = EquivariantScalar(args["embedding_dimension"], args["activation"]) if is_equivariant else Scalar(args["embedding_dimension"], args["activation"])

        model = ManyBodyModel(
            representation_model,
            output_model,
            uncertainty_model,
            reduce_op=args.reduce_op,
            mean=mean,
            std=std,
            derivative=args.derivative,
            cutoff=args.cutoff,
            cutoff_pruning=args.cutoff_pruning,
            max_num_neighbors=args.max_num_neighbors,
            max_num_edges_save=args.max_num_edges_save,
            many_body=args.many_body,
            uncertainty=args.evidential
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model


class ManyBodyModel(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        uncertainty_model,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        cutoff=5.0,
        cutoff_pruning=1.6,
        max_num_neighbors=32,
        max_num_edges_save=32,
        many_body=True,
        uncertainty=False
    ):
        super(ManyBodyModel, self).__init__()
        self.representation_model = representation_model
        
        if output_model is not None:
            self.output_model = output_model

        if uncertainty and uncertainty_model is not None:
            self.uncertainty_model = uncertainty_model
            
        self.reduce_op = reduce_op
        self.derivative = derivative
        self.cutoff = cutoff
        self.cutoff_pruning = cutoff_pruning
        self.max_num_neighbors = max_num_neighbors
        self.max_num_edges_save = max_num_edges_save
        self.many_body = many_body
        self.uncertainty = uncertainty

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        if self.representation_model is not None:
            self.output_model.reset_parameters()

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        if self.derivative:
            data.pos.requires_grad_(True)

        x, z, v = self.representation_model(data)  # [num_nodes, hidden_channels], [num_nodes, 8, hidden_channels]
        x = self.output_model.pre_reduce(x, v, z, data.pos, data.batch)    # [num_nodes, 1]
        x = x * self.std

        out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)  # [num_nodes, 1] -> [batch_size, 1]
        out = self.output_model.post_reduce(out)    # [batch_size, 1]
        out = out + self.mean
        if self.uncertainty:
            uncertainty_out = self.uncertainty_model(out)
        
        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [data.pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, -dy, uncertainty_out if self.uncertainty else None
        return out, None, uncertainty_out if self.uncertainty else None


if __name__ == "__main__":
    # Example usage
    args = {
        "model": "many_body",
        "atom_index": 12,
        "lmax": 6,
        "vecnorm_type": "norm",
        "trainable_vecnorm": True,
        "num_heads": 4,
        "num_layers": 4,
        "embedding_dimension": 128,
        "num_rbf": 16,
        "rbf_type": "gaussian",
        "trainable_rbf": True,
        "activation": "silu",
        "attn_activation": "silu",
        "max_z": 100,
        "cutoff": 5.0,
        "cutoff_pruning": 1.6,
        "max_num_neighbors": 32,
        "max_num_edges_save": 32,
        "use_padding": False,
        "many_body": True,
        "reduce_op": 'add',
        'derivative': False
    }
    
    model = create_model(args)
    print(model)