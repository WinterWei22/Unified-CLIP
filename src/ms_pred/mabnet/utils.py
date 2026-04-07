import dgl
import torch
from torch_geometric.data import Data

def dgl_to_pyg(g):
    """
    Convert a DGL graph to a PyG Data object, including batch attribute.

    Parameters:
        g: DGLGraph object
    Returns:
        Data: PyTorch Geometric Data object
    """
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0).long()
    data = Data(edge_index=edge_index)

    # Add batch attribute
    if g.batch_size > 1:  # Batched graph
        batch_sizes = g.batch_num_nodes()
        data.batch = torch.repeat_interleave(
            torch.arange(g.batch_size, dtype=torch.long), batch_sizes
        )
    else:  # Single graph
        data.batch = torch.zeros(g.num_nodes(), dtype=torch.long)

    if g.ndata.get('feat') is not None:
        data.x = g.ndata['feat'].float()
    elif len(g.ndata) > 0:
        for key, value in g.ndata.items():
            data[key] = value.float()

    if g.edata.get('weight') is not None:
        data.edge_attr = g.edata['weight'].float()
    elif len(g.edata) > 0:
        for key, value in g.edata.items():
            data[key] = value.float()

    if hasattr(g, 'graph_data') and g.graph_data is not None:
        for key, value in g.graph_data.items():
            data[key] = value

    return data