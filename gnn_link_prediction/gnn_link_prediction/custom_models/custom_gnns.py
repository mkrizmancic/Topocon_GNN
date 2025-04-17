import torch
import torch_geometric.nn.aggr as torch_aggr
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.nn import (
    GAT,
    GCN,
    GIN,
    MLP,
    PNA,
    GraphSAGE,
    PNAConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from gnn_link_prediction.gnn_utils.utils import count_parameters


# FIXME: Activation function is missing
class MyGCN(torch.nn.Module):
    def __init__(self, input_channels, mp_layers):
        super(MyGCN, self).__init__()

        # Message-passing layers - GCNConv
        layers = []
        for i, layer_size in enumerate(mp_layers):
            if i == 0:
                layers.append((GCNConv(input_channels, layer_size), "x, edge_index -> x"))
            else:
                layers.append((GCNConv(mp_layers[i - 1], layer_size), "x, edge_index -> x"))
            layers.append(torch.nn.ReLU())
        self.mp_layers = Sequential("x, edge_index", layers)

        # Final readout layer
        self.lin = torch.nn.Linear(mp_layers[-1], 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.mp_layers(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        gnn_model,
        in_channels: int,
        hidden_channels: int,
        gnn_layers: int,
        mlp_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.gnn = gnn_model(
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=None, num_layers=gnn_layers, **kwargs
        )
        self.gnn_is_mlp = isinstance(gnn_model, MLP)
        hc = hidden_channels

        mlp_layer_list = []
        for i in range(mlp_layers):
            if i < mlp_layers - 1:
                mlp_layer_list.append(torch.nn.Linear(hc, hidden_channels))
                mlp_layer_list.append(torch.nn.ReLU())
                hc = hidden_channels
            else:
                mlp_layer_list.append(torch.nn.Linear(hc, 1))
        self.classifier = torch.nn.Sequential(*mlp_layer_list)
        self.mlp_layers = mlp_layers

    def forward(self, x, edge_index, edge_label_index, batch):
        if self.gnn_is_mlp:
            x = self.gnn(x=x, batch=batch)
        else:
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)

        x1, x2 = x[edge_label_index[0]], x[edge_label_index[1]]

        # TODO: Different ways to combine the two node embeddings into one edge embedding
        # x_edge = torch.cat([x1, x2], dim=-1)
        # out = self.classifier(x_edge)

        # out = (x1 * x2).sum(dim=-1)  # Inner product

        x_edge = x1 + x2
        out = self.classifier(x_edge)

        return out

    @property
    def descriptive_name(self):
        try:
            # Base name of the used GNN
            name = [f"{self.gnn.__class__.__name__}"]
            # Number of layers and size of hidden channel or hidden channels list
            if hasattr(self.gnn, "channel_list"):
                name.append(f"{self.gnn.num_layers}x{self.gnn.channel_list[-1]}_{self.mlp_layers}")
            else:
                name.append(f"{self.gnn.num_layers}x{self.gnn.hidden_channels}_{self.mlp_layers}")
            # Activation function
            name.append(f"{self.gnn.act.__class__.__name__}")
            # Dropout
            if isinstance(self.gnn.dropout, torch.nn.Dropout):
                name.append(f"D{self.gnn.dropout.p:.2f}")
            else:
                name.append(f"D{self.gnn.dropout[0]:.2f}")
            # Pooling layer: either a function or a class
            if hasattr(self.pool, "__name__"):
                name.append(f"{self.pool.__name__}")
            else:
                name.append(f"{self.pool.__class__.__name__}")
            # Number of parameters
            name.append(f"{count_parameters(self)}")
            # Join all parts and return.
            name = "-".join(name)
            return name

        except Exception as e:
            raise ValueError(f"Error in descriptive_name: {e}")


premade_gnns = {x.__name__: x for x in [MLP, GCN, GraphSAGE, GIN, GAT, PNA]}
custom_gnns = {x.__name__: x for x in [MyGCN]}