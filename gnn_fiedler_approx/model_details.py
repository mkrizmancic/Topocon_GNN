import os
import pathlib

import torch
from algebraic_connectivity_dataset import ConnectivityDataset
from my_graphs_dataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (MLP, GAT, GCN, GIN, GCNConv, GraphSAGE, Sequential,
                                global_mean_pool, global_max_pool, global_add_pool,
                                summary)


GLOBAL_POOLINGS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "add": global_add_pool
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ***************************************
# *************** MODELS ****************
# ***************************************
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
    def __init__(self, gnn_model, in_channels: int, hidden_channels: int, gnn_layers: int, mlp_layers: int=1, pool="mean", **kwargs):
        super().__init__()
        self.gnn = gnn_model(in_channels=in_channels,
                             hidden_channels=hidden_channels,
                             out_channels=hidden_channels,
                             num_layers=gnn_layers,
                             jk="lstm",
                             **kwargs)
        self.pool = GLOBAL_POOLINGS[pool]
        # self.classifier = torch.nn.Linear(hidden_channels, 1)
        mlp_layer_list = []
        for i in range(mlp_layers):
            if i < mlp_layers - 1:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, hidden_channels))
                mlp_layer_list.append(torch.nn.ReLU())
            else:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, 1))
        self.classifier = torch.nn.Sequential(*mlp_layer_list)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x

    @property
    def descriptive_name(self):
        name = [f"{self.gnn.__class__.__name__}"]
        if hasattr(self.gnn, "channel_list"):
            name.append(f"{self.gnn.num_layers}x{self.gnn.channel_list[-1]}")
        else:
            name.append(f"{self.gnn.num_layers}x{self.gnn.hidden_channels}")
        name.append(f"{self.gnn.act.__class__.__name__}")
        if isinstance(self.gnn.dropout, torch.nn.Dropout):
            name.append(f"D{self.gnn.dropout.p:.2f}")
        else:
            name.append(f"D{self.gnn.dropout[0]:.2f}")
        name.append(f"{self.pool.__name__}")

        name = "-".join(name)

        return name


premade_gnns = {x.__name__: x for x in [MLP, GCN, GraphSAGE, GIN, GAT]}
custom_gnns = {x.__name__: x for x in [MyGCN]}


# ***************************************
# ************* FUNCTIONS ***************
# ***************************************
def generate_model(architecture, in_channels, hidden_channels, gnn_layers, **kwargs):
    """Generate a Neural Network model based on the architecture and hyperparameters."""
    # GLOBALS: device, premade_gnns, custom_gnns
    # if architecture == "GIN":
    #     model = GIN(
    #         in_channels=in_channels,
    #         hidden_channels=hidden_channels,
    #         out_channels=hidden_channels,
    #         num_layers=gnn_layers,
    #         jk="last",
    #     )
    if architecture in premade_gnns:
        model = GNNWrapper(
            gnn_model=premade_gnns[architecture],
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            gnn_layers=gnn_layers,
            **kwargs,
        )
    else:
        MyGNN = custom_gnns[architecture]
        model = MyGNN(input_channels=in_channels, mp_layers=[hidden_channels] * gnn_layers)
    # model = model.to(device)
    return model


def main():
    # Set up dataset.
    selected_graph_sizes = {3: -1, 4: -1}
    root = pathlib.Path(os.getcwd()) / "Dataset"
    graphs_loader = GraphDataset(selection=selected_graph_sizes)
    dataset = ConnectivityDataset(root, graphs_loader)

    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")

    # Batch and load data.
    batch_size = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    data = next(iter(train_loader))

    # Set up the model, optimizer, and criterion.
    model = generate_model(
        "GraphSAGE",
        in_channels=dataset.num_features,
        hidden_channels=32,
        gnn_layers=5,
        mlp_layers=1,
        act="relu",
        dropout=0.0,
        pool="mean"
    )

    print(summary(model, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))
    print(f"Number of parameters: {count_parameters(model)}")



if __name__ == "__main__":
    main()
