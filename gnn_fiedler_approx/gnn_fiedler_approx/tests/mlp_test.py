import os
import pathlib

import torch
from gnn_fiedler_approx import ConnectivityDataset
from my_graphs_dataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (MLP, global_add_pool, summary)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MlpNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(MlpNet, self).__init__()
        self.conv1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, 10)

    def forward(self, x, edge_index, batch):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = global_add_pool(x, batch)
        x = torch.tanh(self.fc1(x))
        return x


class GNNWrapper(torch.nn.Module):
    def __init__(self, gnn_model, in_channels: int, hidden_channels: int, gnn_layers: int, mlp_layers: int=1, **kwargs):
        super().__init__()
        self.gnn = gnn_model(in_channels=in_channels,
                             hidden_channels=hidden_channels,
                             out_channels=hidden_channels,
                             num_layers=gnn_layers,
                             **kwargs)
        self.pool = global_add_pool
        # self.classifier = torch.nn.Linear(hidden_channels, 1)
        mlp_layer_list = []
        for i in range(mlp_layers):
            if i < mlp_layers - 1:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, hidden_channels))
                mlp_layer_list.append(torch.nn.Tanh())
            else:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, 10))
        self.classifier = torch.nn.Sequential(*mlp_layer_list)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up dataset.
    selected_graph_sizes = {3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}
    root = pathlib.Path(os.getcwd()) / "Dataset"
    graphs_loader = GraphDataset(selection=selected_graph_sizes)
    dataset = ConnectivityDataset(root, graphs_loader)

    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print()

    # Batch and load data.
    batch_size = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    data = next(iter(train_loader))
    data = data.to(device)

    # Balcilar et al. (2021) claim that this model can distinguish 99.5% of the graph8c graphs.
    embedding_model_orig = MlpNet(in_channels=dataset.num_features, hidden_channels=64).to(device)
    print("Original model:")
    print(summary(embedding_model_orig, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))
    print(f"Number of parameters: {count_parameters(embedding_model_orig)}\n")

    # This is a similar model according to my wrapper, but it lacks activation functions at some places.
    embedding_model_my = GNNWrapper(MLP, in_channels=dataset.num_features, hidden_channels=64, gnn_layers=3, mlp_layers=1).to(device)
    print("My model:")
    print(summary(embedding_model_my, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))
    print(f"Number of parameters: {count_parameters(embedding_model_my)}\n")

    # Balcilar et al. (2021) take 10-dimensional embeddings from a single pass of the model
    #   and compare how many pairs of embeddings are similar.
    # If this single pass through MLP can produce embeddings that uniquely identify the graphs, then a simple downstream
    #   MLP model should be able to map these embeddings to the graph algebraic connectivity, according to the universal
    #   approximation theorem.

    embedding_orig = embedding_model_orig(data.x, data.edge_index, data.batch).detach()
    embedding_my = embedding_model_my(data.x, data.edge_index, data.batch).detach()

    neurons = 256
    downstream_model = torch.nn.Sequential(
        torch.nn.Linear(10, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1),
    ).to(device)

    epochs = 5000
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=0.001)

    # Train the model on the original embeddings.
    downstream_model.train()
    for epoch in range(epochs):
        out = downstream_model(embedding_orig)  # Perform a single forward pass.
        loss = criterion(out.squeeze(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, "
                f"Train Loss: {loss.item():.6f}, "
            )
        if loss.item() < 0.0001:
            break

    downstream_model.eval()
    out = downstream_model(embedding_orig)
    criterion = torch.nn.L1Loss()
    loss = criterion(out.squeeze(), data.y)
    print(f"Final loss: {loss.item():.4f}")

    # Loss does not go below 0.15, which is much worse than what we have now.
    #   I guess that just because we have (more or less) unique embeddings and MLP follows the universal approximation
    #   theorem, it does not mean that there actually exists a mapping fromm embeddings to the algebraic connectivity.


if __name__ == "__main__":
    main()
