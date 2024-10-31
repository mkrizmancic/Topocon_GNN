import os
import pathlib

import torch
from gnn_fiedler_approx import ConnectivityDataset
from my_graphs_dataset import GraphDataset
from gnn_fiedler_approx.custom_models import SEGK
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (MLP, global_add_pool, summary)


EMBEDDING_SIZE = 10


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MlpNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(MlpNet, self).__init__()
        self.conv1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, EMBEDDING_SIZE)

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
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, EMBEDDING_SIZE))
        self.classifier = torch.nn.Sequential(*mlp_layer_list)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x

def generate_downstream_model():
    neurons = 256
    downstream_model = torch.nn.Sequential(
        torch.nn.Linear(EMBEDDING_SIZE, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1),
    ).to(device)
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=0.001)
    return downstream_model, optimizer


def train(model, embedding, y, epochs, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        out = model(embedding)  # Perform a single forward pass.
        loss = criterion(out.squeeze(), y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        # if epoch % 10 == 0:
        #     print(
        #         f"Epoch: {epoch:03d}, "
        #         f"Train Loss: {loss.item():.6f}, "
        #     )
        if loss.item() < 0.0001:
            break

    return loss.item()


def eval(model, embedding, y, criterion):
    model.eval()
    out = model(embedding)
    loss = criterion(out.squeeze(), y)
    return loss.item()


def main(selected_graph_sizes):


    # Set up dataset.
    root = pathlib.Path(os.getcwd()) / "Dataset"
    graphs_loader = GraphDataset(selection=selected_graph_sizes)
    dataset = ConnectivityDataset(root, graphs_loader)
    num_features = dataset.num_features

    # print()
    # print(f"Dataset: {dataset}:")
    # print("====================")
    # print(f"Number of graphs: {len(dataset)}")
    # print(f"Number of features: {dataset.num_features}")
    # print()

    torch.manual_seed(0)
    dataset = dataset.shuffle()
    train_size = round(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    # Batch and load data.
    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=train_size, shuffle=False)  # type: ignore

    # If the whole dataset fits in memory, we can use the following lines to get a single large batch.
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_data = train_batch.to(device)
    test_data = test_batch.to(device)

    # Balcilar et al. (2021) claim that this model can distinguish 99.5% of the graph8c graphs.
    embedding_model_mlp = MlpNet(in_channels=num_features, hidden_channels=64).to(device)
    # print("Original model:")
    # print(summary(embedding_model_orig, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))
    # print(f"Number of parameters: {count_parameters(embedding_model_orig)}\n")

    # This is a similar model according to my wrapper, but it lacks activation functions at some places.
    embedding_model_gnn = GNNWrapper(MLP, in_channels=num_features, hidden_channels=64, gnn_layers=3, mlp_layers=1).to(device)
    # print("My model:")
    # print(summary(embedding_model_my, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))
    # print(f"Number of parameters: {count_parameters(embedding_model_my)}\n")

    # Balcilar et al. (2021) take 10-dimensional embeddings from a single pass of the model
    #   and compare how many pairs of embeddings are similar.
    # If this single pass through MLP can produce embeddings that uniquely identify the graphs, then a simple downstream
    #   MLP model should be able to map these embeddings to the graph algebraic connectivity, according to the universal
    #   approximation theorem.

    # SEGK model
    embedding_model_segk = SEGK(radius=3, dim=EMBEDDING_SIZE, kernel="weisfeiler_lehman")

    embedding_mlp_train = embedding_model_mlp(train_data.x, train_data.edge_index, train_data.batch).detach()
    embedding_gnn_train = embedding_model_gnn(train_data.x, train_data.edge_index, train_data.batch).detach()
    embedding_segk_train = embedding_model_segk(train_data).detach().to(device)

    embedding_mlp_test = embedding_model_mlp(test_data.x, test_data.edge_index, test_data.batch).detach()
    embedding_gnn_test = embedding_model_gnn(test_data.x, test_data.edge_index, test_data.batch).detach()
    embedding_segk_test = embedding_model_segk(test_data).detach().to(device)

    epochs = 5000
    criterion = torch.nn.L1Loss()

    print("MLP model  |", end=" ")
    downstream_model, optimizer = generate_downstream_model()
    train_loss = train(downstream_model, embedding_mlp_train, train_data.y, epochs, criterion, optimizer)
    test_loss = eval(downstream_model, embedding_mlp_test, test_data.y, criterion)
    print(f"Train loss: {train_loss}, Test loss: {test_loss}")

    print("GNN model  |", end=" ")
    downstream_model, optimizer = generate_downstream_model()
    train_loss = train(downstream_model, embedding_gnn_train, train_data.y, epochs, criterion, optimizer)
    test_loss = eval(downstream_model, embedding_gnn_test, test_data.y, criterion)
    print(f"Train loss: {train_loss}, Test loss: {test_loss}")

    print("SEGK model |", end=" ")
    downstream_model, optimizer = generate_downstream_model()
    embedding_segk_train = global_add_pool(embedding_segk_train, train_data.batch)
    embedding_segk_test = global_add_pool(embedding_segk_test, test_data.batch)
    train_loss = train(downstream_model, embedding_segk_train, train_data.y, epochs, criterion, optimizer)
    test_loss = eval(downstream_model, embedding_segk_test, test_data.y, criterion)
    print(f"Train loss: {train_loss}, Test loss: {test_loss}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_graph_sizes = [
        {3: -1, 4: -1, 5: -1},
        {3: -1, 4: -1, 5: -1, 6: -1},
        {3: -1, 4: -1, 5: -1, 6: -1, 7: -1},
        {3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}
    ]

    for sgs in selected_graph_sizes:
        print(f"Selected graph sizes: {sgs}")
        main(sgs)
        print()
