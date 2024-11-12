import os
import pathlib

from my_graphs_dataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary, PNAConv

from gnn_fiedler_approx import ConnectivityDataset
from gnn_fiedler_approx.algebraic_connectivity_script import generate_model
from gnn_fiedler_approx.tests.segk_test import train


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = "cpu"
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
        gnn_layers=2,
        mlp_layers=1,
        act="relu",
        dropout=0.0,
        pool="mean",
        # towers=1,
        # aggregators=["min", "max", "mean", "std"],
        # scalers=["identity", "amplification", "attenuation"],
        # deg=PNAConv.get_degree_histogram(train_loader)
    )

    # sorted_edge_index = torch_geometric.utils.sort_edge_index(data.edge_index, sort_by_row=False)
    # data = data.sort(sort_by_row=False)
    print(summary(model, data.x, data.edge_index, batch=data.batch, max_depth=10, leaf_module=None))
    print(f"Number of parameters: {count_parameters(model)}")



if __name__ == "__main__":
    main()
