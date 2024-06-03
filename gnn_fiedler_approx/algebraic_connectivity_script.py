import argparse
from ast import arg
import concurrent.futures
import os
import pathlib
from re import A

import codetiming
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import wandb
import yaml
from algebraic_connectivity_dataset import ConnectivityDataset
from my_graphs_dataset import GraphDataset
from torch.nn import Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GCN, GIN, GCNConv, GraphSAGE, Sequential, global_mean_pool
from utils import create_graph_wandb, extract_graphs_from_batch, graphs_to_tuple


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
            layers.append(ReLU())
        self.mp_layers = Sequential("x, edge_index", layers)

        # Final readout layer
        self.lin = Linear(mp_layers[-1], 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.mp_layers(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GNNWrapper(torch.nn.Module):
    def __init__(self, gnn_model, in_channels: int, hidden_channels: int, num_layers: int, **kwargs):
        super().__init__()
        self.gnn = gnn_model(in_channels, hidden_channels, num_layers, **kwargs)
        self.classifier = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


premade_gnns = {x.__name__: x for x in [GCN, GraphSAGE, GIN, GAT]}
custom_gnns = {x.__name__: x for x in [MyGCN]}


# ***************************************
# ************* FUNCTIONS ***************
# ***************************************
def generate_model(architecture, in_channels, hidden_channels, num_layers):
    """Generate a Neural Network model based on the architecture and hyperparameters."""
    # GLOBALS: device, premade_gnns, custom_gnns
    if architecture in premade_gnns:
        model = GNNWrapper(
            gnn_model=premade_gnns[architecture],
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        ).to(device)
    else:
        MyGNN = custom_gnns[architecture]
        model = MyGNN(input_channels=in_channels, mp_layers=[hidden_channels] * num_layers).to(device)
    return model


def generate_optimizer(model, optimizer, lr):
    """Generate optimizer object based on the model and hyperparameters."""
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Only Adam optimizer is currently supported.")


def training_pass(model, batch, optimizer, criterion):
    """Perofrm a single training pass over the batch."""
    data = batch.to(device)  # Move to CUDA if available.
    out = model.forward(data.x, data.edge_index, batch=data.batch)  # Perform a single forward pass.
    loss = criterion(out.squeeze(), data.y)  # Compute the loss.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.


def testing_pass(model, batch, criterion):
    """Perform a single testing pass over the batch."""
    with torch.no_grad():
        data = batch.to(device)
        out = model.forward(data.x, data.edge_index, batch=data.batch)
        loss = criterion(out.squeeze(), data.y).item()  # Compute the loss.
    return loss


def do_train(model, data, optimizer, criterion):
    """Train the model on individual batches or the entire dataset."""
    model.train()

    if isinstance(data, DataLoader):
        for batch in data:  # Iterate in batches over the training dataset.
            training_pass(model, batch, optimizer, criterion)
    elif isinstance(data, Data):
        training_pass(model, data, optimizer, criterion)
    else:
        raise ValueError("Data must be a DataLoader or a Batch object.")


def do_test(model, data, criterion):
    """Test the model on individual batches or the entire dataset."""
    model.eval()

    if isinstance(data, DataLoader):
        for batch in data:
            loss = testing_pass(model, batch, criterion)
    elif isinstance(data, Data):
        loss = testing_pass(model, data, criterion)
    else:
        raise ValueError("Data must be a DataLoader or a Batch object.")

    return loss


def train(model, optimizer, criterion, num_epochs=100, is_sweep=False):
    # GLOBALS: device, dataset, train_data_obj, test_data_obj

    # Prepare for training.
    train_losses = np.zeros(num_epochs)
    test_losses = np.zeros(num_epochs)

    # Start the training loop with timer.
    training_timer = codetiming.Timer(logger=None)
    epoch_timer = codetiming.Timer(logger=None)
    training_timer.start()
    epoch_timer.start()
    for epoch in range(1, num_epochs + 1):
        # Perform one pass over the training set and then test on both sets.
        do_train(model, train_data_obj, optimizer, criterion)
        train_loss = do_test(model, train_data_obj, criterion)
        test_loss = do_test(model, test_data_obj, criterion)

        # Store the losses.
        train_losses[epoch - 1] = train_loss
        test_losses[epoch - 1] = test_loss
        wandb.log({"train_loss": train_loss, "test_loss": test_loss})

        # Print the losses every 10 epochs.
        if epoch % 10 == 0 and not is_sweep:
            print(
                f"Epoch: {epoch:03d}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Avg. duration: {epoch_timer.stop() / 10:.4f} s"
            )
            epoch_timer.start()
    epoch_timer.stop()
    duration = training_timer.stop()

    results = {"train_losses": train_losses, "test_losses": test_losses, "duration": duration}
    return results


def plot_training_curves(num_epochs, train_losses, test_losses, criterion):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_losses, mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=test_losses, mode="lines", name="Test Loss"))
    fig.update_layout(title="Training and Test Loss", xaxis_title="Epoch", yaxis_title=criterion)
    fig.show()


def evaluate(model, plot_graphs=False, is_sweep=False):
    # GLOBALS: dataset_config, train_loader, test_loader

    df = pd.DataFrame()

    # Evaluate the model on the test set.
    model.eval()
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            # Make predictions.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            predictions = out.cpu().numpy().squeeze()
            ground_truth = data.y.cpu().numpy()

            # Extract graphs and create visualizations.
            nx_graphs = extract_graphs_from_batch(data)
            graphs, node_nums, edge_nums = zip(*graphs_to_tuple(nx_graphs))
            # FIXME: This is the only way to parallelize in Jupyter but runs out of memory.
            # with concurrent.futures.ProcessPoolExecutor(4) as executor:
            #     graph_visuals = executor.map(create_graph_wandb, nx_graphs, chunksize=10)
            if plot_graphs:
                graph_visuals = [create_graph_wandb(g) for g in nx_graphs]
            else:
                graph_visuals = ["N/A"] * len(nx_graphs)

            # Store to pandas DataFrame.
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "GraphVis": graph_visuals,
                            "Graph": graphs,
                            "Nodes": node_nums,
                            "Edges": edge_nums,
                            "True": ground_truth,
                            "Predicted": predictions,
                        }
                    ),
                ]
            )

    # Calculate the statistics.
    df["Error"] = df["True"] - df["Predicted"]
    df["Error %"] = 100 * df["Error"] / df["True"]
    df["abs(Error)"] = np.abs(df["Error"])
    err_mean = np.mean(df["abs(Error)"])
    err_stddev = np.std(df["abs(Error)"])

    # Create a W&B table.
    table = wandb.Table(dataframe=df)

    # Print and plot.
    # df = df.sort_values(by="abs(Error)")
    fig_abs_err = px.histogram(df, x="Error")
    fig_rel_err = px.histogram(df, x="Error %")

    if not is_sweep:
        print(f"Mean error: {err_mean:.4f}\n" f"Std. dev.: {err_stddev:.4f}\n")
        fig_abs_err.show()
        fig_rel_err.show()
        df = df.sort_values(by="Nodes")
        print(df)

    results = {
        "mean_err": err_mean,
        "stddev_err": err_stddev,
        "fig_abs_err": fig_abs_err,
        "fig_rel_err": fig_rel_err,
        "table": table,
    }
    return results


def main(config, skip_evaluation=False):
    # GLOBALS: device, dataset, train_data_obj, test_data_obj

    is_sweep = isinstance(config, str)

    if is_sweep:
        with open(pathlib.Path(__file__).parent / config) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    # Set up the run
    run = wandb.init(project="gnn_fiedler_approx", tags=["lambda2", "fiedler", "baseline"], config=config)
    config = wandb.config
    if is_sweep:
        print(f"Running sweep with config: {config}...")

    # Set up the model, optimizer, and criterion.
    model = generate_model(
        config["architecture"], dataset.num_features, config["hidden_channels"], config["num_layers"]
    )
    optimizer = generate_optimizer(model, config["optimizer"], config["learning_rate"])
    criterion = torch.nn.L1Loss()

    # Run training.
    train_results = train(model, optimizer, criterion, config["epochs"], is_sweep=is_sweep)
    run.summary["best_train_loss"] = min(train_results["train_losses"])
    run.summary["best_test_loss"] = min(train_results["test_losses"])
    run.summary["duration"] = train_results["duration"]
    if not is_sweep:
        plot_training_curves(
            config["epochs"], train_results["train_losses"], train_results["test_losses"], type(criterion).__name__
        )

    # Run evaluation.
    if not skip_evaluation:
        eval_results = evaluate(model, plot_graphs=not is_sweep, is_sweep=is_sweep)
        run.summary["mean_err"] = eval_results["mean_err"]
        run.summary["stddev_err"] = eval_results["stddev_err"]
        run.log({"abs_err_hist": eval_results["fig_abs_err"], "rel_err_hist": eval_results["fig_rel_err"]})
        run.log({"results_table": eval_results["table"]})

    if is_sweep:
        print(f"    ...DONE. "
              f"Mean error: {eval_results['mean_err']:.4f}, "
              f"Std. dev.: {eval_results['stddev_err']:.4f}, "
              f"Duration: {train_results['duration']:.4f} s.")

    return run


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train a GNN model to predict the algebraic connectivity of graphs.")
    args.add_argument("--skip-evaluation", action="store_true", help="Skip the evaluation step.")
    args.add_argument("--standalone", action="store_true", help="Run the script as a standalone.")
    args.add_argument("--config", type=str, default="", help="Path to the configuration file.")
    args = args.parse_args()


    # ***************************************
    # *************** LOADING ***************
    # ***************************************
    # Get available device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded torch. Using *{device}* device.")

    # Set up dataset.
    selected_graph_sizes = {
        3: -1,
        4: -1,
        5: -1,
        6: -1,
        7: -1,
        8: -1,
        # 9:  100000,
        # 10: 100000
    }
    dataset_config = {
        "name": "ConnectivityDataset",
        "selected_graphs": str(selected_graph_sizes),
        "split": 0.8,
        "batch_size": 0,
    }

    # Load the dataset.
    graphs_loader = GraphDataset(selection=selected_graph_sizes)
    dataset = ConnectivityDataset(pathlib.Path(os.getcwd()) / "Dataset", graphs_loader)

    # General information
    if args.standalone:
        print()
        print(f"Dataset: {dataset}:")
        print("====================")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")

    # Store information about the dataset.
    dataset_config.update({"num_graphs": len(dataset), "features": dataset.features})

    # Shuffle and split the dataset.
    torch.manual_seed(seed := 42)
    dataset = dataset.shuffle()

    train_size = round(dataset_config["split"] * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    if args.standalone:
        print()
        print(f"Number of training graphs: {len(train_dataset)}")
        print(f"Number of test graphs: {len(test_dataset)}")

    # Batch and load data.
    # TODO: Batch size?
    batch_size = dataset_config["batch_size"] if dataset_config["batch_size"] > 0 else len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_batch = None
    test_batch = None
    # If the whole dataset fits in memory, we can use the following lines to get a single large batch.
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_data_obj = train_batch if train_batch is not None else train_loader
    test_data_obj = test_batch if test_batch is not None else test_loader

    if args.standalone:
        print()
        print("Batches:")
        for step, data in enumerate(train_loader):
            print(f"Step {step + 1}:")
            print("=======")
            print(f"Number of graphs in the current batch: {data.num_graphs}")
            print(data)
            print()

# ***************************************
# ***************** RUN *****************
# ***************************************
    if args.standalone:
        global_config = {
            "seed": seed,
            "dataset": dataset_config,
            "architecture": "GraphSAGE",
            "hidden_channels": 10,
            "num_layers": 3,
            "optimizer": "adam",
            "learning_rate": 0.01,
            "epochs": 500,
        }
        run = main(global_config)
        run.finish()
    else:
        run = main(args.config)
