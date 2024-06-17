import argparse
from collections import Counter
import datetime
import json
import os
import pathlib

import codetiming
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import wandb
from algebraic_connectivity_dataset import ConnectivityDataset
from my_graphs_dataset import GraphDataset
from torch.nn import Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GAT, GCN, GIN, GCNConv, GraphSAGE, Sequential,
                                global_mean_pool, global_max_pool, global_add_pool)
from utils import create_graph_wandb, extract_graphs_from_batch, graphs_to_tuple


GLOBAL_POOLINGS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "add": global_add_pool
}

BEST_MODEL_PATH = pathlib.Path(__file__).parent / "models"
BEST_MODEL_PATH.mkdir(exist_ok=True, parents=True)
BEST_MODEL_PATH /= "best_model.pth"

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
    def __init__(self, gnn_model, in_channels: int, hidden_channels: int, num_layers: int, pool="mean", **kwargs):
        super().__init__()
        self.gnn = gnn_model(in_channels, hidden_channels, num_layers, **kwargs)
        self.pool = GLOBAL_POOLINGS[pool]
        self.classifier = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x

    @property
    def descriptive_name(self):
        return (f"{self.gnn.__class__.__name__}-"
                f"{self.gnn.num_layers}x{self.gnn.hidden_channels}-"
                f"{self.gnn.act.__class__.__name__}-"
                f"D{self.gnn.dropout.p:.2f}-"
                f"{self.pool.__name__}")



premade_gnns = {x.__name__: x for x in [GCN, GraphSAGE, GIN, GAT]}
custom_gnns = {x.__name__: x for x in [MyGCN]}


# ***************************************
# *************** DATASET ***************
# ***************************************
def load_dataset(selected_graph_sizes, selected_features=[], split=0.8, batch_size=0, seed=42, suppress_output=False):
    dataset_config = {
        "name": "ConnectivityDataset",
        "selected_graphs": str(selected_graph_sizes),
        "split": split,
        "batch_size": batch_size,
        "seed": seed,
    }

    # Load the dataset.
    root = pathlib.Path(os.getcwd()) / "Dataset"
    graphs_loader = GraphDataset(selection=selected_graph_sizes)
    dataset = ConnectivityDataset(root, graphs_loader, selected_features=selected_features)

    # General information
    if not suppress_output:
        print()
        print(f"Dataset: {dataset}:")
        print("====================")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")

    # Store information about the dataset.
    dataset_config["num_graphs"] = len(dataset)
    features = selected_features if selected_features else dataset.features

    # Shuffle and split the dataset.
    torch.manual_seed(seed)
    dataset = dataset.shuffle()

    train_size = round(dataset_config["split"] * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    if not suppress_output:
        train_counter = Counter([data.x.shape[0] for data in train_dataset]) # type: ignore
        test_counter = Counter([data.x.shape[0] for data in test_dataset]) # type: ignore
        print()
        print(f"Training dataset: {train_counter} ({train_counter.total()})")
        print(f"Testing dataset : {test_counter} ({test_counter.total()})")

    # Batch and load data.
    batch_size = dataset_config["batch_size"] if dataset_config["batch_size"] > 0 else len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # type: ignore

    train_batch = None
    test_batch = None
    # If the whole dataset fits in memory, we can use the following lines to get a single large batch.
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_data_obj = train_batch if train_batch is not None else train_loader
    test_data_obj = test_batch if test_batch is not None else test_loader

    if not suppress_output:
        print()
        print("Batches:")
        for step, data in enumerate(train_loader):
            print(f"Step {step + 1}:")
            print("=======")
            print(f"Number of graphs in the current batch: {data.num_graphs}")
            print(data)
            print()

    return train_data_obj, test_data_obj, dataset_config, features


# ***************************************
# ************* FUNCTIONS ***************
# ***************************************
def generate_model(architecture, in_channels, hidden_channels, num_layers, **kwargs):
    """Generate a Neural Network model based on the architecture and hyperparameters."""
    # GLOBALS: device, premade_gnns, custom_gnns
    if architecture in premade_gnns:
        model = GNNWrapper(
            gnn_model=premade_gnns[architecture],
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            **kwargs,
        )
    else:
        MyGNN = custom_gnns[architecture]
        model = MyGNN(input_channels=in_channels, mp_layers=[hidden_channels] * num_layers)
    model = model.to(device)
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


def train(
    model, optimizer, criterion, train_data_obj, test_data_obj, num_epochs=100, suppress_output=False, save_best=False
):
    # GLOBALS: device

    # Prepare for training.
    train_losses = np.zeros(num_epochs)
    test_losses = np.zeros(num_epochs)
    best_loss = float("inf")

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

        # Save the best model.
        if save_best and epoch >= 0.3 * num_epochs and test_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss

        # Print the losses every 10 epochs.
        if epoch % 10 == 0 and not suppress_output:
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


def eval_batch(model, batch, plot_graphs=False):
    # Make predictions.
    data = batch.to(device)
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
    return pd.DataFrame(
        {
            "GraphVis": graph_visuals,
            "Graph": graphs,
            "Nodes": node_nums,
            "Edges": edge_nums,
            "True": ground_truth,
            "Predicted": predictions,
        }
    )


def evaluate(model, test_data, plot_graphs=False, suppress_output=False):
    # GLOBALS: dataset_config, train_loader, test_loader

    # Evaluate the model on the test set.
    model.eval()
    df = pd.DataFrame()

    with torch.no_grad():
        if isinstance(test_data, DataLoader):
            for batch in test_data:
                df = pd.concat([df, eval_batch(model, batch, plot_graphs)])
        elif isinstance(test_data, Data):
            df = eval_batch(model, test_data, plot_graphs)
        else:
            raise ValueError("Data must be a DataLoader or a Batch object.")

    # Calculate the statistics.
    df["Error"] = df["True"] - df["Predicted"]
    df["Error %"] = 100 * df["Error"] / df["True"]
    df["abs(Error)"] = np.abs(df["Error"])
    err_mean = np.mean(df["abs(Error)"])
    err_stddev = np.std(df["abs(Error)"])

    good_within = {
        "99": len(df[df["Error %"].between(-1, 1)]) / len(df) * 100,
        "95": len(df[df["Error %"].between(-5, 5)]) / len(df) * 100,
        "90": len(df[df["Error %"].between(-10, 10)]) / len(df) * 100,
        "80": len(df[df["Error %"].between(-20, 20)]) / len(df) * 100,
    }

    # Create a W&B table.
    table = wandb.Table(dataframe=df)

    # Print and plot.
    # df = df.sort_values(by="abs(Error)")
    fig_abs_err = px.histogram(df, x="Error")
    fig_rel_err = px.histogram(df, x="Error %")

    if not suppress_output:
        print(
            f"Mean error: {err_mean:.4f}\n"
            f"Std. dev.: {err_stddev:.4f}\n"
            f"Error brackets: {json.dumps(good_within, indent=4)}\n"
        )
        fig_abs_err.show()
        fig_rel_err.show()
        df = df.sort_values(by="Nodes")
        print(df)

    results = {
        "mean_err": err_mean,
        "stddev_err": err_stddev,
        "good_within": good_within,
        "fig_abs_err": fig_abs_err,
        "fig_rel_err": fig_rel_err,
        "table": table,
    }
    return results


def main(config=None, evaluation=False, no_wandb=False, is_best_run=False):
    # GLOBALS: device

    is_sweep = config is None
    wandb_mode = "disabled" if no_wandb else "online"
    tags = ["lambda2", "baseline"]
    if is_best_run:
        tags.append("BEST")

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

    # Set up the run
    run = wandb.init(mode=wandb_mode, project="gnn_fiedler_approx", tags=tags, config=config)
    config = wandb.config
    if is_sweep:
        print(f"Running sweep with config: {config}...")

    # Load the dataset.
    train_data_obj, test_data_obj, dataset_config, features = load_dataset(
        selected_graph_sizes, selected_features=config.get("selected_features", []), suppress_output=is_sweep
    )

    wandb.config["dataset"] = dataset_config
    if "selected_features" not in wandb.config or not wandb.config["selected_features"]:
        wandb.config["selected_features"] = features

    # Set up the model, optimizer, and criterion.
    model = generate_model(
        config["architecture"],
        len(wandb.config["selected_features"]),
        config["hidden_channels"],
        config["num_layers"],
        act=config["activation"],
        dropout=config["dropout"],
        pool=config["aggr"],
    )
    optimizer = generate_optimizer(model, config["optimizer"], config["learning_rate"])
    criterion = torch.nn.L1Loss()

    # Run training.
    train_results = train(
        model, optimizer, criterion, train_data_obj, test_data_obj, config["epochs"],
        suppress_output=is_sweep,
        save_best=is_best_run
    )
    run.summary["best_train_loss"] = min(train_results["train_losses"])
    run.summary["best_test_loss"] = min(train_results["test_losses"])
    run.summary["duration"] = train_results["duration"]
    if not is_sweep:
        plot_training_curves(
            config["epochs"], train_results["train_losses"], train_results["test_losses"], type(criterion).__name__
        )

    # Run evaluation.
    if evaluation != "none":
        if evaluation == "best":
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            model.eval()

        eval_results = evaluate(model, test_data_obj, plot_graphs=is_best_run, suppress_output=is_sweep)
        run.summary["mean_err"] = eval_results["mean_err"]
        run.summary["stddev_err"] = eval_results["stddev_err"]
        run.summary["good_within"] = eval_results["good_within"]
        run.log({"abs_err_hist": eval_results["fig_abs_err"], "rel_err_hist": eval_results["fig_rel_err"]})

        if evaluation in  ["detailed", "best"]:
            run.log({"results_table": eval_results["table"]})

        if is_best_run:
            # Name the model with the current time and date to make it uniq
            model_name = f"{model.descriptive_name}-{datetime.datetime.now().strftime('%d%m%y%H%M')}"
            artifact = wandb.Artifact(name=model_name, type='model')
            artifact.add_file(str(BEST_MODEL_PATH))
            run.log_artifact(artifact)

    if is_sweep:
        print(
            f"    ...DONE. "
            f"Mean error: {eval_results['mean_err']:.4f}, "
            f"Std. dev.: {eval_results['stddev_err']:.4f}, "
            f"Duration: {train_results['duration']:.4f} s."
        )

    return run


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train a GNN model to predict the algebraic connectivity of graphs.")
    args.add_argument("--standalone", action="store_true", help="Run the script as a standalone.")
    args.add_argument("--evaluation", action="store", choices=["basic", "best", "detailed", "none"],
                      help="Evaluate the model.")
    args.add_argument("--no-wandb", action="store_true", help="Do not use W&B for logging.")
    args.add_argument("--best", action="store_true", help="Plot the graphs with the best model.")
    args = args.parse_args()

    # Get available device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded torch. Using *{device}* device.")

    if args.standalone:
        global_config = {
            "seed": 42,
            "architecture": "GraphSAGE",
            "hidden_channels": 32,
            "num_layers": 5,
            "activation": "tanh",
            "dropout": 0.0,
            "aggr": "max",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "epochs": 2000,
        }
        run = main(global_config, evaluation=args.evaluation, no_wandb=args.no_wandb, is_best_run=args.best)
        run.finish()
    else:
        run = main()
