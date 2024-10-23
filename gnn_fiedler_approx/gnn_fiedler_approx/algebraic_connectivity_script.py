import argparse
from collections import Counter
import datetime
import enum
import json
import os
import pathlib

import codetiming
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
# import torchexplorer
import wandb
from algebraic_connectivity_dataset import ConnectivityDataset
from my_graphs_dataset import GraphDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    GAT,
    GCN,
    GIN,
    GCNConv,
    GraphSAGE,
    Sequential,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from gnn_utils.utils import create_graph_wandb, extract_graphs_from_batch, graphs_to_tuple, count_parameters


GLOBAL_POOLINGS = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}

BEST_MODEL_PATH = pathlib.Path(__file__).parents[1] / "models"
BEST_MODEL_PATH.mkdir(exist_ok=True, parents=True)
BEST_MODEL_PATH /= "best_model.pth"

SORT_DATA = True


class EvalType(enum.Enum):
    NONE = 0
    BASIC = 1
    DETAILED = 2
    FULL = 3


class EvalTarget(enum.Enum):
    LAST = "last"
    BEST = "best"


# ***************************************
# *************** MODELS ****************
# ***************************************
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
        pool="mean",
        **kwargs,
    ):
        super().__init__()
        self.gnn = gnn_model(
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=None, num_layers=gnn_layers, **kwargs
        )
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
        name.append(f"{count_parameters(self)}")

        name = "-".join(name)

        return name


premade_gnns = {x.__name__: x for x in [MLP, GCN, GraphSAGE, GIN, GAT]}
custom_gnns = {x.__name__: x for x in [MyGCN]}


class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs((target - input) / target))


# ***************************************
# *************** DATASET ***************
# ***************************************
def load_dataset(selected_graph_sizes, selected_features=[], split=0.8, batch_size=1.0, seed=42, suppress_output=False):
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
    # TODO: Splitting after shuffle gives relatively balanced splits between the graph sizes, but it's not perfect.
    # torch.manual_seed(seed)
    dataset = dataset.shuffle()

    train_size = round(dataset_config["split"] * len(dataset))
    train_dataset = dataset[:train_size]
    if len(dataset) - train_size > 0:
        test_dataset = dataset[train_size:]
    else:
        test_dataset = train_dataset

    if not suppress_output:
        train_counter = Counter([data.x.shape[0] for data in train_dataset])  # type: ignore
        test_counter = Counter([data.x.shape[0] for data in test_dataset])  # type: ignore
        splits_per_size = {
            size: round(train_counter[size] / (train_counter[size] + test_counter[size]), 2)
            for size in set(train_counter + test_counter)
        }
        print()
        print(f"Training dataset: {train_counter} ({train_counter.total()})")
        print(f"Testing dataset : {test_counter} ({test_counter.total()})")
        print(f"Dataset splits  : {splits_per_size}")

    # Batch and load data.
    batch_size = int(np.ceil(dataset_config["batch_size"] * len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # type: ignore

    # If the whole dataset fits in memory, we can use the following lines to get a single large batch.
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_data_obj = train_batch if batch_size == train_size else train_loader
    test_data_obj = test_batch if batch_size == train_size else test_loader

    if not suppress_output:
        print()
        print("Batches:")
        for step, data in enumerate(train_loader):
            print(f"Step {step + 1}:")
            print("=======")
            print(f"Number of graphs in the current batch: {data.num_graphs}")
            print(data)
            print(f"Average target: {torch.mean(data.y)}")
            print()

    return train_data_obj, test_data_obj, dataset_config, features, dataset.num_features  # type: ignore


# ***************************************
# ************* FUNCTIONS ***************
# ***************************************
def generate_model(architecture, in_channels, hidden_channels, gnn_layers, **kwargs):
    """Generate a Neural Network model based on the architecture and hyperparameters."""
    # GLOBALS: device, premade_gnns, custom_gnns
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
    model = model.to(device)
    return model


def generate_optimizer(model, optimizer, lr, **kwargs):
    """Generate optimizer object based on the model and hyperparameters."""
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError("Only Adam optimizer is currently supported.")


def training_pass(model, batch, optimizer, criterion):
    """Perofrm a single training pass over the batch."""
    if SORT_DATA:
        batch = batch.sort(sort_by_row=False)

    data = batch.to(device)  # Move to CUDA if available.
    out = model(data.x, data.edge_index, batch=data.batch)  # Perform a single forward pass.
    loss = criterion(out.squeeze(), data.y)  # Compute the loss.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.
    return loss.item()


def testing_pass(model, batch, criterion):
    """Perform a single testing pass over the batch."""
    if SORT_DATA:
        batch = batch.sort(sort_by_row=False)

    with torch.no_grad():
        data = batch.to(device)
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = criterion(out.squeeze(), data.y).item()  # Compute the loss.
    return loss


def do_train(model, data, optimizer, criterion):
    """Train the model on individual batches or the entire dataset."""
    model.train()

    if isinstance(data, DataLoader):
        avg_loss = 0
        for batch in data:  # Iterate in batches over the training dataset.
            avg_loss += training_pass(model, batch, optimizer, criterion)
        loss = avg_loss / len(data)
    elif isinstance(data, Data):
        loss = training_pass(model, data, optimizer, criterion)
    else:
        raise ValueError("Data must be a DataLoader or a Batch object.")

    return loss


def do_test(model, data, criterion):
    """Test the model on individual batches or the entire dataset."""
    model.eval()

    if isinstance(data, DataLoader):
        avg_loss = 0
        for batch in data:
            avg_loss += testing_pass(model, batch, criterion)
        loss = avg_loss / len(data)
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
    test_loss = 0

    # Start the training loop with timer.
    training_timer = codetiming.Timer(logger=None)
    epoch_timer = codetiming.Timer(logger=None)
    training_timer.start()
    epoch_timer.start()
    for epoch in range(1, num_epochs + 1):
        # Perform one pass over the training set and then test on both sets.
        train_loss = do_train(model, train_data_obj, optimizer, criterion)
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
    if SORT_DATA:
        batch = batch.sort(sort_by_row=False)

    # Make predictions.
    data = batch.to(device)
    out = model(data.x, data.edge_index, data.batch)
    predictions = out.cpu().numpy().squeeze()
    ground_truth = data.y.cpu().numpy()

    # Extract graphs and create visualizations.
    # TODO: These two lines are very slow.
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


def evaluate(model, test_data, plot_graphs=False, make_table=False, suppress_output=False):
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
    table = wandb.Table(dataframe=df) if make_table else None

    # Print and plot.
    fig_abs_err = px.histogram(df, x="Error")
    fig_rel_err = px.histogram(df, x="Error %")

    plot_df = pd.DataFrame()
    plot_df["abs(Error %)"] = np.abs(df["Error %"])
    plot_df.sort_values(by="abs(Error %)", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)
    fig_err_curve = px.line(plot_df, x="abs(Error %)", y=(plot_df.index + 1) / len(plot_df) * 100, title="Error curve")
    fig_err_curve.update_xaxes(showspikes=True, tickvals=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig_err_curve.update_yaxes(showspikes=True, nticks=10, title_text="Percentage of graphs")

    if not suppress_output:
        print(
            f"Mean error: {err_mean:.4f}\n"
            f"Std. dev.: {err_stddev:.4f}\n"
            f"Error brackets: {json.dumps(good_within, indent=4)}\n"
        )
        fig_abs_err.show()
        fig_rel_err.show()
        fig_err_curve.show()
        df = df.sort_values(by="Nodes")
        print(df)

    results = {
        "mean_err": err_mean,
        "stddev_err": err_stddev,
        "good_within": good_within,
        "fig_abs_err": fig_abs_err,
        "fig_rel_err": fig_rel_err,
        "fig_err_curve": fig_err_curve,
        "table": table,
    }
    return results


def main(config=None, eval_type=EvalType.NONE, eval_target=EvalTarget.LAST, no_wandb=False, is_best_run=False):
    # GLOBALS: device

    save_best = eval_target == EvalTarget.BEST
    plot_graphs = eval_type == EvalType.FULL
    make_table = eval_type.value > EvalType.BASIC.value

    is_sweep = config is None
    wandb_mode = "disabled" if no_wandb else "online"
    tags = ["GraphSAGE"]
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
        # 9:  10000,
        # 10: 10000
    }

    # Set up the run
    # torchexplorer.setup()
    run = wandb.init(mode=wandb_mode, project="gnn_fiedler_approx_v2", tags=tags, config=config)
    config = wandb.config
    if is_sweep:
        print(f"Running sweep with config: {config}...")

    # For this combination of parameters, the model is too large to fit in memory, so we need to reduce the batch size.
    model_kwargs = config.get("model_kwargs", {})
    if model_kwargs and model_kwargs["aggr"] == "lstm" and model_kwargs["project"]:
        bs = 0.5
    else:
        bs = 1.0

    # Load the dataset.
    train_data_obj, test_data_obj, dataset_config, features, feature_dim = load_dataset(
        selected_graph_sizes,
        selected_features=config.get("selected_features", []),
        batch_size=bs,
        split=config.get("dataset", {}).get("split", 0.8),
        suppress_output=is_sweep,
    )

    wandb.config["dataset"] = dataset_config
    if "selected_features" not in wandb.config or not wandb.config["selected_features"]:
        wandb.config["selected_features"] = features

    # Set up the model, optimizer, and criterion.
    model = generate_model(
        config["architecture"],
        feature_dim,
        config["hidden_channels"],
        config["gnn_layers"],
        mlp_layers=config["mlp_layers"],
        act=config["activation"],
        dropout=float(config["dropout"]),
        pool=config["pool"],
        jk=config["jk"] if config["jk"] != "none" else None,
        **model_kwargs,
    )
    optimizer = generate_optimizer(model, config["optimizer"], config["learning_rate"])
    criterion = torch.nn.L1Loss()

    wandb.watch(model, criterion, log="all", log_freq=100)
    # torchexplorer.watch(model, backend="wandb")

    # Run training.
    train_results = train(
        model,
        optimizer,
        criterion,
        train_data_obj,
        test_data_obj,
        config["epochs"],
        suppress_output=is_sweep,
        save_best=save_best,
    )
    run.summary["best_train_loss"] = min(train_results["train_losses"])
    run.summary["best_test_loss"] = min(train_results["test_losses"])
    run.summary["duration"] = train_results["duration"]
    if not is_sweep:
        plot_training_curves(
            config["epochs"], train_results["train_losses"], train_results["test_losses"], type(criterion).__name__
        )

    # Run evaluation.
    if eval_type != EvalType.NONE:
        if eval_target == EvalTarget.BEST:
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            model.eval()

        eval_results = evaluate(model, test_data_obj, plot_graphs, make_table, suppress_output=is_sweep)
        run.summary["mean_err"] = eval_results["mean_err"]
        run.summary["stddev_err"] = eval_results["stddev_err"]
        run.summary["good_within"] = eval_results["good_within"]
        run.log(
            {
                "abs_err_hist": eval_results["fig_abs_err"],
                "rel_err_hist": eval_results["fig_rel_err"],
                "err_curve": eval_results["fig_err_curve"],
            }
        )

        if eval_type.value > EvalType.BASIC.value:
            run.log({"results_table": eval_results["table"]})

        if is_best_run:
            # Name the model with the current time and date to make it uniq
            model_name = f"{model.descriptive_name}-{datetime.datetime.now().strftime('%d%m%y%H%M')}"
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(str(BEST_MODEL_PATH))
            run.log_artifact(artifact)

    if is_sweep:
        print("    ...DONE.")
        if eval_type != EvalType.NONE:
            print(f"Mean error: {eval_results['mean_err']:.4f}")
            print(f"Std. dev.: {eval_results['stddev_err']:.4f}")
        print(f"Duration: {train_results['duration']:.4f} s.")
    return run


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train a GNN model to predict the algebraic connectivity of graphs.")
    args.add_argument("--standalone", action="store_true", help="Run the script as a standalone.")
    # These are the options for model performance evaluation.
    # - none: Evaluation is skipped.
    # - basic: Will calculate all metrics and plot the graphs, but will not upload the results table to W&B.
    # - detailed: Same as basic, but will also upload the results table to W&B.
    # - full: Same as detailed, but will also plot the graphs inside the results table.
    args.add_argument(
        "--eval-type",
        action="store",
        choices=["basic", "detailed", "full", "none"],
        help="Level of detail for model evaluation.",
    )
    # Evaluate the model from the last epoch or the best.
    args.add_argument(
        "--eval-target", action="store", choices=["best", "last"], default="last", help="Which model to evaluate."
    )
    args.add_argument("--no-wandb", action="store_true", help="Do not use W&B for logging.")
    # If best is set, the script will evaluate the best model, add the BEST tag, plot the graphs inside the results
    # table and upload everything to W&B.
    args.add_argument("--best", action="store_true", help="Mark and store the best model.")
    args = args.parse_args()

    eval_type = EvalType[args.eval_type.upper()] if args.eval_type else EvalType.NONE
    eval_target = EvalTarget[args.eval_target.upper()]

    # Get available device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded torch. Using *{device}* device.")

    if args.standalone:
        global_config = {
            ## Model configuration
            "architecture": "GraphSAGE",
            "hidden_channels": 32,
            "gnn_layers": 5,
            "mlp_layers": 1,
            "activation": "tanh",
            "pool": "max",
            "jk": None,
            "dropout": 0.0,
            ## Training configuration
            "optimizer": "adam",
            "learning_rate": 0.01,
            "epochs": 2000,
            ## Dataset configuration
            # "selected_features": ["random1"]
        }
        run = main(global_config, eval_type, eval_target, args.no_wandb, args.best)
        run.finish()
    else:
        run = main(None, eval_type, eval_target, False, args.best)
