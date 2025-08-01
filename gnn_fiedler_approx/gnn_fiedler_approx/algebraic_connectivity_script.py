import argparse
import datetime
import enum
import importlib
import json
import math
import os
import pathlib
import random
from typing import Union

import codetiming
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import wandb
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from my_graphs_dataset import GraphDataset, GraphType
from gnn_fiedler_approx import ConnectivityDataset, inspect_dataset, inspect_graphs
from gnn_fiedler_approx.custom_models import GNNWrapper, premade_gnns, custom_gnns
from gnn_fiedler_approx.gnn_utils.utils import (
    create_combined_histogram,
    create_graph_wandb,
    extract_graphs_from_batch,
    graphs_to_tuple,
    print_dataset_splits
)
from gnn_fiedler_approx.gnn_utils.transformations import DatasetTransformer, resolve_transform

pio.renderers.default = "browser"  # Use browser for Plotly visualizations.


# GLOBAL VARIABLES
BEST_MODEL_PATH = pathlib.Path(__file__).parents[1] / "models"
BEST_MODEL_PATH.mkdir(exist_ok=True, parents=True)
BEST_MODEL_NAME = "best_model.pth"

SORT_DATA = False
USE_HYBRID_LOADING = False

if "PBS_O_HOME" in os.environ:
    # We are on the HPC - adjust for the CPU count and VRAM.
    ON_HPC = True
    BATCH_SIZE = "100%"
    NUM_WORKERS = 8
else:
    ON_HPC = False
    BATCH_SIZE = "100%"
    NUM_WORKERS = 8


class EvalType(enum.Enum):
    NONE = 0
    BASIC = 1
    DETAILED = 2
    FULL = 3


class EvalTarget(enum.Enum):
    LAST = "last"
    BEST = "best"


# ***************************************
# ************** MODULES ****************
# ***************************************
class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs((target - input) / target))


# ***************************************
# *************** DATASET ***************
# ***************************************
def load_dataset(
    selected_graph_sizes,
    selected_features=None,
    label_normalization=None,
    transform=None,
    split=0.8,
    batch_size: Union[int, str] = "100%",
    seed=42,
    suppress_output=False,
):
    # Save dataset configuration.
    dataset_config = {
        "name": "ConnectivityDataset",
        "selected_graphs": str(selected_graph_sizes),
        "split": split,
        "batch_size": batch_size,
        "seed": seed,
    }

    # Resolve the transforms.
    transform = resolve_transform(transform)

    # Load the dataset.
    try:
        root = pathlib.Path(__file__).parents[1] / "Dataset"  # For standalone script.
    except NameError:
        root = pathlib.Path().cwd().parents[1] / "Dataset"  # For Jupyter notebook.
    graphs_loader = GraphDataset(selection=selected_graph_sizes, seed=seed)
    dataset = ConnectivityDataset(root, graphs_loader, selected_features=selected_features, transform=transform)

    # Display general information about the dataset.
    if not suppress_output:
        inspect_dataset(dataset)

    # Compute any necessary or optional dataset properties.
    dataset_props = {}
    dataset_props["feature_dim"] = dataset.num_features  # type: ignore

    dataset_config["num_graphs"] = len(dataset)
    features = selected_features if selected_features else dataset.features
    dynamic_features = dataset.dynamic_features

    # Shuffle and split the dataset.
    # TODO: Splitting after shuffle gives relatively balanced splits between the graph sizes, but it's not perfect.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = dataset.shuffle()

    # Flexible dataset splitting. Can be split to train/test or train/val/test.
    if isinstance(dataset_config["split"], tuple):
        train_size, val_size = dataset_config["split"]
        train_size = round(train_size * len(dataset))
        val_size = round(val_size * len(dataset))
    else:
        train_size = round(dataset_config["split"] * len(dataset))
        val_size = len(dataset) - train_size

    train_dataset = dataset[:train_size]
    if val_size > 0:
        val_dataset = dataset[train_size:train_size + val_size]
    else:
        val_dataset = train_dataset
    val_size = len(val_dataset)
    test_dataset = dataset[train_size + val_size:]
    test_size = len(test_dataset)

    if not suppress_output:
        print_dataset_splits(train_dataset, val_dataset, test_dataset)

    # Perform optional transformations.
    # NOTE: From this point on, the dataset is a list of Data objects, not a Dataset object.
    dataset_transformer = DatasetTransformer(label_normalization)
    train_dataset, test_dataset = dataset_transformer.normalize_labels(train_dataset, test_dataset)
    print(dataset_transformer.params)
    dataset_props["transformation"] = dataset_transformer

    # Batch and load data.
    max_dataset_len = max(train_size, val_size, test_size)
    if isinstance(batch_size, str):
        bs = float(batch_size.strip("%")) / 100.0
        batch_size = int(np.ceil(bs * max_dataset_len))
    loader_kwargs = {"num_workers": NUM_WORKERS, "persistent_workers": True if NUM_WORKERS > 0 else False}
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **loader_kwargs)  # type: ignore
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, **loader_kwargs)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)  # type: ignore

    # If the whole dataset fits in memory, we can use the following lines to get a single large batch.
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader)) if test_size else None

    train_data_obj = train_batch if (train_size <= batch_size and not dynamic_features) else train_loader
    if USE_HYBRID_LOADING:
        val_data_obj = val_batch if val_size <= batch_size else [val_batch for val_batch in val_loader]
        test_data_obj = test_batch if test_size <= batch_size else [test_batch for test_batch in test_loader]
    else:
        val_data_obj = val_batch if val_size <= batch_size else val_loader
        test_data_obj = test_batch if test_size <= batch_size else test_loader

    dataset_props["num_batches"] = math.ceil(max_dataset_len / batch_size)
    if not suppress_output:
        print()
        print(f"Batches: ({dataset_props['num_batches']})")
        print("========================================")
        for step, data in enumerate(train_loader):
            print(f"Step {step + 1}:")
            print(f"Number of graphs in the batch: {data.num_graphs}")
            print(data)
            print(f"Average target: {torch.mean(data.y)}")
            print()

            if step == 5:
                print("The rest of batches are not displayed...")
                break
        print("========================================\n")

    # dataset_props["in_deg_hist"] = PNAConv.get_degree_histogram(train_loader)

    return train_data_obj, val_data_obj, test_data_obj, dataset_config, features, dataset_props


# ***************************************
# ************* FUNCTIONS ***************
# ***************************************
def generate_model(architecture, in_channels, hidden_channels, gnn_layers, **kwargs):
    """Generate a Neural Network model based on the architecture and hyperparameters."""
    # GLOBALS: device, premade_gnns, custom_gnns
    if architecture in premade_gnns:
        model = GNNWrapper(
            gnn_model=getattr(importlib.import_module("torch_geometric.nn"), architecture),
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
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError("Only Adam optimizer is currently supported.")


def training_pass(model, batch, optimizer, criterion):
    """Perform a single training pass over the batch."""
    if SORT_DATA:
        batch = batch.sort(sort_by_row=False)

    data = batch.to(device)  # Move to CUDA if available.
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index, batch=data.batch)  # Perform a single forward pass.
    loss = criterion(out.squeeze(dim=1), data.y)  # Compute the loss.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def testing_pass(model, batch, criterion):
    """Perform a single testing pass over the batch."""
    if SORT_DATA:
        batch = batch.sort(sort_by_row=False)

    with torch.no_grad():
        data = batch.to(device)
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = criterion(out.squeeze(dim=1), data.y)  # Compute the loss.
    return loss


def do_train(model, data, optimizer, criterion):
    """Train the model on individual batches or the entire dataset."""
    model.train()

    if isinstance(data, (DataLoader, list)):
        avg_loss = torch.tensor(0.0, device=device)
        for batch in data:  # Iterate in batches over the training dataset.
            avg_loss += training_pass(model, batch, optimizer, criterion)
        loss = avg_loss / len(data)  # TODO: Doesn't work for different batch sizes?
    elif isinstance(data, Data):
        loss = training_pass(model, data, optimizer, criterion)
    else:
        raise ValueError(f"Data must be a DataLoader or a Batch object, but got: {type(data)}.")

    return loss.item()


def do_test(model, data, criterion):
    """Test the model on individual batches or the entire dataset."""
    model.eval()

    if isinstance(data, (DataLoader, list)):
        avg_loss = torch.tensor(0.0, device=device)
        for batch in data:
            avg_loss += testing_pass(model, batch, criterion)
        loss = avg_loss / len(data)
    elif isinstance(data, Data):
        loss = testing_pass(model, data, criterion)
    else:
        raise ValueError(f"Data must be a DataLoader or a Batch object, but got: {type(data)}.")

    return loss.item()


def train(
    model, optimizer, criterion, train_data_obj, val_data_obj, num_epochs=100, log_freq=10, suppress_output=False, save_best=False
):
    # GLOBALS: device

    # Prepare for training.
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    best_loss = float("inf")
    val_loss = 0

    # This is for the hybrid approach described below.
    train_data = train_data_obj

    # Start the training loop with timer.
    training_timer = codetiming.Timer(logger=None)
    epoch_timer = codetiming.Timer(logger=None)
    training_timer.start()
    epoch_timer.start()
    for epoch in range(1, num_epochs + 1):
        # Hybrid approach for batching:
        #   run set number of epochs with one permutation and then get new batches from the DataLoader.
        if USE_HYBRID_LOADING and (epoch - 1) % 10 == 0 and isinstance(train_data_obj, DataLoader):
            train_data = [batch for batch in train_data_obj]

        # Perform one pass over the training set and then test on both sets.
        train_loss = do_train(model, train_data, optimizer, criterion)
        val_loss = do_test(model, val_data_obj, criterion)

        # Store the losses.
        train_losses[epoch - 1] = train_loss
        val_losses[epoch - 1] = val_loss

        # Save the best model.
        if save_best and epoch >= 0.3 * num_epochs and val_loss < best_loss:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, BEST_MODEL_PATH)
            best_loss = val_loss

        # Log the losses to W&B.
        if log_freq == 1:
            avg_train_loss = train_loss
            avg_val_loss = val_loss
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        elif epoch % log_freq == 0:
            avg_train_loss = np.mean(train_losses[max(0, epoch - log_freq):epoch])
            avg_val_loss = np.mean(val_losses[max(0, epoch - log_freq):epoch])
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)

        # Print the losses every log_freq epochs.
        if epoch % log_freq == 0 and not suppress_output:
            print(
                f"Epoch: {epoch:03d}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Avg. duration: {epoch_timer.stop() / 10:.4f} s"
            )
            epoch_timer.start()

    epoch_timer.stop()
    duration = training_timer.stop()

    results = {"train_losses": train_losses, "val_losses": val_losses, "duration": duration}
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
    predictions = out.cpu().squeeze(dim=1).numpy()
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


def baseline(train_data, val_data, test_data, criterion):
    if isinstance(train_data, DataLoader) or isinstance(val_data, DataLoader) or isinstance(test_data, DataLoader):
        return np.inf, np.inf, np.inf

    # Average target value on the given data
    avg = torch.mean(train_data.y)

    # Mean absolute error
    train = criterion(train_data.y, avg * torch.ones_like(train_data.y)).item()
    val = criterion(val_data.y, avg * torch.ones_like(val_data.y)).item()
    test = criterion(test_data.y, avg * torch.ones_like(test_data.y)).item() if test_data else np.inf

    return train, val, test


def evaluate(
    model, epoch, criterion, train_data, test_data, dst, plot_graphs=False, make_table=False, suppress_output=False
):
    model.eval()
    df = pd.DataFrame()

    # Loss on the train set.
    train_loss = do_test(model, train_data, criterion)
    test_loss = do_test(model, test_data, criterion)

    # Build a detailed results DataFrame.
    with torch.no_grad():
        if isinstance(test_data, DataLoader):
            for batch in test_data:
                df = pd.concat([df, eval_batch(model, batch, plot_graphs)])
        elif isinstance(test_data, Data):
            df = eval_batch(model, test_data, plot_graphs)
        else:
            raise ValueError("Data must be a DataLoader or a Batch object.")

    df["True"] = dst.reverse_transform(df["True"])
    df["Predicted"] = dst.reverse_transform(df["Predicted"])

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
    fig_err_vs_true = create_combined_histogram(df, "True", "Error %")

    plot_df = pd.DataFrame()
    plot_df["abs(Error %)"] = np.abs(df["Error %"])
    plot_df.sort_values(by="abs(Error %)", inplace=True)
    plot_df.reset_index(drop=True, inplace=True)
    fig_err_curve = px.line(plot_df, x="abs(Error %)", y=(plot_df.index + 1) / len(plot_df) * 100, title="Error curve")
    fig_err_curve.update_xaxes(showspikes=True, tickvals=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig_err_curve.update_yaxes(showspikes=True, nticks=10, title_text="Percentage of graphs")

    if not suppress_output:
        print(f"Evaluating model at epoch {epoch}.\n")
        print(
            f"Train loss: {train_loss:.8f}\n"
            f"Eval loss : {test_loss:.8f}\n"
            f"Mean error: {err_mean:.8f}\n"
            f"Std. dev. : {err_stddev:.8f}\n\n"
            f"Error brackets: {json.dumps(good_within, indent=4)}\n"
        )
        fig_abs_err.show()
        fig_rel_err.show()
        fig_err_vs_true.show()
        fig_err_curve.show()
        df = df.sort_values(by="Nodes")
        print("\nDetailed results:")
        print("==================")
        print(df)

    results = {
        "mean_err": err_mean,
        "stddev_err": err_stddev,
        "good_within": good_within,
        "fig_abs_err": fig_abs_err,
        "fig_rel_err": fig_rel_err,
        "fig_err_curve": fig_err_curve,
        "fig_err_vs_true": fig_err_vs_true,
        "table": table,
    }
    return results


def main(config=None, eval_type=EvalType.NONE, eval_target=EvalTarget.LAST, no_wandb=False, is_best_run=False):
    # GLOBALS: device

    # Helper boolean flags.
    is_sweep = config is None
    save_best = eval_target == EvalTarget.BEST
    plot_graphs = eval_type == EvalType.FULL
    make_table = eval_type.value > EvalType.BASIC.value
    suppress_output = is_sweep or ON_HPC

    # Tags for W&B.
    wandb_mode = "disabled" if no_wandb else "online"
    tags = ["norm_layers"]
    if ON_HPC:
        tags.append("HPC")
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

    # PNA dataset
    # selected_graph_sizes = {
    #     GraphType.RANDOM_MIX: (640, range(15, 25))
    # }

    # Set up the run
    run = wandb.init(mode=wandb_mode, project="gnn_fiedler_approx_v2", tags=tags, config=config)
    config = wandb.config
    if is_sweep:
        print(f"Running sweep with config: {config}...")

    if ON_HPC:
        # We are on the HPC - paralel runs use the same disk.
        global BEST_MODEL_PATH
        BEST_MODEL_PATH /= f"{run.id}_{BEST_MODEL_NAME}"
    else:
        BEST_MODEL_PATH /= BEST_MODEL_NAME

    # TODO: It would be better to define these conditions in the config file.
    if config["batch_size"] == "100%":
        config.update({"learning_rate": 0.0012199, "dropout": 0.05}, allow_val_change=True)
    elif config["batch_size"] == 32:
        config.update({"learning_rate": 0.00045955}, allow_val_change=True)
    elif config["batch_size"] == 256:
        config.update({"learning_rate": 0.001036}, allow_val_change=True)

    # Set up model configuration.
    model_kwargs = config.get("model_kwargs", {})
    pool_kwargs = dict()

    if config["architecture"] == "GIN":
        model_kwargs.update({"train_eps": True})
    elif config["architecture"] == "GAT":
        model_kwargs.update({"v2": True})
    # TODO: Check

    bs = config.get("batch_size", BATCH_SIZE)

    # For this combination of parameters, the model is too large to fit in memory, so we need to reduce the batch size.
    if model_kwargs and model_kwargs.get("aggr") == "lstm" and model_kwargs.get("project") and bs > 0.5:
        bs = "50%"

    # Load the dataset.
    train_data_obj, val_data_obj, test_data_obj, dataset_config, features, dataset_props = load_dataset(
        selected_graph_sizes,
        selected_features=config.get("selected_features", None),
        label_normalization=config.get("label_normalization"),
        transform=config.get("transform"),
        batch_size=bs,
        split=config.get("dataset", {}).get("split", 0.8),
        suppress_output=suppress_output,
    )

    wandb.config["dataset"] = dataset_config
    if "selected_features" not in wandb.config or not wandb.config["selected_features"]:
        wandb.config["selected_features"] = features

    # PNA DegreeScalerAggregation requires the in-degree histogram for normalization.
    if config["pool"] == "PNA":
        pool_kwargs["deg"] = dataset_props["in_deg_hist"]
    if config["architecture"] == "PNA":
        model_kwargs["deg"] = dataset_props["in_deg_hist"]

    # Split the normalization layer string.
    norm = config.get("norm")
    norm_kwargs = dict()
    if norm is not None and "_" in norm:
        norm_type, norm_param = norm.split("_", 1)
        if norm_type == "layer":
            norm_kwargs["mode"] = norm_param
        else:
            raise ValueError(f"Unknown normalization type: {norm}.")
        norm = norm_type

    # Update number of epochs for fair evaluation.
    if "iterations" in config:
        config.update({"epochs": round(config["iterations"] / dataset_props["num_batches"])}, allow_val_change=True)

    # Set up the model, optimizer, and criterion.
    model = generate_model(
        config["architecture"],
        dataset_props["feature_dim"],
        config["hidden_channels"],
        config["gnn_layers"],
        mlp_layers=config.get("mlp_layers", 1),
        act=config.get("activation", "relu"),
        dropout=float(config.get("dropout", 0.0)),
        pool=config.get("pool", "max"),
        pool_kwargs=pool_kwargs,
        norm=norm,
        norm_kwargs=norm_kwargs,
        jk=config.get("jk"),
        **model_kwargs,
    )
    optimizer = generate_optimizer(model, config["optimizer"], config["learning_rate"])
    criterion = torch.nn.L1Loss()

    # Print baseline results.
    baseline_results = baseline(train_data_obj, val_data_obj, test_data_obj, criterion)
    print("Baseline results:")
    print("=================")
    print(f"Train baseline: {baseline_results[0]:.8f}")
    print(f"Val baseline: {baseline_results[1]:.8f}")
    print(f"Test baseline: {baseline_results[2]:.8f}")
    print()

    wandb.watch(model, criterion, log="all", log_freq=100)
    # torchexplorer.watch(model, backend="wandb")

    # Run training.
    print("Training...")
    print("===========")
    train_results = train(
        model,
        optimizer,
        criterion,
        train_data_obj,
        val_data_obj,
        config["epochs"],
        log_freq=10 if bs == "100%" else 1,
        suppress_output=suppress_output,
        save_best=save_best,
    )
    run.summary["best_train_loss"] = min(train_results["train_losses"])
    run.summary["best_val_loss"] = min(train_results["val_losses"])
    run.summary["best_test_loss"] = min(train_results["val_losses"])  # For compatibility with earlier experiments.
    run.summary["duration"] = train_results["duration"]
    if not suppress_output:
        plot_training_curves(
            config["epochs"], train_results["train_losses"], train_results["val_losses"], type(criterion).__name__
        )

    # Run evaluation.
    if eval_type != EvalType.NONE:
        epoch = config["epochs"]
        if eval_target == EvalTarget.BEST:
            checkpoint = torch.load(BEST_MODEL_PATH, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]

        print("\nEvaluation results:")
        print("===================")
        eval_results = evaluate(
            model,
            epoch,
            criterion,
            train_data_obj,
            test_data_obj or val_data_obj,
            dataset_props["transformation"],
            plot_graphs,
            make_table,
            suppress_output=suppress_output
        )
        run.summary["mean_err"] = eval_results["mean_err"]
        run.summary["stddev_err"] = eval_results["stddev_err"]
        run.summary["good_within"] = eval_results["good_within"]
        run.log(
            {
                "abs_err_hist": eval_results["fig_abs_err"],
                "rel_err_hist": eval_results["fig_rel_err"],
                "err_curve": eval_results["fig_err_curve"],
                "err_vs_true_hist": eval_results["fig_err_vs_true"],
            }
        )

        if eval_type.value > EvalType.BASIC.value:
            run.log({"results_table": eval_results["table"]})

        if is_best_run:
            # Name the model with the current time and date to make it uniq
            model_name = f"{model.descriptive_name}-{datetime.datetime.now().strftime('%d%m%y_%H%M')}"
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(str(BEST_MODEL_PATH))
            run.log_artifact(artifact)

    print(f"Duration: {train_results['duration']:.8f} s.")
    if is_sweep:
        print("    ...DONE.")
        if eval_type != EvalType.NONE:
            print(f"Mean error: {eval_results['mean_err']:.8f}")
            print(f"Std. dev.: {eval_results['stddev_err']:.8f}")
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
            "mlp_layers": 2,
            "activation": "tanh",
            "pool": "softmax",
            "norm": "message",
            "jk": "cat",
            "dropout": 0.0,
            ## Training configuration
            "optimizer": "adam",
            "learning_rate": 0.005,
            "batch_size": 32,
            "epochs": 5000,
            ## Dataset configuration
            "label_normalization": None,
            "transform": "normalize_features",
            "selected_features": ["degree", "degree_centrality", "core_number", "triangles", "clustering", "close_centrality"]
        }
        run = main(global_config, eval_type, eval_target, args.no_wandb, args.best)
        run.finish()
    else:
        run = main(None, eval_type, eval_target, False, args.best)
