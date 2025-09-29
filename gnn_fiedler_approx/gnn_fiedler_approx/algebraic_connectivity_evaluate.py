import pathlib
import plotly.io as pio
import torch
import yaml
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader

from gnn_fiedler_approx.custom_models.custom_gnns import GNNWrapper
from gnn_fiedler_approx.algebraic_connectivity_script import EvalType, EvalTarget, load_dataset, generate_loss_function, baseline, evaluate


def combine_data_objects(data_objs):
    data_list = []
    for data_obj in data_objs:
        if isinstance(data_obj, (DataLoader, list)):
            data_list.extend(data_obj)
        else:
            data_list.append(data_obj)

    return data_list


def main(config):
    # GLOBALS: device

    seed_everything(42)

    # Helper boolean flags.
    is_sweep = False
    plot_graphs = False
    make_table = False
    plot_embeddings = False
    suppress_output = is_sweep

    bs = config["batch_size"]


    # Load the dataset. Parse selected features and graph sizes.
    if sfi := config.get("selected_features_id") is not None:
        assert "available_features" in config, "Available features must be provided for selected_features_id."
        indicators = [int(bit) for bit in f"{sfi:0{len(config['available_features'])}b}"]
        config["selected_features"] = [feat for feat, ind in zip(config["available_features"], indicators) if ind]

    sgs = config.get("dataset", {}).get("selected_graphs", None)
    selected_graph_sizes = yaml.safe_load(sgs) if sgs is not None else None

    train_data_obj, val_data_obj, test_data_obj, dataset_config, features, dataset_props = load_dataset(
        selected_graph_sizes,
        selected_features=config.get("selected_features", None),
        label_normalization=config.get("label_normalization"),
        transform=config.get("transform"),
        batch_size=bs,
        split=config.get("dataset", {}).get("split", (0.6, 0.2)),
        suppress_output=suppress_output,
    )

    # Set up the model, optimizer, and criterion.
    model = GNNWrapper.load(config["model_path"], device=device)
    criterion = generate_loss_function(config["loss"])

    # Print baseline results.
    baseline_results = baseline(train_data_obj, val_data_obj, test_data_obj, criterion)
    print("Baseline results:")
    print("=================")
    print(f"Train baseline: {baseline_results[0]:.8f}")
    print(f"Val baseline: {baseline_results[1]:.8f}")
    print(f"Test baseline: {baseline_results[2]:.8f}")
    print()

    epoch = -1

    datasets_for_evaluation = ["validation", "test", "entire"]
    dataobjects_for_evaluation = [val_data_obj, test_data_obj, combine_data_objects([train_data_obj, val_data_obj, test_data_obj])]

    for dataset, eval_data_obj in zip(datasets_for_evaluation, dataobjects_for_evaluation):
        header = f"Evaluation results for {dataset} dataset:"
        print(f"\n{header}")
        print("=" * len(header))
        evaluate(
            model,
            epoch,
            criterion,
            train_data_obj,
            eval_data_obj,
            dataset_props["transformation"],
            title=f"Results on the {dataset} dataset",
            plot_graphs_wandb=plot_graphs,
            plot_embeddings=plot_embeddings,
            make_table_wandb=make_table,
            suppress_output=suppress_output
        )


if __name__ == "__main__":
    pio.renderers.default = "browser"  # Use browser for Plotly visualizations.

    # Get available device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded torch. Using *{device}* device.")
    __builtins__.device = device  # A hack to allow imported functions to use the device

    global_config = {
        ## Model and training configuration
        "model_path": pathlib.Path(__file__).parents[1] / "models" / "best_model_full.pth",
        "loss": "MAPE",
        "batch_size": "100%",
        ## Dataset configuration
        "label_normalization": None,
        "transform": None,
        "selected_features": ["degree", "degree_centrality", "triangles", "clustering", "local_density"],
        # "dataset": {
        #     "selected_graphs": "{3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}",
        # },
    }
    main(global_config)
