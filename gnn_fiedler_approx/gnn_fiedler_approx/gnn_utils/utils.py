from collections import Counter

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import torch_geometric.utils as pygUtils
from sklearn.manifold import TSNE

# from umap import UMAP


def extract_graphs_from_batch(data):
    """Convert the PyG batch object to a list of NetworkX graphs."""
    nx_graphs = [pygUtils.to_networkx(d, to_undirected=True) for d in data.to_data_list()]
    return nx_graphs


def graphs_to_tuple(nx_graphs):
    """Convert a list of NetworkX graphs to a list of tuples (graph6, number of nodes, number of edges)."""
    return [
        (nx.to_graph6_bytes(g, header=False).decode("ascii").strip("\n"), g.number_of_nodes(), g.number_of_edges())
        for g in nx_graphs
    ]


def count_parameters(model):
    """Return the number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_dataset_splits(train_dataset, val_dataset, test_dataset):
    train_counter = Counter([data.x.shape[0] for data in train_dataset])  # type: ignore
    val_counter = Counter([data.x.shape[0] for data in val_dataset])  # type: ignore
    test_counter = Counter([data.x.shape[0] for data in test_dataset])  # type: ignore

    sizes = set(train_counter + val_counter + test_counter)
    total_per_size = {size: train_counter[size] + val_counter[size] + test_counter[size] for size in sizes}
    train_splits_per_size = {size: round(train_counter.get(size, 0) / total_per_size[size], 2) for size in sizes}
    val_splits_per_size = {size: round(val_counter.get(size, 0) / total_per_size[size], 2) for size in sizes}
    test_splits_per_size = {size: round(test_counter.get(size, 0) / total_per_size[size], 2) for size in sizes}

    data = []
    for size in sorted(sizes):
        data.append([size, train_counter.get(size, 0), val_counter.get(size, 0), test_counter.get(size, 0),
                    train_splits_per_size[size], val_splits_per_size[size], test_splits_per_size[size]])

    df = pd.DataFrame(data, columns=["Size", "Train", "Val", "Test", "Train Split", "Val Split", "Test Split"])
    df.loc["Total"] = ["Total", train_counter.total(), val_counter.total(), test_counter.total(), "", "", ""]
    df = df.set_index("Size").T  # Transpose the dataframe

    df_str = df.to_string()
    lines = df_str.split('\n')
    separator = '-' * len(lines[0])
    lines.insert(1, separator)  # Insert separator after the first row
    lines.insert(5, separator)  # Insert separator after the fourth row
    lines.append(separator)  # Append separator at the end
    print('\n'.join(lines))


def visualize_embeddings(embeddings, labels, method="tsne"):
    """
    Visualize embeddings using t-SNE or UMAP.

    Args:
        embeddings (dict or torch.Tensor): A dictionary of embeddings for each epoch or a single tensor.
        labels (torch.Tensor): Labels corresponding to the embeddings.
        method (str): The dimensionality reduction method to use, either "tsne" or "umap".
    """

    if not isinstance(embeddings, dict):
        embeddings = {0: embeddings}

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, early_exaggeration=15.0)
    elif method == "umap":
        reducer = UMAP(random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'tsne' or 'umap'.")

    epochs = []
    reduced_embeddings = []
    for epoch in embeddings:
        input_embedding = embeddings[epoch].detach().cpu().numpy()
        reduced_embedding = reducer.fit_transform(input_embedding)
        reduced_embeddings.append(reduced_embedding)
        epochs.append(epoch)

    # Create a scatter plot
    scatter_options = dict(mode='markers', marker=dict(size=7, color=labels, colorscale='Viridis', showscale=True))
    frame_duration = 500  # Duration for each frame in milliseconds
    fig = {
        "data": [go.Scatter(x=reduced_embeddings[0][:, 0], y=reduced_embeddings[0][:, 1], **scatter_options)],
        "layout": {},
        "frames": []
    }
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Epoch:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": frame_duration, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    xlimits = [float('inf'), float('-inf')]
    ylimits = [float('inf'), float('-inf')]
    for epoch, reduced_embedding in zip(epochs, reduced_embeddings):
        fig["frames"].append(
            go.Frame(
                data=[go.Scatter(x=reduced_embedding[:, 0], y=reduced_embedding[:, 1], **scatter_options)],
                name=str(epoch)
            )
        )
        xlimits[0] = min(xlimits[0], reduced_embedding[:, 0].min() * 1.1)
        xlimits[1] = max(xlimits[1], reduced_embedding[:, 0].max() * 1.1)
        ylimits[0] = min(ylimits[0], reduced_embedding[:, 1].min() * 1.1)
        ylimits[1] = max(ylimits[1], reduced_embedding[:, 1].max() * 1.1)

        slider_step = {"args": [
            [epoch],
            {"frame": {"duration": frame_duration, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": frame_duration}}
        ],
            "label": epoch,
            "method": "animate"}
        sliders["steps"].append(slider_step)

    fig["layout"]["sliders"] = [sliders]
    fig["layout"]["xaxis"] = {"range": xlimits}
    fig["layout"]["yaxis"] = {"range": ylimits}

    fig = go.Figure(fig)

    return fig
