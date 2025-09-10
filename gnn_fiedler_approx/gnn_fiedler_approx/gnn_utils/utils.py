import multiprocessing as mp
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch_geometric.utils as pygUtils
import wandb
from sklearn.manifold import TSNE
# from umap import UMAP

from .visualize import GraphVisualization


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


def create_graph_vis(G, features=None):
    """Create a Plotly figure of a NetworkX graph."""
    pos = nx.spring_layout(G)
    vis = GraphVisualization(
        G, pos, node_text_position='top left', node_size=20,
    )
    fig = vis.create_figure()

    if features:
        bar_traces = add_feature_visualization(pos, features["data"], features["names"])
        fig.add_traces(bar_traces)

    return fig
    # # Edges
    # edge_x = []
    # edge_y = []
    # for edge in G.edges():
    #     x0, y0 = pos[edge[0]]
    #     x1, y1 = pos[edge[1]]
    #     edge_x.extend([x0, x1, None])
    #     edge_y.extend([y0, y1, None])

    # edge_trace = go.Scatter(
    #     x=edge_x, y=edge_y,
    #     line=dict(width=0.5, color='#888'),
    #     hoverinfo='none',
    #     mode='lines'
    # )
    # # Nodes
    # node_x = []
    # node_y = []
    # for node in G.nodes():
    #     x, y = pos[node]
    #     node_x.append(x)
    #     node_y.append(y)

    # node_trace = go.Scatter(
    #     x=node_x, y=node_y,
    #     mode='markers',
    #     hoverinfo='text',
    #     line_width=2
    # )
    # # Put it all together.
    # fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout())
    # return fig


def create_graph_wandb(G):
    """Convert a Plotly figure of the NetworkX graph to a W&B  html representation."""
    # return wandb.Html(plotly.io.to_html(create_graph_vis(G)))
    # return wandb.Image(create_graph_vis(G))
    fig = create_graph_vis(G)
    return wandb.Html(fig.to_html(auto_play=False, full_html=False, include_plotlyjs='cdn'))


def create_graph_vis_parallel(graphs):
    """Create a Plotly figure of a NetworkX graph with parallel processing."""
    with mp.Pool() as pool:
        graph_visuals = pool.imap(create_graph_vis, graphs, chunksize=100)

    return list(graph_visuals)


def add_feature_visualization(pos, data, features):
    """Add a bar chart visualization of node features to a Plotly figure."""
    # Create bar charts for each node
    bar_charts = []
    for i, p in pos.items():
        bar_charts.append(
            go.Bar(x=features, y=data[i], orientation='h')
        )
    return bar_charts


def count_parameters(model):
    """Return the number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_combined_histogram(df, bars, line, option="boxplot", title=""):
    """
    Create a chart for visualizing the distribution of a metric across a dataset labels.

    Specifically, this function creates a histogram of the frequency of dataset labels (bars)
    and a line plot of the average error according to some metric (line) across the dataset labels.
    For example, you can see how much the model makes mistakes for poorly or well connected graphs.
    """
    max_val = int(df[bars].max())
    if max_val == 1:
        nbins = 51
        rangebins = (-0.01, max_val + 0.01)
    else:
        nbins = int(df[bars].max() * 5) + 1
        rangebins = (-0.1, df[bars].max() + 0.1)

    df[bars] = df[bars].round(5)

    hist = np.histogram(df[bars], range=rangebins, bins=nbins)
    bin_edges = np.round(hist[1], 3)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    temp_df = df[[bars, line]].copy()
    temp_df["bin"] = pd.cut(df[bars], bins=bin_edges) # type: ignore
    stats = temp_df.groupby("bin", observed=False)[line].agg(['mean', 'std'])
    mean_per_bin = stats['mean']
    std_per_bin = stats['std']

    fig = go.Figure()

    # Add histogram (count)
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist[0],
            name="Frequency",
            hovertemplate="Range: %{customdata[0]} - %{customdata[1]}<br>Count: %{y}<extra></extra>",
            customdata=[(str(round(bin_edges[i], 3)), str(round(bin_edges[i + 1], 3))) for i in range(len(bin_edges) - 1)]
        )
    )

    if option == "mean":
        # Add line plot (average metric)
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=mean_per_bin,
            mode='lines+markers',
            name=f"Average {line}",
            line=dict(color=px.colors.qualitative.Plotly[1]),
            marker=dict(size=8),
            yaxis="y2"
        ))

        # Update layout for dual y-axes
        fig.update_layout(
            xaxis_title=f"{bars} Value",
            yaxis_title="Count",
            yaxis2=dict(
                title=f"Average {line}",
                overlaying='y',
                side='right'
            ),
            bargap=0,
        )

    elif option == "std":
        upper_bound = mean_per_bin + std_per_bin
        lower_bound = mean_per_bin - std_per_bin

        fig.add_trace(go.Scatter(
            x=np.concatenate([bin_centers, bin_centers[::-1]]),  # Fill between upper and lower bounds
            y=np.nan_to_num(np.concatenate([upper_bound, lower_bound[::-1]])),
            fill='toself',
            fillcolor=px.colors.qualitative.Plotly[1],  # Red shade with transparency
            opacity=0.2,
            line=dict(color='rgba(255, 0, 0, 0)'),  # No line for the shaded area
            hoverinfo="skip",  # Skip hover info for the shaded area
            name="Std Deviation",
            yaxis="y2"
        ))

        # Add line plot for mean
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=mean_per_bin,
            mode='lines+markers',
            name="Mean",
            line=dict(color=px.colors.qualitative.Plotly[1]),
            marker=dict(size=8),
            yaxis="y2"
        ))

        # Update layout for the plot
        fig.update_layout(
            xaxis_title=f"{bars} Value",
            yaxis_title="Count",
            yaxis2=dict(
                title=f"{line} Metrics",
                overlaying='y',
                side='right'
            ),
            bargap=0.0,  # Controls the gap between bars
            barmode='overlay',  # Overlay the bars on top of each other
        )

    elif option == "boxplot":
        # Add boxplots for errors
        for i, bin_label in enumerate(temp_df["bin"].cat.categories):
            # Extract errors for the current bin
            bin_errors = temp_df.loc[temp_df["bin"] == bin_label, line]

            # Add a boxplot for each bin
            fig.add_trace(go.Box(
                y=bin_errors,
                x=[bin_centers[i]] * len(bin_errors),  # Align with bin center
                name=f"Bin {i + 1}",
                marker=dict(color=px.colors.qualitative.Plotly[1]),
                boxmean=True,  # Show mean as a marker
                showlegend=(i == 0),  # Show legend only for the first boxplot
                width=(bin_edges[1] - bin_edges[0]) * 0.8,
                yaxis="y2"
            ))

        # Update layout for the plot
        fig.update_layout(
            title=title,
            xaxis_title=f"{bars} Value",
            yaxis_title="Count",
            yaxis2=dict(
                title=f"{line} Distribution",
                overlaying='y',
                side='right'
            ),
            bargap=0,  # Controls the gap between bars
            barmode='overlay',  # Overlay the bars on top of each other
            boxmode='group',  # Group the boxplots
        )

    return fig


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
