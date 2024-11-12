import multiprocessing as mp

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch_geometric.utils as pygUtils
import wandb

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


def create_combined_histogram(df, bars, line):
    """
    Create a chart for visualizing the distribution of a metric across a dataset labels.

    Specifically, this function creates a histogram of the frequency of dataset labels (bars)
    and a line plot of the average error according to some metric (line) across the dataset labels.
    For example, you can see how much the model makes mistakes for poorly or well connected graphs.
    """
    nbins = int(df[bars].max() * 5)

    hist = np.histogram(df[bars], range=(0, df[bars].max()), bins=nbins)
    bin_edges = hist[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    temp_df = df[[bars, line]].copy()
    temp_df["bin"] = pd.cut(df[bars], bins=bin_edges) # type: ignore
    avg_metric_per_bin = temp_df.groupby("bin", observed=False)[line].mean()

    fig = go.Figure()

    # Add histogram (count)
    fig.add_trace(go.Histogram(
        x=df[bars],
        name='Count',
        nbinsx=nbins,
        xbins=dict(
            start=0,
            end=df[bars].max(),
        )
    ))

    # Add line plot (average metric)
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=avg_metric_per_bin,
        mode='lines+markers',
        name=f"Average {line}",
        line=dict(color="red"),
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

    return fig
