import math
import warnings

import codetiming
import concurrent.futures
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from my_graphs_dataset import GraphDataset
from gnn_fiedler_approx.gnn_utils.timeout import timeout


warnings.filterwarnings("ignore")


functions = {
    'lambda_2': lambda x: nx.laplacian_spectrum(x)[1],
    'nodes': lambda G: G.number_of_nodes(),
    'edges': lambda G: G.number_of_edges(),
    'assortativity': nx.degree_assortativity_coefficient,
    'asteroidal': nx.is_at_free,
    'has_bridges': nx.has_bridges,
    'chordal': nx.is_chordal,
    'transitivity': nx.transitivity,
    'average_clustering': nx.average_clustering,
    'node_connectivity': nx.node_connectivity,
    'edge_connectivity': nx.edge_connectivity,
    'density': nx.density,
    'girth': nx.girth,
    'diameter': nx.diameter,
    'radius': nx.radius,
    'effective_graph_resistance': nx.effective_graph_resistance,
    'kemeny_constant': nx.kemeny_constant,
    # 'is_distance_regular': nx.is_distance_regular,
    # 'is_strongly_regular': nx.is_strongly_regular,
    'local_efficiency': nx.local_efficiency,
    'global_efficiency': nx.global_efficiency,
    'is_planar': nx.is_planar,
    # 'is_regular': nx.is_regular,
    's_metric': nx.s_metric,
    # 'is_tree': nx.is_tree,
    'wiener_index': nx.wiener_index,
    'schultz_index': nx.schultz_index,
    'gutman_index': nx.gutman_index,
    'lambda_n': lambda G: nx.laplacian_spectrum(G)[-1],  # Largest eigenvalue of the Laplacian
}


def compute_metrics(G):
    if not nx.is_connected(G):
        # If the graph is not connected, return NaN for all metrics
        return {name: float('nan') for name in functions.keys()}

    results = dict.fromkeys(functions.keys(), 0)
    for name, func in functions.items():
            with timeout(10):
                results[name] = func(G)

    return results


def find_grid(total):
    # Find ncols and nrows to match 16:9 ratio as close as possible
    best_ncols, best_nrows = 1, total
    best_diff = float('inf')
    target_ratio = 16 / 9

    for ncols in range(1, total + 1):
        nrows = math.ceil(total / ncols)
        ratio = ncols / nrows
        diff = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_ncols, best_nrows = ncols, nrows

    return best_ncols, best_nrows


def main():
    # selection = {3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}; filename = "l2_correlation_3-8.png"
    # selection = {"09-25_mix_750": -1};     filename = "l2_correlation_09-25.png"
    # selection = {"26-50_mix_100": -1}; filename = "l2_correlation_26-50.png"
    selection = {"03-25_mix_750": -1, "26-50_mix_100": -1}; filename = "l2_correlation_03-50.png"
    loader = GraphDataset(selection=selection, seed=42, graph_format="graph6", retries=100)

    # records = []
    # with codetiming.Timer(name="Processing time"):
    #     for G in loader.graphs(raw=False, batch_size=1):
    #         records.append(compute_metrics(G))

    records = []
    with codetiming.Timer(name="Processing time"), concurrent.futures.ProcessPoolExecutor() as executor:
        for graphs in loader.graphs(raw=False, batch_size=10000):
            result = executor.map(compute_metrics, graphs)
            records.extend(result)

    df = pd.DataFrame.from_records(records)
    df = df.dropna()  # Drop rows with NaN values
    # Reverse the order of rows so that points with fewer nodes are plotted on top
    df = df.iloc[::-1].reset_index(drop=True)

    # Plot the correlation scatter plots
    metrics = [col for col in df.columns if col != 'lambda_2']
    n_metrics = len(metrics)

    ncols, nrows = find_grid(n_metrics)

    # Create matplotlib figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(19.2, 10.8))
    fig.suptitle("Scatter plots: lambda_2 vs other metrics", fontsize=16)

    # Flatten axes array for easier indexing
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    node_counts = df['nodes'] if 'nodes' in df.columns else None

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create scatter plot with color mapping based on node counts
        scatter = ax.scatter(
            df[metric],
            df['lambda_2'],
            c=node_counts,
            cmap='viridis',
            alpha=0.6,
            s=20
        )

        ax.set_xlabel(metric)

        # Show y axis label only in the first column
        if idx % ncols == 0:
            ax.set_ylabel('lambda_2')
        else:
            ax.set_yticklabels([])

        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Add colorbar only once, positioned on the right side
    if node_counts is not None:
        # Adjust layout to make room for colorbar
        plt.subplots_adjust(right=0.9)
        # Add colorbar to the right of all subplots
        cbar = plt.colorbar(scatter, ax=axes, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('# Nodes', rotation=270, labelpad=20)

    plt.savefig(filename, format="png", bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
