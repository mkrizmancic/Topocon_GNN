import base64
import functools
import hashlib
import json
import random
from pathlib import Path

import codetiming
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as tg_transforms
import torch_geometric.utils as tg_utils
import yaml
from torch_geometric.data import Data, InMemoryDataset

from my_graphs_dataset import GraphDataset
from gnn_link_prediction.gnn_utils.utils import canonical_label



class FeatureFilterTransform(tg_transforms.BaseTransform):
    """
    A transform that filters features of a graph's node attributes based on a given mask.

    Args:
        feature_index_mask (list or np.ndarray): A boolean mask or list of indices to filter the features.
    Methods:
        forward(data: Data) -> Data:
            Applies the feature filter to the node attributes of the input data object.
        __repr__() -> str:
            Returns a string representation of the transform with the mask.
    Example:
        >>> transform = FeatureFilterTransform([0, 2, 4])
        >>> data = Data(x=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        >>> transformed_data = transform(data)
        >>> print(transformed_data.x)
        tensor([[ 1,  3,  5],
                [ 6,  8, 10]])
    """
    # NOTE: This is a better way than doing self._data.x = self._data.x[:, mask]
    # See https://github.com/pyg-team/pytorch_geometric/discussions/7684.
    # This transform function will be automatically applied to each data object
    # when it is accessed. It might be a bit slower, but tensor slicing
    # shouldn't affect the performance too much. It is also following the
    # intended Dataset design.
    def __init__(self, feature_index_mask):
        self.mask = feature_index_mask

    def forward(self, data: Data) -> Data:
        if self.mask is not None:
            assert data.x is not None
            data.x = data.x[:, self.mask]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mask})'

class LinkPredictionDataset(InMemoryDataset):
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/gdelt.py
    # If you want to define different graphs for training and testing.
    def __init__(
        self, root, loader: GraphDataset | None = None, transform=None, pre_transform=None, pre_filter=None, **kwargs
    ):
        if loader is None:
            loader = GraphDataset()
        self.loader = loader

        self.graphs_memo = {}

        print("*****************************************")
        print(f"** Creating dataset with ID {self.hash_representation} **")
        print("*****************************************")

        # Calls InMemoryDataset.__init__ -> calls Dataset.__init__  -> calls Dataset._process -> calls self.process
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=kwargs.get("force_reload", False))

        self.load(self.processed_paths[0])

        selected_features = kwargs.get("selected_features")
        selected_extra_feature = kwargs.get("selected_extra_feature")
        mask = self.get_features_mask(selected_features, selected_extra_feature)

        # Add a feature filter transform to the transform pipeline, if needed.
        if not np.all(mask):
            feature_filter = FeatureFilterTransform(mask)
            if self.transform is not None:
                self.transform = tg_transforms.Compose([self.transform, feature_filter])
            else:
                self.transform = feature_filter

    @property
    def raw_dir(self):
        return str(self.loader.raw_files_dir.resolve())

    @property
    def raw_file_names(self):
        """
        Return a list of all raw files in the dataset.

        This method has two jobs. The returned list with raw files is compared
        with the files currently in raw directory. Files that are missing are
        automatically downloaded using download method. The second job is to
        return the list of raw file names that will be used in the process
        method.
        """
        with open(Path(self.root) / "file_list.yaml", "r") as file:
            raw_file_list = sorted(yaml.safe_load(file))
        return raw_file_list

    @property
    def processed_file_names(self):
        """
        Return a list of all processed files in the dataset.

        If a processed file is missing, it will be automatically created using
        the process method.

        That means that if you want to reprocess the data, you need to delete
        the processed files and reimport the dataset.
        """
        return [f"data_{self.hash_representation}.pt"]

    @property
    def hash_representation(self):
        dataset_props = json.dumps(
            [self.loader.hashable_selection, self.features, self.target_function.__name__, self.loader.seed]
        )
        sha256_hash = hashlib.sha256(dataset_props.encode("utf-8")).digest()
        hash_string = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")[:10]
        return hash_string

    def download(self):
        """Automatically download raw files if missing."""
        # TODO: Should check and download only missing files.
        # zip_file = Path(self.root) / "raw_data.zip"
        # zip_file.unlink(missing_ok=True)  # Delete the exising zip file.
        # download_url(raw_download_url, self.root, filename="raw_data.zip")
        # extract_zip(str(zip_file.resolve()), self.raw_dir)
        raise NotImplementedError("Automatic download is not implemented yet.")

    def process(self):
        """Process the raw files into a graph dataset."""
        # Read data into huge `Data` list.
        data_list = []
        for graph in self.loader.graphs(batch_size=1, raw=False):
            data_list.append(self.make_data(graph))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    # *************************
    # *** Feature functions ***
    # *************************
    @staticmethod
    def one_hot_degree(G):
        degrees = G.degree
        ohd = {}
        for node, deg in degrees:
            ohd[node] = [1.0 if i == deg else 0.0 for i in range(10)]
        return ohd

    feature_functions = {
        # "zero": lambda g: dict.fromkeys(g.nodes(), 0),
        # "one": lambda g: dict.fromkeys(g.nodes(), 1),
        # "random1": lambda g: {n: np.random.uniform() for n in g.nodes()},
        # "random2": lambda g: {n: np.random.uniform() for n in g.nodes()},
        "degree": lambda g: {n: float(g.degree(n)) for n in g.nodes()},
        "degree_centrality": nx.degree_centrality,
        # "core_number": nx.core_number,
        # "triangles": nx.triangles,
        # "clustering": nx.clustering,
        # "close_centrality": nx.closeness_centrality,
        "betweenness_centrality": nx.betweenness_centrality,
        # "one_hot_degree": one_hot_degree,
    }
    extra_feature_functions = {}
    # *************************

    # ************************
    # *** Target functions ***
    # ************************
    @staticmethod
    def algebraic_connectivity(G):
        L = nx.laplacian_matrix(G).toarray()
        lambdas = sorted(np.linalg.eigvalsh(L))
        return lambdas[1]

    @staticmethod
    def spectral_radius(G):
        L = nx.laplacian_matrix(G).toarray()
        lambdas = np.linalg.eigvalsh(L)
        return max(abs(lambdas))

    @staticmethod
    def normalized_algebraic_connectivity(G):
        return LinkPredictionDataset.algebraic_connectivity(G) / (G.number_of_nodes())

    target_function = staticmethod(algebraic_connectivity)

    # def target_function(self, G):
    #     func = self.algebraic_connectivity
    #     # func = self.normalized_algebraic_connectivity
    #     # func = self.spectral_radius
    #     # func = nx.node_connectivity
    #     # func = nx.effective_graph_resistance

    #     if G is None:
    #         return func.__name__
    #     return func(G)
    # ************************


    def make_data(self, G: nx.Graph) -> Data:
        """Create a PyG data object from a graph object."""
        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        # Convert the graph to a PyG data object and prepare the edge labels and index.
        # edge_index holds edges in both directions (PyG's way of storing undirected graphs).
        # edge_label_index only holds one of the edges (networkx's way of storing undirected graphs).
        torch_G = tg_utils.from_networkx(G, group_node_attrs=self.features)
        torch_G.edge_label_index = torch.tensor(list(G.edges()), dtype=torch.long).T
        torch_G.edge_label = torch.zeros(G.number_of_edges())

        # Target value of the graph before removing edges.
        initial_target = self.target_function(G)

        # Create a list of all possible graph variants by removing one edge at a time.
        # Each variant is a copy of the original graph with one edge removed.
        # The graph6 representation is used to uniquely identify each variant.
        variants = []
        for i, (u, v) in enumerate(G.edges()):
            new_G = G.copy()
            new_G.remove_edge(u, v)
            graph6 = GraphDataset.to_graph6(new_G)
            variants.append((i, new_G, graph6))

        # Convert the graph6 representations to a canonical form.
        # This ensures that the same graph structure is not duplicated.
        canonical = canonical_label([c[2] for c in variants], Path(__file__).parents[1] / "labelg").split()

        # For each variant, compute the target value and store it in the edge_label attribute.
        # The edge_label is the difference between the target value of the variant and the initial target value.
        # Use canonical labels to avoid redundant calculations.
        for (i, new_G, graph6), can in zip(variants, canonical):
            if can in self.graphs_memo:
                result = self.graphs_memo[can]
            else:
                result = self.target_function(new_G)
                self.graphs_memo[can] = result
            torch_G.edge_label[i] = result - initial_target

        return torch_G

    @property
    def features(self):
        return list(self.feature_functions.keys())

    @property
    def extra_features(self):
        return list(self.extra_feature_functions.keys())

    @functools.cached_property
    def feature_dims(self):
        """
        Calculate the dimensions of the features.

        Some features (like one-hot encoding and random) may have variable
        dimensions so dataset.num_features != len(dataset.features).
        """
        feature_dims = {}
        G = tg_utils.to_networkx(self[0], to_undirected=True)
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            try:
                feature_dims[feature] = len(feature_val[0])
            except TypeError:
                feature_dims[feature] = 1
        return feature_dims

    def get_features_mask(self, selected_features, selected_extra_feature):
        """Filter out features that are not in the selected features."""
        flags = []
        for feature in self.features:
            val = selected_features is None or feature in selected_features
            flags.extend([val] * self.feature_dims[feature])
        for feature in self.extra_features:
            val = selected_extra_feature is None or feature == selected_extra_feature
            flags.append(val)
        return np.array(flags)


def draw_graph(data):
    """
    Visualize a graph using Plotly.

    Args:
        data: A PyG data object containing graph data.
    """
    import plotly.graph_objects as go
    from plotly.express.colors import sample_colorscale

    nx_graph = tg_utils.to_networkx(data, to_undirected=True)

    # Plot nodes.
    pos = nx.spring_layout(nx_graph)
    node_trace = go.Scatter(
        x=[pos[node][0] for node in nx_graph.nodes()],
        y=[pos[node][1] for node in nx_graph.nodes()],
        mode="markers",
        marker=dict(size=20, color="blue"),
        text=[f"Node: {node}" for node in nx_graph.nodes()],
        hoverinfo="text",
    )

    # Prepare edges.
    x_edges = []
    y_edges = []
    for edge in nx_graph.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])

    # Calculate edge colors depending on the edge labels.
    edge_labels = data.edge_label.numpy()
    graph_value = LinkPredictionDataset.algebraic_connectivity(nx_graph)
    edge_labels = (edge_labels + graph_value) / (0 + graph_value)
    edge_colors = sample_colorscale("viridis", edge_labels)
    colorbar_trace  = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale="viridis",
            showscale=True,
            cmin=-graph_value,
            cmax=0,
        )
    )

    # Create individual edge traces with hover text and colors.
    edge_traces = []
    for edge, label, color in zip(nx_graph.edges(), edge_labels, edge_colors):
        x_edge = [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edge = [pos[edge[0]][1], pos[edge[1]][1], None]
        edge_hover_text = f"Edge: {edge}, Value: {label:.4f}"

        edge_trace = go.Scatter(
            x=x_edge,
            y=y_edge,
            line=dict(width=10, color=color),
            hoverinfo="text",
            hovertext=edge_hover_text,
            mode="lines",
        )
        edge_traces.append(edge_trace)

    fig = go.Figure(data=edge_traces + [node_trace, colorbar_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        title=f"{GraphDataset.to_graph6(nx_graph)} | l2 = {graph_value:.4f}"
    )
    fig.show()


def inspect_dataset(dataset):
    if isinstance(dataset, InMemoryDataset):
        dataset_name = dataset.__repr__()
        y_name = dataset.target_function.__name__
        num_features = dataset.num_features
        features = dataset.features
        assert len(features) == num_features
    else:
        dataset_name = "N/A"
        y_name = "N/A"
        num_features = dataset[0].x.shape[1]
        features = "N/A"

    print()
    header = f"Dataset: {dataset_name}"
    print(header)
    print("=" * len(header))
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {num_features} ({features})")
    print(f"Target: {y_name}")
    print("=" * len(header))
    print()


def inspect_graphs(dataset, graphs:int | list=1):
    """
    Inspect and display information about graphs in a dataset.

    This function prints detailed information about one or more graph objects
    from the given dataset, including their structural properties and features.

    Args:
        dataset: A dataset object containing graph data.
        graphs (int | list, optional): Specifies which graphs to inspect.
                    If an integer is provided, that many random graphs will be
                    selected from the dataset. If a list of indices is provided,
                    the graphs at those indices will be inspected. Defaults to 1.
    Example:
        >>> inspect_graphs(my_dataset, graphs=3)
        >>> inspect_graphs(my_dataset, graphs=[0, 5, 10])
    """

    try:
        y_name = dataset.target_function.__name__
    except AttributeError:
        y_name = "Target value"

    if isinstance(graphs, int):
        graphs = random.sample(range(len(dataset)), graphs)

    for i in graphs:
    # for i in random.sample(range(len(dataset)), num_graphs):
        data = dataset[i]  # Get a random graph object

        print()
        header = f"{i} - {data}"
        print(header)
        print("=" * len(header))

        # Gather some statistics about the graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"{y_name}: {data.edge_label}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
        print(f"Features:\n{data.x}")
        print("=" * len(header))
        print()
        draw_graph(data)



def main():
    root = Path(__file__).parents[1] / "Dataset"
    selected_graph_sizes = {
        3: -1,
        4: -1,
        5: -1,
        6: -1,
        7: -1,
        8: -1,
        # 9:  10000,
        # 10: 100000
    }
    loader = GraphDataset(selection=selected_graph_sizes, seed=42)

    with codetiming.Timer():
        dataset = LinkPredictionDataset(root, loader, selected_features=None)

    inspect_dataset(dataset)
    inspect_graphs(dataset, graphs=5)



if __name__ == "__main__":
    main()