import random
from pathlib import Path

import codetiming
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pygUtils
import yaml
from torch_geometric.data import InMemoryDataset

from my_graphs_dataset import GraphDataset


class ConnectivityDataset(InMemoryDataset):
    def __init__(
        self, root, loader: GraphDataset | None = None, transform=None, pre_transform=None, pre_filter=None, **kwargs
    ):
        if loader is None:
            loader = GraphDataset()
        self.loader = loader

        super().__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])

        if selected_features := kwargs.get("selected_features"):
            self.filter_features(selected_features)

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
        # TODO: Automatically detect changes in the dataset.
        #       We could come up with a namig scheme that will differentiate
        #       which graph families (and/or sizes) and features were used to
        #       generate the dataset. This way, we could detect changes and
        #       reprocess the dataset when needed.
        return ["data.pt"]

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
        for graph in self.loader.graphs(batch_size=1):
            data_list.append(self.make_data(graph))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    # Define features in use.
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

    def make_data(self, G):
        """Create a PyG data object from a graph object."""
        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        torch_G = pygUtils.from_networkx(G, group_node_attrs=self.features)
        torch_G.y = torch.tensor(self.algebraic_connectivity(G), dtype=torch.float32)

        return torch_G

    @property
    def features(self):
        return list(self.feature_functions.keys())

    def filter_features(self, selected_features):
        """Filter out features that are not in the selected features."""
        mask = np.array([name in selected_features for name in self.features])
        # FIXME: This is not a proper way, but I don't know what else to do.
        # https://github.com/pyg-team/pytorch_geometric/discussions/7684
        assert self._data is not None
        self._data.x = self._data.x[:, mask]

    @staticmethod
    def algebraic_connectivity(G):
        L = nx.laplacian_matrix(G).toarray()
        lambdas = sorted(np.linalg.eigvalsh(L))
        return lambdas[1]


def inspect_dataset(dataset, num_graphs=1):
    for i in random.sample(range(len(dataset)), num_graphs):
        data = dataset[i]  # Get a random graph object

        print()
        print(data)
        print("=============================================================")

        # Gather some statistics about the first graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Algrebraic connectivity: {data.y.item():.5f}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
        print(f"Features: {data.x}")


def main():
    root = Path(__file__).parents[1] / "Dataset"
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
    loader = GraphDataset(selection=selected_graph_sizes)

    with codetiming.Timer():
        dataset = ConnectivityDataset(root, loader, selected_features=[])

    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")

    inspect_dataset(dataset, num_graphs=1)


if __name__ == "__main__":
    main()
