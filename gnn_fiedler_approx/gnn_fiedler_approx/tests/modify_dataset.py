import pathlib
from my_graphs_dataset import GraphDataset
from gnn_fiedler_approx import ConnectivityDataset, inspect_dataset, inspect_graphs


def option1(dataset):
    mean = dataset.y.mean()
    std = dataset.y.std()

    dataset.y = (dataset.y - mean) / std

    return dataset


def option2(dataset):
    mean = dataset.y.mean()
    std = dataset.y.std()

    dataset.data.y = (dataset.data.y - mean) / std

    return dataset


def option3(dataset):
    mean = dataset.y.mean()
    std = dataset.y.std()

    dataset._data.y = (dataset._data.y - mean) / std

    return dataset


def load_dataset():
    seed = 42
    split = 0.8

    # Load the dataset.
    root = pathlib.Path(__file__).parents[2] / "Dataset"  # For standalone script.
    graphs_loader = GraphDataset(selection={3: -1, 4: -1, 5: -1}, seed=100)
    dataset = ConnectivityDataset(root, graphs_loader)

    dataset = dataset.shuffle()

    inspect_dataset(dataset)
    inspect_graphs(dataset, num_graphs=1)

    print("Option 1")
    dataset1 = option1(dataset)
    inspect_dataset(dataset1)
    inspect_graphs(dataset1, num_graphs=1)

    print("Option 2")
    dataset2 = option2(dataset)
    inspect_dataset(dataset2)
    inspect_graphs(dataset2, num_graphs=1)

    print("Option 3")
    dataset3 = option3(dataset)
    inspect_dataset(dataset3)
    inspect_graphs(dataset3, num_graphs=1)



if __name__ == "__main__":
    load_dataset()