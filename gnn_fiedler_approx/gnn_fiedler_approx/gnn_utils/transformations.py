import copy

import torch
import numpy as np
import pandas as pd
import torch_geometric.transforms as tg_transforms
from torch_geometric.data import Data
from scipy.special import inv_boxcox
from scipy.stats import boxcox


def resolve_transform(transform_names, **transform_kwargs):
    """
    Resolves the transform class based on the provided name and arguments.

    Args:
        transform_name (str): The name of the transform to resolve.
        **transform_kwargs: Additional keyword arguments for the transform.

    Returns:
        A new instance of the resolved transform class.
    """
    if transform_names is None:
        return None

    if isinstance(transform_names, str):
        transform_names = [transform_names]

    options = {
        "normalize_features": tg_transforms.NormalizeFeatures,}

    transforms = []
    for name in transform_names:
        if isinstance(name, str):
            transform_name = name.lower()
        else:
            raise ValueError(f"Transform name must be a string, got {type(name)}.")

        try:
            transforms.append(options[transform_name](**transform_kwargs))
        except KeyError:
            raise ValueError(f"Transform '{transform_name}' is not recognized. Available transforms: {list(options.keys())}.")

    return tg_transforms.Compose(transforms)


class DatasetTransformer:
    def __init__(self, method):
        assert method in [None, "min-max", "z-score", "boxcox", "cuberoot"], f"Invalid normalization method '{method}'."

        self.method = method
        self.params = {}

    def normalize_labels(self, train_dataset, *other_datasets):
        # No need to do anything.
        if self.method is None:
            return train_dataset, *other_datasets

        # Calculate dataset statistics for fitting.
        self.fit(train_dataset.y)

        # Transform the labels with the calculated statistics.
        # We need to make a data_list out of Dataset object and then modify individual data.
        # https://github.com/pyg-team/pytorch_geometric/issues/839
        train_data_list = [data for data in train_dataset]
        for data in train_data_list:
            data.y = self.transform(data.y)

        other_data_list = [[data for data in dataset] for dataset in other_datasets]
        for dataset in other_data_list:
            for data in dataset:
                data.y = self.transform(data.y)

        return train_data_list, *other_data_list

    def denormalize_labels(self, dataset, inplace=False):
        # No need to do anything.
        if self.method is None:
            return dataset

        if not inplace:
            dataset = [copy.copy(data) for data in dataset]

        for data in dataset:
            data.y = self.reverse_transform(data.y)

        return dataset

    def fit(self, data: torch.Tensor):
        if self.method == "z-score":
            self.params["mean"] = data.mean().item()
            self.params["std"] = data.std().item()
        elif self.method == "min-max":
            self.params["min"] = data.min().item()
            self.params["max"] = data.max().item()
        elif self.method == "boxcox":
            _, lambd = boxcox(data)
            self.params["lambda"] = lambd
        elif self.method == "cuberoot":
            pass

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        # TODO: Handle tensor, numpy, pd.Series, etc.
        if self.method == "z-score":
            return (data - self.params["mean"]) / self.params["std"]
        elif self.method == "min-max":
            return (data - self.params["min"]) / (self.params["max"] - self.params["min"])
        elif self.method == "boxcox":
            return torch.Tensor(boxcox(data, self.params["lambda"]))
        elif self.method == "cuberoot":
            return torch.pow(data, 1 / 3)
        else:
            return data

    def reverse_transform(self, data: pd.Series):
        # TODO: Handle tensor, numpy, pd.Series, etc.
        try:
            if self.method == "z-score":
                return data * self.params["std"] + self.params["mean"]
            elif self.method == "min-max":
                return data * (self.params["max"] - self.params["min"]) + self.params["min"]
            elif self.method == "boxcox":
                return inv_boxcox(data, self.params["lambda"])
            elif self.method == "cuberoot":
                return np.power(data, 3)
            else:
                return data
        except KeyError as e:
            raise ValueError(f"{e} You must first transform the data using `normalize_*` method.")


class EigenvectorFlipperTransform(tg_transforms.BaseTransform):
    """
    A transform that randomly flips the sign of the eigenvector of a graph's Laplacian matrix.

    Args:
        available_features (list): A list of available features.
        selected_features (list, optional): A list of selected features to be used.
        feature_dims (dict): A dictionary of features in use and their corresponding dimensions.
        feature_name (str, optional): The name of the feature to be transformed.
    """
    def __init__(self, available_features, selected_features, feature_dims, feature_name="k_normalized_laplacian"):
        self.start_idx = 0
        self.end_idx = 0

        for feature in feature_dims:
            if feature != feature_name:
                self.start_idx += feature_dims[feature]
            else:
                self.end_idx = self.start_idx + feature_dims[feature_name]
                break

        self.vector_dim = max(self.end_idx - self.start_idx, 0)

    def __new__(cls, available_features, selected_features, feature_dims, feature_name="k_normalized_laplacian"):
        if feature_name not in available_features:
            return None
        if selected_features is not None and feature_name not in selected_features:
            return None

        instance = super().__new__(cls)
        return instance

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        data.x[:, self.start_idx:self.end_idx] *= torch.sign(torch.randn(self.vector_dim))
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.start_idx}, {self.end_idx})'


class RandomNodeFeaturesTransform(tg_transforms.BaseTransform):
    """
    A transform that updates the random node features.

    Args:
        available_features (list): A list of available features.
        selected_features (list, optional): A list of selected features to be used.
        feature_dims (dict): A dictionary of features in use and their corresponding dimensions.
        feature_name (str, optional): The name of the feature to be transformed.
    """
    def __init__(self, available_features, selected_features, feature_dims, feature_name="random"):
        self.start_idx = 0
        self.end_idx = 0

        for feature in feature_dims:
            if feature != feature_name:
                self.start_idx += feature_dims[feature]
            else:
                self.end_idx = self.start_idx + feature_dims[feature_name]
                break

        self.vector_dim = max(self.end_idx - self.start_idx, 0)

    def __new__(cls, available_features, selected_features, feature_dims, feature_name="random"):
        if feature_name not in available_features:
            return None
        if selected_features is not None and feature_name not in selected_features:
            return None

        instance = super().__new__(cls)
        return instance

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        data.x[:, self.start_idx:self.end_idx] = torch.rand(data.x.shape[0], self.vector_dim)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.start_idx}, {self.end_idx})'