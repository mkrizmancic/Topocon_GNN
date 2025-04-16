import copy

import torch
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox


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