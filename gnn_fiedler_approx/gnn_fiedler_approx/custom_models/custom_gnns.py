import importlib
import torch
import torch_geometric.nn as torchg_nn
import torch_geometric.nn.aggr as torchg_aggr
from gnn_fiedler_approx.gnn_utils.utils import count_parameters


def get_global_pooling(name, **kwargs):
    options = {
        "mean": torchg_aggr.MeanAggregation(),
        "max": torchg_aggr.MaxAggregation(),
        "add": torchg_aggr.SumAggregation(),
        "min": torchg_aggr.MinAggregation(),
        "median": torchg_aggr.MedianAggregation(),
        "var": torchg_aggr.VarAggregation(),
        "std": torchg_aggr.StdAggregation(),
        "softmax": torchg_aggr.SoftmaxAggregation(learn=True),
        "s2s": torchg_aggr.Set2Set,
        "minmax": torchg_aggr.MultiAggregation(aggrs=["min", "max"]),
        "multi": torchg_aggr.MultiAggregation(aggrs=["min", "max", "mean", "std"]),
        "multi++": torchg_aggr.MultiAggregation,
        "PNA": torchg_aggr.DegreeScalerAggregation,
        # "powermean": PowerMeanAggregation(learn=True),  # Results in NaNs and error
        # "mlp": MLPAggregation,  # NOT a permutation-invariant operator
        # "sort": SortAggregation,  # Requires sorting node representations
    }

    hc = kwargs["hidden_channels"]
    if name is None:
        return None, kwargs["hidden_channels"]

    pool = options[name]

    if name == "s2s":
        pool = pool(in_channels=hc, processing_steps=4)
        hc = 2 * hc
    elif name == "PNA":
        pool = pool(
            aggr=["mean", "min", "max", "std"], scaler=["identity", "amplification", "attenuation"], deg=kwargs["deg"]
        )
        hc = len(pool.aggr.aggrs) * len(pool.scaler) * hc
    elif name in ["multi", "minmax"]:
        hc = len(pool.aggrs) * hc
    elif name == "multi++":
        pool = pool(
            aggrs=[
                torchg_aggr.Set2Set(in_channels=hc, processing_steps=4),
                torchg_aggr.SoftmaxAggregation(learn=True),
                torchg_aggr.MinAggregation(),
            ]
        )
        hc = (2 + 1 + 1) * hc

    return pool, hc


class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        architecture: str,
        in_channels: int,
        hidden_channels: int,
        gnn_layers: int,
        **kwargs,
    ):
        super().__init__()

        # Store args and kwargs used to initialize the model.
        self.config = {"architecture": architecture,
                       "in_channels": in_channels,
                       "hidden_channels": hidden_channels,
                       "gnn_layers": gnn_layers,
                       }
        self.config.update(kwargs)

        # Sanitize and load kwargs.
        if kwargs.get("jk") == "none":
            kwargs["jk"] = None

        self.mlp_layers = kwargs.pop("mlp_layers", 1)
        pool = kwargs.pop("pool", "mean")
        pool_kwargs = kwargs.pop("pool_kwargs", {})

        self.save_embeddings_freq = kwargs.pop("save_embeddings_freq", float("inf"))
        self.embeddings = {}

        # Build the model: 1) pre-scaler, 2) GNN, 3) pooling, 4) final regression predictor
        self.pre_scaler = None
        if kwargs.pop("pre_scaler", False):
            # If pre-scaling is enabled, the input channels are scaled to hidden channels
            self.pre_scaler = torch.nn.Linear(in_channels, hidden_channels)
            in_channels = hidden_channels

        gnn_model = getattr(importlib.import_module("torch_geometric.nn"), architecture)
        self.gnn = gnn_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=gnn_layers,
            **kwargs,
        )

        self.pool, hc = get_global_pooling(pool, hidden_channels=hidden_channels, **pool_kwargs)

        mlp_layer_list = []
        for i in range(self.mlp_layers):
            if i < self.mlp_layers - 1:
                mlp_layer_list.append(torch.nn.Linear(hc, hidden_channels))
                mlp_layer_list.append(torch.nn.ReLU())
                hc = hidden_channels
            else:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, 1))
        self.predictor = torch.nn.Sequential(*mlp_layer_list)

        # Store other class variables.
        self.gnn_is_mlp = architecture == "MLP"


    def forward(self, x, edge_index, batch, epoch=-1):
        if self.pre_scaler is not None:
            x = self.pre_scaler(x)  # Pre-scaling the input features
        if self.gnn_is_mlp:
            x = self.gnn(x=x, batch=batch)
        else:
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)

        x = self.pool(x, batch) if self.pool is not None else x

        if epoch == 1 or epoch % self.save_embeddings_freq == 0:
            if epoch not in self.embeddings:
                self.embeddings[epoch] = x.detach().cpu()
            else:
                self.embeddings[epoch] = torch.cat((self.embeddings[epoch], x.detach().cpu()), dim=0)

        x = self.predictor(x)

        return x

    @property
    def descriptive_name(self):
        try:
            # Base name of the used GNN
            name = [f"{self.gnn.__class__.__name__}"]
            # Pre-scaling layer
            if self.pre_scaler is not None:
                name.append("lin")
            # Number of layers and size of hidden channel or hidden channels list
            if hasattr(self.gnn, "channel_list"):
                name.append(f"{self.gnn.num_layers}x{self.gnn.channel_list[-1]}_{self.mlp_layers}")
            else:
                name.append(f"{self.gnn.num_layers}x{self.gnn.hidden_channels}_{self.mlp_layers}")
            # Activation function
            name.append(f"{self.gnn.act.__class__.__name__}")
            # Dropout
            if isinstance(self.gnn.dropout, torch.nn.Dropout):
                name.append(f"D{self.gnn.dropout.p:.2f}")
            else:
                name.append(f"D{self.gnn.dropout[0]:.2f}")
            # Pooling layer: either a function or a class
            if hasattr(self.pool, "__name__"):
                name.append(f"{self.pool.__name__}")
            else:
                name.append(f"{self.pool.__class__.__name__}")
            # Number of parameters
            name.append(f"{count_parameters(self)}")
            # Join all parts and return.
            name = "-".join(name)
            return name

        except Exception as e:
            raise ValueError(f"Error in descriptive_name: {e}")

    def save(self, path):
        model_dict = {
            "config": self.config,
            "model": self.state_dict(),
        }
        torch.save(model_dict, path)

    @classmethod
    def load(cls, path, device):
        model_dict = torch.load(path, map_location=device)
        model = cls(**model_dict["config"])
        model.load_state_dict(model_dict["model"])
        model.to(device)
        return model


premade_gnns = torchg_nn.models.classes
custom_gnns = {}
