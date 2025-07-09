import torch_geometric.nn.models.basic_gnn

from typing import Any, Callable, Dict, List, Optional

import torch
import torch_geometric
import torch_geometric.nn as torchg_nn
import torch_geometric.nn.aggr as torchg_aggr
from torch import Tensor
from torch_geometric.nn import (
    MLP,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.resolver import normalization_resolver

from gnn_fiedler_approx.gnn_utils.utils import count_parameters


# TODO: Add BatchNorm
class StupidFFN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: Callable, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels * 2)
        self.lin2 = torch.nn.Linear(out_channels * 2, out_channels)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.act = act

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.act(self.lin1(x)))
        x = self.dropout2(self.lin2(x))
        return x


class FFN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Callable,
        norm: Optional[str] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, out_channels)
        self.act = act

        self.norm = None
        if norm is not None:
            self.norm = normalization_resolver(
                norm,
                out_channels,
                **(norm_kwargs or {}),
            )

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin2(self.act(self.lin1(x))) + x
        x = self.norm(x) if self.norm is not None else x
        return x


OriginalBasicGNN = torch_geometric.nn.models.basic_gnn.BasicGNN
_original_init = OriginalBasicGNN.__init__
_original_reset_parameters = OriginalBasicGNN.reset_parameters


def extended_init(self, *args, residual: bool = False, ffn: bool = False, **kwargs):
    r"""Extended initialization method to support additional parameters."""
    _original_init(self, *args, **kwargs)
    print("Using BasicGNNPlus with residual connections and FFN support")

    hidden_channels: int = kwargs["hidden_channels"]
    num_layers: int = kwargs["num_layers"]

    self.residual = residual
    self.ffn = ffn

    _ffn_impl = FFN
    self.ffns = []
    if self.ffn:
        self.ffns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.ffns.append(
                _ffn_impl(
                    hidden_channels,
                    hidden_channels,
                    self.act,
                    norm=kwargs.get("norm", None),
                    norm_kwargs=kwargs.get("norm_kwargs", None),
                    dropout=kwargs.get("dropout", 0.0),
                )
            )


def extended_reset_parameters(self):
    r"""Resets all learnable parameters of the module."""
    _original_reset_parameters(self)
    if self.ffn is not None:
        if hasattr(self.ffn, "reset_parameters"):
            self.ffn.reset_parameters()


def extended_forward(
    self,
    x: Tensor,
    edge_index: Adj,
    edge_weight: OptTensor = None,
    edge_attr: OptTensor = None,
    batch: OptTensor = None,
    batch_size: Optional[int] = None,
    num_sampled_nodes_per_hop: Optional[List[int]] = None,
    num_sampled_edges_per_hop: Optional[List[int]] = None,
) -> Tensor:
    r"""Forward pass.

    Args:
        x (torch.Tensor): The input node features.
        edge_index (torch.Tensor or SparseTensor): The edge indices.
        edge_weight (torch.Tensor, optional): The edge weights (if
            supported by the underlying GNN layer). (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): The edge features (if supported
            by the underlying GNN layer). (default: :obj:`None`)
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each element to a specific example.
            Only needs to be passed in case the underlying normalization
            layers require the :obj:`batch` information.
            (default: :obj:`None`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given.
            Only needs to be passed in case the underlying normalization
            layers require the :obj:`batch` information.
            (default: :obj:`None`)
        num_sampled_nodes_per_hop (List[int], optional): The number of
            sampled nodes per hop.
            Useful in :class:`~torch_geometric.loader.NeighborLoader`
            scenarios to only operate on minimal-sized representations.
            (default: :obj:`None`)
        num_sampled_edges_per_hop (List[int], optional): The number of
            sampled edges per hop.
            Useful in :class:`~torch_geometric.loader.NeighborLoader`
            scenarios to only operate on minimal-sized representations.
            (default: :obj:`None`)
    """
    if (num_sampled_nodes_per_hop is not None
            and isinstance(edge_weight, Tensor)
            and isinstance(edge_attr, Tensor)):
        raise NotImplementedError("'trim_to_layer' functionality does not "
                                    "yet support trimming of both "
                                    "'edge_weight' and 'edge_attr'")

    xs: List[Tensor] = []
    assert len(self.convs) == len(self.norms)
    for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
        layer_input_x = x
        if not torch.jit.is_scripting() and num_sampled_nodes_per_hop is not None:
            x, edge_index, value = self._trim(
                i,
                num_sampled_nodes_per_hop,
                num_sampled_edges_per_hop,
                x,
                edge_index,
                edge_weight if edge_weight is not None else edge_attr,
            )
            if edge_weight is not None:
                edge_weight = value
            else:
                edge_attr = value

        # Tracing the module is not allowed with *args and **kwargs :(
        # As such, we rely on a static solution to pass optional edge
        # weights and edge attributes to the module.
        if self.supports_edge_weight and self.supports_edge_attr:
            x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        elif self.supports_edge_weight:
            x = conv(x, edge_index, edge_weight=edge_weight)
        elif self.supports_edge_attr:
            x = conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = conv(x, edge_index)

        if i < self.num_layers - 1 or self.jk_mode is not None:
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = self.dropout(x)

            if self.residual:
                x = x + layer_input_x

            if self.ffn:
                x = self.ffns[i](x)

            if hasattr(self, "jk"):
                xs.append(x)

    x = self.jk(xs) if hasattr(self, "jk") else x
    x = self.lin(x) if hasattr(self, "lin") else x

    return x


def get_global_pooling(name, **kwargs):
    options = {
        "mean": global_mean_pool,
        "max": global_max_pool,
        "add": global_add_pool,
        "min": torchg_aggr.MinAggregation(),
        "median": torchg_aggr.MedianAggregation(),
        "var": torchg_aggr.VarAggregation(),
        "std": torchg_aggr.StdAggregation(),
        "softmax": torchg_aggr.SoftmaxAggregation(learn=True),
        "s2s": torchg_aggr.Set2Set,
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
    elif name == "multi":
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
        gnn_model,
        in_channels: int,
        hidden_channels: int,
        gnn_layers: int,
        mlp_layers: int = 1,
        pool="mean",
        pool_kwargs={},
        norm=None,
        norm_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        pre_scaler = kwargs.pop("pre_scaler", False)
        residual = kwargs.get("residual", False)
        ffn = kwargs.get("ffn", False)

        if residual and not pre_scaler:
            raise ValueError(
                "Residual connections require 'pre_scaler=True' option to transform input features to hidden channels dimension."
            )

        self.pre_scaler = None
        if pre_scaler:
            # If pre-scaling is enabled, the input channels are scaled to hidden channels
            self.pre_scaler = torch.nn.Linear(in_channels, hidden_channels)
            in_channels = hidden_channels

        if kwargs.get("jk") == "none":
            kwargs["jk"] = None

        self.save_embeddings_freq = kwargs.pop("save_embeddings_freq", float("inf"))
        self.embeddings = {}

        # We monkey-patched the BasicGNN to include residual connections and feed-forward networks.
        # Now we replace the BasicGNN with this modified version so that all other models can inherit from it.
        gnn_model.__init__ = extended_init
        gnn_model.forward = extended_forward
        gnn_model.reset_parameters = extended_reset_parameters

        self.gnn = gnn_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=gnn_layers,
            norm=norm,
            norm_kwargs=norm_kwargs,
            **kwargs,
        )
        self.gnn_is_mlp = isinstance(self.gnn, MLP)

        self.pool, hc = get_global_pooling(pool, hidden_channels=hidden_channels, **pool_kwargs)

        mlp_layer_list = []
        for i in range(mlp_layers):
            if i < mlp_layers - 1:
                mlp_layer_list.append(torch.nn.Linear(hc, hidden_channels))
                mlp_layer_list.append(torch.nn.ReLU())
                hc = hidden_channels
            else:
                mlp_layer_list.append(torch.nn.Linear(hidden_channels, 1))
        self.classifier = torch.nn.Sequential(*mlp_layer_list)
        self.mlp_layers = mlp_layers

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

        x = self.classifier(x)

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


premade_gnns = torchg_nn.models.classes
custom_gnns = {x.__name__: x for x in []}
