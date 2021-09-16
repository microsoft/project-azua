import torch
from torch_geometric.nn.conv import GATConv, MessagePassing
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor, SparseTensor
from torch_geometric.utils import softmax


class ConvModel(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_dim: int,
        aggregate_function: str,
        normalize_embeddings: bool,
        node_update_dropout: float,
    ) -> None:
        """
        Args
            in_channels: Input dimensionality for the message passing. Corresponds to node_input_dim for the
                            first layer and then to the node embedding size.
            out_channels: Output dimensionality for the message passing. Corresponds to the node embedding size.
            message_dim: Message dimensionality.
            aggregate_function: Aggregate function for the message passing.
                             Valid options are 'add', 'mean', and 'max'.
            normalize_embeddings: Whether to normalize the node embeddings or not during message passing.
            node_update_dropout: dropout rate for msg. passing and aggregation.
        """
        super(ConvModel, self).__init__(aggr=aggregate_function)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_dim = message_dim
        self.message_lin = nn.Sequential(
            Linear(in_channels + message_dim, out_channels), nn.ReLU(), nn.Dropout(node_update_dropout)
        )
        self.aggregation_lin = nn.Sequential(
            Linear(in_channels + out_channels, out_channels), nn.ReLU(), nn.Dropout(node_update_dropout)
        )

        self.normalize_embeddings = normalize_embeddings

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
    ):
        """
        Args
            x: Input nodes of shape (num_nodes, node_input_dim).
            edge_index: Edge indices of shape (2, num_edges_minibatch).
            edge_attr: Edge attributes of shape (num_edges_minibatch, edge_input_dim).
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, size: int,
    ):
        """
        Args
            x_i, x_j : Node embeddings of shape (num_edges_minibatch, node_dim)
            edge_attr: Edge attributes of shape (num_edges_minibatch, edge_input_dim).
            edge_index: Edge indices of shape (2, num_edges_minibatch).
            size: number of nodes:
        """
        message_j_to_i = torch.cat((x_j, edge_attr), dim=-1)
        message_j_to_i = self.message_lin(message_j_to_i)
        return message_j_to_i

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        """
        Args
            aggr_out: Before and after of the node embeddings going through the non-linearlity and updating.
            x: Input nodes of shape (num_nodes, node_input_dim).
        """
        aggr_out = self.aggregation_lin(torch.cat((aggr_out, x[1]), dim=-1))
        if self.normalize_embeddings:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class GATModel(GATConv):
    """
    Graph Attention Networks

    <https://arxiv.org/abs/1710.10903>

    Modified GATConv for recommender systems. A large portion of the code is from the URL below:
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_dim: int,
        aggregate_function: str,
        normalize_embeddings: bool,
        concat: bool = True,
        heads: int = 1,
        **kwargs,
    ) -> None:
        """
        Args
            in_channels: Input dimensionality for the message passing. Corresponds to node_input_dim for the
                            first layer and then to the node embedding size.
            out_channels: Output dimensionality for the message passing. Corresponds to the node embedding size.
            message_dim: Message dimensionality.
            aggregate_function: Aggregate function for the message passing.
                             Valid options are 'add', 'mean', and 'max'.
            normalize_embeddings: Whether to normalize the node embeddings or not during message passing.
            concat: Whether to concatenate the information from the multi-head attentions or not.
            heads: Number of heads for the multi-head attentions.
        """
        super(GATModel, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggregate_function,
            concat=concat,
            heads=heads,
            **kwargs,
        )

        self.message_dim = message_dim

        self.lin_l = Linear(in_channels, self.heads * out_channels, bias=False)
        self.lin_r = self.lin_l

        self.message_lin = Linear(out_channels + self.message_dim, out_channels)

        self.aggregation_lin = Linear(2 * out_channels, out_channels)

        self.normalize_embeddings = normalize_embeddings

        self.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: Adj,
        size: Size = None,
        return_attention_weights=None,
    ):
        """
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l, x_r = x[0], x[1]
        assert x[0].dim() == 2, "Static graphs not supported in `GATConv`."
        x_l = self.lin_l(x_l).view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        x_r = self.lin_r(x_r).view(-1, H, C)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        out = self.propagate(
            edge_index=edge_index, edge_attr=edge_attr, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size
        )

        self._alpha: OptTensor
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_j: torch.Tensor,
        x_i: Optional[torch.Tensor],
        alpha_j: torch.Tensor,
        alpha_i: Optional[torch.Tensor],
        index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ):
        """
        Args
            x_i, x_j : Node embeddings of shape (num_edges_minibatch, node_dim), from j to i.
            edge_attr: Edge attributes of shape (num_edges_minibatch, edge_input_dim).
            edge_index: Edge indices of shape (2, num_edges_minibatch).
            size: number of nodes:
        """
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        edge_attr_expanded = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)

        message_j_to_i = torch.cat((x_j, edge_attr_expanded), dim=-1)
        message_j_to_i = F.leaky_relu(self.message_lin(message_j_to_i), self.negative_slope) * alpha.unsqueeze(-1)
        return message_j_to_i

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        """
        Args
            aggr_out: Before and after of the node embeddings going through the non-linearlity and updating.
            x: Input nodes of shape (num_nodes, node_input_dim).
        """
        aggr_out = F.leaky_relu(self.aggregation_lin(torch.cat((aggr_out, x[1]), dim=-1)), self.negative_slope)
        if self.normalize_embeddings:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
