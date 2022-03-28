from abc import ABC, abstractmethod
from typing import List, Optional, Type

import torch
from torch import nn
from torch.nn import Module

from ...utils.torch_utils import generate_fully_connected


class ContractiveInvertibleGNN(nn.Module):
    """
    Given x, we can easily compute the exog noise z that generates it.
    """

    # TODO: remove mode_f_sem option
    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        mode_f_sem: str = "gnn_i",
        norm_layer: Optional[Type[Module]] = None,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
            mode_f_sem: Mode used for function. Admits {"gnn_i"}. gnn_i is a an encoder-decoder based autoregressive additive SEM.
        """
        super().__init__()
        self.group_mask = group_mask.to(device)
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self._device = device
        self.mode_f_sem = mode_f_sem
        self.W = self._initialize_W()
        self.f = FGNNI(
            self.group_mask,
            self._device,
            norm_layer=norm_layer,
            res_connection=res_connection,
            layers_g=encoder_layer_sizes,
            layers_f=decoder_layer_sizes,
        )

    def _initialize_W(self) -> torch.Tensor:
        """
        Creates and initializes the weight matrix for adjacency.

        Returns:
            Matrix of size (num_nodes, num_nodes) initialized with zeros.

        Question: Initialize to zeros??
        """
        W = torch.zeros(self.num_nodes, self.num_nodes, device=self._device)
        return nn.Parameter(W, requires_grad=True)

    def get_weighted_adjacency(self) -> torch.Tensor:
        """
        Returns the weights of the adjacency matrix.
        """
        W_adj = self.W * (1.0 - torch.eye(self.num_nodes, device=self._device))  # Shape (num_nodes, num_nodes)
        return W_adj

    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Gives the prediction of each variable given its parents.

        Args:
            X: Output of the GNN after reaching a fixed point, batched. Array of size (batch_size, processed_dim_all).
            W_adj: Weighted adjacency matrix, possibly normalized.
        
        Returns:
            predict: Predictions, batched, of size (B, n) that reconstructs X using the SEM.
        """
        return self.f.feed_forward(X, W_adj)  # Shape (batch_size, processed_dim_all)

    def simulate_SEM(
        self,
        Z: torch.Tensor,
        W_adj: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
        gumbel_max_regions: Optional[List[List]] = None,
        gt_zero_region: Optional[List[List]] = None,
    ):
        """
        Given exogenous noise Z, computes the corresponding set of observations X, subject to an optional intervention
        generates it.
        For discrete variables, the exogeneous noise uses the Gumbel max trick to generate samples.

        Args:
            Z: Exogenous noise vector, batched, of size (B, n) 
            W_adj: Weighted adjacency matrix, possibly normalized. (n, n) if a single matrix should be used for all batch elements. Otherwise (B, n, n)
            intervention_mask: torch.Tensor of shape (num_nodes) optional array containing binary flag of nodes that have been intervened.
            intervention_values: torch.Tensor of shape (processed_dim_all) optional array containing values for variables that have been intervened.
            gumbel_max_regions: a list of index lists `a` such that each subarray X[a] represents a one-hot encoded discrete random variable that must be
                sampled by applying the max operator.
            gt_zero_region: a list of indices such that X[a] should be thresholded to equal 1, if positive, 0 if negative. This is used to sample
                binary random variables. This also uses the Gumbel max trick implicitly
        
        Returns:
             X: Output of the GNN after reaching a fixed point, batched. Array of size (batch_size, processed_dim_all).
        """

        X = torch.zeros_like(Z)

        for _ in range(self.num_nodes):
            if intervention_mask is not None:
                X[:, intervention_mask] = intervention_values.unsqueeze(0)
            X = self.f.feed_forward(X, W_adj) + Z
            if gumbel_max_regions is not None:
                for region in gumbel_max_regions:
                    maxes = X[:, region].max(-1, keepdim=True)[0]
                    X[:, region] = (X[:, region] >= maxes).float()
            if gt_zero_region is not None:
                X[:, gt_zero_region] = (X[:, gt_zero_region] > 0).float()

        if intervention_mask is not None:
            X[:, intervention_mask] = intervention_values.unsqueeze(0)
        return X


class FunctionSEM(ABC, nn.Module):
    """
    Function SEM. Defines the (possibly nonlinear) function f for the additive noise SEM.
    """

    def __init__(
        self, group_mask: torch.Tensor, device: torch.device,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
        """
        super().__init__()
        self.group_mask = group_mask
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self._device = device

    @abstractmethod
    def feed_forward(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Computes function f(X) using the given weighted adjacency matrix.

        Args:
            X: Batched inputs, size (B, n).
            W_adj: Weighted adjacency matrix, size (n, n).
        """
        raise NotImplementedError()

    def initialize_embeddings(self) -> torch.Tensor:
        """
        Initialize the node embeddings.
        """
        aux = torch.randn(self.num_nodes, self.embedding_size, device=self._device) * 0.01  # (N, E)
        return nn.Parameter(aux, requires_grad=True)


# TODO: remove parent class, as it now has a single child
class FGNNI(FunctionSEM):
    """
    Defines the function f for the SEM. For each variable x_i we use
    f_i(x) = f(e_i, sum_{k in pa(i)} g(e_k, x_k)), where e_i is a learned embedding
    for node i.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        embedding_size: int = None,
        out_dim_g: int = None,
        norm_layer: Optional[Type[Module]] = None,
        res_connection: bool = False,
        layers_g: List[int] = None,
        layers_f: List[int] = None,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
            embedding_size: Size of the embeddings used by each node. If none, default is processed_dim_all.
            out_dim_g: Output dimension of the "inner" NN, g. If none, default is embedding size.
            layers_g: Size of the layers of NN g. Does not include input not output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10).
            layers_f: Size of the layers of NN f. Does not include input nor output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10)
        """
        super().__init__(group_mask, device)
        # Initialize embeddings
        self.embedding_size = embedding_size or self.processed_dim_all
        self.embeddings = self.initialize_embeddings()  # Shape (input_dim, embedding_size)
        # Set value for out_dim_g
        out_dim_g = out_dim_g or self.embedding_size
        # Set NNs sizes
        a = max(4 * self.processed_dim_all, self.embedding_size, 64)
        layers_g = layers_g or [a, a]
        layers_f = layers_f or [a, a]
        in_dim_g = self.embedding_size + self.processed_dim_all
        in_dim_f = self.embedding_size + out_dim_g
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=layers_g,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self._device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=self.processed_dim_all,
            hidden_dims=layers_f,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self._device,
            normalization=norm_layer,
            res_connection=res_connection,
        )

    def feed_forward(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Computes non-linear function f(X, W) using the given weighted adjacency matrix.

        Args:
            X: Batched inputs, size (batch_size, processed_dim_all).
            W_adj: Weighted adjacency matrix, size (processed_dim_all, processed_dim_all) or size (batch_size, n, n).
        """

        if len(W_adj.shape) == 2:
            W_adj = W_adj.unsqueeze(0)

        # g takes inputs of size (*, embedding_size + processed_dim_all) and outputs (*, out_dim_g)
        # the input will be appropriately masked to correspond to one variable group
        # f takes inputs of size (*, embedding_size + out_dim_g) and outputs (*, processed_dim_all)
        # the ouptut is then masked to correspond to one variable

        # Generate required input for g (concatenate X and embeddings)
        X = X.unsqueeze(1)  # Shape (batch_size, 1, processed_dim_all)
        # Pointwise multiply X : shape (batch_size, 1, processed_dim_all) with self.group_mask : shape (num_nodes, processed_dim_all)
        X_masked = X * self.group_mask  # Shape (batch_size, num_nodes, processed_dim_all)
        E = self.embeddings.expand(X.shape[0], -1, -1)  # Shape (batch_size, num_nodes, embedding_size)
        X_in_g = torch.cat([X_masked, E], dim=2)  # Shape (batch_size, num_nodes, embedding_size + processed_dim_all)
        X_emb = self.g(X_in_g)  # Shape (batch_size, num_nodes, out_dim_g)
        # Aggregate sum and generate input for f (concatenate X_aggr and embeddings)
        X_aggr_sum = torch.matmul(W_adj.transpose(-1, -2), X_emb)  # Shape (batch_size, num_nodes, out_dim_g)
        # return vmap(torch.mm, in_dims=(None, 0))(W_adj.t(), X_emb)  # Shape (batch_size, num_nodes, out_dim_g)
        X_in_f = torch.cat([X_aggr_sum, E], dim=2)  # Shape (batch_size, num_nodes, out_dim_g + embedding_size)
        # Run f
        X_rec = self.f(X_in_f)  # Shape (batch_size, num_nodes, processed_dim_all)
        # Mask and aggregate
        X_rec = X_rec * self.group_mask  # Shape (batch_size, num_nodes, processed_dim_all)
        return X_rec.sum(1)  # Shape (batch_size, processed_dim_all)
