from abc import ABC, abstractmethod

import torch
from torch import nn

import torch.nn.functional as F
import torch.distributions as td
import numpy as np


class AdjMatrix(ABC):
    """
    Adjacency matrix interface for DECI
    """

    @abstractmethod
    def get_adj_matrix(self, round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q. In this case 0.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_A(self) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()


class VarDistA(AdjMatrix, nn.Module):
    """
    Abstract class representing a variational distribution over binary adjacency matrix.
    """

    def __init__(
        self, device: torch.device, input_dim: int, tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__()
        self._device = device
        self.input_dim = input_dim
        self.tau_gumbel = tau_gumbel

    @abstractmethod
    def _get_logits_softmax(self) -> torch.Tensor:
        """
        Returns the (softmax) logits.
        """
        raise NotImplementedError()

    def _build_bernoulli(self) -> td.Distribution:
        """
        Builds and returns the bernoulli distributions obtained using the (softmax) logits.
        """
        logits = self._get_logits_softmax()  # (2, n, n)
        logits_bernoulli_1 = logits[1, :, :] - logits[0, :, :]  # (n, n)
        # Diagonal elements are set to 0
        logits_bernoulli_1 -= 1e10 * torch.eye(self.input_dim, device=self._device)
        dist = td.Independent(td.Bernoulli(logits=logits_bernoulli_1), 2)
        return dist

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q.
        """
        return self._build_bernoulli().entropy()

    def sample_A(self) -> torch.Tensor:
        """
        Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
        and returns hard samples (straight through gradient estimator). Adjacency returned always has
        zeros in its diagonal (no self loops).

        V1: Returns one sample to be used for the whole batch.
        """
        logits = self._get_logits_softmax()
        sample = F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=True, dim=0)  # (2, n, n) binary
        sample = sample[1, :, :]  # (n, n)
        sample = sample * (1 - torch.eye(self.input_dim, device=self._device))  # Force zero diagonals
        return sample

    def log_prob_A(self, A: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the variational distribution q(A) at a sampled adjacency A.

        Args:
            A: A binary adjacency matrix, size (input_dim, input_dim).

        Returns:
            The log probability of the sample A. A number if A has size (input_dim, input_dim).
        """
        return self._build_bernoulli().log_prob(A)

    @abstractmethod
    def get_adj_matrix(self, round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()


class DeterministicAdjacency(AdjMatrix):
    """
     Deterministic adjacency matrix used for DECI + true Graph
    """

    def __init__(
        self, device: torch.device,
    ):
        """
        Args:
            device: Device used.
        """
        self.adj_matrix = None
        self.device = device

    def set_adj_matrix(self, adj_matrix: np.ndarray) -> None:
        """
        Set fixed adjacency matrix
        """
        self.adj_matrix = nn.Parameter(torch.from_numpy(adj_matrix).to(self.device), requires_grad=False)

    def get_adj_matrix(self, round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        return self.adj_matrix

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q. In this case 0.
        """
        return torch.zeros(1, device=self.device)

    def sample_A(self) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        return self.adj_matrix


class VarDistA_Simple(VarDistA):
    """
    Variational distribution for the binary adjacency matrix. Parameterizes the probability of each edge 
    (including orientation).
    """

    def __init__(
        self, device: torch.device, input_dim: int, tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super(VarDistA_Simple, self).__init__(device, input_dim, tau_gumbel)
        self.logits = self._initialize_params()

    def _initialize_params(self) -> torch.Tensor:
        """
        Returns the initial logits to sample A, a tensor of shape (2, input_dim, input_dim).
        Right now initialize all to zero. Could change. Could also change parameterization
        to be similar to the paper Cheng sent (https://arxiv.org/pdf/2107.10483.pdf).
        """
        logits = torch.zeros(2, self.input_dim, self.input_dim, device=self._device)  # Shape (2, input_dim, input_dim)
        return nn.Parameter(logits, requires_grad=True)

    def _get_logits_softmax(self) -> torch.Tensor:
        """
        Returns the (softmax) logits.
        """
        return self.logits

    def get_adj_matrix(self, round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        probs_1 = F.softmax(self.logits, dim=0)[1, :, :]  # Shape (input_dim, input_dim)
        probs_1 *= (1 - torch.eye(self.input_dim, device=self._device))
        if round:
            return probs_1.round()
        return probs_1

    def get_print_params(self):
        """
        Will go away, returs parameters to print.
        """
        return self.logits


class VarDistA_ENCO(VarDistA):
    """
    Variational distribution for the binary adjacency matrix, following the parameterization from
    the ENCO paper (https://arxiv.org/pdf/2107.10483.pdf). For each edge, parameterizes the existence
    and orientation separately. Main benefit is that it avoids length 2 cycles automatically.
    Orientation is somewhat over-parameterized.
    """

    def __init__(
        self, device: torch.device, input_dim: int, tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__(device, input_dim, tau_gumbel)
        self.logits_edges = self._initialize_edge_logits()
        self.params_orient = self._initialize_orient_params()

    def _initialize_edge_logits(self) -> torch.Tensor:
        """
        Returns the initial logits that characterize the presence of an edge (gamma in the ENCO paper),
        a tensor of shape (2, n, n).
        """
        logits = torch.zeros(2, self.input_dim, self.input_dim, device=self._device)  # Shape (2, input_dim, input_dim)
        logits[1, :, :] -= 1
        return nn.Parameter(logits, requires_grad=True)

    def _initialize_orient_params(self) -> torch.Tensor:
        """
        Returns the initial logits that characterize the orientation (theta in the ENCO paper),
        a tensor of shape (n, n). Right now initialize all to zero. Could change.
        This will be processed so as to keep only strictly upper triangular, so some of
        these parameters are not trained.
        """
        params = torch.zeros(self.input_dim, self.input_dim, device=self._device)  # (n, n)
        return nn.Parameter(params, requires_grad=True)

    def _build_logits_orient(self) -> torch.Tensor:
        """
        Auxiliary function that computes the (softmax) logits to sample orientation for the edges given the parameters.
        """
        logits_0 = torch.zeros(self.input_dim, self.input_dim, device=self._device)  # Shape (input_dim, input_dim)
        # Get logits_1 strictly upper triangular
        logits_1 = torch.triu(self.params_orient)
        logits_1 = logits_1 * (1.0 - torch.eye(self.input_dim, self.input_dim, device=self._device))
        logits_1 = logits_1 - torch.transpose(logits_1, 0, 1)  # Make logit_ij = -logit_ji
        return torch.stack([logits_0, logits_1])

    def _get_logits_softmax(self) -> torch.Tensor:
        """
        Auxiliary function to compute the (softmax) logits from both edge logits and orientation logits. Notice
        the logits for the softmax are computed differently than those for Bernoulli (latter uses sigmoid, equivalent
        if the logits for zero filled with zeros).

        Simply put, to sample an edge i->j you need to both sample the precense of that edge, and sample its orientation.
        """
        logits_edges = self.logits_edges  # Shape (2, input_dim, input_dim)
        logits_orient = self._build_logits_orient()  # Shape (2, input_dim, input_dim)
        logits_1 = logits_edges[1, :, :] + logits_orient[1, :, :]  # Shape (input_dim, input_dim)
        aux = torch.stack(
            [
                logits_edges[1, :, :] + logits_orient[0, :, :],
                logits_edges[0, :, :] + logits_orient[1, :, :],
                logits_edges[0, :, :] + logits_orient[0, :, :],
            ]
        )  # Shape (3, input_dim, input_dim)
        logits_0 = torch.logsumexp(aux, dim=0)  # Shape (input_dim, input_dim)
        logits = torch.stack([logits_0, logits_1])  # Shape (2, input_dim, input_dim)
        return logits

    def get_adj_matrix(self, round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        probs_edges = F.softmax(self.logits_edges, dim=0)[1, :, :]  # Shape (input_dim, input_dim)
        logits_orient = self._build_logits_orient()
        probs_orient = F.softmax(logits_orient, dim=0)[1, :, :]  # Shape (input_dim, input_dim)
        probs_1 = probs_edges * probs_orient
        probs_1 = probs_1 * (1.0 - torch.eye(self.input_dim, device=self._device))
        if round:
            return probs_1.round()
        return probs_1

    def get_print_params(self):
        """
        Will go away, returs parameters to print.
        """
        return self.logits_edges, self.params_orient


class ThreeWayGraphDist(AdjMatrix, nn.Module):
    """
    An alternative variational distribution for graph edges. For each pair of nodes x_i and x_j
    where i < j, we sample a three way categorical C_ij. If C_ij = 0, we sample the edge
    x_i -> x_j, if C_ij = 1, we sample the edge x_j -> x_i, and if C_ij = 2, there is no
    edge between these nodes. This variational distribution is faster to use than ENCO
    because it avoids any calls to `torch.stack`.

    Sampling is performed with `torch.gumbel_softmax(..., hard=True)` to give
    binary samples and a straight-through gradient estimator.
    """

    def __init__(
        self, device: torch.device, input_dim: int, tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__()
        # We only use n(n-1)/2 random samples
        # For each edge, sample either A->B, B->A or no edge
        # We convert this to a proper adjacency matrix using torch.tril_indices
        self.logits = nn.Parameter(torch.zeros(3, (input_dim * (input_dim - 1)) // 2, device=device), requires_grad=True)
        self.tau_gumbel = tau_gumbel
        self.input_dim = input_dim
        self._device = device
        self.lower_idxs = torch.unbind(torch.tril_indices(self.input_dim, self.input_dim, offset=-1, device=self._device), 0)

    def _triangular_vec_to_matrix(self, vec):
        """
        Given an array of shape (k, n(n-1)/2) where k in {2, 3}, creates a matrix of shape
        (n, n) where the lower triangular is filled from vec[0, :] and the upper
        triangular is filled from vec[1, :].
        """
        output = torch.zeros((self.input_dim, self.input_dim), device=self._device)
        output[self.lower_idxs[0], self.lower_idxs[1]] = vec[0, ...]
        output[self.lower_idxs[1], self.lower_idxs[0]] = vec[1, ...]
        return output

    def get_adj_matrix(self, round: bool = False) -> torch.Tensor:
        """
        Returns the adjacency matrix of edge probabilities.
        """
        probs = F.softmax(self.logits, dim=0)  # (3, n(n-1)/2) probabilities
        out_probs = self._triangular_vec_to_matrix(probs)
        if round:
            return out_probs.round()
        else:
            return out_probs

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q, which is a collection of n(n-1) categoricals on 3 values.
        """
        dist = td.Categorical(logits=self.logits.transpose(0, -1))
        entropies = dist.entropy()
        return entropies.sum()

    def sample_A(self) -> torch.Tensor:
        """
        Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
        and returns hard samples (straight through gradient estimator). Adjacency returned always has
        zeros in its diagonal (no self loops).

        V1: Returns one sample to be used for the whole batch.
        """
        sample = F.gumbel_softmax(self.logits, tau=self.tau_gumbel, hard=True, dim=0)  # (3, n(n-1)/2) binary
        return self._triangular_vec_to_matrix(sample)
