# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, Optional, Any, Callable, List, Union
from typing import Tuple

import networkx as nx
import numpy as np
import scipy
import torch
import torch.distributions as td
from dependency_injector.wiring import Provide
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .base_distributions import GaussianBase, DiagonalFLowBase, CategoricalLikelihood, BinaryLikelihood
from .generation_functions import ContractiveInvertibleGNN
from .variational_distributions import VarDistA_Simple, VarDistA_ENCO, DeterministicAdjacency, ThreeWayGraphDist
from ...datasets.dataset import CausalDataset, Dataset, TemporalDataset
from ...datasets.variables import Variables
from ...experiment.azua_context import AzuaContext
from ..imodel import (
    IModelForInterventions,
    IModelForImputation,
    IModelForCausalInference,
    IModelForCounterfactuals,
)
from ..torch_model import TorchModel
from ...utils.causality_utils import (
    calculate_ite,
    intervene_graph,
    get_ate_from_samples,
    get_cate_from_samples,
    intervention_to_tensor,
    process_adjacency_mats,
    get_mask_from_idxs,
)
from ...utils.data_mask_utils import to_tensors
from ...utils.nri_utils import edge_prediction_metrics_multisample
from ...utils.fast_data_loader import FastTensorDataLoader
from ...utils.torch_utils import generate_fully_connected
from ...utils.training_objectives import get_input_and_scoring_masks


class DECI(TorchModel, IModelForInterventions, IModelForCausalInference, IModelForImputation, IModelForCounterfactuals):
    """
    Flow-based VISL model, which does causal discovery using a contractive and
    invertible GNN. The adjacency is a random variable over which we do inference.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        imputation: bool = True,
        lambda_dag: float = 1.0,
        lambda_sparse: float = 1.0,
        lambda_prior: float = 1.0,
        tau_gumbel: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        var_dist_A_mode: str = "enco",
        imputer_layer_sizes: Optional[List[int]] = None,
        mode_f_sem: str = "gnn_i",
        mode_adjacency: str = "learn",
        norm_layers: bool = True,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        cate_rff_n_features: int = 3000,
        cate_rff_lengthscale: Union[int, float, List[float], Tuple[float]] = (0.1, 1),
        prior_A: Union[torch.Tensor, np.ndarray] = None,
        prior_A_confidence: float = 0.5,
        prior_mask: Union[torch.Tensor, np.ndarray] = None,
        graph_constraint_matrix: Optional[np.ndarray] = None,
    ):
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device to load model to.
            imputation: Whether to train an imputation network simultaneously with the DECI network.
            lambda_dag: Coefficient for the prior term that enforces DAG.
            lambda_sparse: Coefficient for the prior term that enforces sparsity.
            lambda_prior: Coefficient for the prior term that enforces prior.
            tau_gumbel: Temperature for the gumbel softmax trick.
            base_distribution_type: which type of base distribution to use for the additive noise SEM
            Options:
                fixed_gaussian: Gaussian with non-leanrable mean of 0 and variance of 1
                gaussian: Gaussian with fixed mean of 0 and learnable variance
                spline: learnable flow transformation which composes an afine layer a spline and another affine layer
            spline_bins: How many bins to use for spline flow base distribution if the 'spline' choice is made
            var_dist_A_mode: Variational distribution for adjacency matrix. Admits {"simple", "enco", "true", "three"}.
                             "simple" parameterizes each edge (including orientation) separately. "enco" parameterizes
                             existence of an edge and orientation separately. "true" uses the true graph.
                             "three" uses a 3-way categorical sample for each (unordered) pair of nodes.
            imputer_layer_sizes: Number and size of hidden layers for imputer NN for variational distribution.
            mode_f_sem: Mode used for function. Admits {"linear", "lrelu", "gnn_i"}. The first one
                        is a linear function, the second leaky relu. The third one described in pdf.
            mode_adjacency: In {"upper", "lower", "learn"}. If "learn", do our method as usual. If
                            "upper"/"lower" fix adjacency matrix to strictly upper/lower triangular.
            norm_layers: bool indicating whether all MLPs should use layer norm
            res_connection:  bool indicating whether all MLPs should use layer norm
            encoder_layer_sizes: Optional list indicating width of layers in GNN encoder MLP
            decoder_layer_sizes: Optional list indicating width of layers in GNN decoder MLP,
            cate_rff_n_features: number of random features to use in functiona pproximation when estimating CATE,
            cate_rff_lengthscale: lengthscale of RBF kernel used when estimating CATE,
            prior_A: prior adjacency matrix,
            prior_A_confidence: degree of confidence in prior adjacency matrix enabled edges between 0 and 1,
            graph_constraint_matrix: a matrix imposing constraints on the graph. If the entry (i, j) of the constraint
                            matrix is 0, then there can be no edge i -> j, if the entry is 1, then an edge i -> j
                            must exist, if the entry is `nan`, then an edge i -> j may be learned.
                            By default, only self-edges are constrained to not exist.
        """
        super().__init__(model_id, variables, save_dir, device)
        self.data_processor._squash_input = False  # Avoid squashing data to [0, 1]

        self._device = device
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.lambda_prior = lambda_prior

        self.cate_rff_n_features = cate_rff_n_features
        self.cate_rff_lengthscale = cate_rff_lengthscale

        # DECI treats *groups* as distinct nodes in the graph
        self.validate_variable_groups(variables)
        self.num_nodes = variables.num_query_groups
        self.processed_dim_all = variables.num_processed_non_aux_cols

        # set up soft prior over graphs
        if prior_A is not None:
            self.exist_prior = True
            self.prior_A = nn.Parameter(torch.tensor(prior_A, device=self._device), requires_grad=False)
            self.prior_mask = nn.Parameter(torch.tensor(prior_mask, device=self._device), requires_grad=False)
        else:
            self.exist_prior = False
            self.prior_A = nn.Parameter(
                torch.zeros((self.num_nodes, self.num_nodes), device=self._device), requires_grad=False
            )
            self.prior_mask = nn.Parameter(
                torch.zeros((self.num_nodes, self.num_nodes), device=self._device), requires_grad=False
            )
        assert prior_A_confidence >= 0 and prior_A_confidence <= 1
        self.prior_A_confidence = prior_A_confidence

        # Set up the Neural Nets
        self.res_connection = res_connection
        self.norm_layer = nn.LayerNorm if norm_layers else None
        self.ICGNN = ContractiveInvertibleGNN(
            torch.tensor(variables.group_mask),
            device,
            mode_f_sem,
            norm_layer=self.norm_layer,
            res_connection=self.res_connection,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
        )

        self.spline_bins = spline_bins
        self.likelihoods = nn.ModuleDict(self._generate_error_likelihoods(base_distribution_type, variables))
        self.variables = variables

        self.imputation = imputation
        if self.imputation:
            self.all_cts = all(var.type == "continuous" for var in variables)
            imputation_input_dim = 2 * self.processed_dim_all
            imputation_output_dim = 2 * self.processed_dim_all
            imputer_layer_sizes = imputer_layer_sizes or [max(80, imputation_input_dim)] * 2
            self.imputer_network = generate_fully_connected(
                input_dim=imputation_input_dim,
                output_dim=imputation_output_dim,
                hidden_dims=imputer_layer_sizes,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=self._device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        self.mode_adjacency = mode_adjacency
        if var_dist_A_mode == "simple":
            self.var_dist_A = VarDistA_Simple(device=device, input_dim=self.num_nodes, tau_gumbel=tau_gumbel)
        elif var_dist_A_mode == "enco":
            self.var_dist_A = VarDistA_ENCO(device=device, input_dim=self.num_nodes, tau_gumbel=tau_gumbel)
        elif var_dist_A_mode == "true":
            self.var_dist_A = DeterministicAdjacency(device=device)
        elif var_dist_A_mode == "three":
            self.var_dist_A = ThreeWayGraphDist(device=device, input_dim=self.num_nodes, tau_gumbel=tau_gumbel)
        else:
            raise NotImplementedError()

        self.set_graph_constraint(graph_constraint_matrix)

    def set_graph_constraint(self, graph_constraint_matrix: Optional[np.ndarray]):
        if graph_constraint_matrix is None:
            self.neg_constraint_matrix = 1.0 - torch.eye(self.num_nodes, device=self._device)
            self.pos_constraint_matrix = torch.zeros((self.num_nodes, self.num_nodes), device=self._device)
        else:
            negative_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=1.0)
            self.neg_constraint_matrix = torch.tensor(negative_constraint_matrix, device=self._device)
            self.neg_constraint_matrix *= 1.0 - torch.eye(self.num_nodes, device=self._device)
            positive_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=0.0)
            self.pos_constraint_matrix = torch.tensor(positive_constraint_matrix, device=self._device)

    def networkx_graph(self):

        """
        This samples the most probable graphs and conver to networkx digraph
        Returns:
            A networkx digraph object
        """

        # Return the most probable graph from a fitted DECI model in networkx.digraph form
        adj_mat = self.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
        # Check if non DAG adjacency matrix
        assert (np.trace(scipy.linalg.expm(adj_mat)) - self.num_nodes) == 0, "Generate non DAG graph"
        return nx.convert_matrix.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)

    def sample_graph_posterior(self, round: bool = True, samples: int = 100):

        """
        This sample the adjacency matrix from the posterior and convert them to networkx format without duplicates
        Args:
            round (bool):  If we round the probability during sampling.
            samples (int): The number of adjacency matrix sampled from the posterior

        Returns:
            A list of networkx digraph object.
        """

        # Return a list of DAG networkx digraph without duplicates
        adj_mats = self.get_adj_matrix(round=round, samples=samples)
        # Remove the duplicates and non DAG adjacency matrix
        adj_mats, adj_weights = process_adjacency_mats(adj_mats, self.num_nodes)
        graph_list = [nx.convert_matrix.from_numpy_matrix(adj_mat, create_using=nx.DiGraph) for adj_mat in adj_mats]
        return graph_list, adj_weights

    @staticmethod
    def validate_variable_groups(variables):
        for group in variables.query_group_idxs:
            # Assert group has a single type
            group_type = variables[group[0]].type
            assert all(
                variables[i].type == group_type for i in group
            ), f"Group '{variables[group[0]].group_name}' contains a mixture of types, which is not supported."
            # We only support vector-valued nodes (non-singleton groups) for certain data types
            if group_type not in ["continuous"]:
                assert len(group) == 1, "Vector-valued nodes of type {group_type} are not currently supported."

    def _generate_error_likelihoods(self, base_distribution_string: str, variables: Variables):
        """
        Instantiate error likelihood models for each variable in the SEM.
        For continuous variables, the likelihood is for an additive noise model whose exact form is determined by
        the `base_distribution_string` argument (see below for exact options). For vector-valued continuous variables,
        we assume the error model factorises over dimenions.
        For discrete variables, the likelihood directly parametrises the per-class probabilities.
        For sampling these variables, we rely on the Gumbel max trick. For this to be implemented correctly, this method
        also returns a list of subarrays which should be treated as single categorical variables and processed with a
        `max` operation during sampling.

        Args:
            base_distribution_string: which type of base distribution to use for the additive noise SEM
            Options:
                fixed_gaussian: Gaussian with non-leanrable mean of 0 and variance of 1
                gaussian: Gaussian with fixed mean of 0 and learnable variance
                spline: learnable flow transformation which composes an afine layer a spline and another affine layer
        Returns:
            A dictionary from variable type to the likelihood distribution(s) for variables of that type.
        """

        conditional_dists = {}
        typed_regions = variables.processed_cols_by_type
        # Continuous
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            if base_distribution_string == "fixed_gaussian":
                dist = GaussianBase(len(continuous_range), device=self._device, train_base=False)
            if base_distribution_string == "gaussian":
                dist = GaussianBase(len(continuous_range), device=self._device, train_base=True)
            elif base_distribution_string == "spline":
                dist = DiagonalFLowBase(
                    len(continuous_range), device=self._device, num_bins=self.spline_bins, flow_steps=1
                )
            else:
                raise NotImplementedError("Base distribution type not recognised")
            conditional_dists["continuous"] = dist

        # Binary
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            conditional_dists["binary"] = BinaryLikelihood(len(binary_range), device=self._device)

        # Categorical
        if "categorical" in typed_regions:
            conditional_dists["categorical"] = nn.ModuleList(
                [CategoricalLikelihood(len(region), device=self._device) for region in typed_regions["categorical"]]
            )

        return conditional_dists

    @classmethod
    def name(cls) -> str:
        return "deci"

    def _get_adj_matrix_tensor(
        self, round: bool = True, samples: int = 100, most_likely_graph: bool = False
    ) -> torch.Tensor:
        if self.mode_adjacency == "learn":
            if most_likely_graph:
                assert samples == 1, "When passing most_likely_graph, only 1 sample can be returned."
                A_samples = [self.var_dist_A.get_adj_matrix(round=round)]
            else:
                A_samples = [self.var_dist_A.sample_A() for _ in range(samples)]
                if round:
                    A_samples = [A.round() for A in A_samples]
            adj = torch.stack(A_samples, dim=0)
        elif self.mode_adjacency == "upper":
            adj = (
                torch.triu(torch.ones(self.num_nodes, self.num_nodes), diagonal=1)
                .to(self._device)
                .expand(samples, -1, -1)
            )
        elif self.mode_adjacency == "lower":
            adj = (
                torch.tril(torch.ones(self.num_nodes, self.num_nodes), diagonal=-1)
                .to(self._device)
                .expand(samples, -1, -1)
            )
        else:
            raise NotImplementedError("Adjacency mode %s not implemented" % self.mode_adjacency)
        return self._apply_constraints(adj)

    def _apply_constraints(self, G: torch.Tensor) -> torch.Tensor:
        # Set all entries where self.neg_contraint_matrix=0 to 0, leave elements where self.neg_constraint_matrix=1 unchanged
        G = G * self.neg_constraint_matrix
        # Set all entries where self.pos_contraint_matrix=1 to 1, leave elements where self.pos_constraint_matrix=0 unchanged
        G = 1.0 - (1.0 - G) * (1.0 - self.pos_constraint_matrix)
        return G

    def get_adj_matrix(
        self, round: bool = True, samples: int = 100, most_likely_graph: bool = False, squeeze: bool = False
    ) -> np.ndarray:
        """
        Returns the adjacency matrix (or several) as a numpy array.
        """
        adj_matrix = self._get_adj_matrix_tensor(round, samples, most_likely_graph)

        if squeeze and samples == 1:
            adj_matrix = adj_matrix.squeeze(0)
        # Here we have the cast to np.float64 because the original type
        # np.float32 has some issues with json, when saving causality results
        # to a file.
        return adj_matrix.detach().cpu().numpy().astype(np.float64)

    def get_weighted_adj_matrix(
        self, round: bool = True, samples: int = 100, most_likely_graph: bool = False, squeeze: bool = False
    ) -> np.ndarray:
        """
        Returns the weighted adjacency matrix (or several) as a numpy array.
        """
        A_samples = self._get_adj_matrix_tensor(round, samples, most_likely_graph)

        W_adjs = A_samples * self.ICGNN.get_weighted_adjacency().unsqueeze(0)

        if squeeze and samples == 1:
            W_adjs = W_adjs.squeeze(0)

        return W_adjs

    def dagness_factor(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the dag penalty for matrix A as trace(expm(A)) - dim.

        Args:
            A: Binary adjacency matrix, size (input_dim, input_dim).
        """
        return torch.trace(torch.matrix_exp(A)) - self.num_nodes

    def _log_prior_A(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the prior for adjacency matrix A, which consitst on a term encouraging DAGness
        and another encouraging sparsity (see https://arxiv.org/pdf/2106.07635.pdf).

        Args:
            A: Adjancency matrix of shape (input_dim, input_dim), binary.

        Returns:
            Log probability of A for prior distribution, a number.
        """

        sparse_term = -self.lambda_sparse * A.abs().sum()
        if self.exist_prior:
            prior_term = (
                -self.lambda_prior * (self.prior_mask * (A - self.prior_A_confidence * self.prior_A)).abs().sum()
            )
            return sparse_term + prior_term
        else:
            return sparse_term

    def get_auglag_penalty(self, tracker_dag_penalty: List) -> float:
        """
        Computes DAG penalty for augmented Lagrangian update step as the average of dag factors of binary
        adjacencies sampled during this inner optimization step.
        """
        return torch.mean(torch.Tensor(tracker_dag_penalty)).item()

    def _log_prob(
        self, x: torch.Tensor, predict: torch.Tensor, intervention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the log probability of the observed data given the predictions from the SEM.

        Args:
            x: Array of size (processed_dim_all) or (batch_size, processed_dim_all), works both ways (i.e. single sample
            or batched).
            predict: tensor of the same shape as x.
            intervention_mask (num_nodes): optional array containing indicators of variables that have been intervened upon.
            These will not be considered for log probability computation.

        Returns:
            Log probability of non intervened samples. A number if x has shape (input_dim), or an array of
            shape (batch_size) is x has shape (batch_size, input_dim).
        """

        typed_regions = self.variables.processed_cols_by_type
        # Continuous
        cts_bin_log_prob = torch.zeros_like(x)
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                x[..., continuous_range] - predict[..., continuous_range]
            )

        # Binary
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            cts_bin_log_prob[..., binary_range] = self.likelihoods["binary"].log_prob(
                x[..., binary_range], predict[..., binary_range]
            )

        if intervention_mask is not None:
            cts_bin_log_prob[..., intervention_mask] = 0.0

        log_prob = cts_bin_log_prob.sum(-1)

        # Categorical
        if "categorical" in typed_regions:
            for region, likelihood, idx in zip(
                typed_regions["categorical"],
                self.likelihoods["categorical"],
                self.variables.var_idxs_by_type["categorical"],
            ):
                # Can skip likelihood computation completely if intervened
                if (intervention_mask is None) or (intervention_mask[idx] is False):
                    log_prob += likelihood.log_prob(x[..., region], predict[..., region])

        return log_prob

    def _sample_base(self, Nsamples: int) -> torch.Tensor:
        """
        Draw samples from the base distribution.

        Args:
            Nsamples: Number of samples to draw

        Returns:
            torch.Tensor z of shape (batch_size, input_dim).
        """
        sample = torch.zeros((Nsamples, self.processed_dim_all), device=self._device)
        typed_regions = self.variables.processed_cols_by_type

        # Continuous
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            sample[:, continuous_range] = self.likelihoods["continuous"].sample(Nsamples)

        # Binary
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            sample[:, binary_range] = self.likelihoods["binary"].sample(Nsamples)

        # Categorical
        if "categorical" in typed_regions:
            for region, likelihood in zip(typed_regions["categorical"], self.likelihoods["categorical"]):
                sample[:, region] = likelihood.sample(Nsamples)

        return sample

    def cate(
        self,
        intervention_idxs: Union(torch.Tensor, np.ndarray),
        intervention_values: Union(torch.Tensor, np.ndarray),
        reference_values: Optional[Union(torch.Tensor, np.ndarray)] = None,
        effect_idxs: Optional[Union(torch.Tensor, np.ndarray)] = None,
        conditioning_idxs: Optional[Union(torch.Tensor, np.ndarray)] = None,
        conditioning_values: Optional[Union(torch.Tensor, np.ndarray)] = None,
        Nsamples_per_graph: int = 2,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
    ):
        """
        Returns average treatment effect for a given intervention. Optionally, can condition on additional variables

        Args:
            intervention_idxs: torch.Tensor of shape (input_dim) array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) array containing values for variables that have been intervened.
            reference_values: optional torch.Tensor containing a reference value for the treatment. If specified will compute
                E[Y | do(T=k), X] - E[Y | do(T=k'), X]. Otherwise will compute E[Y | do(T=k), X] - E[Y | X] = E[Y | do(T=k), X] - E[Y | do(T=mu_T), X]
            effect_idxs:  optional torch.Tensor containing indices on which treatment effects should be evaluated
            conditioning_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that we condition on.
            conditioning_values: torch.Tensor of shape (input_dim) optional array containing values for variables that we condition on.
            Nsamples_per_graph: int containing number of samples to draw
            Ngraphs: Number of different graphs to sample for graph posterior marginalisation. If None, defaults to Nsamples
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample


        Returns:
            (ate, ate_norm): (np.ndarray, np.ndarray) both of size (input_dim) average treatment effect computed on regular and normalised data
        """
        assert Nsamples_per_graph > 1, "Nsamples_per_graph must be greater than 1"
        intervention_idxs, _, intervention_values = intervention_to_tensor(
            intervention_idxs, intervention_values, group_mask=self.variables.group_mask, device=self._device
        )
        Nsamples = Ngraphs * Nsamples_per_graph

        with torch.no_grad():
            if reference_values is None:
                reference_samples = self.sample(
                    Nsamples=Nsamples, most_likely_graph=most_likely_graph, samples_per_graph=Nsamples_per_graph
                ).detach()
            else:
                reference_samples = self.sample(
                    Nsamples=Nsamples,
                    most_likely_graph=most_likely_graph,
                    intervention_idxs=intervention_idxs,
                    intervention_values=reference_values,
                    samples_per_graph=Nsamples_per_graph,
                ).detach()

            model_intervention_samples = self.sample(
                Nsamples=Nsamples,
                most_likely_graph=most_likely_graph,
                intervention_idxs=intervention_idxs,
                intervention_values=intervention_values,
                samples_per_graph=Nsamples_per_graph,
            ).detach()

            # CATE computation
            # We use pytorch to fit the CATE models to DECI samples and cast back to numpy at the end
            if conditioning_idxs is not None and conditioning_values is not None:

                assert effect_idxs is not None, "need to specify effect indices for CATE computation"
                effect_mask = get_mask_from_idxs(effect_idxs, group_mask=self.variables.group_mask, device=self._device)

                conditioning_idxs, conditioning_mask, conditioning_values = intervention_to_tensor(
                    conditioning_idxs, conditioning_values, self.variables.group_mask, device=self._device
                )

                # Reshape samples to be asociated with the graph used to generate them
                reference_samples = reference_samples.view(
                    int(Ngraphs), int(Nsamples_per_graph), -1
                )  # (Ngraphs, samples_per_graph, input_dim)
                model_intervention_samples = model_intervention_samples.view(
                    int(Ngraphs), int(Nsamples_per_graph), -1
                )  # (Ngraphs, samples_per_graph, input_dim)

                model_cate = get_cate_from_samples(
                    intervened_samples=model_intervention_samples,
                    baseline_samples=reference_samples,
                    conditioning_mask=conditioning_mask,
                    conditioning_values=conditioning_values,
                    effect_mask=effect_mask,
                    variables=self.variables,
                    normalise=False,
                    rff_lengthscale=self.cate_rff_lengthscale,
                    rff_n_features=self.cate_rff_n_features,
                )

                model_norm_cate = get_cate_from_samples(
                    intervened_samples=model_intervention_samples,
                    baseline_samples=reference_samples,
                    conditioning_mask=conditioning_mask,
                    conditioning_values=conditioning_values,
                    effect_mask=effect_mask,
                    variables=self.variables,
                    normalise=True,
                    rff_lengthscale=self.cate_rff_lengthscale,
                    rff_n_features=self.cate_rff_n_features,
                )

                return model_cate.cpu().numpy().astype(np.float64), model_norm_cate.cpu().numpy().astype(np.float64)

            # ATE computation
            # Computation of ATE from samples is cheap. We cast the samples to numpy and then compute ATE.
            else:

                model_ate = get_ate_from_samples(
                    model_intervention_samples.cpu().numpy().astype(np.float64),
                    reference_samples.cpu().numpy().astype(np.float64),
                    self.variables,
                    normalise=False,
                    processed=True,
                )
                model_norm_ate = get_ate_from_samples(
                    model_intervention_samples.cpu().numpy().astype(np.float64),
                    reference_samples.cpu().numpy().astype(np.float64),
                    self.variables,
                    normalise=True,
                    processed=True,
                )
                # We always compute intervention effect on all variable by model construction. We then throw away info we dont need
                if effect_idxs is not None:
                    effect_mask = get_mask_from_idxs(
                        effect_idxs, group_mask=self.variables.group_mask, device="cpu"
                    ).numpy()
                    model_ate, model_norm_ate = model_ate[effect_mask], model_norm_ate[effect_mask]

                return model_ate, model_norm_ate

    def ite(
        self,
        X: torch.Tensor,
        intervention_idxs: Optional[Union(torch.Tensor, np.ndarray)] = None,
        intervention_values: Optional[Union(torch.Tensor, np.ndarray)] = None,
        reference_values: Optional[Union(torch.Tensor, np.ndarray)] = None,
        Ngraphs: int = 100,
        most_likely_graph: bool = False,
    ) -> np.ndarray:
        """Calculate the individual treatment effect on interventions on observations X.

        Args:
            X: torch.Tensor of shape (Nsamples, input_dim) containing the observations we want to evaluate
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.
            reference_values: optional torch.Tensor containing a reference value for the treatment. If specified will compute
                E_{graph}[X_{T=k} | X, T_obs] - E_{graph}[X_{T=k'} | X, T_obs]. Otherwise will compute E_{graph}[X_{T=k} | X, T_obs] - X
            Nsamples: int containing number of graph samples to draw.
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior instead of sampling graphs

        Returns:
            np.ndarray of shape (Nsamples, input_dim) containing the individual treatment effect on interventions on observations X.
        """
        with torch.no_grad():
            W_adjs = self.get_weighted_adj_matrix(round=True, samples=Ngraphs, most_likely_graph=most_likely_graph)

            # Get counterfactual samples
            counterfactuals = [
                self._counterfactual(X, W_adj, intervention_idxs, intervention_values) for W_adj in W_adjs
            ]

            # Calculate the difference between the counterfactuals and the baseline. This currently only supports continuous variables.
            if reference_values is None:
                ite = torch.stack(counterfactuals, dim=0).mean(dim=0) - X
            else:
                reference_counterfactuals = [
                    self._counterfactual(X, W_adj, intervention_idxs, reference_values) for W_adj in W_adjs
                ]

                ite = calculate_ite(
                    torch.stack(counterfactuals, dim=0).mean(dim=0),
                    torch.stack(reference_counterfactuals, dim=0).mean(dim=0),
                )

        return ite.cpu().numpy().astype(np.float64)

    def sample(
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union(torch.Tensor, np.ndarray)] = None,
        intervention_values: Optional[Union(torch.Tensor, np.ndarray)] = None,
        samples_per_graph: int = 1,
        max_batch_size: int = 1024,
    ) -> torch.Tensor:
        """
        Draws samples from the causal flow model. Optionally these can be subject to an intervention of some variables

        Args:
            Nsamples: int containing number of samples to draw
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.
            samples_per_graph: how many samples to draw per graph sampled from the posterior. If most_likely_graph is true, this variable will default to 1,
            max_batch_size: maximum batch size to use for DECI when drawing samples. Larger is faster but more memory intensive
        Returns:
            Samples: torch.Tensor of shape (Nsamples, input_dim).
        """
        if most_likely_graph:
            assert (
                Nsamples == samples_per_graph
            ), "When setting most_likely_graph=True, the number of samples should equal the number of samples per graph"
        else:
            assert Nsamples % samples_per_graph == 0, "Nsamples must be a multiple of samples_per_graph"

        intervention_idxs, intervention_mask, intervention_values = intervention_to_tensor(
            intervention_idxs, intervention_values, self.variables.group_mask, device=self._device
        )
        gumbel_max_regions = self.variables.processed_cols_by_type["categorical"]
        gt_zero_region = [j for i in self.variables.processed_cols_by_type["binary"] for j in i]

        with torch.no_grad():

            num_graph_samples = Nsamples // samples_per_graph
            W_adj_samples = self.get_weighted_adj_matrix(
                round=most_likely_graph, samples=num_graph_samples, most_likely_graph=most_likely_graph
            )
            if most_likely_graph:
                W_adj_samples = W_adj_samples.expand(Nsamples, -1, -1)
            else:
                W_adj_samples = torch.repeat_interleave(W_adj_samples, repeats=int(samples_per_graph), dim=0)
            # W_adj_samples shape (Nsamples, input_dim, input_dim)

            # Z shape (Nsamples, input_dim)
            Z = self._sample_base(Nsamples)

            X = []
            for W_adj_batch, Z_batch in zip(
                torch.split(W_adj_samples, max_batch_size, dim=0), torch.split(Z, max_batch_size, dim=0)
            ):
                X.append(
                    self.ICGNN.simulate_SEM(
                        Z_batch, W_adj_batch, intervention_mask, intervention_values, gumbel_max_regions, gt_zero_region
                    ).detach()
                )
            samples = torch.cat(X, dim=0)
            assert samples.shape[0] == Nsamples
            return samples

    def _counterfactual(
        self,
        X: torch.Tensor,
        W_adj: torch.Tensor,
        intervention_idxs: Union(torch.Tensor, np.ndarray) = None,
        intervention_values: Union(torch.Tensor, np.ndarray) = None,
    ) -> torch.Tensor:
        """Calculates a counterfactual for a given input X and given graph A_sample.

        Args:
            X: torch.Tensor of shape (Nsamples, input_dim) containing the observations we want to evaluate
            W_adj: torch.Tensor of shape (input_dim, input_dim) containing the weighted adjacency matrix of the graph
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.

        Returns:
            counterfactual: torch.Tensor of shape (Nsamples, input_dim) containing the counterfactuals
        """

        (X,) = to_tensors(X, device=self._device, dtype=torch.float)

        intervention_idxs, intervention_mask, intervention_values = intervention_to_tensor(
            intervention_idxs, intervention_values, self.variables.group_mask, device=self._device
        )

        gumbel_max_regions = self.variables.processed_cols_by_type["categorical"]
        gt_zero_region = [j for i in self.variables.processed_cols_by_type["binary"] for j in i]

        # Infer exogeneous noise variables - this currently only supports continuous variables.
        predict = self.ICGNN.predict(X, W_adj)
        noise_variable_posterior_samples = torch.zeros_like(X)

        typed_regions = self.variables.processed_cols_by_type
        # Continuous
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            # Additive Noise Model applies
            noise_variable_posterior_samples[..., continuous_range] = (
                X[..., continuous_range] - predict[..., continuous_range]
            )

        # Categorical
        if "categorical" in typed_regions:
            for region, likelihood in zip(typed_regions["categorical"], self.likelihoods["categorical"]):
                noise_variable_posterior_samples[..., region] = likelihood.posterior(
                    X[..., region], predict[..., region]
                )

        # Binary: this operation can be done in parallel, so no loop over individual variables
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            noise_variable_posterior_samples[..., binary_range] = self.likelihoods["binary"].posterior(
                X[..., binary_range], predict[..., binary_range]
            )

        # Get counterfactual by intervening on the graph and forward propagating using the inferred noise variable
        X_cf = self.ICGNN.simulate_SEM(
            noise_variable_posterior_samples,
            W_adj,
            intervention_mask,
            intervention_values,
            gumbel_max_regions,
            gt_zero_region,
        )

        return X_cf

    def log_prob(
        self,
        X: torch.Tensor,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union(torch.Tensor, np.ndarray)] = None,
        intervention_values: Optional[Union(torch.Tensor, np.ndarray)] = None,
    ) -> np.ndarray:

        """
        Evaluate log-probability of observations X. Optionally this evaluation can be subject to an intervention on our causal model.
        Then, the log probability of non-intervened variables is computed.

        Args:
            X: torch.Tensor of shape (Nsamples, input_dim) containing the observations we want to evaluate
            Nsamples: int containing number of graph samples to draw.
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior instead of sampling graphs
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.

        Returns:
            log_prob: torch.tensor  (Nsamples)
        """

        # TODO: move these lines into .log_prob(), add dimmention check to sampling / ate code as well

        (X,) = to_tensors(X, device=self._device, dtype=torch.float)

        intervention_idxs, intervention_mask, intervention_values = intervention_to_tensor(
            intervention_idxs, intervention_values, self.variables.group_mask, device=self._device
        )

        if intervention_mask is not None and intervention_values is not None:
            X[:, intervention_mask] = intervention_values

        with torch.no_grad():

            log_prob_samples = []

            if most_likely_graph:
                Nsamples = 1

            for _ in range(Nsamples):

                W_adj = self.get_weighted_adj_matrix(
                    round=most_likely_graph, samples=1, most_likely_graph=most_likely_graph, squeeze=True
                )
                # This sets certain elements of W_adj to 0, to respect the intervention
                W_adj = intervene_graph(W_adj, intervention_idxs, copy_graph=False)

                predict = self.ICGNN.predict(X, W_adj)
                log_prob_samples.append(self._log_prob(X, predict, intervention_mask))  # (B)

            log_prob_samples = torch.stack(log_prob_samples, dim=0)
            log_prob = torch.logsumexp(log_prob_samples, dim=0) - np.log(Nsamples)
            return log_prob.detach().cpu().numpy().astype(np.float64)

    def get_params_variational_distribution(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of samples x with missing values, returns the mean and scale of variational
        approximation over missing values.
        """
        if not self.all_cts:
            raise NotImplementedError("Imputation is only implemented for all-continuous data.")
        x_obs = x * mask
        input_net = torch.cat([x_obs, mask], dim=1)  # Shape (batch_size, 2 * processed_dim_all)
        out = self.imputer_network(input_net)
        mean = out[:, : self.processed_dim_all]  # Shape (batch_size, processed_dim_all)
        logscale = out[:, self.processed_dim_all :]  # Shape (batch_size, processed_dim_all)
        logscale = torch.clip(logscale, min=-20, max=5)
        scale = torch.exp(logscale)
        return mean, scale

    def impute_and_compute_entropy(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of samples x with missing values, returns the reparameterization filled in data
        and the entropy term for the variational bound.
        """
        # If fully observed do nothing (for efficicency)
        if (mask == 1).all():
            return x, torch.Tensor([0.0]).to(self._device), torch.Tensor([0.0]).to(self._device)
        mean, scale = self.get_params_variational_distribution(x, mask)
        var_dist = td.Normal(mean, scale)
        sample = var_dist.rsample()  # Shape (batch_size, input_dim)
        x_orig = x * mask
        x_filling = sample * (1 - mask)
        x_filled = x_orig + x_filling  # Shape (batch_size, input_dim)
        entropy = var_dist.entropy() * (1 - mask)  # Shape (batch_size, input_dim)
        entropy = entropy.sum(dim=1)  # Shape (batch_size)
        return x_filled, entropy, mean

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        *,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:
        data = torch.from_numpy(data.astype(np.float32)).to(self._device)
        mask = torch.from_numpy(mask.astype(np.float32)).to(self._device)
        mean, scale = self.get_params_variational_distribution(data, mask)
        if average:
            sample = data * mask + mean * (1.0 - mask)
            return mean.detach().cpu().numpy()
        else:
            var_dist = td.Normal(mean, scale)
            samples = []
            for _ in range(impute_config_dict["sample_count"]):
                sample = data * mask + var_dist.sample() * (1.0 - mask)
                samples.append(sample)
            return torch.stack(samples).detach().cpu().numpy()

    def _ELBO_terms(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes all terms involved in the ELBO.

        Args:
            X: Batched samples from the dataset, size (batch_size, input_dim).

        Returns:
            Tuple (penalty_dag, log_p_A, log_p_base, log_q_A)
        """
        # Get adjacency matrix with weights
        A_sample = self._get_adj_matrix_tensor(round=False, samples=1, most_likely_graph=False).squeeze(0)
        if self.mode_adjacency == "learn":
            factor_q = 1.0
        elif self.mode_adjacency in ["upper", "lower"]:
            factor_q = 0.0
        else:
            raise NotImplementedError("Adjacency mode %s not implemented" % self.mode_adjacency)
        W_adj = A_sample * self.ICGNN.get_weighted_adjacency()
        predict = self.ICGNN.predict(X, W_adj)
        log_p_A = self._log_prior_A(A_sample)  # A number
        penalty_dag = self.dagness_factor(A_sample)  # A number
        log_p_base = self._log_prob(X, predict)  # (B)
        log_q_A = -self.var_dist_A.entropy()  # A number
        return penalty_dag, log_p_A, log_p_base, log_q_A * factor_q

    def compute_loss(
        self,
        step: int,
        x: torch.Tensor,
        mask_train_batch: torch.Tensor,
        input_mask: torch.Tensor,
        num_samples: int,
        tracker: Dict,
        train_config_dict: Dict[str, Any] = {},
        alpha: float = None,
        rho: float = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes loss and updates trackers of different terms. mask_train_batch is the mask indicating
        which values are missing in the dataset, and input_mask indicates which additional values
        are artificially masked.
        """
        if self.imputation:
            # Get mean for imputation with artificially masked data, "regularizer" for imputation network
            _, _, mean_rec = self.impute_and_compute_entropy(x, input_mask)
            # Compute reconstruction loss on artificially dropped data
            scoring_mask = mask_train_batch
            reconstruction_term = scoring_mask * (mean_rec - x) ** 2  # Shape (batch_size, input_dim)
            reconstruction_term = reconstruction_term.sum(dim=1)  # Shape (batch_size)
            # Fill in missing data with amortized variational approximation for causality (without artificially masked data)
            x_fill, entropy_filled_term, _ = self.impute_and_compute_entropy(
                x, mask_train_batch
            )  # x_fill (batch_size, input_dim)  entropy  (batch_size)
            imputation_entropy = entropy_filled_term.mean(dim=0)  # (1)
            avg_reconstruction_err = reconstruction_term.mean(dim=0)
        else:
            x_fill = x
            imputation_entropy = avg_reconstruction_err = torch.tensor(0.0, device=self._device)

        #  Compute remaining terms
        penalty_dag, log_p_A_sparse, log_p_base, log_q_A = self._ELBO_terms(
            x_fill
        )  # penalty_dag (1), log_p_A_sparse (1), log_p_base (batch_size), log_q_A (1)
        log_p_term = log_p_base.mean(dim=0)  # batch average density under base distribution. (1)
        log_p_A_term = log_p_A_sparse / num_samples  # (1)
        log_q_A_term = log_q_A / num_samples  # (1)

        penalty_dag_term = penalty_dag * alpha / num_samples
        penalty_dag_term += penalty_dag * penalty_dag * rho / (2 * num_samples)

        if train_config_dict["anneal_entropy"] == "linear":
            ELBO = log_p_term + imputation_entropy + log_p_A_term - log_q_A_term / max(step - 5, 1) - penalty_dag_term
        elif train_config_dict["anneal_entropy"] == "noanneal":
            ELBO = log_p_term + imputation_entropy + log_p_A_term - log_q_A_term - penalty_dag_term
        loss = -ELBO + avg_reconstruction_err * train_config_dict["reconstruction_loss_factor"]

        tracker["loss"].append(loss.item())
        tracker["penalty_dag"].append(penalty_dag.item())
        tracker["penalty_dag_weighed"].append(penalty_dag_term.item())
        tracker["log_p_A_sparse"].append(log_p_A_term.item())
        tracker["log_p_x"].append(log_p_term.item())
        tracker["imputation_entropy"].append(imputation_entropy.item())
        tracker["log_q_A"].append(log_q_A_term.item())
        tracker["reconstruction_mse"].append(avg_reconstruction_err.item())
        return loss, tracker

    def print_tracker(self, inner_step: int, tracker: Dict) -> None:
        """
        Prints formatted contents of loss terms that are being tracked.
        """
        loss = np.mean(tracker["loss"])
        log_p_x = np.mean(tracker["log_p_x"])
        penalty_dag = np.mean(tracker["penalty_dag"])
        log_p_A_sparse = np.mean(tracker["log_p_A_sparse"])
        log_q_A = np.mean(tracker["log_q_A"])
        h_filled = np.mean(tracker["imputation_entropy"])
        reconstr = np.mean(tracker["reconstruction_mse"])
        print(
            f"Inner Step: {inner_step}, loss: {loss:.2f}, log p(x|A): {log_p_x:.2f}, dag: {penalty_dag:.8f}, \
                log p(A)_sp: {log_p_A_sparse:.2f}, log q(A): {log_q_A:.3f}, H filled: {h_filled:.3f}, rec: {reconstr:.3f}"
        )

    def process_dataset(
        self, dataset: Dataset, train_config_dict: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the training data and mask.
        Args:
            dataset: Dataset to use.
            train_config_dict: Dictionary with training hyperparameters.
        Returns:
            Tuple with data and mask tensors.
        """
        processed_dataset = self.data_processor.process_dataset(dataset)

        if isinstance(self.var_dist_A, DeterministicAdjacency):
            assert isinstance(dataset, CausalDataset)
            self.var_dist_A.set_adj_matrix(dataset.get_adjacency_data_matrix().astype(np.float32))

        data, mask = processed_dataset.train_data_and_mask
        data = data.astype(np.float32)
        # Should be careful if missing values?
        # TODO: if used, this has to also be applied to test data. Should be in data processor
        if train_config_dict["stardardize_data_mean"]:
            data = data - data.mean(axis=0)
        if train_config_dict["stardardize_data_std"]:
            data = data / data.std(axis=0)
        return data, mask

    def _create_dataloader_for_deci(
        self, dataset: Union[CausalDataset, TemporalDataset], train_config_dict: Dict[str, Any]
    ) -> Tuple[DataLoader, int]:
        """Create a data loader. For static deci, return an instance of FastTensorDataLoader and the number of samples.

        Args:
            dataset (Union[CausalDataset, TemporalDataset]): Dataset to generate a dataloader for.
            train_config_dict (Dict[str, Any]): Dictionary with training hyperparameters.

        Returns:
            DataLoader: DataLoader for the dataset.
            int: Number of samples in the dataset.
        """

        data, mask = self.process_dataset(dataset, train_config_dict)
        dataloader = FastTensorDataLoader(
            *to_tensors(data, mask, device=self._device), batch_size=train_config_dict["batch_size"], shuffle=True
        )
        return dataloader, len(data)

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Dict[str, Any] = {},
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
        azua_context: AzuaContext = Provide[AzuaContext],
    ) -> None:
        """
        Runs training.
        """

        dataloader, num_samples = self._create_dataloader_for_deci(dataset, train_config_dict)

        # initialise logging machinery
        train_output_dir = os.path.join(self.save_dir, "train_output")
        os.makedirs(train_output_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(train_output_dir, "summary"), flush_secs=1)
        metrics_logger = azua_context.metrics_logger()

        rho = train_config_dict["rho"]
        alpha = train_config_dict["alpha"]
        progress_rate = train_config_dict["progress_rate"]

        base_lr = train_config_dict["learning_rate"]

        # This allows the setting of the starting learning rate of each of the different submodules in the config, e.g. "likelihoods_learning_rate".
        parameter_list = [
            {"params": module.parameters(), "lr": train_config_dict.get(f"{name}_learning_rate", base_lr)}
            for name, module in self.named_children()
        ]

        self.opt = torch.optim.Adam(parameter_list)

        # Outer optimization loop
        base_idx = 0
        dag_penalty_prev = float("inf")
        num_below_tol = 0
        num_max_rho = 0
        num_not_done = 0
        for step in range(train_config_dict["max_steps_auglag"]):

            # stopping if DAG conditions satisfied
            if num_below_tol >= 5 or num_max_rho >= 3:
                break

            if rho >= train_config_dict["safety_rho"]:
                num_max_rho += 1

            # Inner loop
            print("Auglag Step: %i" % step)
            # Optimize adjacency for fixed rho and alpha
            outer_step_start_time = time.time()
            done_inner, tracker_loss_terms = self.optimize_inner_auglag(
                rho, alpha, step, num_samples, dataloader, train_config_dict
            )
            outer_step_time = time.time() - outer_step_start_time
            dag_penalty = np.mean(tracker_loss_terms["penalty_dag"])

            print("Dag penalty after inner: %.10f" % dag_penalty)
            print("Time taken for this step", outer_step_time)
            print(self.get_adj_matrix(round=True, most_likely_graph=True, samples=1))

            # Update alpha (and possibly rho) if inner optimization done or if 2 consecutive not-done
            if done_inner or num_not_done == 1:
                num_not_done = 0
                if dag_penalty < train_config_dict["tol_dag"]:
                    num_below_tol += 1
                if report_progress_callback is not None:
                    report_progress_callback(self.model_id, step + 1, train_config_dict["max_steps_auglag"])

                with torch.no_grad():
                    if dag_penalty > dag_penalty_prev * progress_rate:
                        print("Updating rho, dag penalty prev: %.10f" % dag_penalty_prev)
                        rho *= 10.0
                    else:
                        print("Updating alpha.")
                        dag_penalty_prev = dag_penalty
                        alpha += rho * dag_penalty
                        if dag_penalty == 0.0:
                            alpha *= 5
                    if rho >= train_config_dict["safety_rho"]:
                        alpha *= 5
                    rho = min([rho, train_config_dict["safety_rho"]])
                    alpha = min([alpha, train_config_dict["safety_alpha"]])

                    # logging outer progress and adjacency matrix
                    if isinstance(dataset, CausalDataset):
                        adj_matrix = self.get_adj_matrix(round=True, samples=100)
                        adj_metrics = edge_prediction_metrics_multisample(
                            dataset.get_adjacency_data_matrix(),
                            adj_matrix,
                            adj_matrix_mask=dataset.get_known_subgraph_mask_matrix(),
                            compute_mean=False,
                        )
                    else:
                        adj_metrics = None

                    base_idx = _log_epoch_metrics(
                        metrics_logger, writer, tracker_loss_terms, adj_metrics, step, outer_step_time, base_idx
                    )

            else:
                num_not_done += 1
                print("Not done inner optimization.")

            self.save()

            if dag_penalty_prev is not None:
                print("Dag penalty: %.15f" % dag_penalty)
                print("Rho: %.2f, alpha: %.2f" % (rho, alpha))

    def optimize_inner_auglag(
        self,
        rho: float,
        alpha: float,
        step: int,
        num_samples: int,
        dataloader,
        train_config_dict: Dict[str, Any] = {},
    ) -> Tuple[bool, float]:
        """
        Optimizes for a given alpha and rho.
        """

        def get_lr():
            for param_group in self.opt.param_groups:
                return param_group["lr"]

        def set_lr(factor):
            for param_group in self.opt.param_groups:
                param_group["lr"] = param_group["lr"] * factor

        def initialize_lr(val):
            for param_group in self.opt.param_groups:
                param_group["lr"] = val

        lim_updates_down = 3
        num_updates_lr_down = 0
        auglag_inner_early_stopping_lag = train_config_dict.get("auglag_inner_early_stopping_lag", 1500)
        auglag_inner_reduce_lr_lag = train_config_dict.get("auglag_inner_reduce_lr_lag", 500)
        initialize_lr(train_config_dict["learning_rate"])
        print("LR:", get_lr())
        best_loss = np.nan
        last_updated = -1
        done_opt = False

        tracker_loss_terms = defaultdict(list)
        inner_step = 0

        while inner_step < train_config_dict["max_auglag_inner_epochs"]:  # and not done_steps:

            for x, mask_train_batch in dataloader:
                input_mask, scoring_mask = get_input_and_scoring_masks(
                    mask_train_batch,
                    max_p_train_dropout=train_config_dict["max_p_train_dropout"],
                    score_imputation=True,
                    score_reconstruction=True,
                )
                loss, tracker_loss_terms = self.compute_loss(
                    step,
                    x,
                    mask_train_batch,
                    input_mask,
                    num_samples,
                    tracker_loss_terms,
                    train_config_dict,
                    alpha,
                    rho,
                )
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                inner_step += 1

                if int(inner_step) % 500 == 0:
                    self.print_tracker(inner_step, tracker_loss_terms)
                    break
                elif inner_step == train_config_dict["max_auglag_inner_epochs"]:
                    break

            # Save if loss improved
            if np.isnan(best_loss) or np.mean(tracker_loss_terms["loss"][-10:]) < best_loss:
                best_loss = np.mean(tracker_loss_terms["loss"][-10:])
                best_inner_step = inner_step
            # Check if has to reduce step size
            if (
                inner_step >= best_inner_step + auglag_inner_reduce_lr_lag
                and inner_step >= last_updated + auglag_inner_reduce_lr_lag
            ):
                last_updated = inner_step
                num_updates_lr_down += 1
                set_lr(0.1)
                print("Reducing lr to %.5f" % get_lr())
                if num_updates_lr_down >= 2:
                    done_opt = True
                if num_updates_lr_down >= lim_updates_down:
                    done_opt = True
                    print("Exiting at innner step %i." % inner_step)
                    # done_steps = True
                    break
            if inner_step >= best_inner_step + auglag_inner_early_stopping_lag:
                done_opt = True
                print("Exiting at innner step %i." % inner_step)
                # done_steps = True
                break
            if np.any(np.isnan(tracker_loss_terms["loss"])):
                print(tracker_loss_terms)
                print("Loss is nan, I'm done.", flush=True)
                # done_steps = True
                break

        print(f"Best model found at innner step {best_inner_step}, with Loss {best_loss:.2f}")
        return done_opt, tracker_loss_terms


# Auxiliary method that loggs training metrics to AML and tensorboard
def _log_epoch_metrics(
    metrics_logger,
    writer: SummaryWriter,
    tracker_loss_terms: dict,
    adj_metrics: Optional[dict],
    step: int,
    epoch_time: float,
    base_idx: int = 0,
):
    """
    Logging method for DECI training loop
    Args:
        metrics_logger: azua context metrics logger
        writer: tensorboard summarywriter used to log experiment results
        tracker_loss_terms: dictionary containing arrays with values generated at each inner-step during the inner optimisation procedure
        adj_metrics: Optional dictionary with adjacency matrixx discovery metrics
        step: outer step number
        epoch_time: time it took to perform outer step,
        base_idx: cummulative inner step number
    """

    # iterate over tracker vectors
    for key, value_list in tracker_loss_terms.items():
        metrics_logger.log_list(f"step_{step}_" + key, value_list)  # AzureML

        for i in range(len(tracker_loss_terms[key])):
            writer.add_scalar(f"step_{step}_" + key, tracker_loss_terms[key][i], i)  # tensorboard
            writer.add_scalar(key, tracker_loss_terms[key][i], i + base_idx)  # tensorboard

    base_idx += len(tracker_loss_terms[key])

    # Log time
    metrics_logger.log_value(f"step_{step}_time", epoch_time)
    writer.add_scalar("step_time", epoch_time, step)

    # log adjacency matrix metrics
    if adj_metrics is not None:
        for key, value in adj_metrics.items():
            writer.add_scalar(key + "_mean", np.mean(value), step)  # tensorboard
            writer.add_scalar(key + "std", np.std(value), step)  # tensorboard

    return base_idx
