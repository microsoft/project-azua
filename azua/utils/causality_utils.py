import torch
import numpy as np
from typing import Dict, List, Optional
from ..datasets.variables import Variables
from ..utils.data_mask_utils import to_tensors
from ..models.imodel import IModelForCausalInference, IModelForInterventions
from ..datasets.intervention_data import IntervetionData


def intervene_graph(adj_matrix: torch.Tensor, intervention_idxs: torch.Tensor, copy_graph: bool = True):
    """
    Simulates an intervention by removing all incoming edges for nodes being intervened

    Args:
        adj_matrix: torch.Tensor of shape (input_dim, input_dim) containing  adjacency_matrix
        intervention_idxs: torch.Tensor containing which variables to intervene
        copy_graph: bool whether the operation should be performed in-place or a new matrix greated
    """
    if intervention_idxs is None or len(intervention_idxs) == 0:
        return adj_matrix

    if copy_graph:
        adj_matrix = adj_matrix.copy()

    adj_matrix[:, intervention_idxs] = 0
    return adj_matrix


def intervention_to_tensor(intervention_idxs, intervention_values, device):
    """
    Maps empty interventions to nan and np.ndarray intervention data to torch tensors
    """

    if intervention_idxs is not None and intervention_values is not None:
        (intervention_idxs,) = to_tensors(intervention_idxs, device=device, dtype=torch.long)
        (intervention_values,) = to_tensors(intervention_values, device=device, dtype=torch.float)

        if intervention_idxs.dim() == 0:
            intervention_idxs = None

        if intervention_values.dim() == 0:
            intervention_values = None

    return intervention_idxs, intervention_values


def get_treatment_data_logprob(
    model: IModelForCausalInference, intervention_datasets: List[IntervetionData], most_likely_graph: bool = False,
):
    """
    Computes the log-probability of test-points sampled from intervened distributions.
    Args:
        model: IModelForInterventions with which we can evaluate the log-probability of points while applying interventions to the generative model
        intervention_datasets: List[IntervetionData] containing intervetions and samples from the ground truth data generating process when the intervention is applied
        most_likely_graph: whether to use the most likely causal graph (True) or to sample graphs (False)
    """
    all_log_probs = []
    per_intervention_log_probs_mean = []
    per_intervention_log_probs_std = []
    for intervention_data in intervention_datasets:

        # if intervention_data.effect_idxs is None:

        intervention_log_probs = model.log_prob(
            X=intervention_data.test_data.astype(float),
            most_likely_graph=most_likely_graph,
            intervention_idxs=intervention_data.intervention_idxs,
            intervention_values=intervention_data.intervention_values,
        )
        # Evaluate log-prob per dimension
        intervention_log_probs = intervention_log_probs / (
            intervention_data.test_data.shape[1] - len(intervention_data.intervention_idxs)
        )

        all_log_probs.append(intervention_log_probs)
        per_intervention_log_probs_mean.append(intervention_log_probs.mean(axis=0))
        per_intervention_log_probs_std.append(intervention_log_probs.std(axis=0))

    if len(all_log_probs) > 0:
        all_log_probs = np.concatenate(all_log_probs, axis=0)
    else:
        all_log_probs = np.array([np.nan])

    return {
        "all_log_probs_mean": all_log_probs.mean(axis=0),
        "all_log_probs_std": all_log_probs.std(axis=0),
        "per_intervention_log_probs_mean": per_intervention_log_probs_mean,
        "per_intervention_log_probs_std": per_intervention_log_probs_std,
    }


def get_ate_rms(
    model: IModelForInterventions,
    test_samples: np.ndarray,
    intervention_datasets: List[IntervetionData],
    variables: Variables,
    most_likely_graph: bool = False,
):
    """
    Computes the rmse between the ground truth ate and the ate predicted by our model across all available interventions 
        for both normalised and unnormalise data.
    Args:
        model: IModelForInterventions from which we can sample points while applying interventions 
        test_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        intervention_datasets: List[IntervetionData] containing intervetions and samples from the ground truth data generating process when the intervention is applied
        variables: Instance of Variables containing metada used for normalisation
        most_likely_graph: whether to use the most likely causal graph (True) or to sample graphs (False)
    """

    error_vec = []
    norm_error_vec = []

    for intervention_data in intervention_datasets:
        if intervention_data.reference_data is not None:
            reference_data = intervention_data.reference_data
        else:
            reference_data = test_samples
        ate = get_ate_from_samples(intervention_data.test_data, reference_data, variables, normalise=False)
        norm_ate = get_ate_from_samples(intervention_data.test_data, reference_data, variables, normalise=True)

        if intervention_data.effect_idxs is not None:
            ate = ate[intervention_data.effect_idxs]
            norm_ate = norm_ate[intervention_data.effect_idxs]

        model_ate, model_norm_ate = model.cate(
            intervention_idxs=intervention_data.intervention_idxs,
            intervention_values=intervention_data.intervention_values,
            reference_values=intervention_data.intervention_reference,
            effect_idxs=intervention_data.effect_idxs,
            most_likely_graph=most_likely_graph,
        )

        error_vec.append(np.abs(model_ate - ate))
        norm_error_vec.append(np.abs(model_norm_ate - norm_ate))

    # error is computed per intervention
    error_vec = np.stack(error_vec, axis=0)  # (N_interventions, N_inputs)
    norm_error_vec = np.stack(norm_error_vec, axis=0)  # (N_interventions, N_inputs)

    # rmse computed over interventions
    rmse_across_interventions = (error_vec ** 2).mean(axis=0) ** 0.5  # (N_inputs)
    norm_rmse_across_interventions = (norm_error_vec ** 2).mean(axis=0) ** 0.5  # (N_inputs)

    # rmse computed over dimensions
    rmse_across_dimensions = (error_vec ** 2).mean(axis=1) ** 0.5  # (N_interventions)
    norm_rmse_across_dimensions = (norm_error_vec ** 2).mean(axis=1) ** 0.5  # (N_interventions)

    # ALL represents average over columns
    all_rmse = (error_vec ** 2).mean(axis=(0, 1)) ** 0.5  # (1)
    all_norm_rmse = (norm_error_vec ** 2).mean(axis=(0, 1)) ** 0.5  # (1)

    return {
        "error_vec": error_vec,
        "norm_error_vec": norm_error_vec,
        "rmse_across_interventions": rmse_across_interventions,
        "norm_rmse_across_interventions": norm_rmse_across_interventions,
        "rmse_across_dimensions": rmse_across_dimensions,
        "norm_rmse_across_dimensions": norm_rmse_across_dimensions,
        "all_rmse": all_rmse,
        "all_norm_rmse": all_norm_rmse,
    }


def get_ate_from_samples(
    intervened_samples: np.ndarray, baseline_samples: np.ndarray, variables: Variables, normalise: bool = False
):
    """
    Computes ATE E[y | do(x)=a] - E[y] from samples of y from p(y | do(x)=a) and p(y)

    Args:
        intervened_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the intervened distribution p(y | do(x)=a)
        intervened_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        variables: Instance of Variables containing metada used for normalisation
        normalise: boolean indicating whether to normalise samples by their maximum and minimum values 
    """
    if normalise:
        # Normalise values between 0 and 1.
        # TODO 18375: can we avoid the repeated (un)normalization of data before/during this function or at least
        # share the normalization logic in both places?
        lowers = np.zeros(variables.num_processed_cols)
        uppers = np.ones(variables.num_processed_cols)
        for region, variable in zip(variables.processed_cols, variables):
            if variable.type == "continuous":
                lowers[region] = variable.lower
                uppers[region] = variable.upper
        intervened_samples = (intervened_samples.copy() - lowers) / (uppers - lowers)
        baseline_samples = (baseline_samples.copy() - lowers) / (uppers - lowers)

    intervened_mean = intervened_samples.mean(axis=0)
    baseline_mean = baseline_samples.mean(axis=0)

    return intervened_mean - baseline_mean


def generate_and_log_intervention_result_dict(
    metrics_logger=None,
    rmse_dict=None,
    rmse_most_likely_dict=None,
    log_prob_dict=None,
    log_prob_most_likely_dict=None,
    test_log_prob_dict=None,
) -> Dict:
    """
    Method that aggregates all causal inference results into a single dictionary. Any subset of causal inference results can be passed.

    Args:
        rmse_dict: output of get_ate_rms when run in graph marginalisation mode
        rmse_most_likely_dict: output of get_ate_rms when run with most likely graph,
        rmse_GT_dict: output of get_ate_rms when run with true graph,
        log_prob_dict: output of get_treatment_data_logprob when run with graph marginalisation,
        log_prob_most_likely_dict: output of get_treatment_data_logprob when run with most likely graph,
        log_prob_GT_dict: output of get_treatment_data_logprob when run with true graph,,
        test_log_prob_dict: output of get_treatment_data_logprob when run on samples from the unintervened test distribution,

    Returns: 
        Dictionary containing all results 

    """

    metric_dict = {}

    if log_prob_dict is not None:

        intervention_log_prob_mean = log_prob_dict["all_log_probs_mean"]
        intervention_log_prob_std = log_prob_dict["all_log_probs_std"]
        per_intervention_log_prob_mean = log_prob_dict["per_intervention_log_probs_mean"]
        per_intervention_log_prob_std = log_prob_dict["per_intervention_log_probs_std"]

        if "all interventions" not in metric_dict.keys():
            metric_dict["all interventions"] = {}

        metric_dict["all interventions"].update(
            {"log prob mean": intervention_log_prob_mean, "log prob std": intervention_log_prob_std,}
        )
        for n_int in range(len(per_intervention_log_prob_mean)):
            if f"Intervention {n_int}" not in metric_dict.keys():
                metric_dict[f"Intervention {n_int}"] = {}
            metric_dict[f"Intervention {n_int}"].update(
                {
                    "log prob mean": per_intervention_log_prob_mean[n_int],
                    "log prob std": per_intervention_log_prob_std[n_int],
                }
            )

        if metrics_logger is not None:
            metrics_logger.log_value("interventions.all.log_prob_mean", intervention_log_prob_mean, False)
            metrics_logger.log_value("interventions.all.log_prob_std", intervention_log_prob_std, False)

    if log_prob_most_likely_dict is not None:

        intervention_log_prob_mean_most_likely_graph = log_prob_most_likely_dict["all_log_probs_mean"]
        intervention_log_prob_std_most_likely_graph = log_prob_most_likely_dict["all_log_probs_std"]
        per_intervention_log_prob_mean_most_likely_graph = log_prob_most_likely_dict["per_intervention_log_probs_mean"]
        per_intervention_log_prob_std_most_likely_graph = log_prob_most_likely_dict["per_intervention_log_probs_std"]

        if "all interventions" not in metric_dict.keys():
            metric_dict["all interventions"] = {}

        metric_dict["all interventions"].update(
            {
                "log prob mean ML Graph": intervention_log_prob_mean_most_likely_graph,
                "log prob std ML Graph": intervention_log_prob_std_most_likely_graph,
            }
        )

        for n_int in range(len(per_intervention_log_prob_mean_most_likely_graph)):
            if f"Intervention {n_int}" not in metric_dict.keys():
                metric_dict[f"Intervention {n_int}"] = {}
            metric_dict[f"Intervention {n_int}"].update(
                {
                    "log prob mean ML Graph": per_intervention_log_prob_mean_most_likely_graph[n_int],
                    "log prob std ML Graph": per_intervention_log_prob_std_most_likely_graph[n_int],
                }
            )

        if metrics_logger is not None:
            metrics_logger.log_value(
                "interventions.all.ML.log_prob_mean", intervention_log_prob_mean_most_likely_graph, False
            )
            metrics_logger.log_value(
                "interventions.all.ML.log_prob_std", intervention_log_prob_std_most_likely_graph, False
            )

    if test_log_prob_dict is not None:
        test_log_prob_mean = test_log_prob_dict["all_log_probs_mean"]
        test_log_prob_std = test_log_prob_dict["all_log_probs_std"]

        metric_dict.update(
            {"test log prob mean": test_log_prob_mean, "test log prob std": test_log_prob_std,}
        )

        if metrics_logger is not None:
            metrics_logger.log_value("interventions.test.log_prob_mean", test_log_prob_mean, False)
            metrics_logger.log_value("interventions.test.log_prob_std", test_log_prob_std, False)

    if rmse_dict is not None:

        error_vec = rmse_dict["error_vec"]
        norm_error_vec = rmse_dict["norm_error_vec"]
        rmse_across_dimensions = rmse_dict["rmse_across_dimensions"]
        norm_rmse_across_dimensions = rmse_dict["norm_rmse_across_dimensions"]
        all_rmse = rmse_dict["all_rmse"]
        all_norm_rmse = rmse_dict["all_norm_rmse"]

        if "all interventions" not in metric_dict.keys():
            metric_dict["all interventions"] = {}

        metric_dict["all interventions"].update(
            {"Normalised ATE RMSE": all_norm_rmse, "ATE RMSE": all_rmse,}
        )

        for n_int in range(len(rmse_across_dimensions)):
            if f"Intervention {n_int}" not in metric_dict.keys():
                metric_dict[f"Intervention {n_int}"] = {}

            if "all columns" not in metric_dict[f"Intervention {n_int}"]:
                metric_dict[f"Intervention {n_int}"]["all columns"] = {}

            metric_dict[f"Intervention {n_int}"]["all columns"].update(
                {"Normalised ATE RMSE": norm_rmse_across_dimensions[n_int], "ATE RMSE": rmse_across_dimensions[n_int],}
            )

            # loop over data dimensions
            for dim in range(error_vec.shape[1]):
                if f"Column {dim}" not in metric_dict[f"Intervention {n_int}"].keys():
                    metric_dict[f"Intervention {n_int}"][f"Column {dim}"] = {}

                metric_dict[f"Intervention {n_int}"][f"Column {dim}"].update(
                    {"Normalised ATE RMSE": norm_error_vec[n_int, dim], "ATE RMSE": error_vec[n_int, dim],}
                )

        if metrics_logger is not None:
            metrics_logger.log_value("interventions.all.ate_rmse", all_rmse, False)

        if rmse_most_likely_dict is not None:

            error_vec_most_likely_graph = rmse_dict["error_vec"]
            norm_error_vec_most_likely_graph = rmse_dict["norm_error_vec"]
            rmse_across_dimensions_most_likely_graph = rmse_dict["rmse_across_dimensions"]
            norm_rmse_across_dimensions_most_likely_graph = rmse_dict["norm_rmse_across_dimensions"]
            all_rmse_most_likely_graph = rmse_dict["all_rmse"]
            all_norm_rmse_most_likely_graph = rmse_dict["all_norm_rmse"]

            if "all interventions" not in metric_dict.keys():
                metric_dict["all interventions"] = {}

            metric_dict["all interventions"].update(
                {
                    "Normalised ATE RMSE ML Graph": all_norm_rmse_most_likely_graph,
                    "ATE RMSE ML Graph": all_rmse_most_likely_graph,
                }
            )

            for n_int in range(len(rmse_across_dimensions_most_likely_graph)):
                if f"Intervention {n_int}" not in metric_dict.keys():
                    metric_dict[f"Intervention {n_int}"] = {}

                if "all columns" not in metric_dict[f"Intervention {n_int}"]:
                    metric_dict[f"Intervention {n_int}"]["all columns"] = {}

                metric_dict[f"Intervention {n_int}"]["all columns"].update(
                    {
                        "Normalised ATE RMSE ML Graph": norm_rmse_across_dimensions_most_likely_graph[n_int],
                        "ATE RMSE ML Graph": rmse_across_dimensions_most_likely_graph[n_int],
                    }
                )

                # loop over data dimensions
                for dim in range(error_vec_most_likely_graph.shape[1]):
                    if f"Column {dim}" not in metric_dict[f"Intervention {n_int}"].keys():
                        metric_dict[f"Intervention {n_int}"][f"Column {dim}"] = {}

                    metric_dict[f"Intervention {n_int}"][f"Column {dim}"].update(
                        {
                            "Normalised ATE RMSE ML Graph": norm_error_vec_most_likely_graph[n_int, dim],
                            "ATE RMSE ML Graph": error_vec_most_likely_graph[n_int, dim],
                        }
                    )

            if metrics_logger is not None:
                metrics_logger.log_value("interventions.all.ML.ate_rmse", all_rmse_most_likely_graph, False)

    return metric_dict


def int2binlist(i: int, n_bits: int):
    """
    Convert integer to list of ints with values in {0, 1}
    """
    assert i < 2 ** n_bits
    str_list = list(np.binary_repr(i, n_bits))
    return [int(i) for i in str_list]


def cpdag2dags(cp_mat: np.ndarray, samples: Optional[int] = None):
    """
    Compute all possible DAGs contained within a Markov equivalence class, given by a CPDAG
    Args:
        cp_mat: adjacency matrix containing both forward and backward edges for edges for which directionality is undetermined
    Returns:
        3 dimensional tensor, where the first indexes all the possible DAGs
    """

    assert len(cp_mat.shape) == 2 and cp_mat.shape[0] == cp_mat.shape[1]

    # matrix composed of just undetermined edges
    cycle_mat = (cp_mat == cp_mat.T) * cp_mat
    # return original matrix if there are no length-1 cycles
    if cycle_mat.sum() == 0:
        return cp_mat[None, :, :]

    # matrix of determined edges
    cp_no_cycles = cp_mat - cycle_mat

    # number of parent nodes for each node under the well determined matrix
    N_in_nodes = cp_no_cycles.sum(axis=0)

    # lower triangular version of cycles edges: only keep cycles in one direction.
    cycles_tril = np.tril(cycle_mat, k=-1)

    # indices of potential new edges
    undetermined_idx_mat = np.array(np.nonzero(cycles_tril)).T  # (N_undedetermined, 2)

    # number of undetermined edges
    N_undetermined = int(cycles_tril.sum())

    # choose random order for mask iteration
    max_dags = 2 ** N_undetermined
    if samples is None:
        samples = max_dags
    mask_indices = list(np.random.permutation(np.arange(max_dags)))

    # iterate over list of all potential combinations of new edges. 0 represents keeping edge from upper triangular and 1 from lower triangular
    dag_list = []
    while mask_indices and len(dag_list) < samples:

        mask_index = mask_indices.pop()
        mask = np.array(int2binlist(mask_index, N_undetermined))

        # extract list of indices which our new edges are pointing into
        incoming_edges = np.take_along_axis(undetermined_idx_mat, mask[:, None], axis=1).squeeze()

        # check if multiple edges are pointing at same node
        _, unique_counts = np.unique(incoming_edges, return_index=False, return_inverse=False, return_counts=True)

        # check if new colider has been created by checkig if multiple edges point at same node or if new edge points at existing child node
        new_colider = np.any(unique_counts > 1) or np.any(N_in_nodes[incoming_edges] > 0)

        if not new_colider:

            # get indices of new edges by sampling from lower triangular mat and upper triangular according to indices
            edge_selection = undetermined_idx_mat.copy()
            edge_selection[mask == 0, :] = np.fliplr(edge_selection[mask == 0, :])

            # add new edges to matrix and add to dag list
            new_dag = cp_no_cycles.copy()
            new_dag[(edge_selection[:, 0], edge_selection[:, 1])] = 1

            dag_list.append(new_dag)

    return np.stack(dag_list, axis=0)
