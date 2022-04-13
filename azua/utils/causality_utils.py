from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch

from ..datasets.intervention_data import InterventionData
from ..datasets.variables import Variables
from ..models.imodel import IModelForCausalInference, IModelForInterventions
from ..utils.data_mask_utils import to_tensors
from ..utils.torch_utils import LinearModel, MultiROFFeaturiser
from ..utils.evaluation_dataclasses import IteEvaluationResults


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


def intervention_to_tensor(intervention_idxs, intervention_values, group_mask, device):
    """
    Maps empty interventions to nan and np.ndarray intervention data to torch tensors.
    Converts indices to a mask using the group_mask.
    """
    intervention_mask = None

    if intervention_idxs is not None and intervention_values is not None:
        (intervention_idxs,) = to_tensors(intervention_idxs, device=device, dtype=torch.long)
        (intervention_values,) = to_tensors(intervention_values, device=device, dtype=torch.float)

        if intervention_idxs.dim() == 0:
            intervention_idxs = None

        if intervention_values.dim() == 0:
            intervention_values = None

        intervention_mask = get_mask_from_idxs(intervention_idxs, group_mask, device)

    return intervention_idxs, intervention_mask, intervention_values


def get_mask_from_idxs(idxs, group_mask, device):
    """
    Generate mask for observations or samples from indices using group_mask
    """
    mask = torch.zeros(group_mask.shape[0], device=device, dtype=torch.bool)
    mask[idxs] = 1
    (group_mask,) = to_tensors(group_mask, device=device, dtype=torch.bool)
    mask = (mask.unsqueeze(1) * group_mask).sum(0).bool()
    return mask


def get_treatment_data_logprob(
    model: IModelForCausalInference, intervention_datasets: List[InterventionData], most_likely_graph: bool = False,
):
    """
    Computes the log-probability of test-points sampled from intervened distributions.
    Args:
        model: IModelForInterventions with which we can evaluate the log-probability of points while applying interventions to the generative model
        intervention_datasets: List[InterventionData] containing intervetions and samples from the ground truth data generating process when the intervention is applied
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
        all_log_probs_arr = np.concatenate(all_log_probs, axis=0)
    else:
        all_log_probs_arr = np.array([np.nan])

    return {
        "all_log_probs_mean": all_log_probs_arr.mean(axis=0),
        "all_log_probs_std": all_log_probs_arr.std(axis=0),
        "per_intervention_log_probs_mean": per_intervention_log_probs_mean,
        "per_intervention_log_probs_std": per_intervention_log_probs_std,
    }


def get_ate_rms(
    model: IModelForInterventions,
    test_samples: np.ndarray,
    intervention_datasets: List[InterventionData],
    variables: Variables,
    most_likely_graph: bool = False,
    processed: bool = True,
):
    """
    Computes the rmse between the ground truth ate and the ate predicted by our model across all available interventions 
        for both normalised and unnormalise data.
    Args:
        model: IModelForInterventions from which we can sample points while applying interventions 
        test_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        intervention_datasets: List[InterventionData] containing intervetions and samples from the ground truth data generating process when the intervention is applied
        variables: Instance of Variables containing metadata used for normalisation
        most_likely_graph: whether to use the most likely causal graph (True) or to sample graphs (False)
        processed: whether the data has been processed
    """

    error_vec = []
    norm_error_vec = []

    for intervention_data in intervention_datasets:

        if intervention_data.reference_data is not None:
            reference_data = intervention_data.reference_data
        else:
            reference_data = test_samples

        # conditions are applied to the test data when it is generated. As a result computing ATE on this data returns the CATE.
        ate = get_ate_from_samples(
            intervention_data.test_data, reference_data, variables, normalise=False, processed=processed
        )
        norm_ate = get_ate_from_samples(
            intervention_data.test_data, reference_data, variables, normalise=True, processed=processed
        )

        if intervention_data.effect_idxs is not None:
            if processed:
                effect_mask = get_mask_from_idxs(intervention_data.effect_idxs, variables.group_mask, "cpu").numpy()
                ate = ate[effect_mask]
                norm_ate = norm_ate[effect_mask]
            else:
                ate = ate[intervention_data.effect_idxs]
                norm_ate = norm_ate[intervention_data.effect_idxs]

        # Check for conditioning
        if intervention_data.conditioning_idxs is not None:
            if most_likely_graph:
                Ngraphs = 1
                Nsamples_per_graph = 50000
            else:
                Ngraphs = 10
                Nsamples_per_graph = 5000
        else:
            if most_likely_graph:
                Ngraphs = 1
                Nsamples_per_graph = 20000
            else:
                Ngraphs = 10000
                Nsamples_per_graph = 2

        model_ate, model_norm_ate = model.cate(
            intervention_idxs=intervention_data.intervention_idxs,
            intervention_values=intervention_data.intervention_values,
            reference_values=intervention_data.intervention_reference,
            effect_idxs=intervention_data.effect_idxs,
            conditioning_idxs=intervention_data.conditioning_idxs,
            conditioning_values=intervention_data.conditioning_values,
            most_likely_graph=most_likely_graph,
            Nsamples_per_graph=Nsamples_per_graph,
            Ngraphs=Ngraphs,
        )

        error_vec.append(np.abs(model_ate - ate))
        norm_error_vec.append(np.abs(model_norm_ate - norm_ate))

    # error is computed per intervention
    error_vec_arr = np.stack(error_vec, axis=0)  # (N_interventions, N_inputs)
    norm_error_vec_arr = np.stack(norm_error_vec, axis=0)  # (N_interventions, N_inputs)

    # rmse computed over interventions
    rmse_across_interventions = np.square(error_vec_arr).mean(axis=0) ** 0.5  # (N_inputs)
    norm_rmse_across_interventions = np.square(norm_error_vec_arr).mean(axis=0) ** 0.5  # (N_inputs)

    # rmse computed over dimensions
    rmse_across_dimensions = np.square(error_vec_arr).mean(axis=1) ** 0.5  # (N_interventions)
    norm_rmse_across_dimensions = np.square(norm_error_vec_arr).mean(axis=1) ** 0.5  # (N_interventions)

    # ALL represents average over columns
    all_rmse = np.square(error_vec_arr).mean(axis=(0, 1)) ** 0.5  # (1)
    all_norm_rmse = np.square(norm_error_vec_arr).mean(axis=(0, 1)) ** 0.5  # (1)

    return {
        "error_vec": error_vec_arr,
        "norm_error_vec": norm_error_vec_arr,
        "rmse_across_interventions": rmse_across_interventions,
        "norm_rmse_across_interventions": norm_rmse_across_interventions,
        "rmse_across_dimensions": rmse_across_dimensions,
        "norm_rmse_across_dimensions": norm_rmse_across_dimensions,
        "all_rmse": all_rmse,
        "all_norm_rmse": all_norm_rmse,
    }


def get_ate_from_samples(
    intervened_samples: np.ndarray,
    baseline_samples: np.ndarray,
    variables: Variables,
    normalise: bool = False,
    processed: bool = True,
):
    """
    Computes ATE E[y | do(x)=a] - E[y] from samples of y from p(y | do(x)=a) and p(y)

    Args:
        intervened_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the intervened distribution p(y | do(x)=a)
        baseline_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        variables: Instance of Variables containing metada used for normalisation
        normalise: boolean indicating whether to normalise samples by their maximum and minimum values 
        processed: whether the data has been processed (which affects the column numbering)
    """
    if normalise:
        # Normalise values between 0 and 1.
        # TODO 18375: can we avoid the repeated (un)normalization of data before/during this function or at least
        # share the normalization logic in both places?
        if processed:
            n_cols = variables.num_processed_cols
            cols = variables.processed_cols
        else:
            n_cols = variables.num_unprocessed_cols
            cols = variables.unprocessed_cols
        lowers = np.zeros(n_cols)
        uppers = np.ones(n_cols)
        for region, variable in zip(cols, variables):
            if variable.type == "continuous":
                lowers[region] = variable.lower
                uppers[region] = variable.upper
                
        intervened_samples = np.subtract(intervened_samples.copy(), lowers) / np.subtract(uppers, lowers)
        baseline_samples = np.subtract(baseline_samples.copy(), lowers) / np.subtract(uppers, lowers)

    intervened_mean = intervened_samples.mean(axis=0)
    baseline_mean = baseline_samples.mean(axis=0)

    return intervened_mean - baseline_mean


def get_cate_from_samples(
    intervened_samples: torch.tensor,
    baseline_samples: torch.tensor,
    conditioning_mask: torch.tensor,
    conditioning_values: torch.tensor,
    effect_mask: torch.tensor,
    variables: Variables,
    normalise: bool = False,
    rff_lengthscale: Union[int, float, List[float], Tuple[float, ...]] = (0.1, 1),
    rff_n_features: int = 3000,
):
    """
    Estimate CATE using a functional approach: We fit a function that takes as input the conditioning variables
     and as output the outcome variables using intervened_samples as training points. We do the same while using baseline_samples
     as training data. We estimate CATE as the difference between the functions' outputs when the input is set to conditioning_values.
     As functions we use linear models on a random fourier feature basis. If intervened_samples and baseline_samples are provided for multiple graphs
     the CATE estimate is averaged across graphs. 

    Args:
        intervened_samples: tensor of shape (Ngraphs, Nsamples, Nvariables) sampled from intervened (non-conditional) distribution
        baseline_samples: tensor of shape (Ngraphs, Nsamples, Nvariables) sampled from a reference distribution. Note that this could mean a reference intervention has been applied. 
        conditioning_mask: boolean tensor which indicates which variables we want to condition on
        conditioning_values: tensor containing values of variables we want to condition on
        effect_mask: boolean tensor which indicates which outcome variables for which we want to estimate CATE
        variables: Instance of Variables containing metada used for normalisation
        normalise: boolean indicating whether to normalise samples by their maximum and minimum values 
        rff_lengthscale: either a positive float/int indicating the lengthscale of the RBF kernel or a list/tuple
         containing the lower and upper limits of a uniform distribution over the lengthscale. The latter option is prefereable when there is no prior knowledge about functional form.
        rff_n_features: Number of random features with which to approximate the RBF kernel. Larger numbers result in lower variance but are more computationally demanding.
    Returns:
        CATE_estimates: tensor of shape (len(effect_idxs)) containing our estimates of CATE for outcome variables
    """

    # TODO: we are assuming the conditioning variable is d-connected to the target but we should probably use the networkx dseparation method to check this in the future

    if normalise:
        # Normalise values between 0 and 1.
        # TODO 18375: can we avoid the repeated (un)normalization of data before/during this function or at least
        # share the normalization logic in both places?
        lowers = torch.zeros(
            variables.num_processed_cols, dtype=intervened_samples.dtype, device=intervened_samples.device
        )
        uppers = torch.ones(
            variables.num_processed_cols, dtype=intervened_samples.dtype, device=intervened_samples.device
        )
        for region, variable in zip(variables.processed_cols, variables):
            if variable.type == "continuous":
                lowers[region] = variable.lower
                uppers[region] = variable.upper
        intervened_samples = (intervened_samples.clone() - lowers) / (uppers - lowers)
        baseline_samples = (baseline_samples.clone() - lowers) / (uppers - lowers)

    assert effect_mask.sum() == 1.0, "Only 1d outcomes are supported"

    test_inputs = conditioning_values.unsqueeze(1)

    featuriser = MultiROFFeaturiser(rff_n_features=rff_n_features, lengthscale=rff_lengthscale)
    featuriser.fit(X=intervened_samples.new_ones((1, int(conditioning_mask.sum()))))

    CATE_estimates = []
    for graph_idx in range(intervened_samples.shape[0]):
        intervened_train_inputs = intervened_samples[graph_idx, :, conditioning_mask]
        reference_train_inputs = baseline_samples[graph_idx, :, conditioning_mask]

        featurised_intervened_train_inputs = featuriser.transform(intervened_train_inputs)
        featurised_reference_train_inputs = featuriser.transform(reference_train_inputs)
        featurised_test_input = featuriser.transform(test_inputs)

        intervened_train_targets = intervened_samples[graph_idx, :, effect_mask].reshape(intervened_samples.shape[1])
        reference_train_targets = baseline_samples[graph_idx, :, effect_mask].reshape(intervened_samples.shape[1])

        intervened_predictive_model = LinearModel()
        intervened_predictive_model.fit(features=featurised_intervened_train_inputs, targets=intervened_train_targets)

        reference_predictive_model = LinearModel()
        reference_predictive_model.fit(features=featurised_reference_train_inputs, targets=reference_train_targets)

        CATE_estimates.append(
            intervened_predictive_model.predict(features=featurised_test_input)[0]
            - reference_predictive_model.predict(features=featurised_test_input)[0]
        )

    return torch.stack(CATE_estimates, dim=0).mean(dim=0)


def calculate_ite(intervention_samples: np.ndarray, reference_samples: np.ndarray) -> np.ndarray:
    """
    Calculates individual treatment effect (ITE) between two sets of samples each
    with shape (no. of samples, no. of variables).
    
    Returns: the variable-wise ITE with shape (no. of samples, no. of variables)
    
    """
    
    assert intervention_samples.shape == reference_samples.shape,\
        "Intervention and reference samples must be the shape for ITE calculation"
    return intervention_samples - reference_samples


def calculate_rmse(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the root mean squared error (RMSE) between arrays `a` and `b`.
    
    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean
    
    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.sqrt(np.mean(np.square(np.subtract(a, b)), axis=axis))


def normalise_data(arrs: List[np.ndarray], variables: Variables, processed: bool) -> List[np.ndarray]:
    """
    Normalises all arrays in `arrs` given variable maximums (upper) and minimums (lower)
    in `variables`. Categorical data is excluded from normalization.

    Args:
        arrs (List[ndarray]): A list of ndarrays to normalise
        variables (Variables): A Variables instance containing metadata about arrays in `arrs`
        processed (bool): Whether the data in `arrs` has been processed

    Returns:
        (list(ndarray)) A list of normalised ndarrays corresponding with `arrs`.
    """

    if processed:
        n_cols = variables.num_processed_cols
        col_groups = variables.processed_cols
    else:
        n_cols = variables.num_unprocessed_cols
        col_groups = variables.unprocessed_cols

    assert all(n_cols == arr.shape[1] for arr in arrs)

    # if lower/uppers aren't updated, performs (arr - 0)/(1 - 0), i.e. doesn't normalize
    lowers = np.zeros(n_cols)
    uppers = np.ones(n_cols)

    for cols_idx, variable in zip(col_groups, variables):
        if variable.type == "continuous":
            lowers[cols_idx] = variable.lower
            uppers[cols_idx] = variable.upper

    return [np.divide(np.subtract(arr, lowers), np.subtract(uppers, lowers)) for arr in arrs]


def calculate_per_group_rmse(a: np.ndarray,
                             b: np.ndarray,
                             variables: Variables,
                             processed: bool = False,
                             normalise: bool = False) -> np.ndarray:
    """
    Calculates RMSE group-wise between two ndarrays (`a` and `b`) for all samples.
    Arrays 'a' and 'b' have expected shape (no. of rows, no. of variables).
    
    Args:
        a (ndarray): Array of shape (no. of rows, no. of variables)
        b (ndarray): Array of shape (no. of rows, no. of variables)
        variables (Variables): A Variables object indicating groups
        processed (bool): Whether arrays `a` and `b` have been processed
        normalise (bool): Whether arrays `a` and `b` should be normalised
        
    Returns:
        (ndarrray) RMSE calculated over each group for each sample in `a`/`b`
    """
    if normalise:
        a, b = normalise_data([a, b], variables, processed)

    rmse_array = np.zeros(shape=(a.shape[0], variables.num_query_groups))
    for return_array_idx, group_idxs in enumerate(variables.query_group_idxs):
        # calculate RMSE columnwise for all samples
        rmse_array[:, return_array_idx] = calculate_rmse(a[:, group_idxs], b[:, group_idxs], axis=1)
    return rmse_array


def filter_target_columns(arrs: List[np.ndarray], variables: Variables, processed: bool) -> Tuple[List[np.ndarray], Variables]:
    """
    Returns the columns associated with target variables. If `proccessed` is True, assume
    that arrs has been processed and handle expanded columns appropriately.
    
    Args:
        arrs (List[ndarray]): A list of ndarrays to be filtered
        variables (Variables): A Variables instance containing metadata
        processed (bool): Whether to treat data in `arrs` as having been processed
    
    Returns: A list of ndarrays corresponding to `arrs` with columns relating to target variables, 
        and a new Variables instance relating to target variables
    """    
    if processed: 
        # Get target idxs according to processed data
        target_idxs = []
        for i in variables.target_var_idxs:
            target_idxs.extend(variables.processed_cols[i])
    else:
        target_idxs = variables.target_var_idxs
        
    return_arrs = [a[:, target_idxs] for a in arrs]    
    target_variables = variables.subset(variables.target_var_idxs)
    return return_arrs, target_variables


def get_ite_evaluation_results(model: IModelForInterventions,
                               counterfactual_datasets: List[InterventionData],
                               variables: Variables,
                               normalise: bool,
                               processed: bool,
                               most_likely_graph: bool = False,
                               Ngraphs: int = 100) -> IteEvaluationResults:
    """
    Calculates ITE evaluation metrics.
    Args: 
        model (IModelForinterventions): Trained DECI model
        counterfactual_datasets (list[InterventionData]): a list of counterfactual datasets 
            used to calculate metrics.
        variables (Variables): Variables object indicating variable group membership
        normalise (bool): Whether the data should be normalised prior to calculating RMSE
        processed (bool): Whether the data in `counterfactual_datasets` has been processed
        most_likely_graph (bool): Flag indicating whether to use most likely graph. 
            If false, model-generated counterfactual samples are averaged over `Ngraph` graphs.
        Ngraphs (int): Number of graphs sampled when generating counterfactual samples. Unused if 
            `most_likely_graph` is true.

    Returns:
            IteEvaluationResults object containing ITE evaluation metrics.
    """
    
    per_int_av_ite_rmse = []
    for counterfactual_data in counterfactual_datasets:
        baseline_samples = counterfactual_data.conditioning_values
        reference_samples = counterfactual_data.reference_data
        intervention_samples = counterfactual_data.test_data
        assert intervention_samples is not None
        assert reference_samples is not None
        
        # get sample (ground truth) ite
        sample_ite = calculate_ite(intervention_samples=intervention_samples, 
                                   reference_samples=reference_samples)
        # get model (predicted) ite
        model_ite = model.ite(X=baseline_samples,
                              intervention_idxs=counterfactual_data.intervention_idxs,
                              intervention_values=counterfactual_data.intervention_values,
                              reference_values=counterfactual_data.intervention_reference,
                              most_likely_graph=most_likely_graph,
                              Ngraphs=Ngraphs)
        
        # If there are defined target variables, only use these for evaluation
        if len(variables.target_var_idxs) > 0:
            [sample_ite, model_ite], variables = filter_target_columns([sample_ite, model_ite], variables, processed)
            
        # calculate ite rmse per group         
        # (no. of samples, no. of input variables) -> (no. of samples, no. of groups)
        per_group_rmse = calculate_per_group_rmse(sample_ite, model_ite, variables, processed=processed, normalise=normalise)
        # average over all samples (no. of samples, no. of groups) -> (no. of groups)
        av_per_group_ite_rmse = np.mean(per_group_rmse, axis=0)
        # average over all groups (no. of groups) -> (1)
        av_ite_rmse = np.mean(av_per_group_ite_rmse)

        per_int_av_ite_rmse.append(av_ite_rmse)
    
    # average over all interventions
    av_ite_rmse = np.mean(per_int_av_ite_rmse) 
    
    return IteEvaluationResults(av_ite_rmse, np.stack(per_int_av_ite_rmse))
        
    
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

    metric_dict: Dict[str, Any] = {}

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


def dag_pen_np(X):
    assert X.shape[0] == X.shape[1]
    X = torch.from_numpy(X)
    return (torch.trace(torch.matrix_exp(X)) - X.shape[0]).item()


def int2binlist(i: int, n_bits: int):
    """
    Convert integer to list of ints with values in {0, 1}
    """
    assert i < 2 ** n_bits
    str_list = list(np.binary_repr(i, n_bits))
    return [int(i) for i in str_list]


def approximate_maximal_acyclic_subgraph(adj_matrix: np.ndarray, n_samples: int = 10):
    """
    Compute an (approximate) maximal acyclic subgraph of a directed non-dag but removing at most 1/2 of the edges
    See Vazirani, Vijay V. Approximation algorithms. Vol. 1. Berlin: springer, 2001, Page 7;
    Also Hassin, Refael, and Shlomi Rubinstein. "Approximations for the maximum acyclic subgraph problem."
    Information processing letters 51.3 (1994): 133-140.
    Args:
        adj_matrix: adjacency matrix of a directed graph (may contain cycles)
        n_samples: number of the random permutations generated. Default is 10.
    Returns:
        an adjacency matrix of the acyclic subgraph
    """
    # assign each node with a order
    adj_dag = np.zeros_like(adj_matrix)
    for n in range(n_samples):
        random_order = np.expand_dims(np.random.permutation(adj_matrix.shape[0]), 0)
        # subgraph with only forward edges defined by the assigned order
        adj_forward = ((random_order.T > random_order).astype(int)) * adj_matrix
        # subgraph with only backward edges defined by the assigned order
        adj_backward = ((random_order.T < random_order).astype(int)) * adj_matrix
        # return the subgraph with the least deleted edges
        adj_dag_n = adj_forward if adj_backward.sum() < adj_forward.sum() else adj_backward
        if adj_dag_n.sum() > adj_dag.sum():
            adj_dag = adj_dag_n
    return adj_dag


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
        if dag_pen_np(cp_mat) != 0.0:
            cp_mat = approximate_maximal_acyclic_subgraph(cp_mat)
        return cp_mat[None, :, :]

    # matrix of determined edges
    cp_determined_subgraph = cp_mat - cycle_mat

    # prune cycles if the matrix of determined edges is not a dag
    if dag_pen_np(cp_determined_subgraph.copy()) != 0.0:
        cp_determined_subgraph = approximate_maximal_acyclic_subgraph(cp_determined_subgraph, 1000)

    # number of parent nodes for each node under the well determined matrix
    N_in_nodes = cp_determined_subgraph.sum(axis=0)

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
    dag_list: list = []
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
            new_dag = cp_determined_subgraph.copy()
            new_dag[(edge_selection[:, 0], edge_selection[:, 1])] = 1

            # Check for high order cycles
            if dag_pen_np(new_dag.copy()) == 0.0:
                dag_list.append(new_dag)
    # When all combinations of new edges create cycles, we will only keep determined ones
    if len(dag_list) == 0:
        dag_list.append(cp_determined_subgraph)

    return np.stack(dag_list, axis=0)


def process_adjacency_mats(adj_mats: np.ndarray, num_nodes: int):
    """
    This processes the adjacency matrix in the format [num, variable, variable]. It will remove the duplicates and non DAG adjacency matrix.
    Args:
        adj_mats (np.ndarry): A group of adjacency matrix
        num_nodes (int): The number of variables (dimensions of the adjacency matrix)

    Returns:
        A list of adjacency matrix without duplicates and non DAG
        A np.ndarray storing the weights of each adjacency matrix.
    """

    # This method will get rid of the non DAG and duplicated ones. It also returns a proper weight for each of the adjacency matrix
    if len(adj_mats.shape) == 2:
        # Single adjacency matrix
        assert (np.trace(scipy.linalg.expm(adj_mats)) - num_nodes) == 0, "Generate non DAG graph"
        return adj_mats, np.ones(1)
    else:
        # Multiple adjacency matrix samples
        # Remove non DAG adjacency matrix
        adj_mats = np.array([adj_mat for adj_mat in adj_mats if (np.trace(scipy.linalg.expm(adj_mat)) - num_nodes) == 0])
        assert np.any(adj_mats), "Generate non DAG graph"
        # Remove duplicated adjacency and aggregate the weights
        adj_mats_unique, dup_counts = np.unique(adj_mats, axis=0, return_counts=True)
        # Normalize the weights
        adj_weights = dup_counts / np.sum(dup_counts)
        return adj_mats_unique, adj_weights
