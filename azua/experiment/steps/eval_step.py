"""
Run evaluation on a trained PVAE.

To run: python run_eval.py boston -ic parameters/impute_config.json -md runs/run_name/models/model_id
"""

import datetime as dt
import os
import warnings
from logging import Logger
from typing import Any, Dict, cast, Optional, Union

import numpy as np
from scipy.sparse import issparse, csr_matrix

from ..imetrics_logger import IMetricsLogger
from ...datasets.dataset import Dataset, SparseDataset, CausalDataset, InterventionData, TemporalDataset
from ...models.imodel import IModelForImputation, IModelForObjective, IModelForCausalInference, IModelForInterventions
from ...models.pvae_base_model import PVAEBaseModel
from ...utils.causality_utils import get_ate_rms, get_treatment_data_logprob, generate_and_log_intervention_result_dict
from ...utils.check_information_gain import test_information_gain
from ...utils.imputation import (
    run_imputation_with_stats,
    eval_imputation,
    impute_targets_only,
    plot_pairwise_comparison,
)
from ...utils.metrics import (
    compute_save_additional_metrics,
    compute_target_metrics,
    save_train_val_test_metrics,
)
from ...utils.nri_utils import (
    edge_prediction_metrics,
    edge_prediction_metrics_multisample,
    convert_temporal_adj_matrix_to_static,
)
from ...utils.plot_functions import violin_plot_imputations
from ...utils.torch_utils import set_random_seeds


def run_eval_main(
    logger: Logger,
    model: IModelForImputation,
    dataset: Union[Dataset, SparseDataset],
    vamp_prior_data,
    impute_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    extra_eval: bool,
    seed: int,
    split_type: str = "rows",
    user_id: int = 0,
    metrics_logger: Optional[IMetricsLogger] = None,
    impute_train_data: bool = True,
):
    """
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        model (IModel): Model to use.
        dataset: Dataset or SparseDataset object.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.
        impute_config (dictionary): Dictionary containing options for inference.
        objective_config (dictionary): Dictionary containing objective configuration parameters.
        extra_eval (bool): If True, run evaluation that takes longer (pairwise comparison and information gain check.)
        split_type (str): Whether test data is split by rows ('rows') or elements ('elements'). If 'elements', the test
            set values will be predicted conditioned on the training set values.
        seed (int): Random seed to use when running eval.
        user_id (int): the index of the datapoint to plot the violin plot (uncertainty output).
        impute_train_data (bool): Whether imputation should be run on training data. This can require much more memory.
    """
    start_time = dt.datetime.utcnow()

    train_data, train_mask = dataset.train_data_and_mask
    val_data, val_mask = dataset.val_data_and_mask
    test_data, test_mask = dataset.test_data_and_mask

    # Assert that the test data has at least one row
    assert test_data is not None and test_mask is not None and dataset.variables is not None
    variables = dataset.variables
    assert test_data.shape == test_mask.shape
    user_count, _ = test_data.shape
    assert user_count > 0, "Empty test data array provided for evaluation"
    assert user_id < user_count, "Violin plot data point index out of bounds"
    assert split_type in ["rows", "elements"]

    # Fix evaluation seed
    set_random_seeds(seed)

    # Missing value imputation
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, split_type, vamp_prior_data, impute_config, impute_train_data, seed)

    # Marginal log likelihood
    if extra_eval and issubclass(type(model), PVAEBaseModel):
        model = cast(PVAEBaseModel, model)

        if impute_train_data:
            train_imputation_mll = model.get_marginal_log_likelihood(
                impute_config=impute_config,
                data=train_data,
                observed_mask=train_obs_mask,
                target_mask=train_target_mask,
                evaluate_imputation=True,
            )
            train_metrics["Imputation MLL"] = train_imputation_mll

        if val_data is not None:
            val_imputation_mll = model.get_marginal_log_likelihood(
                impute_config=impute_config,
                data=val_data,
                observed_mask=val_obs_mask,
                target_mask=val_target_mask,
                evaluate_imputation=True,
            )
            val_metrics["Imputation MLL"] = val_imputation_mll

        test_imputation_mll = model.get_marginal_log_likelihood(
            impute_config=impute_config,
            data=test_data,
            observed_mask=test_obs_mask,
            target_mask=test_target_mask,
            evaluate_imputation=True,
        )
        test_metrics["Imputation MLL"] = test_imputation_mll

    save_confusion = len(variables.continuous_idxs) == 0
    save_train_val_test_metrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        save_file=os.path.join(model.save_dir, "results.json"),
        save_confusion=save_confusion,
    )

    # Target value imputation
    if split_type != "elements":
        # Assume no target features if data split is elementwise
        if impute_train_data:
            train_target_imputations = impute_targets_only(
                model, train_data, train_mask, impute_config, vamp_prior_data
            )
            train_target_metrics = compute_target_metrics(train_target_imputations, train_data, variables)
        else:
            train_target_metrics = {}

        if val_data is None:
            val_target_metrics = {}
        else:
            val_target_imputations = impute_targets_only(model, val_data, val_mask, impute_config, vamp_prior_data)
            val_target_metrics = compute_target_metrics(val_target_imputations, val_data, variables)

        test_target_imputations = impute_targets_only(model, test_data, test_mask, impute_config, vamp_prior_data)
        test_target_metrics = compute_target_metrics(test_target_imputations, test_data, variables)

        save_train_val_test_metrics(
            train_metrics=train_target_metrics,
            val_metrics=val_target_metrics,
            test_metrics=test_target_metrics,
            save_file=os.path.join(model.save_dir, "target_results.json"),
            save_confusion=False,
        )
    else:
        train_target_metrics = {}
        val_target_metrics = {}
        test_target_metrics = {}

    if metrics_logger is not None:
        # Log metrics to AzureML
        metrics_logger.log_dict({"train_data.all": train_metrics.get("all", {})})
        metrics_logger.log_dict({"val_data.all": val_metrics.get("all", {})})
        metrics_logger.log_dict({"test_data.all": test_metrics["all"]})

        metrics_logger.log_dict({"train_data.Imputation MLL": train_metrics.get("Imputation MLL", {})})
        metrics_logger.log_dict({"val_data.Imputation MLL": val_metrics.get("Imputation MLL", {})})
        metrics_logger.log_dict({"test_data.Imputation MLL": test_metrics.get("Imputation MLL", {})})

        # Label in AzureML with 'target' otherwise for example, MEDV.RMSE can mean two different things - they are imputations with
        # different masks. We no longer report the non-target metrics per variable, but we used to.
        metrics_logger.log_dict({"train_data.target": train_target_metrics})
        metrics_logger.log_dict({"val_data.target": val_target_metrics})
        metrics_logger.log_dict({"test_data.target": test_target_metrics})

    # Violin plot
    # Plot the violin plot output for one datapoint to check imputation results
    # Only plot for dense data, assume too many features if sparse
    if isinstance(test_obs_mask, np.ndarray):
        imputed_values, imputed_stats = run_imputation_with_stats(
            model, test_data, test_obs_mask, variables, impute_config, vamp_prior_data
        )
        if impute_config["sample_count"] < 10:
            warnings.warn("Imputation sample count is < 10, violin plot may be of low quality.")
        violin_plot_imputations(
            imputed_values,
            imputed_stats,
            test_target_mask,
            variables,
            user_id,
            title="Violin plot for continuous variables",
            save_path=model.save_dir,
            plot_name="imputed_values_violin_plot_user%d" % user_id,
            normalized=True,
        )

    # Extra evaluation
    if extra_eval:
        # Create pair plots
        # We assume all test_data is observed (i.e. no special treatment of input test mask)

        # For ground truth data
        if issparse(test_data):
            # Declare types to fix mypy error
            test_data_: csr_matrix = test_data
            test_mask_: csr_matrix = test_mask
            test_data_dense = test_data_.toarray()
            test_mask_dense = test_mask_.toarray()
        else:
            test_data_dense = test_data
            test_mask_dense = test_mask
        plot_pairwise_comparison(
            test_data_dense, variables, filename_suffix="ground_truth_data", save_dir=model.save_dir
        )

        # Data generation
        # The impute() call is identical to one in AL beside averaging MC samples (which doesn't happen in AL)
        empty_mask = np.zeros_like(test_data_dense, dtype=bool)
        generated_values = model.impute(test_data_dense, empty_mask, impute_config, vamp_prior_data=vamp_prior_data)
        plot_pairwise_comparison(
            generated_values, model.variables, filename_suffix="data_generation", save_dir=model.save_dir
        )

        # Data reconstruction
        reconstructed_values = model.impute(test_data, test_mask, impute_config, vamp_prior_data=vamp_prior_data)
        plot_pairwise_comparison(
            reconstructed_values, model.variables, filename_suffix="data_reconstruction", save_dir=model.save_dir
        )

        # Check information gain is behaving as expected.
        # We enforce model to be of IModelForObjective type here
        # TODO: rewrite, so we don't need to use casting
        assert isinstance(model, IModelForObjective)
        model_for_objective = cast(IModelForObjective, model)
        test_information_gain(
            logger, model_for_objective, test_data_dense, test_mask_dense, vamp_prior_data, impute_config, seed=seed,
        )

    # Additional metrics
    if (
        len(variables.target_var_idxs) == 0
        and not issparse(train_data)
        and model.name() not in ["vicause", "deci", "deci_gaussian", "deci_spline"]
        and model.name() not in ["mean_imputing", "mice", "missforest", "zero_imputing"]
    ):
        # If there is no target then we compute additional metrics that are question quality and difficulty.

        # We enforce model to be of IModelForObjective type here
        # TODO: rewrite, so we don't need to use casting
        # TODO: sparse implementation of additional metrics

        assert isinstance(model, IModelForObjective)
        model_for_objective = cast(IModelForObjective, model)
        compute_save_additional_metrics(
            test_imputations, train_data, model_for_objective, model.save_dir, objective_config=objective_config,
        )
    elif model.name() == "vicause":
        warnings.warn("Question quality and difficulty are not currently implemented for vicause")
    elif model.name() in ["deci", "deci_gaussian", "deci_spline"]:
        warnings.warn("Question quality and difficulty are not currently implemented for deci model")
    elif model.name() in ["mean_imputing", "mice", "missforest", "zero_imputing"]:
        warnings.warn("Question quality and difficulty are not currently implemented for imputation baselines")
    if metrics_logger:
        metrics_logger.log_value("impute/running-time", (dt.datetime.utcnow() - start_time).total_seconds() / 60)


def eval_causal_discovery(
    logger: Logger,
    dataset: CausalDataset,
    model: IModelForCausalInference,
    metrics_logger: Optional[IMetricsLogger] = None,
):
    """
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        dataset: Dataset or SparseDataset object.
        model (IModelForCausalInference): Model to use.

    This requires the model to have a method get_adjacency_data_matrix() implemented, which returns the adjacency
    matrix learnt from the data.
    """
    adj_ground_truth = dataset.get_adjacency_data_matrix()

    # For DECI, the default is to give 100 samples of the graph posterior
    adj_pred = model.get_adj_matrix().astype(float).round()

    # Convert temporal adjacency matrices to static adjacency matrices, currently does not support partially observed ground truth (i.e. subgraph_idx=None).
    if type(dataset) == TemporalDataset:
        adj_ground_truth, adj_pred = convert_temporal_adj_matrix_to_static(adj_ground_truth, adj_pred)
        subgraph_idx = None
    else:
        subgraph_idx = dataset.get_known_subgraph_mask_matrix()

    # save adjacency matrix
    np.save(os.path.join(model.save_dir, "adj_matrices"), adj_pred, allow_pickle=True, fix_imports=True)

    if len(adj_pred.shape) == 2:
        # If predicts single adjacency matrix
        results = edge_prediction_metrics(adj_ground_truth, adj_pred, adj_matrix_mask=subgraph_idx)
    elif len(adj_pred.shape) == 3:
        # If predicts multiple adjacency matrices (stacked)
        results = edge_prediction_metrics_multisample(adj_ground_truth, adj_pred, adj_matrix_mask=subgraph_idx)
    if metrics_logger is not None:
        # Log metrics to AzureML
        metrics_logger.log_value("adjacency.recall", results["adjacency_recall"])
        metrics_logger.log_value("adjacency.precision", results["adjacency_precision"])
        metrics_logger.log_value("adjacency.f1", results["adjacency_fscore"], True)
        metrics_logger.log_value("orientation.recall", results["orientation_recall"])
        metrics_logger.log_value("orientation.precision", results["orientation_precision"])
        metrics_logger.log_value("orientation.f1", results["orientation_fscore"], True)
        metrics_logger.log_value("causal_accuracy", results["causal_accuracy"])
        metrics_logger.log_value("causalshd", results["shd"])
        metrics_logger.log_value("causalnnz", results["nnz"])
    # Save causality results to a file
    save_train_val_test_metrics(
        train_metrics={},
        val_metrics={},
        test_metrics=results,
        save_file=os.path.join(model.save_dir, "target_results_causality.json"),
        save_confusion=False,
    )


def eval_treatment_effects(
    logger: Logger,
    dataset: CausalDataset,
    model: IModelForInterventions,
    metrics_logger: Optional[IMetricsLogger] = None,
    eval_likelihood: bool = True,
    process_dataset: bool = True,
) -> None:
    """
    Run treatment effect experiments: ATE RMSE and interventional distribution log-likelihood with graph marginalisation and most likely (ml) graph.
        Save results as json file and in metrics logger.
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        dataset: Dataset or SparseDataset object.
        model (IModelForInterventions): Model to use.
        metrics_logger: Optional[IMetricsLogger]
        name_prepend: Optional string that will be prepended to the json save name and logged metrics.
             This allows us to distinguish results from end2end models from results computed with downstream models (models that require graph as input, like DoWhy)
        eval_likelihood: Optional bool flag that will disable the log likelihood evaluation
        process_data: Whether to apply the data processor to the interventional data. This is done for Azua-internal models such as DECI, and not for DoWhy.

    This requires the model to implement methods sample() to sample from the model distribution with interventions
     and log_prob() to evaluate the density of test samples under interventions
    """
    # Process test and intervention data in the same way that train data is processed
    if process_dataset:
        processed_dataset = model.data_processor.process_dataset(dataset)
    else:
        processed_dataset = dataset
    test_data, _ = processed_dataset.test_data_and_mask

    # TODO: when scaling continuous variables in `process_dataset` we should call `revert_data` to evaluate RMSE in original space
    rmse_dict = get_ate_rms(
        model,
        test_data.astype(float),
        processed_dataset.get_intervention_data(),
        processed_dataset.variables,
        processed=process_dataset,
    )
    rmse_most_likely_dict = get_ate_rms(
        model,
        test_data.astype(float),
        processed_dataset.get_intervention_data(),
        processed_dataset.variables,
        most_likely_graph=True,
        processed=process_dataset,
    )

    if eval_likelihood:
        log_prob_dict = get_treatment_data_logprob(model, processed_dataset.get_intervention_data())
        log_prob_most_likely_dict = get_treatment_data_logprob(
            model, processed_dataset.get_intervention_data(), most_likely_graph=True
        )

        # Evaluate test log-prob only for models that support it
        if "do" not in model.name():
            base_testset_intervention = [
                InterventionData(
                    intervention_idxs=np.array([]), intervention_values=np.array([]), test_data=test_data.astype(float)
                )
            ]
            test_log_prob_dict = get_treatment_data_logprob(model, base_testset_intervention)
        else:
            test_log_prob_dict = None
    else:
        logger.info("Disable the log likelihood evaluation for this causal model")
        log_prob_dict = None
        log_prob_most_likely_dict = None
        test_log_prob_dict = None

    # Generate result dict
    metric_dict = generate_and_log_intervention_result_dict(
        metrics_logger=metrics_logger,
        rmse_dict=rmse_dict,
        rmse_most_likely_dict=rmse_most_likely_dict,
        log_prob_dict=log_prob_dict,
        log_prob_most_likely_dict=log_prob_most_likely_dict,
        test_log_prob_dict=test_log_prob_dict,
    )

    # prepend modifier to save names
    base_name = "TE"
    savefile = "results_interventions"
    savefile += ".json"

    if metrics_logger is not None:
        # Log metrics to AzureML

        if "all interventions" in metric_dict.keys() and "ATE RMSE" in metric_dict["all interventions"].keys():
            metrics_logger.log_value(base_name + "_rmse", metric_dict["all interventions"]["ATE RMSE"], True)

        if "all interventions" in metric_dict.keys() and "log prob mean" in metric_dict["all interventions"].keys():
            metrics_logger.log_value(
                base_name + "_LL", metric_dict["all interventions"]["log prob mean"], True,
            )
        if "test log prob mean" in metric_dict.keys():
            metrics_logger.log_value("test_LL", metric_dict["test log prob mean"], True)

    # Save intervention results to a file
    save_train_val_test_metrics(
        train_metrics={},
        val_metrics={},
        test_metrics=metric_dict,
        save_file=os.path.join(model.save_dir, savefile),
        save_confusion=False,
    )
