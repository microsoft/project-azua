"""
Run evaluation on a trained PVAE.

To run: python run_eval.py boston -ic parameters/impute_config.json -md runs/run_name/models/model_id
"""

import datetime as dt
import os
import warnings
from logging import Logger
from typing import Any, Dict, Optional, Union, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ...models.pvae_base_model import PVAEBaseModel
from ...datasets.dataset import Dataset, SparseDataset
from ...models.imodel import (
    IModelForImputation,
    IModelForObjective,
)
from ...utils.check_information_gain import test_information_gain
from ...utils.imputation import (
    eval_imputation,
    impute_targets_only,
    plot_pairwise_comparison,
    run_imputation_with_stats,
)
from ...utils.metrics import (
    compute_save_additional_metrics,
    compute_target_metrics,
    save_confusion_plot,
    save_train_val_test_metrics,
)
from ...utils.plot_functions import violin_plot_imputations
from ...utils.torch_utils import set_random_seeds
from ..imetrics_logger import IMetricsLogger


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
    if extra_eval and isinstance(model, PVAEBaseModel):

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

    save_train_val_test_metrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        save_file=os.path.join(model.save_dir, "results.json"),
    )
    if len(variables.continuous_idxs) == 0:
        save_confusion_plot(test_metrics, model.save_dir)

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
            logger,
            model_for_objective,
            test_data_dense,
            test_mask_dense,
            vamp_prior_data,
            impute_config,
            seed=seed,
        )

    # Additional metrics
    if (
        len(variables.target_var_idxs) == 0
        and not issparse(train_data)
        and model.name() != "visl"
        and model.name() not in ["mean_imputing", "mice", "missforest", "zero_imputing"]
    ):
        # If there is no target then we compute additional metrics that are question quality and difficulty.

        # We enforce model to be of IModelForObjective type here
        # TODO: rewrite, so we don't need to use casting
        # TODO: sparse implementation of additional metrics

        assert isinstance(model, IModelForObjective)
        model_for_objective = cast(IModelForObjective, model)
        compute_save_additional_metrics(
            test_imputations,
            train_data,
            model_for_objective,
            model.save_dir,
            objective_config=objective_config,
        )
    elif model.name() == "visl":
        warnings.warn("Question quality and difficulty are not currently implemented for visl")
    elif model.name() in ["mean_imputing", "mice", "missforest", "zero_imputing"]:
        warnings.warn("Question quality and difficulty are not currently implemented for imputation baselines")
    if metrics_logger:
        metrics_logger.log_value("impute/running-time", (dt.datetime.utcnow() - start_time).total_seconds() / 60)
