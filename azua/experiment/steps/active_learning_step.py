from logging import Logger
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from scipy.sparse import csr_matrix

from ..imetrics_logger import IMetricsLogger
from ...models.imodel import IModelForObjective
from ...models.transformer_imputer import TransformerImputer
from ...utils.torch_utils import set_random_seeds
from ...utils.active_learning import (
    run_active_learning,
    compute_rmse_curves,
    plot_and_save_rmse_curves,
    save_metrics_json,
    save_observations,
    plot_info_gain_bar_charts,
    plot_choices_line_chart,
    plot_imputation_violin_plots_active_learning,
)

from ...utils.plot_functions import (
    plot_target_curves,
    plot_mean_target_curves,
    plot_rewards_hist,
)


class ActiveLearningResults:
    """Light-class for active learning results

    Attributes:
        all_info_gains: Dictionary of {active learning strategy: estimated information gain (np array)}.
            The shape of this array is strategy dependent.
            If no information gain for this method, this is None.
            If info gain changes per step (e.g. EDDI, Cond_SING), this is a list of length (step_count) of list of length
                (user_count) of dictionaries {variable_id: info_gain}
            If info gain is constant for all steps and users (e.g. SING) this is a list of length
                (step_count) of dictionaries {variable_id: info_gain}
        metrics: dict TODO: describe nested structure
    """

    def __init__(self, info_gains, metrics):
        self.info_gains = info_gains
        self.metrics = metrics


def run_active_learning_main(
    logger: Logger,
    model: Union[IModelForObjective, TransformerImputer],
    data: Union[np.ndarray, csr_matrix],
    mask: Union[np.ndarray, csr_matrix],
    vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]],
    active_learning_strategies: List[str],
    objective_config: Dict[str, Any],
    impute_config: Dict[str, Any],
    seed=0,
    max_steps=None,
    max_rows=np.inf,
    users_to_plot=[0],
    metrics_logger: Optional[IMetricsLogger] = None,
) -> ActiveLearningResults:
    """
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        model (IModel): model to use.
        data (numpy array of shape (user_count, variable_count)): Data to run active learning on.
        mask (numpy array of shape (user_count, variable_count)): 1 is observed, 0 is missing.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.
        active_learning_strategies (list of str): List of active learning methods to check.
        objective_config (dictionary): Dictionary containing config options for creating Objective.
        seed (int): Random seed to use when running active learning. Defaults to 0.
        max_steps (int): Maximum number of active learning steps to take (default: inf).
        max_rows (int): Maximum number of data rows on which to perform active learning.

    Returns:
        active_learning_results (ActiveLearningResults): the results of active learning
    """
    if data.shape[0] > max_rows:
        data = data[0:max_rows]
        mask = mask[0:max_rows]
    # Fix active learning seed.
    set_random_seeds(seed)

    save_dir = os.path.join(model.save_dir, "active_learning")
    os.makedirs(save_dir, exist_ok=True)

    # When VAMP prior is used, the first step for empty data rows will just load the pre-computed information gain,
    # as vamp_prior_data will always be None here. The first step retunred imputed data is not using vamp prior.
    # This is also due to the duplication of imputation outside and inside the get information gain.
    all_imputed = {}
    all_info_gains = {}
    # return averaged imputed values for eedi application see run_active_learning() in utils/active_learning.py
    average = objective_config["imputation_method"] is not None
    al_delta_dict_all_strategy = {}
    for strategy in active_learning_strategies:
        # Fix active learning seed before running each strategy.
        set_random_seeds(seed)
        logger.info("Running active learning with strategy %s" % strategy)
        strategy_dir = os.path.join(save_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        imputed_values_mc, observations, info_gains = run_active_learning(
            strategy,
            model,
            data,
            mask,
            vamp_prior_data,
            objective_config,
            impute_config,
            max_steps=max_steps,
            average=average,
        )

        if len(imputed_values_mc.shape) > 3:  # If imputed_values_mc includes samples
            # average over the MC non-string samples of the imputations
            # For string variables, take the 1st sample as "mean" (as we can't perform mean over string data)
            # TODO #18668: experiment with calculating mean in text embedding space instead
            imputed_values = np.copy(imputed_values_mc[0])
            non_text_idxs = model.variables.non_text_idxs
            imputed_values[:, :, non_text_idxs] = np.mean(imputed_values_mc[:, :, :, non_text_idxs], axis=0)
        else:
            imputed_values = imputed_values_mc

        all_imputed[strategy] = imputed_values

        save_observations(observations, model.variables, strategy_dir)

        all_info_gains[strategy] = info_gains

        # Save RMSE curve for each variable and this one strategy.
        rmse_curves = compute_rmse_curves({strategy: imputed_values}, data, mask, model.variables, normalise=True)
        plot_and_save_rmse_curves(rmse_curves, strategy_dir, model.variables)

        # Compute AUIC (sum auic over all target variables)
        # TODO inform an user when max_steps<num variables?
        auic_per_target_vars = {k: sum(v[strategy].values()) for (k, v) in rmse_curves.items()}
        save_metrics_json(strategy_dir, "auic", auic_per_target_vars)

        for (k, v) in auic_per_target_vars.items():
            logger.info(f"{k}.AUIC for {strategy} is {v}")
        # log the difference between the active learning steps of eddi. We want the last point lower than the first and the second point in between
        if "all" in rmse_curves:  # Account for a case when no target variable present
            al_delta_dict = {}
            al_target_rmse_curve = rmse_curves["all"][strategy]
            al_rmse_list = list(al_target_rmse_curve.values())
            # we expect all these below are postive
            delta_0end = al_rmse_list[0] - al_rmse_list[-1]
            delta_01 = al_rmse_list[0] - al_rmse_list[1]
            delta_1end = al_rmse_list[1] - al_rmse_list[-1]
            al_delta_dict = {"delta_01": delta_01, "delta_0end": delta_0end, "delta_1end": delta_1end}
            save_metrics_json(strategy_dir, "al_delta", al_delta_dict)
            al_delta_dict_all_strategy[strategy] = al_delta_dict

            for (k, v) in al_delta_dict.items():
                logger.info(f"{k} for {strategy} is {v}")

        # plot the violin plot for each step
        if len(imputed_values_mc.shape) > 3:  # If imputed_values_mc includes samples
            for idx in users_to_plot:
                plot_imputation_violin_plots_active_learning(
                    imputed_values_mc, model.variables, observations, strategy_dir, user_idx=idx,
                )

        # Plot info_gain bar plots.
        if info_gains is not None and strategy in ["eddi", "eddi_mc"]:
            for idx in users_to_plot:
                plot_info_gain_bar_charts(info_gains, model.variables, strategy_dir, user_idx=idx)

        # Plot choices line plot
        plot_choices_line_chart(observations, model.variables, strategy_dir)

        if len(model.variables.target_var_idxs) == 0:
            # difficulty = np.asarray(pd.read_csv(os.path.join(model.save_dir, 'difficulty.csv'), header=None))[:,1]
            # quality = np.asarray(pd.read_csv(os.path.join(model.save_dir, 'quality.csv'), header=None))[:,1]

            # plot_difficulty_curves(strategy, observations, model.variables, difficulty, strategy_dir)
            # plot_quality_curves(strategy, observations, model.variables,  quality, strategy_dir)

            if strategy == "ei":
                plot_rewards_hist(info_gains, strategy_dir)

    # Plot combined strategy plot for each variable.
    all_metrics = compute_rmse_curves(all_imputed, data, mask, model.variables, normalise=True)
    plot_and_save_rmse_curves(all_metrics, save_dir, model.variables)

    # Compute AUIC (sum auic over all target variables) for each strategy
    all_auic_per_target_vars = {
        var: {k: sum(v.values()) for k, v in strategies.items()} for (var, strategies) in all_metrics.items()
    }
    save_metrics_json(save_dir, "auic", all_auic_per_target_vars)

    if objective_config["imputation_method"] is not None:
        # When we dont have a target (eedi) we plot the evolution of the target variable over steps
        if data.shape[0] < 20:
            plot_target_curves(all_imputed, save_dir)
        plot_mean_target_curves(all_imputed, objective_config["max_steps"], save_dir)
        # plot_time_results_hist(all_imputed, objective_config['max_steps'], save_dir)

    if metrics_logger is not None:
        # Log metrics to AzureML
        if "all" in all_auic_per_target_vars:  # Account for no target variables
            metrics_logger.log_dict({"auic": all_auic_per_target_vars["all"]})

        metrics_logger.log_dict(al_delta_dict_all_strategy)

    return ActiveLearningResults(info_gains=all_info_gains, metrics=all_metrics)
