"""
Check information gain is approximating well.

"""

import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from ..objectives.eddi import EDDIObjective
from ..models.imodel import IModelForObjective
from ..utils.metrics import get_metric
from ..utils.torch_utils import set_random_seeds


# TODO #14087: Reuse code for checking information gain
def test_information_gain(
    logger,
    model: IModelForObjective,
    test_data,
    test_mask,
    vamp_prior_data,
    impute_config,
    seed=0,
):
    """
    Check information gain is approximating well.

    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        model (.Model): Model to use.
        test_data (numpy array of shape (user_count, variable_count)): Data to run active learning on.
        test_mask (numpy array of shape (user_count, variable_count)): 1 is observed, 0 is missing.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.
        impute_config (dictionary): Dictionary containing options for inference.
        seed (int): Random seed to use when running active learning. Defaults to 0.
    """
    set_random_seeds(seed)
    user_count, feature_count = test_data.shape

    variables = model.variables
    target_idxs = [idx for idx, var in enumerate(variables) if not var.query]
    empty_mask = np.zeros_like(test_data)

    # Run imputation with one feature added for all test users.
    results = {}
    # Iterate through variables in non-sorted order.
    for var_idx, variable in enumerate(variables):
        if not variable.query:
            continue
        logger.info("Checking variable %s" % variable.name)

        imputed_no_observations = model.impute(test_data, empty_mask, impute_config, vamp_prior_data=vamp_prior_data)
        imputed_predictions = model.impute(test_data, test_mask, impute_config, vamp_prior_data=vamp_prior_data)

        # Then for get metric for targets, given imputed_no_obs and imputed_predictions.
        # Need to generate a mask saying observed this var for all users.
        mask_no_obs = np.zeros((user_count, feature_count), dtype=bool)
        mask_obs = np.zeros((user_count, feature_count), dtype=bool)
        mask_obs[:, var_idx] = 1  # We have observed this feature only.

        # TODO target_idxs or var_idx?
        metric_with_obs_dict = get_metric(variables, imputed_predictions, test_data, mask_obs, target_idxs)
        metric_no_obs_dict = get_metric(variables, imputed_no_observations, test_data, mask_no_obs, target_idxs)

        if len(metric_with_obs_dict) == 1:
            key = list(metric_with_obs_dict.keys())[0]
        else:
            key = "Normalised RMSE"

        metric_with_obs: float = metric_with_obs_dict[key]
        metric_no_obs: float = metric_no_obs_dict[key]
        metric_drop = metric_no_obs - metric_with_obs
        if metric_drop < 0:
            metric_drop = 0
        results[variable.name] = 100 * metric_drop

    # results is a {var: sq_err array of shape (users, variables)}
    # get_next_questions with no data - to get SING order.
    eddi_objective = EDDIObjective(model, sample_count=50, use_vamp_prior=True)
    single_row_data = np.expand_dims(test_data[0, :], axis=0)
    # Make data mask all 1's (can query all features) and obs mask all 0's (no features currently observed).
    single_row_data_mask = np.ones_like(np.expand_dims(empty_mask[0, :], axis=0))
    single_row_obs_mask = np.zeros_like(single_row_data_mask)
    _, information_gain = eddi_objective.get_next_questions(single_row_data, single_row_data_mask, single_row_obs_mask)
    information_gain = information_gain[0]

    info_gain_per_var = {}
    for var_idx, gain in information_gain.items():
        short_desc = variables[var_idx].name
        info_gain_per_var[short_desc] = gain

    # Rank information gain (max) and metric - i.e. RMSE (min)
    # Plot the two on the same bar chart.
    plt.clf()
    # Sort x labels by maximum info gain.
    x_labels = sorted(info_gain_per_var.keys(), key=info_gain_per_var.__getitem__, reverse=True)
    x_ticks = np.arange(len(x_labels))
    bar_width = 0.4

    info_gains = [info_gain_per_var[x_label] for x_label in x_labels]
    metrics = [results[x_label] for x_label in x_labels]
    avg_metric_drop = np.mean(metrics)

    ax1 = plt.subplot(111)
    plt.xticks(x_ticks, x_labels, rotation=45)
    info_gain = ax1.bar(
        x_ticks,
        info_gains,
        width=-bar_width,
        align="edge",
        label="Info gain",
        color="b",
    )
    ax1.set_ylabel("Info gain")

    ax2 = ax1.twinx()
    metric_drop = ax2.bar(
        x_ticks,
        metrics,
        width=bar_width,
        align="edge",
        label="Metric drop (%)",
        color="g",
    )
    ax2.set_ylabel("Metric drop (%)")

    line = plt.axhline(y=avg_metric_drop, color="r", linestyle="--")

    # plt.tight_layout()
    plt.title("Info gain vs Metric drop after first step")
    plt.legend(
        [info_gain, metric_drop, line],
        ["Info gain", "Metric drop (%)", "Avg Metric drop (%)"],
    )

    save_path = os.path.join(model.save_dir, "info_gain_vs_metric_drop.png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    logger.info("Saved plot to %s" % save_path)
