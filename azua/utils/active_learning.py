import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
from scipy.sparse import csr_matrix

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spmatrix
from tqdm import trange

from ..datasets.variables import Variable, Variables
from ..models.imodel import IModelForObjective
from ..utils.helper_functions import maintain_random_state
from .imputation_statistics_utils import ImputationStatistics
from ..utils.io_utils import save_json
from ..utils.metrics import get_metric
from ..utils.plot_functions import violin_plot_imputations
from ..utils.torch_utils import set_random_seeds
from ..models.transformer_imputer import TransformerImputer
from ..objectives.objective import Objective
from ..objectives.objectives_factory import create_objective
from .active_learning_eedi import run_active_learning_strategy_eedi

logger = logging.getLogger(__name__)


def save_observations(observations: np.ndarray, variables: Variables, path):
    """
    Save observations using both IDs and variable names as a JSON file, with a key for each data row.

    Args:
        observations: numpy array of shape (row_count, step_count) with observation variable id taken at each step.
        variables: List of variables objects.
        path: Path to save files to.
    """
    output_dict = {}
    for i, row in enumerate(observations):
        output_dict[i] = row.tolist()
    save_ids_path = os.path.join(path, "observation_ids.json")
    save_names_path = os.path.join(path, "observation_names.csv")

    # Save observation IDs
    save_json(output_dict, save_ids_path)

    # Convert ids to short names.
    grp_id_to_short_name = {grp_id: grp_name for grp_id, grp_name in enumerate(variables.group_names)}
    grp_id_to_short_name[-1] = "None"

    with open(save_names_path, "w") as save_file:
        writer = csv.writer(save_file, delimiter=",")
        for row in observations:
            observation_w_names = [grp_id_to_short_name[grp_id] for grp_id in row]
            writer.writerow(observation_w_names)


def run_active_learning(
    strategy: str,
    model: Union[IModelForObjective, TransformerImputer],
    test_data: np.ndarray,
    test_mask: np.ndarray,
    vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]],
    objective_config: Dict[str, Any],
    impute_config: Dict[str, Any],
    max_steps: Optional[int] = None,
    average: bool = True,
):
    """
    Run active learning over the test dataset.

    Args:
        strategy (str): One of "eddi", "eddi_mc", "sing", "cond_sing" or "rand".
        model (IModel): Model to use.
        test_data (numpy array of shape (user_count, variable_count)): Data to run active learning on.
        test_mask (numpy array of shape (user_count, variable_count)): 1 is observed, 0 is missing.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.
        objective_config (dictionary): Dictionary containing config options for creating Objective.
        max_steps (int): Maximum number of active learning steps to take.
        average (bool): a boolean variable to indicate whether to return an averaged imputation or not.

    Returns:
        imputed values: numpy array of shape (sample_count, user_count, step_count, variable_count) if average=False
                        else have shape (user_count, step_count, variable_count)
        observations: numpy array of shape (user_count, step_count) with observation group id taken at each step.
        info_gains:
            The shape of this is strategy dependent.
            If no information gain for this method, this is None.
            If info gain changes per step (e.g. EDDI), this is a list of length (step_count) of list of length
                (user_count) of dictionaries {variable_id: info_gain}
            If info gain is constant for all users but not steps (e.g. Cond_SING) this is a list of length
                (user_count) of dictionaries {variable_id: info_gain}
            If info gain is constant for all steps and users (e.g. SING) this is a list of length
                (step_count) of dictionaries {variable_id: info_gain}
    """
    # TODO merge with run_active_learning_strategy?
    with maintain_random_state():
        strategy = strategy.lower()
        objective = create_objective(strategy, model, objective_config)

    with maintain_random_state():
        if objective_config["imputation_method"] is None:
            imputed_values, all_observations, info_gains = run_active_learning_strategy(
                objective,
                model,
                test_data,
                test_mask,
                vamp_prior_data,
                max_steps,
                impute_config,
                average=average,
            )
        else:
            # In this case the dataset is eedi
            (imputed_values, all_observations, info_gains,) = run_active_learning_strategy_eedi(
                objective,
                model,
                test_data,
                test_mask,
                vamp_prior_data,
                max_steps,
                impute_config,
                objective_config,
            )

    return imputed_values, all_observations, info_gains


def run_active_learning_strategy(
    objective,
    model: Union[IModelForObjective, TransformerImputer],
    data: Union[np.ndarray, spmatrix],
    mask: Union[np.ndarray, spmatrix],
    vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]],
    max_steps: Optional[int],
    impute_config: Dict[str, Any],
    initial_obs_mask: Optional[np.ndarray] = None,
    average: bool = True,
):
    if max_steps is None:
        max_steps = len(model.variables.group_idxs)
    else:
        max_steps = min(max_steps, len(model.variables.group_idxs))
    user_count, feature_count = data.shape

    # TODO This was introduced to ensure the violin plot was sensible, but do we always want to override like this?
    if "sample_count" not in impute_config or impute_config["sample_count"] < 10:
        print("overiding sample_count to 10 (missing or insufficiently large sample_count in the original case)")
        impute_config["sample_count"] = 10
    sample_count = impute_config["sample_count"]

    all_imputed_values = np.full(
        (sample_count, user_count, max_steps + 1, feature_count), fill_value=None, dtype=object
    )  # Shape (sample_count, user, steps+1, variable)
    all_info_gains = []  # List (step) of list (user) of info gain dicts.
    all_step_ids = np.full((user_count, max_steps), fill_value=-1, dtype=int)  # Shape (user, step)

    if isinstance(data, spmatrix):
        assert isinstance(mask, spmatrix)
        # For the time being, assume we will only run active learning on data that's small enough to store as a dense
        # array.
        assert data.shape[0] < 100000
        data = data.toarray()
        mask = mask.toarray()
    empty_mask = np.zeros_like(data, dtype=bool)
    imputed_no_observations_mc = model.impute(
        data, empty_mask, impute_config, vamp_prior_data=vamp_prior_data, average=False
    )
    all_imputed_values[:, :, 0, :] = imputed_no_observations_mc

    # If no initial mask is given, mark all features as unobserved to begin with
    if initial_obs_mask is None:
        obs_mask = np.zeros_like(data, dtype=bool)
    elif isinstance(initial_obs_mask, int):
        obs_mask = np.zeros_like(data, dtype=bool)
        obs_mask[:, initial_obs_mask] = 1
    else:
        obs_mask = initial_obs_mask

    for step_idx in trange(max_steps):
        next_qs, info_gains = select_feature(data, mask, obs_mask, objective, model.variables.group_idxs)

        # Both next_qs and info_gains are of length user_count
        for idx, row_choices in enumerate(next_qs):
            if len(row_choices) > 0:
                # Assume only one query chosen per row (if available), in principle row_choices could contain multiple choices.
                all_step_ids[idx, step_idx] = row_choices[0]
        if info_gains is not None:
            all_info_gains.append(info_gains)  # TODO: Change to array
        # We will always have some data here, so vamp_prior_data is never used.
        imputed_mc = model.impute(data, obs_mask, impute_config, vamp_prior_data=None, average=False)
        all_imputed_values[:, :, step_idx + 1, :] = imputed_mc

    if len(all_info_gains) == 0:
        all_info_gains = None  # type: ignore
    if average:
        all_imputed_values = all_imputed_values.mean(0)

    return all_imputed_values, all_step_ids, all_info_gains


def select_feature(
    data: np.ndarray,
    mask: np.ndarray,
    obs_mask: np.ndarray,
    objective: Objective,
    group_idxs: List[List[int]],
    question_count: int = 1,
):
    with maintain_random_state():
        next_qs, info_gains = objective.get_next_questions(data, mask, obs_mask, question_count)

        # Update observation mask to mark selected features as observed
        for i in range(len(obs_mask)):
            for j in next_qs[i]:
                # Select all features if we use question_count > 1
                obs_mask[i, group_idxs[j]] = 1

        # We will always have some data here, so vamp_prior_data is never used.
        # imputed_mc = model.impute(data, obs_mask, impute_config, vamp_prior_data=None, average=False)
        # all_imputed_values[:, :, step_idx + 1, :] = imputed_mc
        return next_qs, info_gains


def draw_and_save_active_learning_plots(
    objectives: Dict[str, Objective],
    model: IModelForObjective,
    data: np.ndarray,
    data_mask: np.ndarray,
    impute_config: Dict[str, Any],
    vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_steps: Optional[int] = None,
    initial_mask: Optional[np.ndarray] = None,
    random_seed: int = 0,
    save_dir: Optional[str] = None,
):
    """
    Draws and saves various stepwise plots.

    Args:
        objectives (dict): Dictionary of the form {objective_name : objectives.Objective}
            containing objectives to compare (such as from different models).
        model (IModel): Instance of Model class to use for imputation.
        data (np.ndarray): Numpy array of shape (user_count, feature_count) to use for
            active learning.
        data_mask (np.ndarray): Numpy array of shape (user_count, feature_count) indicating
            which entries in data are observable.
        impute_config (dict): Impute config dictionary for the model.
        vamp_prior_data (): vamp_prior_data to be used for step 0 active learning.
        max_steps (int): Max number of steps to take. If set to None, will make as many
            steps as there are observable groups of variables.
        initial_mask (np.ndarray or int): Mask indicating which observations are observed
            at step 0. If passed as an int, will assume that only that feature index is
            observed. If set to None (default), no features are observed before step 0.
        random_seed (int): Random seed to use.
        save_dir (str): Directory to save plots to. If None, plots are saved in
            model_save_dir/comparative_active_Learning_plots.
    """

    if save_dir is None:
        save_dir = os.path.join(model.save_dir, "comparative_active_learning_plots")
    os.makedirs(save_dir, exist_ok=True)
    imputations = {}
    step_ids = {}
    masks = {}
    info_gains = {}

    set_random_seeds(random_seed)

    for obj in objectives:
        imputations[obj], step_ids[obj], info_gains[obj] = run_active_learning_strategy(
            objectives[obj],
            model,
            data,
            data_mask,
            vamp_prior_data,
            max_steps,
            initial_obs_mask=initial_mask,
            impute_config=impute_config,
        )
        masks[obj] = get_masks_from_step_ids(step_ids[obj], model.variables, initial_mask=initial_mask)

        # Plot choice plots (separate plot for each objective)
        choices_name = "{}_choices.png".format(obj)
        alpha = 10 / data.shape[0]
        plot_choices_line_chart(
            step_ids[obj],
            model.variables,
            save_dir,
            save_name=choices_name,
            alpha=alpha,
        )

    # Plot shared plots
    rmse_curves = compute_rmse_curves(imputations, data, data_mask, model.variables, normalise=True)
    plot_and_save_rmse_curves(rmse_curves, save_dir, model.variables)

    plot_pll_curves(model, data, masks, save_dir)

    # Get a second set of steps from the baseline objective for self-consistency
    base_key = list(objectives.keys())[0]
    _, step_ids[base_key + "_consistency"], _ = run_active_learning_strategy(
        objectives[base_key],
        model,
        data,
        data_mask,
        vamp_prior_data,
        max_steps,
        initial_obs_mask=initial_mask,
        impute_config=impute_config,
    )
    plot_step_accuracy(step_ids, save_dir=save_dir)


def compute_rmse_curves(
    imputed_values_per_strategy: Dict[str, Union[np.ndarray, csr_matrix]],
    test_data: np.ndarray,
    test_mask: np.ndarray,
    variables: Variables,
    normalise: bool,
) -> dict:

    user_count, feature_count = test_data.shape
    variable_metrics = {}

    # Compute the metric for each single target variable, but also for all target variables together (if not empty)
    # TODO can this be cleaned up?
    target_vars_tuples = [([var_idx], [var], var.name) for (var_idx, var) in enumerate(variables) if var.target]
    target_vars_idxs: List[int] = sum([var_idxs for var_idxs, _, _ in target_vars_tuples], [])
    target_vars: List[Variable] = sum([vars for _, vars, _ in target_vars_tuples], [])
    if len(target_vars_idxs) > 0:
        target_vars_tuples.append((target_vars_idxs, target_vars, "all"))
    else:  # If no target var present, compute "all" metrics for all vars
        all_vars_idxs = [var_idx for var_idx, _ in enumerate(variables)]
        all_vars_as_list = [
            var for _, var in enumerate(variables)
        ]  # TODO: figure out right typing of Variables, so we don't need to do things like this
        target_vars_tuples.append((all_vars_idxs, all_vars_as_list, "all"))

    for vars_idxs, vars, vars_name in target_vars_tuples:
        plot_values = {}

        for strategy, imputed_values in imputed_values_per_strategy.items():
            assert imputed_values.ndim == 3  # shape (user_count, step_count, variable_count)

            _, step_count, _ = imputed_values.shape

            steps = np.arange(step_count)

            # Get only info for this variable.
            # imputed_values = imputed_values[:, :, :, var_idx]        # Shape (seed, user, step)
            # imputed_values = np.expand_dims(imputed_values, axis=3)  # Shape (seed, user, step, 1)

            metric_per_step = np.zeros(step_count)
            for step_idx in steps:
                imputed_for_step = imputed_values[:, step_idx, :]  # Shape (seed, user, features)

                metric_dict = get_metric(variables, imputed_for_step, test_data, test_mask, vars_idxs)

                unique_variables_types = list(set([var.type for var in vars]))
                if len(unique_variables_types) > 1:
                    raise ValueError("All of the variables should be of the same type")
                var_type = unique_variables_types[0]
                if var_type == "continuous":
                    metric = metric_dict["Normalised RMSE"] if normalise else metric_dict["RMSE"]
                elif var_type == "categorical" or var_type == "binary":
                    metric = metric_dict["Fraction Incorrectly classified"]
                metric_per_step[step_idx] = metric
            plot_values[strategy] = {step: metric for step, metric in enumerate(metric_per_step)}

        variable_metrics[vars_name] = plot_values
    return variable_metrics


def plot_and_save_rmse_curves(metrics: Dict, save_dir: str, variables: Variables) -> None:
    """

    Plot the RMSE per step and save plotted data into a JSON file.

    Args:
        metrics (dict): metrics to plot and save in form {var: {strategy: {step: val}}}
            or {var: {strategy: {step: {mean: m, std: s, num_samples: n}}}}
        save_dir (str): Directory to save plots to.
        variables: Variables corresponding to the metrics
        dataset_seed: the dataset_seed for aggragation

    """

    target_vars_tuples = [(var.name, var.type) for var in variables if var.target]
    target_vars_tuples.append(("all", ""))
    for var_name, var_type in target_vars_tuples:
        var_metrics = metrics[var_name]

        plt.clf()
        plt.figure(figsize=(6.4, 4.8))

        for strategy, strategy_metrics in var_metrics.items():
            # Strategy metrics is {step: val} or {step: {mean: val, std: val, num_samples: val}}
            step_count = len(strategy_metrics)

            steps = np.arange(step_count)
            # If metrics were written to JSON and then read back, the keys (which are step numbers)
            # will have been converted to strings. Get them back to ints.
            sm = {int(k): v for k, v in strategy_metrics.items()}
            if isinstance(sm[0], dict):
                # Aggregated metrics
                means = np.array([sm[i]["mean"] for i in steps])
                stds = np.array([sm[i]["std"] for i in steps])
                plt.fill_between(steps, means - stds, means + stds, alpha=0.1)
            else:
                means = np.array([sm[i] for i in steps])
            plt.plot(steps, means, label=strategy)

        plt.legend()
        plt.xlabel("Steps")
        plt.title(var_name)
        if var_type == "continuous":
            plt.ylabel("Avg test RMSE")
        elif var_type == "categorical" or var_type == "binary":
            plt.ylabel("Avg test fraction incorrectly classified")
        elif var_type == "" and var_name == "all":  # TODO: pass type of "all" around, so ylabel can be more informative
            plt.ylabel("Avg test RMSE/fraction incorrectly classified")
        else:
            raise ValueError("Variable type %s not supported." % var_type)
        image_path = os.path.join(save_dir, f"{var_name}.png")
        plt.savefig(image_path, format="png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {image_path}")

        save_metrics_json(save_dir, var_name, metrics)


def save_metrics_json(save_dir: str, filename: str, metrics: Dict):
    json_path = os.path.join(save_dir, f"{filename}.json")
    save_json(metrics, json_path)
    logger.info(f"Saved values to {json_path}")


def plot_pll_curves(
    model: IModelForObjective, data, masks, save_dir, sample_count=50, seed=0
):  # TODO: Add multi-seed support
    logger = logging.getLogger()

    full_vars_pll = {}

    step_count = max(len(mask) for mask in masks.values())
    steps = np.arange(step_count)
    for group_name, group_idxs in zip(model.variables.group_names, model.variables.group_idxs):
        plt.clf()
        plt.figure(figsize=(6.4, 4.8))
        save_path = os.path.join(save_dir, "{}_pll.png".format(group_name))
        plot_values_path = os.path.join(save_dir, "{}_pll.json".format(group_name))
        plot_values = {}

        pll_per_strategy = {obj: np.zeros(step_count) for obj in masks}
        for obj in masks:
            set_random_seeds(seed)  # Set this here so that all objectives use the same seeding
            for step, mask in enumerate(masks[obj]):
                # TODO: Fix model.get_model_pll to accept target_idxs list
                pll_per_strategy[obj][step] = model.get_model_pll(
                    data, mask, group_idxs[0], sample_count=sample_count
                ).cpu()
            plt.plot(steps, pll_per_strategy[obj], label=obj)
            plot_values[obj] = {step: pll for step, pll in enumerate(pll_per_strategy[obj])}

        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Avg model predictive log-likelihood")

        # Save everything
        plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved pll plot to {}".format(save_path))
        save_json(plot_values, plot_values_path)

        full_vars_pll[group_name] = pll_per_strategy

    return full_vars_pll


def plot_info_gain_bar_charts(info_gain_list, variables: Variables, save_dir, user_idx=0):
    """
    Plot bar charts of info gain for all groups of variables at each step.

    Args:
        info_gain: List of length (step_count) of  list of length (user_count), dictionaries {group_id: info_gain}
        variables (`Variables`): variable information.
        save_dir (str): Directory to save plots to.
        user_idx (int): User to plot for. Defaults to 0.
    """
    logger = logging.getLogger()

    col_count = 3

    row_count = (len(info_gain_list) // col_count) + 1

    max_info_gain = 0.0
    for info_gain_for_user in info_gain_list:
        if isinstance(info_gain_for_user, list):
            info_gain_for_user = info_gain_for_user[user_idx]
        if len(info_gain_for_user) == 0:
            max_for_step = 0
        else:
            max_for_step = max(info_gain_for_user.values())
        if max_for_step > max_info_gain:
            max_info_gain = max_for_step
    max_info_gain *= 1.5  # To allow space for labels.

    plt.clf()
    plt.figure(figsize=(25, 15))
    plt.subplots_adjust(hspace=0.5)

    for step_idx, info_gain_for_user in enumerate(info_gain_list):
        if isinstance(info_gain_for_user, list):
            info_gain_for_user = info_gain_for_user[user_idx]

        x_labels = []
        y_values = []
        for group_id, group_name in enumerate(variables.group_names):
            x_labels.append(group_name)
            info_gain = info_gain_for_user.get(group_id, 0)  # Get info gain, default to 0 if not present.
            y_values.append(info_gain)
            x_ticks = np.arange(len(x_labels))

        plt.subplot(row_count, col_count, step_idx + 1)
        bar_container = plt.bar(x_ticks, y_values, align="center")
        plt.xticks(x_ticks, x_labels)
        plt.ylabel("Info gain")
        plt.ylim(top=max_info_gain)
        plt.title("User %d, Step %d" % (user_idx, step_idx))

        # Add labels above each bar.
        for rect, y_value in zip(bar_container.patches, y_values):
            y_pos = rect.get_height()
            x_pos = rect.get_x() + rect.get_width() / 2
            label = "%.3g" % y_value

            plt.annotate(
                label,  # Use `label` as label
                (x_pos, y_pos),  # Place label at end of the bar
                xytext=(0, 5),  # Vertically shift label by 5
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha="center",  # Horizontally center label
                va="bottom",
            )  # Align above bar.

    save_path = os.path.join(save_dir, "info_gain_user_" + str(user_idx) + ".png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s" % save_path)


def plot_choices_line_chart(observations, variables: Variables, save_dir, steps=None, save_name=None, alpha=0.2):
    """
    Plot choices made per step for all users.

    Args:
        observations: numpy array of shape (user_count, step_count) with observation group id taken at each step.
        variables: List of variables.
        save_dir: Directory to save plots to.
        steps: Number of steps to plot. Defaults to None - all steps are plotted.
    """
    logger = logging.getLogger()

    if steps is not None:
        observations = observations[:, :steps]
    else:
        _, steps = observations.shape

    xs = np.arange(1, steps + 1)
    y_labels = variables.group_names
    y_range = np.arange(0, len(variables.group_names))

    plt.clf()
    plt.xticks(xs)
    plt.yticks(y_range, y_labels)  # Set y axis labels to be the short names of variables.
    for user in observations:
        # user has shape (steps)
        # Plot transparent lines - same path with multiple occurrences will appear darker.
        plt.plot(xs, user, linestyle="-", marker="o", color="purple", alpha=alpha)

    if save_name is None:
        save_name = "choices.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s" % save_path)


def plot_step_accuracy(step_ids, truth_key=None, save_dir=None):
    if truth_key is None:
        truth_key = list(step_ids.keys())[0]
    truth_steps = step_ids.pop(truth_key).T
    step_accs = {}
    plot_values = {}

    plt.clf()
    for obj in step_ids:
        step_accs[obj] = []
        for truth_step, obj_step in zip(truth_steps, step_ids[obj].T):
            step_accs[obj].append(np.mean(truth_step == obj_step))

        steps = np.arange(len(step_accs[obj]))
        plt.plot(steps, step_accs[obj], label=obj)
        plot_values[obj] = {step: acc for step, acc in enumerate(step_accs[obj])}

    plt.xticks(steps, steps)
    plt.xlabel("Steps")
    plt.ylabel("Step accuracy")
    plt.legend()
    plt.title("Selection accuracy of each step against {} selections".format(truth_key))

    plt.savefig(
        os.path.join(save_dir, "step_accuracy.png"),
        format="png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    save_json(plot_values, os.path.join(save_dir, "step_accuracy.json"))

    return step_accs


def plot_imputation_violin_plots_active_learning(
    all_imputations_mc: np.ndarray,
    variables: Variables,
    all_step_ids: np.ndarray,
    save_dir: str,
    user_idx: int = 0,
):
    """
    Plot violin plot for the imputation of all groups of variables at each step.

    Args:
        all_imputations_mc (np.ndarray): all MC imputations of shape (sample_count, user, steps, feature_count)
        variables (`Variables`): variable information.
        all_step_ids (np.ndarray): of shape (user_count, step_count) with observation group id taken at each step.
        save_dir (str): Directory to save plots to.
        user_idx (int): User to plot for. Defaults to 0.
    """
    logger = logging.getLogger()

    col_count = 3
    _, _, num_steps, _ = all_imputations_mc.shape
    row_count = (num_steps // col_count) + 1

    plt.clf()
    plt.figure(figsize=(25, 15))
    plt.subplots_adjust(hspace=0.5)
    all_masks = get_masks_from_step_ids(all_step_ids, variables, initial_mask=None)

    for step_idx in range(num_steps):
        imputations_step = all_imputations_mc[:, :, step_idx]
        imputations_stats_step = ImputationStatistics.get_statistics(imputations_step, variables)
        mask_step = all_masks[step_idx]

        ax = plt.subplot(row_count, col_count, step_idx + 1)
        violin_plot_imputations(
            imputations_step,
            imputations_stats_step,
            mask_step,
            variables,
            user_idx,
            title="imputation for User %d, Step %d",
            normalized=True,
            ax=ax,
        )

    save_path = os.path.join(save_dir, "imputation_active_learning_user_" + str(user_idx) + ".png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s" % save_path)


def get_masks_from_step_ids(all_step_ids: np.ndarray, variables: Variables, initial_mask=None):
    feature_count = len(variables)
    user_count, steps = all_step_ids.shape
    if initial_mask is None:
        mask = np.zeros((user_count, feature_count))
    elif isinstance(initial_mask, int):
        mask = np.zeros((user_count, feature_count))
        mask[:, initial_mask] = 1
    else:
        mask = initial_mask
    all_masks = [mask.copy()]

    for step in range(steps):
        group_idxs_list = [i for i in variables.group_idxs]
        step_features_idxs = [group_idxs_list[group_id] for group_id in all_step_ids[:, step]]
        for row_idx in range(len(mask)):
            mask[row_idx, step_features_idxs[row_idx]] = 1
        all_masks.append(mask.copy())

    return all_masks


def save_info_gain_normalizer(objective: Objective, model: IModelForObjective):
    """
    Save a scalar normalizing constant for computing normalized information gain esimates. This normalizer is
    computed by taking the largest estimated information gain available when no features are observed, and all
    features can be selected.

    This normalizer is saved to the model's `save_dir` as "info_gain_normalizer.npy".

    Args:
        objective: Objective used to compute information gains.
        model: Model used to compute information gains.
        sample_count: Number of Monte Carlo samples to use when computing the information gain estimates.
    """
    empty_data = Variables.create_empty_data(model.variables)
    # Assume all features can be queried when computing normalizer (e.g. data_mask==1 and obs_mask==0).
    # Also assume each feature can be queried individually (no query groups).
    empty_data_mask = np.ones_like(empty_data, dtype=bool)
    empty_mask = np.zeros_like(empty_data, dtype=bool)

    with maintain_random_state():
        _, zero_step_info_gain = objective.get_next_questions(
            empty_data,
            data_mask=empty_data_mask,
            obs_mask=empty_mask,
            question_count=1,
        )
    max_val = max(zero_step_info_gain[0].values()) if len(zero_step_info_gain[0]) > 0 else 1
    # Set to 1 if max_val is somehow <= 0 to avoid reversing the ordering of info gains/divison by zero
    if max_val <= 0:
        max_val = 1
    np.save(os.path.join(model.save_dir, "info_gain_normalizer.npy"), max_val)
