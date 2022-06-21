import os
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import bernoulli
from tqdm import trange

from .synthetic_imputations import synthetic_fill_variables


# This is on BO based next best action for eedi with heuristic dynamics
def run_active_learning_strategy_eedi(
    objective,
    model,
    data,
    mask,
    vamp_prior_data,
    max_steps,
    impute_config,
    objective_config,
):

    max_steps = objective_config["max_steps"]
    user_count, feature_count = data.shape

    # Get imputatation method for missing values
    imputation_method = objective_config["imputation_method"]
    similarity_metric = objective_config["similarity_metric"]
    data_dir = objective_config["data_dir"]
    use_model = objective_config["use_model"]

    if objective_config["aggregate_values"]:
        question_construct = np.asarray(pd.read_csv(data_dir + "/question_construct.csv", delimiter=",", header=None))
    else:
        question_construct = None

    # Store initial data and mask so that all strategies use the same initial data and mask
    original_mask = mask.copy()
    original_data = data.copy()
    mask = original_mask.copy()
    data = original_data.copy()

    all_imputed_values = np.full(
        (user_count, max_steps + 1, feature_count), fill_value=np.nan
    )  # Shape (user, steps+1, variable)
    all_reward_gains = []  # List (step) of list (user) of info gain dicts.
    all_observations = np.full((user_count, max_steps), fill_value=np.nan)  # Shape (user, step)

    if use_model:
        # Imputing batch based on observed points in the test data
        imputed_no_observations = model.impute(data, mask, impute_config, vamp_prior_data=None)
    else:
        # Synthetic imputations
        imputed_no_observations = synthetic_fill_variables(data, all_observations[:, :0], idx=None, average=False)

    # Store the initial imputation
    all_imputed_values[:, 0, :] = imputed_no_observations

    # Create lists to store rewards in non myopic case
    next_qs_lists = []

    # Mark all features as unobserved to begin with
    obs_mask = np.zeros_like(data, dtype=bool)

    # Initial time
    t0 = time.process_time()
    for step_idx in trange(max_steps):
        # Update the answers we expect to get based on the different heuristics
        previous_imputation = all_imputed_values[:, step_idx, :]

        (X_imputed, imputed_probabilities, initial_imputations) = get_imputations(
            model,
            data,
            mask,
            next_qs_lists,
            previous_imputation,
            imputation_method=imputation_method,
            similarity_metric=similarity_metric,
            data_dir=data_dir,
            impute_config=impute_config,
            delta=objective_config["delta"],
            aggregate_values=objective_config["aggregate_values"],
        )

        # Get the next features to observe
        (next_qs, rewards, updated_X_imputed) = objective.get_next_questions(
            data,
            mask,
            obs_mask,
            X_imputed,
            imputed_probabilities,
            initial_imputations,
            impute_config,
            step_idx,
            all_observations[:, :step_idx],
            use_model,
            previous_imputation,
            question_construct,
        )

        # Use the updated imputations for a batch method
        if updated_X_imputed is not None:
            X_imputed = updated_X_imputed

        next_qs_lists.append(next_qs)

        # Update test mask and test data based on the heuristics
        data, mask = update_data_mask(next_qs, data, mask, X_imputed)

        # Add a dummy value of -1 to empty next_qs lists (i.e. if no features are left to query) to ensure shapes match
        next_qs = [([-1] if len(x) == 0 else x) for x in next_qs]
        # If we are using a batch method we assign all questions observed in one go
        # otherwise we store the next question index
        if objective.name() == "b_ei":
            all_observations[:, :] = np.array(next_qs)
        else:
            all_observations[:, step_idx] = np.array(next_qs).reshape(-1)

        if rewards is not None:
            all_reward_gains.append(rewards)

        # TODO: this may be able to be optimised.
        # Update obs mask
        for row, next_q_idx in enumerate(next_qs):
            # Collect and store one question per step if the method is not batched
            if type(next_q_idx) == int:
                obs_mask[row, next_q_idx] = 1
            else:
                # If a batch method used update the mask with all features collected
                for j in next_q_idx:
                    next_q_idx = int(j)
                    obs_mask[row, next_q_idx] = 1

        # New imputation based on the collected features
        if use_model:
            imputed = model.impute(data, mask, impute_config)
        else:
            # Synthetic imputations
            if objective.name() == "b_ei":
                imputed = synthetic_fill_variables(data, all_observations, idx=None, method="b_ei", average=False)
            else:
                imputed = synthetic_fill_variables(data, all_observations[:, : (step_idx + 1)], idx=None, average=False)
        # Store the imputed values
        all_imputed_values[:, step_idx + 1, :] = imputed
        # If a batch method is used, all imputed values are computed in one step thus we can break the loop after one step
        if objective.name() == "b_ei":
            all_imputed_values = all_imputed_values[:, :2, :]
            break

    # Final time
    t1 = np.asarray(time.process_time() - t0)[np.newaxis]

    # Save running time for one individual
    if user_count == 1:
        save_dir = os.path.join(model.save_dir, "active_learning/", objective.name())
        np.savetxt(save_dir + "/time.csv", t1)

    if len(all_reward_gains) == 0:
        all_reward_gains = None

    return all_imputed_values, all_observations, all_reward_gains


def expected_improvement(
    model,
    data: np.ndarray,
    data_mask: np.ndarray,
    obs_mask: np.ndarray,
    X_imputed: np.ndarray,
    next_qs_lists,
    impute_config,
    use_model,
    previous_imputation,
    combinations_list=None,
    method=None,
    max_steps=None,
) -> np.ndarray:

    """
    Calculate expected improvement for adding each observable feature individually.
    Args:
        model (Model): Trained `Model` class to use.
        data (Torch tensor of shape (batch_size, proc_feature_count)): Contains processed, observed data.
        obs_mask (Torch tensor of shape (batch_size, proc_feature_count)): Processed mask indicating which
            features have already been observed (i.e. which to condition the information gain calculations on).
        sample_count (int): Number of Monte Carlo samples of the latent space to use for the calculation.
        X_imputed (numpy array of shape (batch_size, feature_count)): Contains values of X that we predict to get when collecting a feature
                    and we want to consider when computing the expected rewards
        data_mask (numpy array of shape (batch_size, feature_count)): Contains mask where 1 is observed in the
            underlying data, 0 is missing.
        as_array (bool): When True will return info_gain values as an np.ndarray. When False (default) will return info
            gain values as a List of Dictionaries.
        greedy (bool): When true will use a greedy policy and compute the expected improvement as the predicted target
    Returns:
        rewards (List of Dictionaries): Length (batch_size) of dictionaries of the form {idx : expected_improvement} where
            expected_improvement is np.nan if the variable is observed already (1 in obs_mask) or it is not observable (0 in data_mask)
            If as_array is True, rewards is an array of shape
            (batch_size, feature_count) instead.
    """
    user_count, feature_count = data.shape

    if previous_imputation is None:
        # Get current target value
        old_imputed = np.mean(model.fill_variables(data, data_mask, impute_config), axis=1)
    else:
        old_imputed = np.mean(previous_imputation, axis=1)

    # If not in batch method we create a list of batches that include the features we want to explore
    if combinations_list is None:
        combinations_list = []
        for idx in model.variables.query_var_idxs:
            var = [[idx]] * user_count
            combinations_list.append(var)

    # For batch (or single feature) we compute the reward
    rewards_list = []

    for idx, s in enumerate(combinations_list):
        # Compute data and mask one step ahead
        if method == "b_ei":
            # If we are using a batch method, the imputed values will be different depending on the batch
            data_osa, data_mask_osa = update_data_mask(s, data.copy(), data_mask, X_imputed[idx])
        else:
            data_osa, data_mask_osa = update_data_mask(s, data.copy(), data_mask, X_imputed)

        if use_model:
            # Get new imputations from model
            new_imputed = np.mean(model.impute(data_osa, data_mask_osa, impute_config), axis=1)
        else:
            # Get synthetic imputations
            if method == "b_ei":
                new_imputed = synthetic_fill_variables(data, next_qs_lists, s, method=method)
            else:
                # For each user we are evaluating the same feature so we pass the first element of set
                new_imputed = synthetic_fill_variables(data, next_qs_lists, s[0], method=method)

        # Compute improvement
        rewards = new_imputed - old_imputed
        rewards[rewards < 0.0] = 0.0
        # Concatenate rewards
        if method == "b_ei":
            # If batch method, we concatenate the rewards of all batch
            if max_steps == feature_count:
                # We only have one batch if feature_count equal the batch size
                rewards_list.append(rewards)
            else:
                rewards_list.append(np.hstack(rewards))
        else:
            rewards_list.append(rewards)

    rewards_list_arr = np.transpose(np.vstack(rewards_list))

    if method != "b_ei":
        # Set reward to nan if feature is already observed
        rewards_list_arr[obs_mask == 1.0] = np.nan

    return rewards_list_arr


def update_data_mask(question_ids, data, mask, imputed):
    # This function is updating the test_data and the test_mask with the values in imputed
    # We loop over the users and then over the features we are collecting for each of them
    data_copy = data.copy()
    mask_copy = mask.copy()

    for i in range(len(question_ids)):
        if len(question_ids[i]) == 0:
            continue
        else:
            for j in range(len(question_ids[i])):
                selected_id = np.asarray(question_ids[i][j]).astype(int)
                data_copy[i, selected_id] = imputed[i, selected_id]
                mask_copy[i, selected_id] = 1.0
    return data_copy, mask_copy


def get_similarity(similarity_metric, model, imputations, data_dir, p=2.0):
    """
    Compute similarity among features based on the chosen metric.
    Args:
        similarity_metric (str): Name of the chosen metric. Options are 'qe' (question embeddings), 'topics', 'sbert' and 'cosine' (cosine difference among imputations)
        model (Model): Trained `Model` class to use
        imputations (numpy array of shape (batch_size, feature_count)): Imputed values for the features
        data_dir (str): Directory to get precomputed metrics from.
        p (float): type of distance to use when similarity_metric is 'sbert'

    Returns:
        distance (numpy array of shape (feature_count, feature_count)): matrix of distances among features
        threshold_value (float): threshold value to use to determine if features are similar or not
    """
    if similarity_metric == "qe":
        question_embeddings_weights = model._set_encoder._embedding_weights.detach()
        question_embeddings_bias = model._set_encoder._embedding_bias.detach()
        question_embeddings = torch.cat((question_embeddings_weights, question_embeddings_bias), 1)
        # Compute the similarity between questions as the euclidian distance between the question embeddings
        distance = torch.cdist(question_embeddings, question_embeddings, p=p)
        threshold_value = torch.mean(distance)

    if similarity_metric == "topics":
        matrix_distances = pd.read_csv(data_dir + "/matrix_distances.csv", delimiter=",", header=None)
        distance = np.asarray(matrix_distances)
        threshold_value = 0.0

    if similarity_metric == "sbert":
        matrix_distances = pd.read_csv(data_dir + "/matrix_sbert.csv", delimiter=",", header=None)
        distance = np.asarray(matrix_distances)
        threshold_value = np.mean(np.mean(matrix_distances))

    if similarity_metric == "cosine":
        imputations = np.transpose(imputations)
        distance = 1.0 - sklearn.metrics.pairwise.cosine_similarity(imputations)
        threshold_value = np.mean(np.mean(distance))

    return distance, threshold_value


def question_similarity_imputation(
    user,
    test_mask,
    test_data,
    imputations,
    distance,
    threshold_value,
    delta,
    aggregate_values,
    data_dir,
    next_qs_lists,
):
    """
    Modify the imputations based on features similarities and on construct if aggregate_values is True.
    Args:
        user (int): Index of the user for which we want to modify the imputations
        test_mask (numpy array of shape (user_count, feature_count)): Mask indicating which entries in data are observed.
        test_data (numpy array of shape (user_count, feature_count)): Data to use for active learning.
        imputations (numpy array of shape (batch_size, feature_count)): Imputed values for the features
        distance (numpy array of shape (feature_count, feature_count)): matrix of distances among features
        threshold_value (float): threshold value to use to determine if features are similar or not
        delta (float): percentage increase in the probability to answer a question correclty. This is used to modify the imputations.
        aggregate_values (bool): Whether to consider question constructs or not.
        data_dir (str): Directory to get precomputed metrics from.
        next_qs_lists (list of lists) : List of observed features for all users in test_data.

    Returns:
        imputations (numpy array of shape (batch_size, feature_count)): Modified imputed values for the features
    """

    index_list = np.arange(test_data.shape[1])
    criteria = distance <= threshold_value
    if aggregate_values:
        question_construct = np.asarray(pd.read_csv(data_dir + "/question_construct.csv", delimiter=",", header=None))

    for i in list(index_list):
        # For each feature
        if aggregate_values and len(next_qs_lists) > 0:
            # Modify depending on constructs asked
            observed_construct = np.unique(question_construct[(next_qs_lists[user]).astype(int), 1])
            selected_construct = question_construct[i, 1]
            if selected_construct in observed_construct:
                imputations[user, i] += delta * (1.0 - imputations[user, i])

        for j in range(test_data.shape[1]):
            # For each feature
            if criteria[i][j] and test_mask[user, j] != 0:
                # Check if distance is small and similar features are observed
                imputations[user, i] += delta * (1.0 - imputations[user, i])

    return imputations


def modify_batch_imputation(
    model,
    data,
    data_mask,
    next_qs_lists,
    previous_imputation,
    combinations,
    X_imputed,
    data_dir,
    imputation_method,
    similarity_metric,
    impute_config,
    delta,
    aggregate_values,
):
    """
    Modify the imputations based on features similarities and on construct if aggregate_values is True.
    Args:
        model (Model): Trained `Model` class to use
        data_mask (numpy array of shape (user_count, feature_count)): Mask indicating which entries in data are observed.
        data (numpy array of shape (user_count, feature_count)): Data to use for active learning.
        next_qs_lists (list of lists) : List of observed features for all users in test_data.
        previous_imputation (numpy array of shape (batch_size, feature_count)): Current imputed values for the features
        combinations (list of list): List of sets of features for all users for which we want to compute the joint imputations.
        X_imputed (numpy array of shape (batch_size, feature_count)): X values when features are collected individually and are determined by the heuristic used.
        data_dir (str): Directory to get precomputed metrics from.
        imputation_method (str): heuristic to use for shifting the imputations
        similarity_metric (str): metric of similarity among features to be used when imputation_method is 'question_similarity'
        impute_config (dict): Impute config dictionary for the model.
        delta (float): percentage increase in the probability to answer a question correclty. This is used to modify the imputations.
        aggregate_values (bool): Whether to consider question constructs or not.


    Returns:
        modified_X_imputed (numpy array of shape (batch_size, feature_count)): Modified imputed values when sampling from the joint distribution of a set of features
    """

    modified_X_imputed = np.tile(X_imputed, (len(combinations), 1, 1))

    for i in range(modified_X_imputed.shape[0]):
        # For each batch
        mask_copy = data_mask.copy()
        data_copy = data.copy()

        combinations_all_users = np.vstack(combinations[i])
        if combinations_all_users.shape[1] > 1:
            # There is more than one feature in the batch
            for s in range(combinations_all_users.shape[1]):
                values = np.asarray(combinations_all_users[:, s]).astype(int)
                if s == 0:
                    # Keep the first imputed value
                    for j in range(values.shape[0]):
                        data_copy[j, values[j]] = X_imputed[j, values[j]]
                        mask_copy[j, values[j]] = 1.0
                if s != 0:
                    # Change probabilities for other features in the set
                    if len(next_qs_lists) != 0:
                        additional_qs = np.asarray(
                            [combinations_all_users[i, :s] for i in range(combinations_all_users.shape[0])]
                        ).astype(int)
                        past_questions = np.concatenate((next_qs_lists, additional_qs), axis=1)
                    else:
                        past_questions = np.asarray(
                            [combinations_all_users[i, :s] for i in range(combinations_all_users.shape[0])]
                        ).astype(int)

                    user_imputations, _, _ = get_imputations(
                        model,
                        data_copy,
                        mask_copy,
                        [past_questions],
                        previous_imputation,
                        imputation_method=imputation_method,
                        similarity_metric=similarity_metric,
                        data_dir=data_dir,
                        impute_config=impute_config,
                        delta=delta,
                        aggregate_values=aggregate_values,
                    )
                    # Assign new value
                    for j in range(values.shape[0]):
                        modified_X_imputed[i, j, values[j]] = user_imputations[j, values[j]]
                        data_copy[j, values[j]] = user_imputations[j, values[j]]
                        mask_copy[j, values[j]] = 1.0
    return modified_X_imputed


def get_imputations(
    model,
    test_data,
    test_mask,
    next_qs_lists,
    previous_imputation,
    imputation_method,
    similarity_metric,
    data_dir,
    impute_config,
    delta,
    aggregate_values,
    sample_count=1,
    sample=False,
):
    torch.manual_seed(1)
    imputations = model.impute(test_data, test_mask, impute_config)
    initial_imputations = imputations.copy()

    assert imputation_method in [
        "predictive",
        "unit",
        "increasing_probability",
        "threshold",
        "question_similarity",
    ], "Heuristic not implemented"
    assert similarity_metric in ["qe", "topics", "sbert", "cosine"]

    if imputation_method == "unit":
        # We expect all answer to be correct and impute all of them to 1
        X_imputed = torch.as_tensor(
            np.full((test_data.shape[0] * sample_count, test_data.shape[1]), fill_value=1.0),
            dtype=torch.float,
        )
    else:
        if imputation_method == "increasing_probability":
            # We increase the p from PVAE by a delta
            for s in range(test_data.shape[0]):
                for i in range(test_data.shape[1]):
                    imputations[s, i] = imputations[s, i] + delta * (1.0 - imputations[s, i])

        if imputation_method == "threshold":
            # If probability to answer correclty is greater than 0.5 we impute to 1.
            # Otherwise we sample from Bernoulli with p from the PVAE.
            for s in range(test_data.shape[0]):
                for i in range(test_data.shape[1]):
                    if imputations[s, i] >= 0.5:
                        imputations[s, i] = 1.0

        if imputation_method == "question_similarity":
            # Compute the distance between questions and the threshold we want to use to assess similarity
            distance, threshold_value = get_similarity(similarity_metric, model, previous_imputation, data_dir)

            if len(next_qs_lists) > 0:
                next_qs_lists = np.hstack(next_qs_lists)
            for s in range(test_data.shape[0]):
                user_imputations = question_similarity_imputation(
                    s,
                    test_mask,
                    test_data,
                    imputations,
                    distance,
                    threshold_value,
                    delta,
                    aggregate_values,
                    data_dir,
                    next_qs_lists,
                )
                imputations[s, :] = user_imputations[s, :]

        if sample is True:
            X_imputed = bernoulli.rvs(imputations)
        else:
            X_imputed = imputations

        if len(X_imputed.shape) == 1:
            X_imputed = X_imputed[np.newaxis]

    return X_imputed, imputations, initial_imputations


def get_question_indices(next_question_ids, combinations_list):
    # Extract question indices from batch index
    list_next_question_ids = []
    for idx, index in enumerate(next_question_ids):
        next_combination = combinations_list[index[0]][idx]
        list_next_question_ids.append(list(next_combination))
    return list_next_question_ids
