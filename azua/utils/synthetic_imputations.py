import numpy as np


def synthetic_fill_variables(data, next_qs_lists, idx, method=None, average=True):
    user_count, feature_count = data.shape

    if idx is not None and method is not None and method not in ["b_ei", "bin", "gls"]:
        next_qs_lists = np.concatenate(((next_qs_lists, np.repeat(idx, user_count)[:, np.newaxis])), axis=1)

    if type(next_qs_lists) is list:
        n_feat_col = np.asarray(next_qs_lists).shape[0]
        next_qs_lists = np.tile(np.asarray(next_qs_lists)[np.newaxis], (user_count, 1))
    else:
        n_feat_col = np.asarray(next_qs_lists).shape[1]

    # Initialise increases in probs
    arr = np.zeros((feature_count, feature_count))
    arr[0] = 0.1
    for i in range(feature_count - 1):
        i = i + 1
        arr[i] = arr[i - 1] + 0.1
    percentage_improv = (np.tile(arr, (user_count, 1, 1))).copy() / 2.0
    imputations = np.zeros((user_count, feature_count))

    for i in range(user_count):
        imputations[i] = np.sum(percentage_improv[i, next_qs_lists[i].astype(int), :], axis=0)

        if n_feat_col == 2:
            if next_qs_lists[i, 1] < next_qs_lists[i, 0]:
                imputations[i] += percentage_improv[i, next_qs_lists[i, 1].astype(int), :]

        if n_feat_col == 3 and method != "b_ei" and method != "k_ei":
            if next_qs_lists[i, 1] < next_qs_lists[i, 0]:
                imputations[i] += percentage_improv[i, next_qs_lists[i, 1].astype(int), :]

            if next_qs_lists[i, 1] < next_qs_lists[i, 2]:
                imputations[i] += percentage_improv[i, next_qs_lists[i, 2].astype(int), :]

        if method in ["b_ei", "bin"]:
            if idx is not None:
                imputations[i] += np.sum(percentage_improv[i, np.asarray(idx[i]).astype(int), :], axis=0)

            if idx is None:
                if set([1, 2]).issubset(next_qs_lists[i]):
                    imputations[i] += 0.2
            else:
                if set([1, 2]).issubset(idx[i]):
                    imputations[i] += 0.2

    if average:
        result = np.mean(imputations, axis=1)
    else:
        result = imputations

    return result
